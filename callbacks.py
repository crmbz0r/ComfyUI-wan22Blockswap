"""Callback functions for WAN 2.2 BlockSwap operations.

This module contains the callback functions that are executed during
model loading and cleanup. These callbacks implement the lazy loading
strategy and proper cleanup of swapped blocks.

The cleanup_callback integrates with BlockSwapModelTracker to make smart
cleanup decisions based on session state, preventing cleanup from deleting
blocks that subsequent loops still need.
"""

import torch
import gc
from typing import Any, Optional, Dict
from tqdm import tqdm

from .block_manager import BlockManager, BlockSwapTracker
from .utils import log_debug, sync_gpu, clear_device_caches
from .model_tracker import BlockSwapModelTracker, CleanupDecision


def lazy_load_callback(
    model_patcher: Any,
    device_to: torch.device,
    lowvram_model_memory: int,
    force_patch_weights: bool,
    full_load: bool,
    blocks_to_swap: int,
    offload_txt_emb: bool,
    offload_img_emb: bool,
    use_non_blocking: bool,
    vace_blocks_to_swap: int,
    prefetch_blocks: int,
    block_swap_debug: bool,
) -> None:
    """
    LAZY LOADING: Load blocks directly to target device.

    This prevents the massive VRAM spike from loading the entire
    model to GPU first. Blocks that should be on CPU are loaded
    directly to CPU, never touching GPU VRAM.

    Args:
        model_patcher: The model patcher instance
        device_to: Target device for model loading
        lowvram_model_memory: Memory limit for low VRAM mode
        force_patch_weights: Whether to force weight patching
        full_load: Whether to perform full model loading
        blocks_to_swap: Number of blocks to swap to CPU
        offload_txt_emb: Whether to offload text embeddings
        offload_img_emb: Whether to offload image embeddings
        use_non_blocking: Whether to use non-blocking transfers
        vace_blocks_to_swap: Number of VACE blocks to swap
        prefetch_blocks: Number of blocks to prefetch
        block_swap_debug: Enable debug logging
    """
    # Initialize block tracking for ON_CLEANUP
    blockswap_tracking = BlockSwapTracker()
    blockswap_tracking.blocks_to_swap = blocks_to_swap

    base_model = model_patcher.model
    main_device = torch.device("cuda")
    offload_device = model_patcher.offload_device

    # Detect GGUF model
    model_type_str = str(type(base_model).__module__)
    blockswap_tracking.is_gguf_model = BlockManager.detect_gguf_model(model_type_str)

    if block_swap_debug:
        if blockswap_tracking.is_gguf_model:
            print("[BlockSwap] GGUF model detected - using LAZY LOADING with safe swapping")
        else:
            print("[BlockSwap] Using LAZY LOADING - blocks loaded directly to target device")
        print("[BlockSwap] This prevents VRAM spikes during loading!")

    # Get UNet model
    unet = BlockManager.get_unet_from_model(base_model)
    if unet is None or not hasattr(unet, "blocks"):
        return

    # Calculate swap indices
    swap_start_idx, actual_blocks_to_swap = BlockManager.calculate_swap_indices(
        len(unet.blocks), blocks_to_swap
    )

    if block_swap_debug:
        print(f"[BlockSwap] Total blocks: {len(unet.blocks)}")
        print(f"[BlockSwap] Keeping on GPU: blocks 0-{swap_start_idx-1}")
        print(f"[BlockSwap] Offloading to CPU: blocks {swap_start_idx}-{len(unet.blocks)-1}")

    # Perform block swapping
    gpu_blocks, cpu_blocks, failed_swaps = BlockManager.swap_transformer_blocks(
        unet, swap_start_idx, len(unet.blocks), offload_device, main_device,
        use_non_blocking, block_swap_debug, blockswap_tracking
    )

    # Offload embeddings
    BlockManager.offload_embeddings(
        unet, offload_device, use_non_blocking, offload_txt_emb, offload_img_emb,
        block_swap_debug, blockswap_tracking
    )

    # Handle VACE blocks
    if vace_blocks_to_swap > 0:
        vace_success = BlockManager.handle_vace_blocks(
            unet, vace_blocks_to_swap, offload_device, main_device,
            use_non_blocking, block_swap_debug
        )

    # Final cleanup
    BlockManager._clear_gpu_cache()

    if block_swap_debug:
        print(f"[BlockSwap] ===== LAZY LOADING COMPLETE =====")
        print(f"[BlockSwap] GPU blocks: {gpu_blocks}")
        print(f"[BlockSwap] CPU blocks: {cpu_blocks}")
        if failed_swaps > 0:
            print(f"[BlockSwap] Failed/Skipped: {failed_swaps} (GGUF quantized)")
        print(f"[BlockSwap] Peak VRAM usage should be MUCH lower!")
        print(f"[BlockSwap] =====================================")

    # Attach tracking data to model_patcher for cleanup callback
    model_patcher.attachments['blockswap_tracking'] = blockswap_tracking
    if block_swap_debug:
        print(f"[BlockSwap] Block tracking attached for ON_CLEANUP callback")
        print(f"[BlockSwap] Tracking {len(blockswap_tracking.swapped_indices)} swapped blocks")


def cleanup_callback(model_patcher: Any) -> None:
    """
    ON_CLEANUP callback for comprehensive cleanup operations.

    Handles both GGUF and native models with appropriate cleanup strategies.
    Integrates with BlockSwapModelTracker for smart cleanup decisions.
    """
    try:
        tracking = model_patcher.attachments.get('blockswap_tracking')
        if tracking is None or tracking.cleanup_executed:
            return

        # Get tracker for smart cleanup decisions
        tracker = BlockSwapModelTracker.get_instance()
        model_id = id(model_patcher.model)

        # Get session_id from tracking if available
        session_id = getattr(tracking, 'session_id', None)

        # If no session, also check if tracker has this model
        if session_id is None:
            session_id = tracker.find_session_for_model(model_id)

        # Check with tracker for cleanup decision
        if session_id:
            decision = tracker.get_cleanup_decision(model_id, session_id)

            if decision == CleanupDecision.SKIP:
                if getattr(tracking, 'block_swap_debug', False):
                    print("[BlockSwap] Cleanup skipped (tracker decision)")
                return

            if decision == CleanupDecision.PRESERVE:
                if getattr(tracking, 'block_swap_debug', False):
                    print("[BlockSwap] Blocks preserved for next loop")
                # Clear tracking state but DON'T move/delete blocks
                _clear_callback_tracking(model_patcher)
                return

        tracking.cleanup_executed = True

        if tracking.block_swap_debug:
            print(f"[BlockSwap] ===== ON_CLEANUP CALLBACK EXECUTING =====")

        # Get model info
        base_model = model_patcher.model
        is_gguf = tracking.is_gguf_model
        successfully_swapped = tracking.successfully_swapped_indices

        if tracking.block_swap_debug:
            print(f"[BlockSwap] Model type: {'GGUF' if is_gguf else 'Native'}")
            print(f"[BlockSwap] Successfully swapped blocks: {len(successfully_swapped)}")

        # Get UNet
        unet = BlockManager.get_unet_from_model(base_model)
        main_device = torch.device("cuda")

        # Phase 1: Synchronize GPU before any block operations
        if tracking.block_swap_debug:
            print("[BlockSwap] Phase 1: Synchronizing GPU...")
        sync_gpu(tracking.block_swap_debug)

        # Phase 2: Handle block cleanup based on model type
        if unet is not None and hasattr(unet, "blocks") and len(successfully_swapped) > 0:
            if is_gguf:
                # GGUF: Move blocks back to GPU (don't delete!)
                if tracking.block_swap_debug:
                    print("[BlockSwap] Phase 2 (GGUF): Moving blocks back to GPU")

                moved_back = 0
                move_failures = 0

                for block_idx in successfully_swapped:
                    try:
                        if block_idx < len(unet.blocks):
                            # Move back to GPU with blocking (non_blocking=False) for safety
                            unet.blocks[block_idx].to(main_device, non_blocking=False)
                            moved_back += 1
                    except Exception as e:
                        move_failures += 1
                        if tracking.block_swap_debug:
                            print(f"[BlockSwap] Block {block_idx} move-back failed: {str(e)[:80]}")

                if tracking.block_swap_debug:
                    print(f"[BlockSwap] GGUF: Moved {moved_back}/{len(successfully_swapped)} blocks back to GPU")
                    if move_failures > 0:
                        print(f"[BlockSwap] GGUF: {move_failures} move-back failures (non-critical)")

                # Clear tracking but DON'T delete references for GGUF
                tracking.swapped_blocks_refs.clear()

            else:
                # NATIVE: Aggressive cleanup (delete references)
                if tracking.block_swap_debug:
                    print("[BlockSwap] Phase 2 (Native): Deleting swapped block references")

                blocks_deleted = 0
                for block_idx in successfully_swapped:
                    try:
                        if block_idx in tracking.swapped_blocks_refs:
                            del tracking.swapped_blocks_refs[block_idx]
                            blocks_deleted += 1
                    except Exception as e:
                        if tracking.block_swap_debug:
                            print(f"[BlockSwap] Failed to delete block {block_idx}: {str(e)[:80]}")

                if tracking.block_swap_debug:
                    print(f"[BlockSwap] Native: Deleted {blocks_deleted}/{len(successfully_swapped)} block references")

                tracking.swapped_blocks_refs.clear()

        # Phase 3: Cleanup embeddings (same for both GGUF and Native)
        if tracking.block_swap_debug:
            print("[BlockSwap] Phase 3: Cleaning up embedding references")

        for emb_name in list(tracking.embeddings_offloaded.keys()):
            try:
                del tracking.embeddings_offloaded[emb_name]
                if tracking.block_swap_debug:
                    print(f"[BlockSwap] Deleted embedding: {emb_name}")
            except Exception as e:
                if tracking.block_swap_debug:
                    print(f"[BlockSwap] Failed to delete embedding {emb_name}: {str(e)[:80]}")

        tracking.embeddings_offloaded.clear()

        # Phase 4: Clear tracking indices
        tracking.swapped_indices.clear()
        tracking.successfully_swapped_indices.clear()
        tracking.failed_to_swap_indices.clear()

        # Phase 5: Garbage collection
        if tracking.block_swap_debug:
            print("[BlockSwap] Phase 5: Running garbage collection")
        gc.collect()
        gc.collect()

        # Phase 6: Clear device caches (AFTER operations complete)
        if tracking.block_swap_debug:
            print("[BlockSwap] Phase 6: Clearing device caches")
        clear_device_caches(tracking.block_swap_debug)

        if tracking.block_swap_debug:
            print("[BlockSwap] ===== ON_CLEANUP COMPLETE =====")
            if is_gguf:
                print("[BlockSwap] GGUF: Blocks moved back to GPU safely (no deletion)")
            else:
                print("[BlockSwap] Native: All swapped block references freed from memory")

        # Mark cleanup done in tracker if session active
        if session_id:
            tracker.mark_cleanup_done(model_id, session_id)

    except Exception as e:
        print(f"[BlockSwap] CRITICAL ERROR in cleanup_callback: {str(e)}")
        import traceback
        traceback.print_exc()


def _clear_callback_tracking(model_patcher: Any) -> None:
    """Clear tracking state for preserve mode.

    This is used when the tracker decides to PRESERVE blocks for subsequent
    loops. We clear the tracking state but don't move or delete blocks.

    Args:
        model_patcher: The model patcher instance
    """
    tracking = model_patcher.attachments.get('blockswap_tracking')
    if tracking is None:
        return

    tracking.cleanup_executed = True
    tracking.swapped_indices.clear()
    tracking.successfully_swapped_indices.clear()
    tracking.failed_to_swap_indices.clear()
    tracking.embeddings_offloaded.clear()
