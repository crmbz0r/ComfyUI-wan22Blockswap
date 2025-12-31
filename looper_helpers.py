"""Helper functions for WAN 2.2 BlockSwap looper operations.

This module provides reusable helper functions for loop-aware model preparation,
cleanup, and tensor validation. These functions implement the core logic for
managing BlockSwap state across multiple loop iterations in WanVideoLooper.

Key Functions:
- prepare_model_for_loop: Prepare a model for a specific loop iteration
- create_fresh_blockswap_tracker: Factory for fresh tracking instances
- cleanup_loop_blockswap: Comprehensive cleanup between loop iterations
- validate_tensor_consistency: Ensure tensor device/dtype alignment
- reset_model_blockswap_state: Clear persistent tracking state
- start_blockswap_session: Start a new BlockSwap session for multi-loop workflows
- end_blockswap_session: End a BlockSwap session and perform final cleanup
- update_session_loop_state: Update session loop state

These helpers address the 5 root causes of loop degradation by ensuring proper
state isolation, cleanup, and validation between iterations.
"""

import torch
import gc
from typing import Any, Dict, List, Tuple, Optional

import comfy.model_management as mm
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import ModelPatcher

from .block_manager import BlockSwapTracker, BlockManager
from .callbacks import lazy_load_callback, cleanup_callback
from .utils import log_debug, sync_gpu, clear_device_caches
from .model_tracker import (
    BlockSwapModelTracker,
    CleanupMode,
    CleanupDecision,
)


def prepare_model_for_loop(
    model: ModelPatcher,
    loop_index: int,
    blocks_to_swap: int,
    offload_txt_emb: bool,
    offload_img_emb: bool,
    use_non_blocking: bool,
    vace_blocks_to_swap: int,
    prefetch_blocks: int,
    block_swap_debug: bool = False,
    session_id: Optional[str] = None,
) -> ModelPatcher:
    """
    Prepare a model for a specific loop iteration with fresh BlockSwap configuration.

    This function clones the input model and registers fresh callbacks with a new
    BlockSwapTracker, ensuring clean state entry for each loop iteration. This
    prevents model state pollution and callback double-execution issues.

    When a session_id is provided, the function integrates with BlockSwapModelTracker
    to enable smart cleanup decisions across loop iterations.

    Args:
        model (ModelPatcher): The input model to prepare for the loop
        loop_index (int): The index of the current loop iteration (0-based)
        blocks_to_swap (int): Number of transformer blocks to swap to CPU
        offload_txt_emb (bool): Whether to offload text embeddings to CPU
        offload_img_emb (bool): Whether to offload image embeddings (I2V)
        use_non_blocking (bool): Use non-blocking transfers for speed
        vace_blocks_to_swap (int): VACE blocks to swap (0=auto detection)
        prefetch_blocks (int): Blocks to prefetch ahead for pipeline
        block_swap_debug (bool): Enable debug logging
        session_id (Optional[str]): Session ID for smart cleanup tracking

    Returns:
        ModelPatcher: A cloned model with fresh BlockSwap configuration

    Raises:
        ValueError: If model is None or blocks_to_swap is invalid
    """
    # Input validation
    if model is None:
        raise ValueError("Model cannot be None")

    if blocks_to_swap < 0:
        raise ValueError("blocks_to_swap must be non-negative")

    # Get model tracker for smart cleanup
    tracker_singleton = BlockSwapModelTracker.get_instance()
    model_id = id(model.model)

    # Get patches UUID for change detection
    patches_uuid = getattr(model, 'patches_uuid', None)
    if patches_uuid is not None:
        patches_uuid = str(patches_uuid)

    # Check if we should skip preparation (already prepared with same config)
    if session_id and tracker_singleton.should_skip_preparation(
        model, loop_index, session_id
    ):
        if block_swap_debug:
            print(f"[BlockSwap] Loop {loop_index+1}: Model already prepared, skipping")
        # Increment reference count for this loop
        try:
            tracker_singleton.increment_ref(model)
        except (KeyError, ValueError, AttributeError):
            pass  # Model may not be registered yet
        return model  # Return original, not clone

    if block_swap_debug:
        print(f"[BlockSwap] Loop {loop_index+1}: Preparing model with {blocks_to_swap} blocks to swap")

    # Clone the model to ensure state isolation
    model_copy = model.clone()

    if model_copy is None:
        raise RuntimeError("Failed to clone model")

    if block_swap_debug:
        print(f"[BlockSwap] Loop {loop_index+1}: Model cloned successfully")

    # Register model with tracker if session provided
    if session_id:
        try:
            tracker_singleton.register_model(
                model=model_copy,
                loop_index=loop_index,
                session_id=session_id,
                blocks_to_swap=blocks_to_swap,
            )
            if block_swap_debug:
                print(f"[BlockSwap] Loop {loop_index+1}: Model registered with tracker")
        except ValueError as e:
            if block_swap_debug:
                print(f"[BlockSwap] Loop {loop_index+1}: Tracker registration warning: {e}")

    # Create a fresh BlockSwapTracker for this loop iteration
    tracker = create_fresh_blockswap_tracker(blocks_to_swap, block_swap_debug)

    # Store session_id in tracker for cleanup callback access
    tracker.session_id = session_id

    # Attach the fresh tracker to the model
    model_copy.attachments['blockswap_tracking'] = tracker

    if block_swap_debug:
        print(f"[BlockSwap] Loop {loop_index+1}: Fresh BlockSwapTracker attached")

    # Create callback wrapper with all parameters bound for this loop
    def lazy_load_callback_wrapper(
        model_patcher: ModelPatcher,
        device_to: torch.device,
        lowvram_model_memory: int,
        force_patch_weights: bool,
        full_load: bool
    ) -> None:
        """Wrapper for lazy load callback with loop-specific parameters."""
        lazy_load_callback(
            model_patcher=model_patcher,
            device_to=device_to,
            lowvram_model_memory=lowvram_model_memory,
            force_patch_weights=force_patch_weights,
            full_load=full_load,
            blocks_to_swap=blocks_to_swap,
            offload_txt_emb=offload_txt_emb,
            offload_img_emb=offload_img_emb,
            use_non_blocking=use_non_blocking,
            vace_blocks_to_swap=vace_blocks_to_swap,
            prefetch_blocks=prefetch_blocks,
            block_swap_debug=block_swap_debug,
        )

    # Register both ON_LOAD and ON_CLEANUP callbacks with fresh state
    model_copy.add_callback(CallbacksMP.ON_LOAD, lazy_load_callback_wrapper)
    model_copy.add_callback(CallbacksMP.ON_CLEANUP, cleanup_callback)

    if block_swap_debug:
        print(f"[BlockSwap] Loop {loop_index+1}: Both ON_LOAD and ON_CLEANUP callbacks registered")

    return model_copy


def create_fresh_blockswap_tracker(
    blocks_to_swap: int,
    block_swap_debug: bool = False,
) -> BlockSwapTracker:
    """
    Create a fresh BlockSwapTracker instance configured for a new loop iteration.

    This factory function ensures that each loop iteration starts with a clean
    tracking state, preventing state leakage between iterations.

    Args:
        blocks_to_swap (int): Number of transformer blocks to swap to CPU
        block_swap_debug (bool): Enable debug logging

    Returns:
        BlockSwapTracker: A fresh tracker instance with all state initialized
    """
    # Create new BlockSwapTracker instance
    tracker = BlockSwapTracker()

    # Configure the tracker for this loop
    tracker.blocks_to_swap = blocks_to_swap
    tracker.block_swap_debug = block_swap_debug

    # Initialize all collections as empty
    tracker.swapped_indices = []
    tracker.successfully_swapped_indices = []
    tracker.failed_to_swap_indices = []
    tracker.swapped_blocks_refs = {}
    tracker.embeddings_offloaded = {}

    # Reset all state flags
    tracker.cleanup_executed = False
    tracker.original_device = torch.device('cuda')
    tracker.is_gguf_model = False
    tracker.blocks_restored = False  # For pre-unpatch restoration

    return tracker


def cleanup_loop_blockswap(
    model: ModelPatcher,
    loop_index: int,
    block_swap_debug: bool = False,
    session_id: Optional[str] = None,
) -> None:
    """
    Execute comprehensive cleanup between loop iterations with explicit block restoration.

    This function performs all necessary cleanup operations between loop iterations,
    ensuring that blocks are properly restored to GPU and all state is cleared for
    the next iteration. This prevents block state leakage and tensor misalignment.

    When a session_id is provided, the function integrates with BlockSwapModelTracker
    to make smart cleanup decisions based on session state.

    Args:
        model (ModelPatcher): The model to clean up
        loop_index (int): The index of the current loop (for logging)
        block_swap_debug (bool): Enable debug logging
        session_id (Optional[str]): Session ID for smart cleanup decisions

    Raises:
        RuntimeError: If cleanup fails critically
    """
    if model is None:
        return

    try:
        # Get tracker for smart cleanup decisions
        tracker_singleton = BlockSwapModelTracker.get_instance()

        # Get session_id from tracking if not provided
        tracking = model.attachments.get('blockswap_tracking')
        if session_id is None and tracking is not None:
            session_id = getattr(tracking, 'session_id', None)

        # Check with tracker for cleanup decision
        if session_id:
            decision = tracker_singleton.get_cleanup_decision(model)

            if decision == CleanupDecision.SKIP:
                if block_swap_debug:
                    print(f"[BlockSwap] Loop {loop_index+1}: Cleanup skipped (tracker decision)")
                return

            if decision == CleanupDecision.PRESERVE:
                if block_swap_debug:
                    print(f"[BlockSwap] Loop {loop_index+1}: Blocks preserved for next loop")
                # Clear tracking state but DON'T move/delete blocks
                _clear_tracking_state_only(model, loop_index, block_swap_debug)
                return

        # Continue with normal cleanup for FULL decision or no session
        if tracking is None or tracking.cleanup_executed:
            if block_swap_debug:
                print(f"[BlockSwap] Loop {loop_index+1}: No tracking data or cleanup already executed")
            return

        tracking.cleanup_executed = True

        if block_swap_debug:
            print(f"[BlockSwap] ===== Loop {loop_index+1}: ON_CLEANUP CALLBACK EXECUTING =====")

        # Get model info
        base_model = model.model
        is_gguf = tracking.is_gguf_model
        successfully_swapped = tracking.successfully_swapped_indices

        if block_swap_debug:
            print(f"[BlockSwap] Loop {loop_index+1}: Model type: {'GGUF' if is_gguf else 'Native'}")
            print(f"[BlockSwap] Loop {loop_index+1}: Successfully swapped blocks: {len(successfully_swapped)}")

        # Get UNet for block operations
        unet = BlockManager.get_unet_from_model(base_model)
        main_device = torch.device("cuda")

        # Phase 1: Synchronize GPU before any block operations
        if block_swap_debug:
            print(f"[BlockSwap] Loop {loop_index+1}: Phase 1: Synchronizing GPU...")
        sync_gpu(block_swap_debug)

        # Phase 2: Handle block cleanup based on model type
        if unet is not None and hasattr(unet, "blocks") and len(successfully_swapped) > 0:
            if is_gguf:
                # GGUF: Move blocks back to GPU (don't delete references!)
                if block_swap_debug:
                    print(f"[BlockSwap] Loop {loop_index+1}: Phase 2 (GGUF): Moving blocks back to GPU")

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
                        if block_swap_debug:
                            print(f"[BlockSwap] Loop {loop_index+1}: Block {block_idx} move-back failed: {str(e)[:80]}")

                if block_swap_debug:
                    print(f"[BlockSwap] Loop {loop_index+1}: GGUF: Moved {moved_back}/{len(successfully_swapped)} blocks back to GPU")
                    if move_failures > 0:
                        print(f"[BlockSwap] Loop {loop_index+1}: GGUF: {move_failures} move-back failures (non-critical)")

                # Clear tracking but DON'T delete references for GGUF
                tracking.swapped_blocks_refs.clear()

            else:
                # NATIVE: Aggressive cleanup (delete references)
                if block_swap_debug:
                    print(f"[BlockSwap] Loop {loop_index+1}: Phase 2 (Native): Deleting swapped block references")

                blocks_deleted = 0
                for block_idx in successfully_swapped:
                    try:
                        if block_idx in tracking.swapped_blocks_refs:
                            del tracking.swapped_blocks_refs[block_idx]
                            blocks_deleted += 1
                    except Exception as e:
                        if block_swap_debug:
                            print(f"[BlockSwap] Loop {loop_index+1}: Failed to delete block {block_idx}: {str(e)[:80]}")

                if block_swap_debug:
                    print(f"[BlockSwap] Loop {loop_index+1}: Native: Deleted {blocks_deleted}/{len(successfully_swapped)} block references")

                tracking.swapped_blocks_refs.clear()

        # Phase 3: Cleanup embeddings (same for both GGUF and Native)
        if block_swap_debug:
            print(f"[BlockSwap] Loop {loop_index+1}: Phase 3: Cleaning up embedding references")

        for emb_name in list(tracking.embeddings_offloaded.keys()):
            try:
                del tracking.embeddings_offloaded[emb_name]
                if block_swap_debug:
                    print(f"[BlockSwap] Loop {loop_index+1}: Deleted embedding: {emb_name}")
            except Exception as e:
                if block_swap_debug:
                    print(f"[BlockSwap] Loop {loop_index+1}: Failed to delete embedding {emb_name}: {str(e)[:80]}")

        tracking.embeddings_offloaded.clear()

        # Phase 4: Clear tracking indices
        tracking.swapped_indices.clear()
        tracking.successfully_swapped_indices.clear()
        tracking.failed_to_swap_indices.clear()

        # Phase 5: Garbage collection
        if block_swap_debug:
            print(f"[BlockSwap] Loop {loop_index+1}: Phase 5: Running garbage collection")
        gc.collect()
        gc.collect()

        # Phase 6: Clear device caches (AFTER operations complete)
        if block_swap_debug:
            print(f"[BlockSwap] Loop {loop_index+1}: Phase 6: Clearing device caches")
        clear_device_caches(block_swap_debug)

        if block_swap_debug:
            print(f"[BlockSwap] Loop {loop_index+1}: ===== ON-CLEANUP COMPLETE =====")
            if is_gguf:
                print(f"[BlockSwap] Loop {loop_index+1}: GGUF: Blocks moved back to GPU safely (no deletion)")
            else:
                print(f"[BlockSwap] Loop {loop_index+1}: Native: All swapped block references freed from memory")

    except Exception as e:
        print(f"[BlockSwap] CRITICAL ERROR in cleanup_loop_blockswap: {str(e)}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Critical error during loop cleanup: {str(e)}")


def validate_tensor_consistency(
    latent: Dict[str, torch.Tensor],
    color_match_ref: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    block_swap_debug: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Ensure working tensors stay on consistent devices across loop iterations.

    This function validates and corrects device/dtype alignment for tensors used
    in color matching operations, preventing tensor misalignment issues that cause
    degraded output quality in subsequent loops.

    Args:
        latent (Dict[str, torch.Tensor]): The latent tensor dictionary
        color_match_ref (Optional[torch.Tensor]): Color matching reference tensor
        device (torch.device): Target device for consistency
        dtype (torch.dtype): Target dtype for consistency
        block_swap_debug (bool): Enable debug logging

    Returns:
        Tuple containing corrected latent and color_match_ref tensors

    Raises:
        ValueError: If latent tensor is invalid
    """
    # Validate latent input
    if not isinstance(latent, dict) or 'samples' not in latent:
        raise ValueError("Latent must be a dictionary with 'samples' key")

    latent_tensor = latent['samples']

    if not isinstance(latent_tensor, torch.Tensor):
        raise ValueError("Latent samples must be a torch.Tensor")

    # Validate latent tensor
    current_device = latent_tensor.device
    current_dtype = latent_tensor.dtype

    if current_device != device or current_dtype != dtype:
        if block_swap_debug:
            print(f"[BlockSwap] Tensor consistency: Moving latent from {current_device}/{current_dtype} to {device}/{dtype}")

        latent_tensor = latent_tensor.to(device=device, dtype=dtype, non_blocking=True)
        latent = {'samples': latent_tensor}

    # Validate color reference tensor
    if color_match_ref is not None:
        current_ref_device = color_match_ref.device
        current_ref_dtype = color_match_ref.dtype

        if current_ref_device != device or current_ref_dtype != dtype:
            if block_swap_debug:
                print(f"[BlockSwap] Tensor consistency: Moving color ref from {current_ref_device}/{current_ref_dtype} to {device}/{dtype}")

            color_match_ref = color_match_ref.to(device=device, dtype=dtype, non_blocking=True)

    # Cross-validate both tensors are on same device/dtype
    consistency_ok = True

    if color_match_ref is not None:
        if latent_tensor.device != color_match_ref.device:
            consistency_ok = False
            if block_swap_debug:
                print(f"[BlockSwap] Tensor consistency WARNING: Latent on {latent_tensor.device}, color ref on {color_match_ref.device}")

        if latent_tensor.dtype != color_match_ref.dtype:
            consistency_ok = False
            if block_swap_debug:
                print(f"[BlockSwap] Tensor consistency WARNING: Latent dtype {latent_tensor.dtype}, color ref dtype {color_match_ref.dtype}")

    if block_swap_debug and consistency_ok:
        print(f"[BlockSwap] Tensor consistency: All tensors aligned on {device}/{dtype}")

    return latent, color_match_ref


def reset_model_blockswap_state(model: ModelPatcher) -> None:
    """
    Clear all BlockSwap tracking state from model attachments.

    This function ensures that all BlockSwap state is removed from the model,
    preparing it for the next iteration with a completely clean state. This
    prevents state pollution between loop iterations.

    Args:
        model (ModelPatcher): The model to reset
    """
    if model is None:
        return

    # Safety check for attachments dict
    if not hasattr(model, 'attachments') or model.attachments is None:
        return

    # Clear BlockSwap tracking if present
    if 'blockswap_tracking' in model.attachments:
        tracking = model.attachments['blockswap_tracking']

        if hasattr(tracking, 'cleanup_executed'):
            tracking.cleanup_executed = True

        # Remove from attachments
        del model.attachments['blockswap_tracking']

    # Clear any callback state
    if hasattr(model, '_callbacks'):
        model._callbacks.clear()

    # Force garbage collection to ensure cleanup
    gc.collect()


# =============================================================================
# Private Helper Functions
# =============================================================================


def _clear_tracking_state_only(
    model: ModelPatcher,
    loop_index: int,
    block_swap_debug: bool = False,
) -> None:
    """Clear tracking state without moving or deleting blocks.

    This is used for PRESERVE mode where we want to clear the tracking
    state but keep the blocks in their current locations for the next
    loop iteration.

    Args:
        model: The model to clear tracking state for
        loop_index: Current loop index (for logging)
        block_swap_debug: Enable debug logging
    """
    tracking = model.attachments.get('blockswap_tracking')
    if tracking is None:
        return

    # Clear indices but preserve blocks
    tracking.swapped_indices.clear()
    tracking.successfully_swapped_indices.clear()
    tracking.failed_to_swap_indices.clear()

    # Clear embedding tracking
    tracking.embeddings_offloaded.clear()

    # Mark as executed to prevent double-run
    tracking.cleanup_executed = True

    if block_swap_debug:
        print(f"[BlockSwap] Loop {loop_index+1}: Tracking state cleared (blocks preserved)")


# =============================================================================
# Session Management Helper Functions
# =============================================================================


def start_blockswap_session(
    loop_count: int = 1,
    cleanup_mode: CleanupMode = CleanupMode.SMART,
    block_swap_debug: bool = False,
) -> str:
    """Start a new BlockSwap session for multi-loop workflows.

    Creates a session in the global tracker that will manage model
    preparation and cleanup across multiple loop iterations.

    Args:
        loop_count: Expected number of loops in this workflow
        cleanup_mode: Default cleanup mode for the session
        block_swap_debug: Enable debug logging

    Returns:
        str: Session ID for use in subsequent calls
    """
    tracker = BlockSwapModelTracker.get_instance()
    tracker.set_debug(block_swap_debug)

    session_id = tracker.start_session(expected_loops=loop_count)

    if block_swap_debug:
        print(f"[BlockSwap] Started session {session_id} for {loop_count} loops")

    return session_id


def end_blockswap_session(
    session_id: str,
    block_swap_debug: bool = False,
) -> None:
    """End a BlockSwap session and perform final cleanup.

    Ends the session and cleans up all associated model states.
    This should be called after all loops have completed.

    Args:
        session_id: Session ID to end
        block_swap_debug: Enable debug logging
    """
    tracker = BlockSwapModelTracker.get_instance()

    if block_swap_debug:
        print(f"[BlockSwap] Ending session {session_id}")

    tracker.end_session(session_id)

    # Clean up any expired sessions while we're at it
    expired = tracker.cleanup_expired_sessions()
    if expired > 0 and block_swap_debug:
        print(f"[BlockSwap] Cleaned up {expired} expired sessions")


def update_session_loop_state(
    session_id: str,
    loop_index: int,
    completed: bool = False,
    block_swap_debug: bool = False,
) -> None:
    """Update session loop state.

    Call this at the start of each loop to update the current loop index,
    and at the end of each loop with completed=True to mark it done.

    Args:
        session_id: Session ID to update
        loop_index: Current loop index (0-based)
        completed: Whether this loop just completed
        block_swap_debug: Enable debug logging
    """
    tracker = BlockSwapModelTracker.get_instance()

    # Use the available method - update_loop_progress
    tracker.update_loop_progress(session_id, loop_index)
    
    if completed:
        # Mark session as complete if this is marked as the final loop
        tracker.mark_session_complete(session_id)
        if block_swap_debug:
            print(f"[BlockSwap] Session {session_id} completed loop {loop_index + 1}")
