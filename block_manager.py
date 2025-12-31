"""Block management module for WAN 2.2 BlockSwap operations.

This module handles the core logic for swapping transformer blocks between
GPU and CPU memory to reduce VRAM usage during model loading.
"""

import torch
import gc
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm


class BlockSwapTracker:
    """Tracks block swapping state and cleanup operations.

    This class maintains the state of block swapping operations including
    which blocks were successfully swapped, which failed, and references
    to swapped blocks for proper cleanup.
    """

    def __init__(self):
        self.blocks_to_swap: int = 0  # Number of blocks user wants to swap
        self.swapped_indices: List[int] = []  # All attempted swaps
        self.successfully_swapped_indices: List[int] = []  # Actually moved blocks
        self.failed_to_swap_indices: List[int] = []  # Failed during load
        self.original_device: torch.device = torch.device('cuda')  # Original device
        self.swapped_blocks_refs: Dict[int, Any] = {}  # References to swapped blocks
        self.embeddings_offloaded: Dict[str, Any] = {}  # Offloaded embeddings
        self.cleanup_executed: bool = False  # Whether cleanup was already done
        self.is_gguf_model: bool = False  # Whether model is GGUF quantized
        self.block_swap_debug: bool = False  # Enable debug logging
        self.session_id: Optional[str] = None  # Session ID for tracker


class BlockManager:
    """Manages block swapping operations for WAN models.

    This class contains the core logic for:
    - Detecting model types (GGUF vs native)
    - Calculating which blocks to swap
    - Performing the actual block swapping
    - Handling embeddings and VACE blocks
    - Memory management
    """

    @staticmethod
    def detect_gguf_model(model_type_str: str) -> bool:
        """
        Detect if the model is a GGUF quantized model.

        Args:
            model_type_str: String representation of model type

        Returns:
            bool: True if GGUF model, False otherwise
        """
        return 'gguf' in model_type_str.lower()

    @staticmethod
    def calculate_swap_indices(
        total_blocks: int,
        blocks_to_swap: int
    ) -> Tuple[int, int]:
        """
        Calculate which blocks to swap based on total blocks and swap count.

        Args:
            total_blocks: Total number of transformer blocks
            blocks_to_swap: Number of blocks to swap to CPU

        Returns:
            Tuple of (swap_start_idx, actual_blocks_to_swap)
        """
        actual_blocks_to_swap = min(blocks_to_swap, total_blocks)
        swap_start_idx = total_blocks - actual_blocks_to_swap
        return swap_start_idx, actual_blocks_to_swap

    @staticmethod
    def swap_transformer_blocks(
        unet: Any,
        swap_start_idx: int,
        total_blocks: int,
        offload_device: torch.device,
        main_device: torch.device,
        use_non_blocking: bool,
        block_swap_debug: bool,
        tracking: BlockSwapTracker
    ) -> Tuple[int, int, int]:
        """
        Swap transformer blocks from GPU to CPU.

        Args:
            unet: The UNet model containing the blocks
            swap_start_idx: Index where swapping starts
            total_blocks: Total number of blocks
            offload_device: Target device for swapped blocks
            main_device: Main device (GPU)
            use_non_blocking: Whether to use non-blocking transfers
            block_swap_debug: Enable debug logging
            tracking: BlockSwapTracker instance

        Returns:
            Tuple of (gpu_blocks, cpu_blocks, failed_swaps)
        """
        gpu_blocks = 0
        cpu_blocks = 0
        failed_swaps = 0

        for b, block in tqdm(
            enumerate(unet.blocks),
            total=total_blocks,
            desc="[BlockSwap] Lazy loading blocks",
            disable=not block_swap_debug,
        ):
            try:
                if b < swap_start_idx:
                    # Keep on GPU - load directly to GPU
                    target_device = main_device
                    block.to(target_device, non_blocking=use_non_blocking)
                    gpu_blocks += 1
                else:
                    # Offload to CPU - load directly to CPU
                    target_device = offload_device

                    # For already-loaded blocks, move them
                    # For not-yet-loaded, they'll load to CPU
                    block.to(target_device, non_blocking=use_non_blocking)
                    cpu_blocks += 1

                    # Track swapped block indices and references for cleanup
                    tracking.swapped_indices.append(b)
                    tracking.successfully_swapped_indices.append(b)
                    tracking.swapped_blocks_refs[b] = block

                    # Immediately free GPU memory after each CPU block
                    if b % 5 == 0:  # Every 5 blocks
                        BlockManager._clear_gpu_cache()

            except (RuntimeError, torch.cuda.OutOfMemoryError, Exception) as e:
                failed_swaps += 1
                tracking.failed_to_swap_indices.append(b)
                if block_swap_debug:
                    error_msg = str(e)
                    if "invalid argument" in error_msg.lower():
                        print(f"[BlockSwap] Block {b} - GGUF quantized, skipping")
                    else:
                        print(f"[BlockSwap] Block {b} - Error: {error_msg[:80]}")
                continue

        return gpu_blocks, cpu_blocks, failed_swaps

    @staticmethod
    def offload_embeddings(
        unet: Any,
        offload_device: torch.device,
        use_non_blocking: bool,
        offload_txt_emb: bool,
        offload_img_emb: bool,
        block_swap_debug: bool,
        tracking: BlockSwapTracker
    ) -> None:
        """
        Offload embedding layers to CPU.

        Args:
            unet: The UNet model
            offload_device: Target device for embeddings
            use_non_blocking: Whether to use non-blocking transfers
            offload_txt_emb: Whether to offload text embeddings
            offload_img_emb: Whether to offload image embeddings
            block_swap_debug: Enable debug logging
            tracking: BlockSwapTracker instance
        """
        # Offload text embedding
        if offload_txt_emb and hasattr(unet, "text_embedding"):
            try:
                unet.text_embedding.to(offload_device, non_blocking=use_non_blocking)
                BlockManager._clear_gpu_cache()
                tracking.embeddings_offloaded['text_embedding'] = unet.text_embedding
                if block_swap_debug:
                    print("[BlockSwap] Offloaded text_embedding to CPU")
            except Exception as e:
                if block_swap_debug:
                    print(f"[BlockSwap] Failed to offload text_embedding: {e}")

        # Offload image embedding
        if offload_img_emb and hasattr(unet, "img_emb"):
            try:
                unet.img_emb.to(offload_device, non_blocking=use_non_blocking)
                BlockManager._clear_gpu_cache()
                tracking.embeddings_offloaded['img_emb'] = unet.img_emb
                if block_swap_debug:
                    print("[BlockSwap] Offloaded img_emb to CPU")
            except Exception as e:
                if block_swap_debug:
                    print(f"[BlockSwap] Failed to offload img_emb: {e}")

    @staticmethod
    def handle_vace_blocks(
        unet: Any,
        vace_blocks_to_swap: int,
        offload_device: torch.device,
        main_device: torch.device,
        use_non_blocking: bool,
        block_swap_debug: bool,
    ) -> int:
        """
        Handle VACE model block swapping.

        Args:
            unet: The UNet model
            vace_blocks_to_swap: Number of VACE blocks to swap
            offload_device: Target device for VACE blocks
            main_device: Main device (GPU)
            use_non_blocking: Whether to use non-blocking transfers
            block_swap_debug: Enable debug logging

        Returns:
            Number of successfully swapped VACE blocks
        """
        if vace_blocks_to_swap <= 0 or not hasattr(unet, "vace_blocks"):
            return 0

        actual_vace_swap = min(vace_blocks_to_swap, len(unet.vace_blocks))
        vace_swap_idx = len(unet.vace_blocks) - actual_vace_swap
        vace_success = 0

        for b, block in enumerate(unet.vace_blocks):
            try:
                if b < vace_swap_idx:
                    block.to(main_device, non_blocking=use_non_blocking)
                else:
                    block.to(offload_device, non_blocking=use_non_blocking)
                    vace_success += 1
                    if b % 3 == 0:
                        BlockManager._clear_gpu_cache()
            except Exception:
                if block_swap_debug:
                    print(f"[BlockSwap] VACE block {b} failed - skipping")
                continue

        if block_swap_debug:
            print(f"[BlockSwap] VACE: {vace_success}/{actual_vace_swap} blocks offloaded")

        return vace_success

    @staticmethod
    def _clear_gpu_cache() -> None:
        """Clear GPU cache and run garbage collection."""
        import comfy.model_management as mm
        mm.soft_empty_cache()
        gc.collect()

    @staticmethod
    def get_unet_from_model(base_model: Any) -> Optional[Any]:
        """
        Extract UNet from the base model.

        Args:
            base_model: The base model (WAN21 or other)

        Returns:
            UNet model or None if not found
        """
        from comfy.model_base import WAN21

        unet = None
        if isinstance(base_model, WAN21):
            unet = base_model.diffusion_model
        elif hasattr(base_model, "diffusion_model"):
            unet = base_model.diffusion_model

        return unet
