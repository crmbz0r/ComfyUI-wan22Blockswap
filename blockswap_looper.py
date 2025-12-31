"""Specialized looper node for WAN 2.2 BlockSwap integration with WanVideoLooper.

This module provides a loop-aware BlockSwap node that integrates with WanVideoLooper's
multi-loop architecture. It addresses the 5 root causes of degraded output quality on
subsequent loops by implementing proper state management, cleanup, and tensor validation.

Key Features:
- Loop-aware model preparation with fresh callback registration
- Between-loop cleanup with explicit block restoration
- Tensor consistency validation for color matching chains
- Compatibility with WanVideoLoraSequencer for per-segment variations
- Comprehensive error handling and debug logging

Root Causes Addressed:
1. Model state pollution across loops
2. Callback double-execution and cleanup flag issues
3. Block state leakage between iterations
4. Tensor device/dtype misalignment
5. Embeddings persistence
"""

import torch
import gc
from typing import Any, Dict, List, Tuple, Optional

import comfy.model_management as mm
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import ModelPatcher

from .config import BlockSwapConfig
from .block_manager import BlockSwapTracker, BlockManager
from .callbacks import lazy_load_callback, cleanup_callback
from .looper_helpers import (
    prepare_model_for_loop,
    cleanup_loop_blockswap,
    validate_tensor_consistency,
    reset_model_blockswap_state,
    start_blockswap_session,
    end_blockswap_session,
    update_session_loop_state,
)
from .model_tracker import BlockSwapModelTracker, CleanupMode, CleanupDecision


class WAN22BlockSwapLooper:
    """
    Specialized looper node that integrates BlockSwap with WanVideoLooper's multi-loop architecture.

    This node manages BlockSwap state per loop iteration, preventing model state pollution,
    callback state leakage, and tensor misalignment across subsequent video generation loops.

    The implementation creates fresh BlockSwap state for each loop iteration and ensures
    proper cleanup between iterations, addressing the 5 identified root causes of degraded
    output quality on subsequent loops.

    Integration Points:
    - Works with WanVideoLooperPrompts (no changes needed)
    - Works with WanVideoLoraSequencer for per-segment model variations
    - Returns compatible model format for WanVideoLooper
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Get input types for the WAN22BlockSwapLooper node."""
        return {
            "required": {
                "models_list": (
                    "ANY",
                    {
                        "tooltip": "List or tuple of models from WanVideoLooper or WanVideoLoraSequencer. "
                        "Can be a single model list, or a list of (model_high, model_low, clip) tuples "
                        "from WanVideoLoraSequencer."
                    },
                ),
                "blocks_to_swap": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 48,
                        "step": 1,
                        "tooltip": "Number of transformer blocks to swap to CPU per loop iteration. "
                        "1.3B/5B models: 30 blocks, 14B model: 40 blocks, LongCat: 48 blocks",
                    },
                ),
                "offload_txt_emb": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Offload text_embedding to CPU. "
                        "Reduces VRAM by ~500MB but may impact performance",
                    },
                ),
                "offload_img_emb": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Offload img_emb to CPU (I2V models only). "
                        "Reduces VRAM by ~200MB but may impact performance",
                    },
                ),
            },
            "optional": {
                "use_non_blocking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use non-blocking memory transfers. "
                        "Faster but reserves more RAM (~5-10% additional)",
                    },
                ),
                "vace_blocks_to_swap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 15,
                        "step": 1,
                        "tooltip": "VACE model blocks to swap (0 = auto, "
                        "1-15 = specific count). "
                        "VACE model has 15 blocks total",
                    },
                ),
                "prefetch_blocks": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 40,
                        "step": 1,
                        "tooltip": "Prefetch N blocks ahead for performance. "
                        "Value of 1 usually sufficient to offset "
                        "speed loss from swapping. "
                        "Use debug mode to find optimal value",
                    },
                ),
                "block_swap_debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable debug logging to monitor block swap "
                        "performance and memory usage across loops",
                    },
                ),
            },
        }

    RETURN_TYPES: tuple = ("ANY",)
    RETURN_NAMES: tuple = ("prepared_models",)
    CATEGORY: str = "ComfyUI-wan22Blockswap/looper"
    FUNCTION: str = "prepare_looper_models"
    DESCRIPTION: str = (
        "Prepare models for WanVideoLooper with loop-aware BlockSwap integration. "
        "This node integrates BlockSwap with WanVideoLooper's multi-loop architecture, "
        "preventing model state pollution, callback state leakage, and tensor "
        "misalignment across subsequent video generation loops. "
        "Ensures consistent output quality across all loops by managing BlockSwap "
        "state per iteration with proper cleanup and validation."
    )

    def prepare_looper_models(
        self,
        models_list: Any,
        blocks_to_swap: int,
        offload_txt_emb: bool,
        offload_img_emb: bool,
        use_non_blocking: bool = False,
        vace_blocks_to_swap: int = 0,
        prefetch_blocks: int = 0,
        block_swap_debug: bool = False,
    ) -> Tuple[Any]:
        """
        Prepare models for WanVideoLooper with loop-aware BlockSwap configuration.

        This function processes the input models list and applies loop-aware BlockSwap
        configuration to each model, ensuring proper state isolation between loop iterations.
        It creates a session to track models across loops and prevent premature cleanup.

        Args:
            models_list (Any): List or tuple of models from WanVideoLooper or
                WanVideoLoraSequencer. Can be a single model list, or a list of
                (model_high, model_low, clip) tuples from WanVideoLoraSequencer.
            blocks_to_swap (int): Number of transformer blocks to swap to CPU
            offload_txt_emb (bool): Whether to offload text embeddings to CPU
            offload_img_emb (bool): Whether to offload image embeddings (I2V)
            use_non_blocking (bool): Use non-blocking transfers for speed
            vace_blocks_to_swap (int): VACE blocks to swap (0=auto detection)
            prefetch_blocks (int): Blocks to prefetch ahead for pipeline
            block_swap_debug (bool): Enable performance monitoring

        Returns:
            Tuple containing a dict with 'models' list and 'session_id' for downstream use

        Raises:
            ValueError: If models_list is not a valid iterable or is empty
            TypeError: If models_list contains invalid model types
        """
        # Input validation
        if not hasattr(models_list, '__iter__') or isinstance(models_list, (str, bytes)):
            raise ValueError("models_list must be a list, tuple, or other iterable of models")

        if not models_list:
            raise ValueError("models_list cannot be empty")

        # Determine expected loop count based on input structure
        expected_loops = len(models_list)

        if block_swap_debug:
            print(f"[BlockSwap] ===== WAN22BlockSwapLooper: Preparing {expected_loops} models =====")

        # Start a tracking session for this looper workflow
        session_id = start_blockswap_session(expected_loops, block_swap_debug)

        if block_swap_debug:
            print(f"[BlockSwap] Session started: {session_id}")

        # Process models based on input type
        prepared_models = []

        try:
            for i, model_item in enumerate(models_list):
                if block_swap_debug:
                    print(f"[BlockSwap] Processing model {i+1}/{len(models_list)}")

                # Handle different input formats
                if isinstance(model_item, (list, tuple)) and len(model_item) == 3:
                    # WanVideoLoraSequencer format: (model_high, model_low, clip)
                    model_high, model_low, clip = model_item

                    # Prepare high-noise model with session tracking
                    prepared_high = prepare_model_for_loop(
                        model_high, i, blocks_to_swap, offload_txt_emb, offload_img_emb,
                        use_non_blocking, vace_blocks_to_swap, prefetch_blocks, block_swap_debug,
                        session_id=session_id
                    )

                    # Prepare low-noise model with session tracking
                    prepared_low = prepare_model_for_loop(
                        model_low, i, blocks_to_swap, offload_txt_emb, offload_img_emb,
                        use_non_blocking, vace_blocks_to_swap, prefetch_blocks, block_swap_debug,
                        session_id=session_id
                    )

                    prepared_models.append((prepared_high, prepared_low, clip))

                    if block_swap_debug:
                        print(f"[BlockSwap] Looper: Prepared segment {i+1} with high/low models")

                else:
                    # Single model format (from WanVideoLooper)
                    prepared_model = prepare_model_for_loop(
                        model_item, i, blocks_to_swap, offload_txt_emb, offload_img_emb,
                        use_non_blocking, vace_blocks_to_swap, prefetch_blocks, block_swap_debug,
                        session_id=session_id
                    )
                    prepared_models.append(prepared_model)

                    if block_swap_debug:
                        print(f"[BlockSwap] Looper: Prepared model {i+1}")

            if block_swap_debug:
                print(f"[BlockSwap] ===== WAN22BlockSwapLooper: Preparation complete =====")
                tracker = BlockSwapModelTracker.get_instance()
                stats = tracker.get_session_stats(session_id)
                print(f"[BlockSwap] Session stats: {stats}")

        except Exception as e:
            # Clean up session on error
            if block_swap_debug:
                print(f"[BlockSwap] Error during preparation: {e}")
            end_blockswap_session(session_id, block_swap_debug)
            raise

        # Return both models and session_id for downstream use
        return ({
            "models": prepared_models,
            "session_id": session_id,
            "expected_loops": expected_loops,
        },)


# Node registration
NODE_CLASS_MAPPINGS = {
    "wan22BlockSwapLooper": WAN22BlockSwapLooper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "wan22BlockSwapLooper": "WAN 2.2 BlockSwap Looper (Loop-Aware)",
}
