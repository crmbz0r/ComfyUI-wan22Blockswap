"""Specialized looper nodes for WAN 2.2 BlockSwap integration with WanVideoLooper.

This module provides loop-aware BlockSwap nodes that integrate with WanVideoLooper's
multi-loop architecture. It addresses issues with model state pollution across
loops by implementing proper state management, cleanup, and tensor validation.

Key Features:
- Loop-aware model preparation with fresh callback registration
- Between-loop cleanup with explicit block restoration
- Compatibility with WanVideoLoraSequencer for per-segment variations
- Comprehensive error handling and debug logging

Integration Nodes:
- WAN22BlockSwapLooperModels: Applies BlockSwap to model_high/model_low pair
- WAN22BlockSwapSequencer: Applies BlockSwap to WanVideoLoraSequencer output
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


class WAN22BlockSwapLooperModels:
    """
    Apply BlockSwap to model_high and model_low for WanVideoLooper integration.

    This node takes the model_high and model_low that would normally go directly
    to WanVideoLooper and applies BlockSwap configuration to both. The output
    models are fully compatible with WanVideoLooper's inputs.

    Use this node when you're NOT using WanVideoLoraSequencer - it handles
    the base high/low model pair.

    Workflow:
    1. Load models with WAN22BlockSwapLoader (or standard loader)
    2. Connect to this node to apply BlockSwap callbacks
    3. Connect outputs to WanVideoLooper's model_high/model_low inputs
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Get input types for the WAN22BlockSwapLooperModels node."""
        return {
            "required": {
                "model_high": (
                    "MODEL",
                    {
                        "tooltip": "The high-noise model to apply BlockSwap to. "
                        "Will be used for early sampling steps in WanVideoLooper."
                    },
                ),
                "model_low": (
                    "MODEL",
                    {
                        "tooltip": "The low-noise model to apply BlockSwap to. "
                        "Will be used for later sampling steps in WanVideoLooper."
                    },
                ),
                "blocks_to_swap": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 48,
                        "step": 1,
                        "tooltip": "Number of transformer blocks to swap to CPU. "
                        "1.3B/5B: 30 blocks, 14B: 40 blocks, LongCat: 48 blocks",
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
                        "tooltip": "VACE model blocks to swap (0 = auto). "
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
                        "speed loss from swapping.",
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

    RETURN_TYPES: tuple = ("MODEL", "MODEL")
    RETURN_NAMES: tuple = ("model_high", "model_low")
    CATEGORY: str = "ComfyUI_Wan22Blockswap/looper"
    FUNCTION: str = "apply_blockswap_to_models"
    DESCRIPTION: str = (
        "Apply BlockSwap to high/low noise model pair for WanVideoLooper. "
        "This node applies BlockSwap callbacks to both models, ensuring proper "
        "VRAM management during WanVideoLooper's multi-step sampling. "
        "Connect outputs directly to WanVideoLooper's model_high/model_low inputs."
    )

    def apply_blockswap_to_models(
        self,
        model_high: ModelPatcher,
        model_low: ModelPatcher,
        blocks_to_swap: int,
        offload_txt_emb: bool,
        offload_img_emb: bool,
        use_non_blocking: bool = False,
        vace_blocks_to_swap: int = 0,
        prefetch_blocks: int = 0,
        block_swap_debug: bool = False,
    ) -> Tuple[ModelPatcher, ModelPatcher]:
        """
        Apply BlockSwap to high/low noise models for WanVideoLooper.

        Args:
            model_high: The high-noise model
            model_low: The low-noise model
            blocks_to_swap: Number of transformer blocks to swap to CPU
            offload_txt_emb: Whether to offload text embeddings
            offload_img_emb: Whether to offload image embeddings
            use_non_blocking: Use non-blocking memory transfers
            vace_blocks_to_swap: VACE blocks to swap (0=auto)
            prefetch_blocks: Blocks to prefetch ahead
            block_swap_debug: Enable debug logging

        Returns:
            Tuple of (model_high, model_low) with BlockSwap applied
        """
        if block_swap_debug:
            print("[BlockSwap] ===== WAN22BlockSwapLooperModels: Applying BlockSwap =====")
            print(f"[BlockSwap] blocks_to_swap: {blocks_to_swap}")
            print(f"[BlockSwap] offload_txt_emb: {offload_txt_emb}, offload_img_emb: {offload_img_emb}")

        # Start a session for this pair (2 models expected)
        session_id = start_blockswap_session(loop_count=2, block_swap_debug=block_swap_debug)

        if block_swap_debug:
            print(f"[BlockSwap] Session started: {session_id}")

        try:
            # Prepare high-noise model (loop_index=0)
            prepared_high = prepare_model_for_loop(
                model=model_high,
                loop_index=0,
                blocks_to_swap=blocks_to_swap,
                offload_txt_emb=offload_txt_emb,
                offload_img_emb=offload_img_emb,
                use_non_blocking=use_non_blocking,
                vace_blocks_to_swap=vace_blocks_to_swap,
                prefetch_blocks=prefetch_blocks,
                block_swap_debug=block_swap_debug,
                session_id=session_id,
            )

            if block_swap_debug:
                print("[BlockSwap] High-noise model prepared with BlockSwap callbacks")

            # Prepare low-noise model (loop_index=1)
            prepared_low = prepare_model_for_loop(
                model=model_low,
                loop_index=1,
                blocks_to_swap=blocks_to_swap,
                offload_txt_emb=offload_txt_emb,
                offload_img_emb=offload_img_emb,
                use_non_blocking=use_non_blocking,
                vace_blocks_to_swap=vace_blocks_to_swap,
                prefetch_blocks=prefetch_blocks,
                block_swap_debug=block_swap_debug,
                session_id=session_id,
            )

            if block_swap_debug:
                print("[BlockSwap] Low-noise model prepared with BlockSwap callbacks")
                print("[BlockSwap] ===== WAN22BlockSwapLooperModels: Complete =====")

        except Exception as e:
            if block_swap_debug:
                print(f"[BlockSwap] Error during model preparation: {e}")
            end_blockswap_session(session_id, block_swap_debug)
            raise

        return (prepared_high, prepared_low)


class WAN22BlockSwapSequencer:
    """
    Apply BlockSwap to WanVideoLoraSequencer output for per-segment BlockSwap.

    This node takes the output from WanVideoLoraSequencer (a list of model tuples)
    and applies BlockSwap to each model. The output is compatible with
    WanVideoLooper's model_clip_sequence input.

    Use this when using WanVideoLoraSequencer to have different LoRAs per segment.

    Workflow:
    1. Load models with WAN22BlockSwapLoader (or standard loader)
    2. Use WanVideoLoraSequencer to assign per-segment LoRAs
    3. Connect sequencer output to this node
    4. Connect this node's output to WanVideoLooper's model_clip_sequence
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Get input types for the WAN22BlockSwapSequencer node."""
        return {
            "required": {
                "model_clip_sequence": (
                    "ANY",
                    {
                        "tooltip": "Connect the output from WanVideoLoraSequencer. "
                        "This is a list of (model_high, model_low, clip) tuples."
                    },
                ),
                "blocks_to_swap": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 48,
                        "step": 1,
                        "tooltip": "Number of transformer blocks to swap to CPU. "
                        "1.3B/5B: 30 blocks, 14B: 40 blocks, LongCat: 48 blocks",
                    },
                ),
                "offload_txt_emb": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Offload text_embedding to CPU. "
                        "Reduces VRAM by ~500MB",
                    },
                ),
                "offload_img_emb": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Offload img_emb to CPU (I2V models only). "
                        "Reduces VRAM by ~200MB",
                    },
                ),
            },
            "optional": {
                "use_non_blocking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use non-blocking memory transfers.",
                    },
                ),
                "vace_blocks_to_swap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 15,
                        "step": 1,
                        "tooltip": "VACE model blocks to swap (0 = auto).",
                    },
                ),
                "prefetch_blocks": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 40,
                        "step": 1,
                        "tooltip": "Prefetch N blocks ahead for performance.",
                    },
                ),
                "block_swap_debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable debug logging.",
                    },
                ),
            },
        }

    RETURN_TYPES: tuple = ("ANY",)
    RETURN_NAMES: tuple = ("model_clip_sequence",)
    CATEGORY: str = "ComfyUI_Wan22Blockswap/looper"
    FUNCTION: str = "apply_blockswap_to_sequence"
    DESCRIPTION: str = (
        "Apply BlockSwap to WanVideoLoraSequencer output. "
        "Takes a list of (model_high, model_low, clip) tuples and applies "
        "BlockSwap callbacks to each model. Output is compatible with "
        "WanVideoLooper's model_clip_sequence input."
    )

    def apply_blockswap_to_sequence(
        self,
        model_clip_sequence: List[Tuple],
        blocks_to_swap: int,
        offload_txt_emb: bool,
        offload_img_emb: bool,
        use_non_blocking: bool = False,
        vace_blocks_to_swap: int = 0,
        prefetch_blocks: int = 0,
        block_swap_debug: bool = False,
    ) -> Tuple[List[Tuple]]:
        """
        Apply BlockSwap to a WanVideoLoraSequencer output.

        Args:
            model_clip_sequence: List of (model_high, model_low, clip) tuples
            blocks_to_swap: Number of transformer blocks to swap to CPU
            offload_txt_emb: Whether to offload text embeddings
            offload_img_emb: Whether to offload image embeddings
            use_non_blocking: Use non-blocking memory transfers
            vace_blocks_to_swap: VACE blocks to swap (0=auto)
            prefetch_blocks: Blocks to prefetch ahead
            block_swap_debug: Enable debug logging

        Returns:
            Tuple containing the processed sequence (same format as input)
        """
        # Validate input
        if not isinstance(model_clip_sequence, list):
            raise ValueError("model_clip_sequence must be a list from WanVideoLoraSequencer")

        if block_swap_debug:
            print("[BlockSwap] ===== WAN22BlockSwapSequencer: Processing sequence =====")
            print(f"[BlockSwap] Sequence length: {len(model_clip_sequence)}")
            print(f"[BlockSwap] blocks_to_swap: {blocks_to_swap}")

        # Count total models (2 per segment that has models)
        total_models = 0
        for segment_data in model_clip_sequence:
            if segment_data and isinstance(segment_data, tuple) and len(segment_data) == 3:
                model_high, model_low, _ = segment_data
                if model_high is not None:
                    total_models += 1
                if model_low is not None:
                    total_models += 1

        if block_swap_debug:
            print(f"[BlockSwap] Total models to process: {total_models}")

        # Start a session for this sequence
        session_id = start_blockswap_session(
            loop_count=total_models,
            block_swap_debug=block_swap_debug
        )

        if block_swap_debug:
            print(f"[BlockSwap] Session started: {session_id}")

        processed_sequence = []
        model_counter = 0

        try:
            for i, segment_data in enumerate(model_clip_sequence):
                if segment_data is None:
                    processed_sequence.append(None)
                    continue

                if not isinstance(segment_data, tuple) or len(segment_data) != 3:
                    # Pass through unchanged if not proper format
                    processed_sequence.append(segment_data)
                    continue

                model_high, model_low, clip = segment_data

                # Process high-noise model if present
                if model_high is not None:
                    prepared_high = prepare_model_for_loop(
                        model=model_high,
                        loop_index=model_counter,
                        blocks_to_swap=blocks_to_swap,
                        offload_txt_emb=offload_txt_emb,
                        offload_img_emb=offload_img_emb,
                        use_non_blocking=use_non_blocking,
                        vace_blocks_to_swap=vace_blocks_to_swap,
                        prefetch_blocks=prefetch_blocks,
                        block_swap_debug=block_swap_debug,
                        session_id=session_id,
                    )
                    model_counter += 1
                else:
                    prepared_high = None

                # Process low-noise model if present
                if model_low is not None:
                    prepared_low = prepare_model_for_loop(
                        model=model_low,
                        loop_index=model_counter,
                        blocks_to_swap=blocks_to_swap,
                        offload_txt_emb=offload_txt_emb,
                        offload_img_emb=offload_img_emb,
                        use_non_blocking=use_non_blocking,
                        vace_blocks_to_swap=vace_blocks_to_swap,
                        prefetch_blocks=prefetch_blocks,
                        block_swap_debug=block_swap_debug,
                        session_id=session_id,
                    )
                    model_counter += 1
                else:
                    prepared_low = None

                # CLIP is passed through unchanged (no BlockSwap needed)
                processed_sequence.append((prepared_high, prepared_low, clip))

                if block_swap_debug:
                    print(f"[BlockSwap] Segment {i+1}: Processed")

            if block_swap_debug:
                print(f"[BlockSwap] ===== WAN22BlockSwapSequencer: Complete =====")
                print(f"[BlockSwap] Processed {model_counter} models")

        except Exception as e:
            if block_swap_debug:
                print(f"[BlockSwap] Error during sequence processing: {e}")
            end_blockswap_session(session_id, block_swap_debug)
            raise

        return (processed_sequence,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "wan22BlockSwapLooperModels": WAN22BlockSwapLooperModels,
    "wan22BlockSwapSequencer": WAN22BlockSwapSequencer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "wan22BlockSwapLooperModels": "WAN 2.2 BlockSwap Looper (High/Low Models)",
    "wan22BlockSwapSequencer": "WAN 2.2 BlockSwap Sequencer (For LoRA Sequence)",
}
