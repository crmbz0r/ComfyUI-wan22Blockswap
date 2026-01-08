"""
WANBlockSwapMetaLoader: True zero-VRAM blockswap loader using meta tensors.

This loader implements the WanVideoWrapper approach:
1. Create model skeleton with meta tensors (zero memory)
2. Load weights directly to target devices per-block
3. Swapped blocks NEVER touch GPU

Compatible with ANY KSampler including WanVideoLooper.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import folder_paths
import comfy.model_management as mm
import comfy.utils
from comfy.model_patcher import ModelPatcher

from .loader_helpers import (
    detect_model_format,
    load_state_dict_to_cpu,
    create_model_skeleton,
    assign_weights_with_routing,
    create_model_patcher,
)
from .model_detection import detect_wan_config

logger = logging.getLogger("WANBlockSwapMetaLoader")


class WANBlockSwapMetaLoader:
    """
    WAN model loader with TRUE block swap - no initial VRAM spike.
    
    Uses meta tensors + per-block device routing to load swapped blocks
    directly to CPU, never touching GPU at all.
    
    This is the WanVideoWrapper approach adapted for ComfyUI compatibility.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model_path": (folder_paths.get_filename_list("diffusion_models"),),
                "blocks_to_swap": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 48,
                    "step": 1,
                    "display": "slider",
                }),
                "offload_txt_emb": ("BOOLEAN", {"default": False}),
                "offload_img_emb": ("BOOLEAN", {"default": False}),
                "fp8_optimization": ([
                    "disabled",
                    "e4m3fn",
                    "e5m2"
                ], {"default": "disabled"}),
                "noise_level": (["high_noise", "low_noise"], {
                    "default": "high_noise",
                    "tooltip": (
                        "Which noise level this model handles. "
                        "'high_noise' loads immediately (used first in sampling). "
                        "'low_noise' uses lazy loading (loads only when sampler needs it)."
                    )
                }),
                "enable_debug": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    CATEGORY = "ComfyUI_Wan22Blockswap"
    FUNCTION = "load_model"
    DESCRIPTION = (
        "Load WAN model with TRUE block swap using meta tensors. "
        "Swapped blocks are loaded directly to CPU/RAM and NEVER touch GPU. "
        "No initial VRAM spike! Compatible with ANY KSampler including WanVideoLooper."
    )
    
    def load_model(
        self,
        model_path: str,
        blocks_to_swap: int,
        offload_txt_emb: bool,
        offload_img_emb: bool,
        fp8_optimization: str,
        noise_level: str,
        enable_debug: bool,
    ) -> Tuple[ModelPatcher]:
        """
        Load WAN model with meta tensor-based block swapping.
        
        Args:
            model_path: Path to model file (safetensors or GGUF)
            blocks_to_swap: Number of blocks from end to load to CPU
            offload_txt_emb: Offload text embeddings to CPU
            offload_img_emb: Offload image embeddings to CPU
            fp8_optimization: FP8 quantization mode
            noise_level: "high_noise" (load immediately) or "low_noise" (lazy load)
            enable_debug: Print debug information
            
        Returns:
            ModelPatcher with properly routed blocks
        """
        # For low_noise models, wrap in lazy loader to defer loading
        if noise_level == "low_noise":
            if enable_debug:
                logger.info("=" * 60)
                logger.info("WAN BlockSwap Meta Loader: LAZY LOAD (low_noise model)")
                logger.info(f"  Model: {model_path}")
                logger.info("  Model will load when sampler first needs it")
                logger.info("=" * 60)
            
            # Import LazyModelPatcher from blockswap_loader
            from .blockswap_loader import LazyModelPatcher
            
            # Create lazy loader that defers actual loading
            lazy_patcher = LazyModelPatcher(
                loader_func=self._do_load_model,
                loader_args={
                    "model_path": model_path,
                    "blocks_to_swap": blocks_to_swap,
                    "offload_txt_emb": offload_txt_emb,
                    "offload_img_emb": offload_img_emb,
                    "fp8_optimization": fp8_optimization,
                    "enable_debug": enable_debug,
                }
            )
            return (lazy_patcher,)
        
        # High noise model - load immediately
        return self._do_load_model(
            model_path=model_path,
            blocks_to_swap=blocks_to_swap,
            offload_txt_emb=offload_txt_emb,
            offload_img_emb=offload_img_emb,
            fp8_optimization=fp8_optimization,
            enable_debug=enable_debug,
        )
    
    def _do_load_model(
        self,
        model_path: str,
        blocks_to_swap: int,
        offload_txt_emb: bool,
        offload_img_emb: bool,
        fp8_optimization: str,
        enable_debug: bool,
    ) -> Tuple[ModelPatcher]:
        """Internal method that does the actual loading."""
        # Get full path
        full_path = folder_paths.get_full_path("diffusion_models", model_path)
        if enable_debug:
            logger.info(f"Loading model from: {full_path}")
        
        # Detect format
        model_format = detect_model_format(full_path)
        if enable_debug:
            logger.info(f"Detected format: {model_format}")
        
        # Load state dict to CPU
        if enable_debug:
            logger.info("Loading state dict to CPU...")
        state_dict, gguf_reader, metadata = load_state_dict_to_cpu(full_path, model_format)
        
        if enable_debug:
            logger.info(f"State dict loaded with {len(state_dict)} keys")
            # Sample some keys to see the pattern
            sample_keys = list(state_dict.keys())[:20]
            logger.info("Sample state dict keys:")
            for k in sample_keys:
                logger.info(f"  {k}")
            # Check for img_emb keys
            img_emb_keys = [k for k in state_dict.keys() if 'img_emb' in k]
            if img_emb_keys:
                logger.info(f"Found {len(img_emb_keys)} img_emb keys: {img_emb_keys}")
            # Check highest block number
            block_keys = [k for k in state_dict.keys() if 'blocks.' in k]
            if block_keys:
                block_nums = []
                for k in block_keys:
                    try:
                        parts = k.split('blocks.')[1].split('.')[0]
                        block_nums.append(int(parts))
                    except:
                        pass
                if block_nums:
                    logger.info(f"Block range in state dict: {min(block_nums)}-{max(block_nums)}")
        
        # Detect model configuration
        if enable_debug:
            logger.info("Detecting model configuration...")
        wan_config = detect_wan_config(state_dict, metadata)
        if enable_debug:
            logger.info(f"Detected config: {wan_config}")
        
        total_blocks = wan_config.get("num_layers", 30)
        if blocks_to_swap > total_blocks:
            logger.warning(f"blocks_to_swap ({blocks_to_swap}) > total_blocks ({total_blocks}), clamping")
            blocks_to_swap = total_blocks
        
        # Determine devices
        main_device = mm.get_torch_device()
        offload_device = torch.device('cpu')
        
        # Determine dtype
        if model_format == "gguf":
            base_dtype = torch.float16  # GGUF uses BF16 internally
        else:
            base_dtype = torch.bfloat16 if wan_config.get("wan_version") == "2.2" else torch.float16
        
        if enable_debug:
            logger.info(f"Creating model skeleton with meta tensors...")
            logger.info(f"Total blocks: {total_blocks}, Swapping: {blocks_to_swap}")
            logger.info(f"Main device: {main_device}, Offload device: {offload_device}")
        
        # Create model skeleton with meta tensors (ZERO memory!)
        is_gguf = model_format == "gguf"
        model = create_model_skeleton(
            wan_config=wan_config,
            dtype=base_dtype,
            device="meta",  # Meta device = no memory allocated
            is_gguf=is_gguf,
        )
        
        if enable_debug:
            logger.info("Model skeleton created with zero memory")
            logger.info("Assigning weights with per-block device routing...")
        
        # Determine key prefix based on state dict keys
        # Safetensors files have "diffusion_model." prefix, GGUF files don't
        sample_keys = list(state_dict.keys())[:5]
        has_prefix = any(k.startswith("diffusion_model.") for k in sample_keys)
        key_prefix = "diffusion_model." if has_prefix else ""
        
        if enable_debug:
            if key_prefix:
                logger.info(f"State dict has 'diffusion_model.' prefix, will strip it")
            else:
                logger.info(f"State dict has no prefix, using keys as-is")
        
        # Assign weights with per-block device routing
        # This is where the magic happens - swapped blocks go directly to CPU
        block_device_map = assign_weights_with_routing(
            model=model,
            state_dict=state_dict,
            total_blocks=total_blocks,
            blocks_to_swap=blocks_to_swap,
            main_device=main_device,
            offload_device=offload_device,
            gguf_reader=gguf_reader,
            fp8_optimization=fp8_optimization,
            key_prefix=key_prefix,
        )
        
        if enable_debug:
            logger.info("Weight assignment complete")
            logger.info(f"Block device map: {block_device_map}")
        
        # Optionally offload embeddings
        if offload_txt_emb and hasattr(model, 'txt_in'):
            if enable_debug:
                logger.info("Offloading text embeddings to CPU")
            model.txt_in.to(offload_device)
        
        if offload_img_emb and hasattr(model, 'img_in'):
            if enable_debug:
                logger.info("Offloading image embeddings to CPU")
            model.img_in.to(offload_device)
        
        # Create model patcher
        if enable_debug:
            logger.info("Creating model patcher...")
        
        model_patcher = create_model_patcher(
            diffusion_model=model,
            wan_config=wan_config,
            load_device=main_device,
            offload_device=offload_device,
            blocks_to_swap=blocks_to_swap,
            is_gguf=is_gguf,
        )
        
        # Store metadata so other nodes know this model was pre-routed
        if not hasattr(model_patcher, "model_options"):
            model_patcher.model_options = {}
        
        model_patcher.model_options["wan22_blockswap_meta_loaded"] = True
        model_patcher.model_options["wan22_blockswap_info"] = {
            "blocks_to_swap": blocks_to_swap,
            "total_blocks": total_blocks,
            "offload_txt_emb": offload_txt_emb,
            "offload_img_emb": offload_img_emb,
            "block_device_map": block_device_map,
        }
        
        if enable_debug:
            logger.info("Model loaded successfully with meta tensor block swap!")
            logger.info(f"GPU blocks: 0-{total_blocks - blocks_to_swap - 1}")
            logger.info(f"CPU blocks: {total_blocks - blocks_to_swap}-{total_blocks - 1}")
        
        return (model_patcher,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "WANBlockSwapMetaLoader": WANBlockSwapMetaLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANBlockSwapMetaLoader": "WAN BlockSwap Meta Loader (Zero VRAM Spike)",
}
