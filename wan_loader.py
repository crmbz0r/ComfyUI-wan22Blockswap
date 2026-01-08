"""
WANModelLoader: GGUF-focused WAN model loader.

This loader handles GGUF WAN models with automatic version and variant
detection. Uses the same stable loading approach as ComfyUI-GGUF's
Advanced loader to avoid CUDA corruption issues.

Supports:
- WAN 2.1 (1.3B, 5B, 14B) and WAN 2.2 models
- All variants: T2V, I2V, VACE, Camera, S2V, Humo, Animate
- GGUF quantized format only (optimized for BlockSwap)
"""

import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import folder_paths
import comfy.model_management as mm
import comfy.utils
import comfy.sd

from .model_detection import detect_wan_config

logger = logging.getLogger("WANModelLoader")


def _register_gguf_extension():
    """Add GGUF extension support to diffusion_models folder type."""
    if "diffusion_models" in folder_paths.folder_names_and_paths:
        paths, extensions = folder_paths.folder_names_and_paths["diffusion_models"]
        if ".gguf" not in extensions:
            extensions.add(".gguf")
            folder_paths.folder_names_and_paths["diffusion_models"] = (paths, extensions)
            if hasattr(folder_paths, "filename_list_cache"):
                if "diffusion_models" in folder_paths.filename_list_cache:
                    del folder_paths.filename_list_cache["diffusion_models"]
            logger.info("Registered .gguf extension for diffusion_models folder")

_register_gguf_extension()


class WANModelLoader:
    """
    GGUF-focused WAN model loader.

    Loads WAN 2.1/2.2 models in GGUF format with automatic configuration
    detection. Uses ComfyUI-GGUF's proven stable loading approach.

    For BlockSwap functionality, connect the output to a
    WAN22BlockSwap node.

    Features:
    - Auto-detects WAN version (2.1/2.2) and variant (T2V/I2V/etc)
    - GGUF quantized models only (safetensors removed for stability)
    - Advanced GGUF options for stable multi-run operation
    - Clean, simple interface
    """

    CATEGORY = "WAN"
    FUNCTION = "load_model"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define node inputs."""
        try:
            all_models = folder_paths.get_filename_list("diffusion_models")
        except Exception:
            all_models = []

        # Filter for GGUF only
        gguf_models = sorted([
            p for p in all_models
            if p.lower().endswith(".gguf")
        ])

        if not gguf_models:
            gguf_models = ["no gguf models found"]

        return {
            "required": {
                "gguf_model": (gguf_models, {
                    "tooltip": "GGUF quantized model from diffusion_models folder"
                }),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {
                    "default": "default",
                    "tooltip": "Data type for dequantized weights. 'default'=float16, 'target'=match input."
                }),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {
                    "default": "default",
                    "tooltip": "Data type for LoRA patches. 'default'=float16, 'target'=match input."
                }),
                "patch_on_device": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep patches on GPU. Uses more VRAM but faster with LoRAs."
                }),
            },
            "optional": {
                "wan_version": (["auto", "2.1", "2.2"], {
                    "default": "auto",
                    "tooltip": "WAN model version. 'auto' detects from weights."
                }),
                "model_variant": (
                    ["auto", "t2v", "i2v", "vace", "camera", "s2v", "humo", "animate"],
                    {
                        "default": "auto",
                        "tooltip": "WAN model variant. 'auto' detects from weights."
                    }
                ),
            }
        }

    def load_model(
        self,
        gguf_model: str,
        dequant_dtype: str = "default",
        patch_dtype: str = "default",
        patch_on_device: bool = False,
        wan_version: str = "auto",
        model_variant: str = "auto",
    ) -> Tuple[Any]:
        """
        Load a GGUF WAN model using ComfyUI-GGUF's stable loading approach.

        Args:
            gguf_model: GGUF model filename
            dequant_dtype: Dequantization dtype
            patch_dtype: LoRA patch dtype
            patch_on_device: Keep patches on GPU
            wan_version: WAN version ("auto", "2.1", "2.2")
            model_variant: Model variant type

        Returns:
            Tuple containing ModelPatcher
        """
        if gguf_model == "no gguf models found":
            raise FileNotFoundError(
                "No GGUF models found. Add .gguf files to models/diffusion_models/"
            )

        full_path = folder_paths.get_full_path("diffusion_models", gguf_model)
        if not full_path:
            raise FileNotFoundError(f"Model not found: {gguf_model}")

        logger.info("=" * 60)
        logger.info("WAN Model Loader (GGUF)")
        logger.info(f"  Model: {gguf_model}")
        logger.info(f"  dequant_dtype: {dequant_dtype}")
        logger.info(f"  patch_dtype: {patch_dtype}")
        logger.info(f"  patch_on_device: {patch_on_device}")
        logger.info("=" * 60)

        # Memory cleanup
        mm.unload_all_models()
        mm.soft_empty_cache()

        # Import GGMLOps from ComfyUI-GGUF
        try:
            import sys
            import os
            gguf_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ComfyUI-GGUF")
            if gguf_folder not in sys.path:
                sys.path.insert(0, gguf_folder)
            from ops import GGMLOps
            from loader import gguf_sd_loader
            from nodes import GGUFModelPatcher
        except ImportError as e:
            raise ImportError(
                f"ComfyUI-GGUF not found: {e}\n"
                "GGUF support requires ComfyUI-GGUF custom node:\n"
                "  https://github.com/city96/ComfyUI-GGUF\n"
                "Install via ComfyUI Manager or git clone into custom_nodes/"
            )

        # Create GGMLOps instance with proper settings (same as Advanced loader)
        ops = GGMLOps()

        if dequant_dtype in ("default", None):
            ops.Linear.dequant_dtype = None
        elif dequant_dtype == "target":
            ops.Linear.dequant_dtype = dequant_dtype
        else:
            ops.Linear.dequant_dtype = getattr(torch, dequant_dtype)

        if patch_dtype in ("default", None):
            ops.Linear.patch_dtype = None
        elif patch_dtype == "target":
            ops.Linear.patch_dtype = patch_dtype
        else:
            ops.Linear.patch_dtype = getattr(torch, patch_dtype)

        # Load GGUF state dict
        logger.info("Loading GGUF model...")
        sd = gguf_sd_loader(full_path)
        logger.info(f"Loaded {len(sd)} tensors")

        # Detect WAN config from state dict
        logger.info("Detecting WAN configuration...")
        try:
            wan_config = detect_wan_config(
                sd,
                wan_version=wan_version,
                model_variant=model_variant,
            )
        except ValueError as e:
            raise ValueError(f"Not a valid WAN model: {e}")

        logger.info(
            f"Detected: WAN {wan_config['wan_version']}, "
            f"variant={wan_config['model_variant']}, "
            f"size={wan_config.get('model_size', 'unknown')}, "
            f"blocks={wan_config['num_layers']}, "
            f"dim={wan_config['dim']}"
        )

        # Load model using ComfyUI's standard loader with our custom ops
        # This is the same approach as UnetLoaderGGUFAdvanced
        logger.info("Creating model with GGMLOps...")
        model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}
        )

        if model is None:
            logger.error(f"ERROR UNSUPPORTED UNET {full_path}")
            raise RuntimeError(f"ERROR: Could not detect model type of: {full_path}")

        # Convert to GGUFModelPatcher for proper GGUF handling
        model = GGUFModelPatcher.clone(model)
        model.patch_on_device = patch_on_device

        # Store WAN config on model for BlockSwap nodes
        model.wan_config = wan_config

        # Cleanup
        del sd
        mm.soft_empty_cache()

        logger.info("=" * 60)
        logger.info("Model loaded successfully!")
        logger.info("  Use WAN22BlockSwap node for block swapping if needed")
        logger.info("=" * 60)

        return (model,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "WANModelLoader": WANModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANModelLoader": "WAN Model Loader (GGUF)",
}
