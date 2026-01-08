"""
WAN22BlockSwapLoader: A model loader with integrated BlockSwap support.

This loader prevents VRAM overflow by routing transformer blocks to their
target devices DURING the initial weight loading process, rather than
loading everything to GPU first and then moving blocks to CPU.

The key innovation is that swap blocks NEVER touch GPU memory - they are
loaded directly from disk to CPU, preventing the VRAM spike that occurs
with traditional loading approaches.

Usage:
    1. Place WAN22BlockSwapLoader node in workflow
    2. Connect to sampler nodes
    3. Optionally connect to wan22BlockSwap for dynamic swapping
    
    The loader will route blocks during load based on blocks_to_swap:
    - Blocks 0 to (total - blocks_to_swap - 1) → GPU
    - Blocks (total - blocks_to_swap) to (total - 1) → CPU
"""

import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import folder_paths
import comfy.model_management as mm
import comfy.utils

from .loader_helpers import (
    detect_model_format,
    load_state_dict_to_cpu,
    create_model_skeleton,
    assign_weights_with_routing,
    create_model_patcher,
    verify_block_devices,
)
from .model_detection import detect_wan_config

logger = logging.getLogger("WAN22BlockSwapLoader")


# Register GGUF extension at module load time
# This must happen before any folder_paths.get_filename_list calls
def _register_gguf_extension():
    """Add GGUF extension support to diffusion_models folder type."""
    if "diffusion_models" in folder_paths.folder_names_and_paths:
        paths, extensions = folder_paths.folder_names_and_paths["diffusion_models"]
        if ".gguf" not in extensions:
            extensions.add(".gguf")
            folder_paths.folder_names_and_paths["diffusion_models"] = (paths, extensions)
            # Invalidate cache to force re-scan with new extension
            if hasattr(folder_paths, "filename_list_cache"):
                if "diffusion_models" in folder_paths.filename_list_cache:
                    del folder_paths.filename_list_cache["diffusion_models"]
            logger.info("Registered .gguf extension for diffusion_models folder")

_register_gguf_extension()


class WAN22BlockSwapLoader:
    """
    Load WAN 2.1/2.2 models with integrated BlockSwap support.
    
    This node loads transformer blocks directly to their target devices
    based on the blocks_to_swap parameter, preventing VRAM overflow
    that would occur if all blocks were loaded to GPU first.
    
    Key Features:
    - Loads safetensors and GGUF models
    - Routes blocks to GPU/CPU during weight loading (not after)
    - Prevents VRAM spikes by never loading swap blocks to GPU
    - Stores routing info for downstream BlockSwap node detection
    - Supports WAN 2.1 (1.3B, 5B, 14B) and WAN 2.2 models
    - Supports all WAN variants: T2V, I2V, VACE, Camera, S2V, Humo, Animate
    
    Memory Profile:
    - State dict load: ~8GB CPU (same as standard)
    - Model creation: ~0GB GPU (uses meta tensors)
    - Weight assignment: GPU usage = (total_blocks - blocks_to_swap) / total_blocks
    
    Example with 30 blocks and blocks_to_swap=20:
    - Blocks 0-9 → GPU (10 blocks)
    - Blocks 10-29 → CPU (20 blocks)
    - GPU memory: ~40% of full model
    """
    
    CATEGORY = "WAN22BlockSwap"
    FUNCTION = "load_model"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define node inputs.
        
        Returns:
            Dict containing required and optional input specifications.
        """
        # GGUF extension already registered at module load time
        # Get available model files - includes GGUF
        try:
            all_models = folder_paths.get_filename_list("diffusion_models")
        except Exception:
            all_models = []
        
        # Filter safetensors files
        safetensors_extensions = [".safetensors", ".sft"]
        safetensors_models = sorted([
            p for p in all_models 
            if any(p.lower().endswith(ext) for ext in safetensors_extensions)
        ])
        
        # Filter GGUF files
        gguf_models = sorted([
            p for p in all_models 
            if p.lower().endswith(".gguf")
        ])
        
        # Add fallback if no models found
        if not safetensors_models:
            safetensors_models = ["no safetensors models found"]
        if not gguf_models:
            gguf_models = ["no gguf models found"]
        
        return {
            "required": {
                "model_type": (["safetensors", "gguf"], {
                    "default": "safetensors",
                    "tooltip": (
                        "Model format to load. "
                        "'safetensors' loads from models/diffusion_models or models/unet. "
                        "'gguf' loads GGUF quantized models from the same folders."
                    )
                }),
                "safetensors_model": (safetensors_models, {
                    "tooltip": "Safetensors model from diffusion_models/unet folder"
                }),
                "gguf_model": (gguf_models, {
                    "tooltip": "GGUF quantized model from diffusion_models/unet folder"
                }),
                "blocks_to_swap": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 48,  # Maximum for LongCat models
                    "step": 1,
                    "tooltip": (
                        "Number of transformer blocks to keep on CPU during loading. "
                        "Higher = less VRAM usage, slower inference. "
                        "Recommended: 20 for 12GB, 15 for 16GB, 10 for 24GB"
                    )
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
                "fp8_optimization": (["disabled", "e4m3fn", "e5m2"], {
                    "default": "disabled",
                    "tooltip": (
                        "Apply FP8 quantization to GPU blocks for additional memory savings. "
                        "e4m3fn is recommended for best quality/speed balance."
                    )
                }),
                "weight_dtype": (["auto", "fp16", "bf16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Data type for model weights. 'auto' uses model's native dtype."
                }),
                "noise_level": (["high_noise", "low_noise"], {
                    "default": "high_noise",
                    "tooltip": (
                        "Which noise level this model handles. "
                        "'high_noise' loads immediately (used first in sampling). "
                        "'low_noise' uses lazy loading (loads only when sampler needs it)."
                    )
                }),
            }
        }
    
    def load_model(
        self,
        model_type: str,
        safetensors_model: str,
        gguf_model: str,
        blocks_to_swap: int,
        wan_version: str = "auto",
        model_variant: str = "auto",
        fp8_optimization: str = "disabled",
        weight_dtype: str = "auto",
        noise_level: str = "high_noise",
    ) -> Tuple[Any]:
        """
        Load WAN model with integrated block swapping.
        
        This method implements the key innovation: routing blocks to their
        target devices DURING weight loading, so swap blocks never touch GPU.
        
        Flow:
        1. Detect model format (safetensors/GGUF)
        2. Load state dict to CPU (no GPU touch)
        3. Detect WAN version and variant
        4. Create model skeleton with meta tensors (no memory)
        5. Assign weights with per-block device routing
        6. Create ModelPatcher with pre-routing info
        
        For low_noise models, a lazy-loading wrapper is returned that defers
        actual model loading until the sampler first accesses it.
        
        Args:
            model_type: "safetensors" or "gguf" - determines which folder to use
            safetensors_model: Model name from diffusion_models folder
            gguf_model: Model name from unet folder
            blocks_to_swap: Number of blocks to keep on CPU
            wan_version: WAN version ("auto", "2.1", "2.2")
            model_variant: WAN variant type
            fp8_optimization: FP8 quantization mode
            weight_dtype: Weight data type
            noise_level: "high_noise" (load immediately) or "low_noise" (lazy load)
        
        Returns:
            Tuple containing ModelPatcher with pre-routed blocks
        
        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model format unsupported or not a WAN model
            RuntimeError: If loading fails
        """
        # For low_noise models, wrap in lazy loader to defer loading
        if noise_level == "low_noise":
            logger.info("=" * 60)
            logger.info("WAN BlockSwap Loader: LAZY LOAD (low_noise model)")
            logger.info(f"  Model: {gguf_model if model_type == 'gguf' else safetensors_model}")
            logger.info("  Model will load when sampler first needs it")
            logger.info("=" * 60)
            
            # Create lazy loader that defers actual loading
            lazy_patcher = LazyModelPatcher(
                loader_func=self._do_load_model,
                loader_args={
                    "model_type": model_type,
                    "safetensors_model": safetensors_model,
                    "gguf_model": gguf_model,
                    "blocks_to_swap": blocks_to_swap,
                    "wan_version": wan_version,
                    "model_variant": model_variant,
                    "fp8_optimization": fp8_optimization,
                    "weight_dtype": weight_dtype,
                },
            )
            return (lazy_patcher,)
        
        # For high_noise, load immediately
        return (self._do_load_model(
            model_type=model_type,
            safetensors_model=safetensors_model,
            gguf_model=gguf_model,
            blocks_to_swap=blocks_to_swap,
            wan_version=wan_version,
            model_variant=model_variant,
            fp8_optimization=fp8_optimization,
            weight_dtype=weight_dtype,
        ),)
    
    def _do_load_model(
        self,
        model_type: str,
        safetensors_model: str,
        gguf_model: str,
        blocks_to_swap: int,
        wan_version: str = "auto",
        model_variant: str = "auto",
        fp8_optimization: str = "disabled",
        weight_dtype: str = "auto",
    ) -> Any:
        """
        Actually perform the model loading.
        
        This is the core loading logic, extracted so it can be called
        immediately (high_noise) or deferred (low_noise via LazyModelPatcher).
        """
        # Step 1: Determine model path based on model_type
        if model_type == "gguf":
            model_path = gguf_model
            if model_path == "no gguf models found":
                raise FileNotFoundError(
                    "No GGUF models found. "
                    "Please add .gguf files to ComfyUI/models/diffusion_models/ or ComfyUI/models/unet/"
                )
        else:  # safetensors
            model_path = safetensors_model
            if model_path == "no safetensors models found":
                raise FileNotFoundError(
                    "No safetensors models found. "
                    "Please add .safetensors files to ComfyUI/models/diffusion_models/"
                )
        
        # Step 2: Get full path and validate (uses diffusion_models which includes unet folder)
        full_path = folder_paths.get_full_path("diffusion_models", model_path)
        if not full_path:
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )
        
        logger.info("=" * 60)
        logger.info(f"WAN BlockSwap Loader: Loading {model_type} model")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Blocks to swap: {blocks_to_swap}")
        logger.info("=" * 60)
        
        # Step 2: Memory cleanup before loading
        logger.info("Cleaning up memory before load...")
        mm.unload_all_models()
        mm.soft_empty_cache()
        
        # Step 3: Detect format and load state dict to CPU
        model_format = detect_model_format(full_path)
        logger.info(f"Model format: {model_format}")
        
        logger.info("Loading state dict to CPU (no GPU touch)...")
        state_dict, _, metadata = load_state_dict_to_cpu(full_path, model_format)
        logger.info(f"Loaded {len(state_dict)} tensors to CPU")
        
        # Detect if this is a GGUF model (contains GGMLTensor objects)
        is_gguf_model = False
        for tensor in state_dict.values():
            if hasattr(tensor, 'tensor_type') and tensor.tensor_type is not None:
                is_gguf_model = True
                break
        
        if is_gguf_model:
            logger.info("Detected GGUF model with GGMLTensor objects")
        
        # Step 4: Detect WAN configuration
        logger.info("Detecting WAN model configuration...")
        try:
            wan_config = detect_wan_config(
                state_dict,
                wan_version=wan_version,
                model_variant=model_variant,
            )
        except ValueError as e:
            raise ValueError(
                f"Failed to detect WAN model configuration: {e}. "
                f"Ensure this is a valid WAN model file."
            )
        
        logger.info(
            f"Detected: WAN {wan_config['wan_version']}, "
            f"variant={wan_config['model_variant']}, "
            f"size={wan_config.get('model_size', 'unknown')}, "
            f"blocks={wan_config['num_layers']}, "
            f"dim={wan_config['dim']}"
        )
        
        # Step 5: Validate and adjust blocks_to_swap
        total_blocks = wan_config["num_layers"]
        if blocks_to_swap > total_blocks:
            logger.warning(
                f"blocks_to_swap ({blocks_to_swap}) > total_blocks ({total_blocks}). "
                f"Clamping to {total_blocks}."
            )
            blocks_to_swap = total_blocks
        
        if blocks_to_swap < 0:
            logger.warning(f"blocks_to_swap ({blocks_to_swap}) < 0. Setting to 0.")
            blocks_to_swap = 0
        
        # Step 6: Get devices
        main_device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        gpu_blocks = total_blocks - blocks_to_swap
        cpu_blocks = blocks_to_swap
        
        logger.info(f"┌─────────────────────────────────────────")
        logger.info(f"│ Block Routing Plan:")
        logger.info(f"│   Main device (GPU): {main_device}")
        logger.info(f"│   Offload device (CPU): {offload_device}")
        logger.info(f"│   Blocks 0-{gpu_blocks-1} → GPU ({gpu_blocks} blocks)")
        logger.info(f"│   Blocks {gpu_blocks}-{total_blocks-1} → CPU ({cpu_blocks} blocks)")
        logger.info(f"│   Estimated GPU usage: {100 * gpu_blocks / total_blocks:.1f}%")
        logger.info(f"└─────────────────────────────────────────")
        
        # Step 7: Determine weight dtype
        if weight_dtype == "auto":
            # Use model's native dtype or default to fp16
            # For GGUF models (GGMLTensor), the underlying data is quantized bytes
            # but inference will use fp16/bf16 after dequantization
            sample_tensor = next(iter(state_dict.values()))
            
            # Check if this is a GGMLTensor (has tensor_type for quantization)
            is_ggml = hasattr(sample_tensor, 'tensor_type') and sample_tensor.tensor_type is not None
            
            if is_ggml:
                # GGUF model - default to fp16 for dequantization
                model_dtype = torch.float16
            elif sample_tensor.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                model_dtype = sample_tensor.dtype
            else:
                model_dtype = torch.float16
        else:
            dtype_map = {
                "fp16": torch.float16,
                "bf16": torch.bfloat16,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(weight_dtype, torch.float16)
        
        logger.info(f"Using weight dtype: {model_dtype}")
        
        # Step 8: Create model skeleton with empty weights
        logger.info("Creating model skeleton with meta tensors...")
        model = create_model_skeleton(
            wan_config=wan_config,
            dtype=model_dtype,
            device="meta",
            is_gguf=is_gguf_model,  # Use GGMLOps for GGUF models
        )
        
        # Step 9: Assign weights with block-level routing
        logger.info("Assigning weights with per-block device routing...")
        block_device_map = assign_weights_with_routing(
            model=model,
            state_dict=state_dict,
            total_blocks=total_blocks,
            blocks_to_swap=blocks_to_swap,
            main_device=main_device,
            offload_device=offload_device,
            gguf_reader=None,  # No longer used - GGMLTensor handles everything
            fp8_optimization=fp8_optimization,
        )
        
        # Step 10: Verify block devices
        logger.debug("Verifying block device placement...")
        devices_correct = verify_block_devices(model, total_blocks, blocks_to_swap)
        if not devices_correct:
            logger.warning("Some blocks may not be on expected devices")
        
        # Step 11: Create ModelPatcher (use GGUFModelPatcher for GGUF models)
        logger.info("Creating ModelPatcher with pre-routing info...")
        model_patcher = create_model_patcher(
            diffusion_model=model,
            wan_config=wan_config,
            load_device=main_device,
            offload_device=offload_device,
            blocks_to_swap=blocks_to_swap,
            is_gguf=is_gguf_model,
        )
        
        # Step 12: Free state dict memory
        del state_dict
        mm.soft_empty_cache()
        
        # Log success
        logger.info("=" * 60)
        logger.info("Model loaded successfully with pre-routed blocks!")
        logger.info(f"  GPU blocks: {gpu_blocks}, CPU blocks: {cpu_blocks}")
        logger.info("  Pre-routing info stored for downstream BlockSwap detection")
        logger.info("=" * 60)
        
        return model_patcher


class LazyModelPatcher:
    """
    A wrapper that defers model loading until actual inference begins.
    
    This is used for low_noise models so they don't load until the sampler
    needs them (after high_noise model completes its passes).
    
    Key insight: ComfyUI and downstream nodes access .model and other attributes
    during workflow setup. We must NOT trigger loading for these accesses.
    Loading should ONLY happen when:
    1. comfy.model_management loads the model to GPU for inference
    2. The sampler actually calls methods on the model
    
    Strategy: Return placeholder/dummy values for introspection, and only
    load when methods that require the actual model weights are called.
    """
    
    # Attributes that return placeholder values (don't trigger loading)
    _INTROSPECTION_ATTRS = {
        'model', 'model_options', 'is_clone', 'model_size',
        'load_device', 'offload_device', 'current_device', 'model_dtype',
        'patches', 'object_patches', 'object_patches_backup',
        'weight_inplace_update', 'model_lowvram', 'lowvram_patch_counter',
        'attachments', 'get_model_object',  # Added to avoid premature loading
    }
    
    # Model attributes that should trigger loading when accessed
    # These are used during sampling, not just setup
    _MODEL_SAMPLING_ATTRS = {
        'latent_format',  # Accessed by fix_empty_latent_channels before sampling
    }
    
    # Methods that can be called WITHOUT loading (setup/configuration methods)
    _SAFE_METHODS = {
        'add_object_patch',  # Used by _apply_sigma_shift - stores patches for later
        'set_model_unet_function_wrapper',  # Wrapper setup
    }
    
    # Methods that trigger actual loading (inference-related)
    _LOAD_TRIGGERS = {
        'patch_model', 'unpatch_model', 'patch_model_lowvram', 
        'calculate_weight', 'add_patches', 'get_key_patches',
        'model_state_dict', 'add_callback', 'apply_model',  # Added apply_model as key trigger
    }
    
    # Special methods that should return self (allow chaining without loading)
    _RETURN_SELF_METHODS = {
        'clone', 'to',  # clone and to() should return a new lazy wrapper
    }
    
    def __init__(self, loader_func, loader_args: Dict[str, Any]):
        """
        Initialize lazy loader.
        
        Args:
            loader_func: Function to call to load the actual model
            loader_args: Arguments to pass to loader_func
        """
        # Use object.__setattr__ to avoid triggering __setattr__ override
        object.__setattr__(self, '_loader_func', loader_func)
        object.__setattr__(self, '_loader_args', loader_args)
        object.__setattr__(self, '_real_patcher', None)
        object.__setattr__(self, '_loading', False)
        object.__setattr__(self, '_load_triggered_by', None)
        # Placeholder attributes for introspection
        object.__setattr__(self, '_placeholder_model_options', {})
        object.__setattr__(self, '_placeholder_attachments', {})
    
    def _ensure_loaded(self, trigger_name: str = "unknown"):
        """Load the model if not already loaded."""
        if self._real_patcher is None and not self._loading:
            object.__setattr__(self, '_loading', True)
            object.__setattr__(self, '_load_triggered_by', trigger_name)
            
            model_name = self._loader_args.get('gguf_model') or self._loader_args.get('safetensors_model')
            logger.info("=" * 60)
            logger.info("LazyModelPatcher: Loading low_noise model NOW")
            logger.info(f"  Model: {model_name}")
            logger.info(f"  Triggered by: {trigger_name}")
            logger.info("=" * 60)
            
            try:
                patcher = self._loader_func(**self._loader_args)
                object.__setattr__(self, '_real_patcher', patcher)
                
                # Apply any deferred object patches
                attachments = object.__getattribute__(self, '_placeholder_attachments')
                if attachments:
                    logger.debug(f"Applying {len(attachments)} deferred object patches")
                    for key, value in attachments.items():
                        patcher.add_object_patch(key, value)
            finally:
                object.__setattr__(self, '_loading', False)
        
        return self._real_patcher
    
    def _is_loaded(self) -> bool:
        """Check if model is already loaded."""
        return object.__getattribute__(self, '_real_patcher') is not None
    
    def __getattr__(self, name):
        """Proxy attribute access, with smart loading deferral."""
        # Don't proxy private attributes
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Check if model already loaded - if so, always proxy
        real_patcher = object.__getattribute__(self, '_real_patcher')
        if real_patcher is not None:
            return getattr(real_patcher, name)
        
        # Log ALL attribute accesses on lazy model for debugging
        logger.debug(f"LazyModelPatcher.__getattr__('{name}') - model not loaded yet")
        
        # ===== NOT LOADED YET - be careful about what triggers loading =====
        
        # Return placeholder values for introspection attributes
        if name in LazyModelPatcher._INTROSPECTION_ATTRS:
            if name == 'model':
                # Return a smart placeholder that triggers loading for sampling-related attributes
                lazy_patcher_ref = self  # Capture reference to trigger loading if needed
                
                class PlaceholderModel:
                    def parameters(self):
                        # Return a single placeholder parameter with CPU device
                        # This prevents StopIteration when WanVideoLooper calls next()
                        placeholder_param = torch.nn.Parameter(torch.zeros(1, device='cpu', dtype=torch.float16))
                        return iter([placeholder_param])
                    def __repr__(self):
                        return "LazyModel(not_loaded)"
                    def __getattr__(self, name):
                        # Check if this attribute access should trigger model loading
                        if name in LazyModelPatcher._MODEL_SAMPLING_ATTRS:
                            # This is a sampling-related attribute - trigger loading NOW
                            logger.info(f"LazyModelPatcher: Triggering load due to access of model.{name}")
                            real_patcher = lazy_patcher_ref._ensure_loaded(trigger_name=f"model.{name} access")
                            if real_patcher is not None:
                                return getattr(real_patcher.model, name)
                            return None
                        
                        # For .model_config access, return a placeholder
                        if name == 'model_config':
                            class PlaceholderModelConfig:
                                """Placeholder model config that returns safe defaults for all attributes."""
                                def __init__(self):
                                    # Common attributes accessed during setup
                                    self.sampling_settings = {
                                        "beta_schedule": "sqrt_linear",
                                        "linear_start": 0.00085,
                                        "linear_end": 0.012,
                                    }
                                    # Latent format with proper attributes
                                    class PlaceholderLatentFormat:
                                        latent_channels = 16  # WAN default
                                        def process_in(self, latent):
                                            return latent
                                        def process_out(self, latent):
                                            return latent
                                    
                                    self.latent_format = PlaceholderLatentFormat()
                                    self.unet_config = {}
                                
                                def __getattr__(self, name):
                                    # Return None for any unknown attributes
                                    return None
                            
                            return PlaceholderModelConfig()
                        # For other attributes, return None
                        return None
                return PlaceholderModel()
            elif name == 'model_options':
                return object.__getattribute__(self, '_placeholder_model_options')
            elif name == 'attachments':
                return object.__getattribute__(self, '_placeholder_attachments')
            elif name == 'get_model_object':
                # Return a function that returns None (for _apply_sigma_shift checks)
                return lambda key: None
            elif name == 'is_clone':
                return False
            elif name == 'model_size':
                return 0  # Will be correct after loading
            elif name in ('load_device', 'current_device'):
                return torch.device('cpu')
            elif name == 'offload_device':
                return torch.device('cpu')
            elif name == 'model_dtype':
                return torch.bfloat16
            elif name in ('patches', 'object_patches', 'object_patches_backup'):
                return {}
            elif name in ('weight_inplace_update', 'model_lowvram'):
                return False
            elif name == 'lowvram_patch_counter':
                return 0
            else:
                return None
        
        # Handle methods that should return self (clone, to)
        if name in LazyModelPatcher._RETURN_SELF_METHODS:
            if name == 'clone':
                # Return self without loading - cloning a lazy loader returns another lazy loader
                logger.debug(f"LazyModelPatcher.clone() called - returning self without loading")
                return lambda: self
            elif name == 'to':
                # Return self without loading - device moves are deferred
                return lambda *args, **kwargs: self
        
        # Handle safe methods that can be called without loading
        # These store configuration for later use when model loads
        if name in LazyModelPatcher._SAFE_METHODS:
            # Store method calls to be replayed when model loads
            def safe_method_wrapper(*args, **kwargs):
                logger.debug(f"LazyModelPatcher.{name}() called - deferring until load")
                # For now, just store in placeholder dicts
                # When model loads, these patches will be applied during actual inference
                if name == 'add_object_patch':
                    # Store object patches
                    object_patches = object.__getattribute__(self, '_placeholder_attachments')
                    if len(args) >= 2:
                        object_patches[args[0]] = args[1]
                return self
            return safe_method_wrapper
        
        # Trigger loading for inference-related methods
        if name in LazyModelPatcher._LOAD_TRIGGERS:
            patcher = self._ensure_loaded(trigger_name=f"method '{name}' called")
            if patcher is None:
                raise RuntimeError("Failed to load model")
            return getattr(patcher, name)
        
        # For unknown attributes, trigger loading (safer default)
        patcher = self._ensure_loaded(trigger_name=f"getattr('{name}')")
        if patcher is None:
            raise RuntimeError("Failed to load model")
        return getattr(patcher, name)
    
    def __setattr__(self, name, value):
        """Proxy attribute setting, with loading deferral for safe attrs."""
        if name.startswith('_'):
            object.__setattr__(self, name, value)
            return
        
        # Check if model already loaded
        real_patcher = object.__getattribute__(self, '_real_patcher')
        if real_patcher is not None:
            setattr(real_patcher, name, value)
            return
        
        # Allow setting model_options without loading (for BlockSwap info)
        if name == 'model_options':
            object.__setattr__(self, '_placeholder_model_options', value)
            return
        
        if name == 'attachments':
            object.__setattr__(self, '_placeholder_attachments', value)
            return
        
        # Other setattr triggers loading
        patcher = self._ensure_loaded(trigger_name=f"setattr('{name}')")
        if patcher is None:
            raise RuntimeError("Failed to load model")
        setattr(patcher, name, value)
    
    def __repr__(self):
        if self._real_patcher is not None:
            return f"LazyModelPatcher(loaded={repr(self._real_patcher)})"
        return f"LazyModelPatcher(not_loaded, model={self._loader_args.get('gguf_model') or self._loader_args.get('safetensors_model')})"


# Node registration info
NODE_CLASS_MAPPINGS = {
    "WAN22BlockSwapLoader": WAN22BlockSwapLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WAN22BlockSwapLoader": "WAN BlockSwap Model Loader",
}
