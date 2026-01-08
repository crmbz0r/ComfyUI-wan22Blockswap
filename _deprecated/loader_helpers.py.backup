"""
Helper functions for WAN22BlockSwapLoader.

Provides core utilities for:
- Model format detection (safetensors, GGUF)
- State dict loading to CPU
- Model skeleton creation with empty weights
- Per-block device routing during weight assignment
- ModelPatcher creation
"""
import os
import sys
import torch
import torch.nn as nn
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Set
from contextlib import contextmanager

import comfy.utils
import comfy.model_management
import comfy.ops
from comfy.model_patcher import ModelPatcher

logger = logging.getLogger("WAN22BlockSwapLoader")


def detect_model_format(file_path: str) -> str:
    """
    Detect model file format from extension.
    
    Args:
        file_path: Path to model file
    
    Returns:
        Format string: "safetensors" or "gguf"
    
    Raises:
        ValueError: If format is not supported
    """
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    if suffix in [".safetensors", ".sft"]:
        return "safetensors"
    elif suffix == ".gguf":
        return "gguf"
    elif suffix in [".pt", ".pth", ".bin", ".ckpt"]:
        return "pytorch"
    else:
        raise ValueError(
            f"Unsupported model format: {suffix}. "
            f"Supported: .safetensors, .sft, .gguf, .pt, .pth, .bin, .ckpt"
        )


def load_state_dict_to_cpu(
    file_path: str,
    model_format: str,
) -> Tuple[Dict[str, torch.Tensor], Optional[Any], Optional[Dict[str, Any]]]:
    """
    Load model state dict to CPU memory only.
    
    This function loads the model weights to CPU, never touching GPU.
    For GGUF models, returns the reader for streaming tensor access.
    
    Args:
        file_path: Path to model file
        model_format: "safetensors", "gguf", or "pytorch"
    
    Returns:
        Tuple of (state_dict, gguf_reader or None, metadata or None)
    """
    if model_format == "safetensors":
        # Load safetensors to CPU using ComfyUI's utility
        sd, metadata = comfy.utils.load_torch_file(file_path, device="cpu", return_metadata=True)
        return sd, None, metadata
    
    elif model_format == "pytorch":
        # Load pytorch checkpoint to CPU
        sd = comfy.utils.load_torch_file(file_path, device="cpu")
        return sd, None, None
    
    elif model_format == "gguf":
        # GGUF Loading Strategy:
        # Currently we use ComfyUI-GGUF's loader as it properly integrates with
        # ComfyUI's weight assignment system. Our native loader needs more work
        # to properly handle deferred dequantization during weight assignment.
        #
        # Note: ComfyUI-GGUF's GGMLTensor may cause CUDA issues with block swap
        # after 2-3 consecutive runs. For best GGUF + BlockSwap experience,
        # consider using WanVideoWrapper's loader which has better GGUF handling.
        
        # Use ComfyUI-GGUF's loader
        try:
            # Import from ComfyUI-GGUF - the folder is ComfyUI-GGUF (with hyphen)
            # and the function is gguf_sd_loader in loader.py
            import sys
            import os
            
            # Add custom_nodes to path if needed
            custom_nodes_path = os.path.dirname(os.path.dirname(__file__))
            if custom_nodes_path not in sys.path:
                sys.path.insert(0, custom_nodes_path)
            
            # Try direct import first (works if ComfyUI-GGUF was loaded first)
            try:
                from importlib import import_module
                gguf_loader = import_module("ComfyUI-GGUF.loader")
                gguf_sd_loader = gguf_loader.gguf_sd_loader
            except (ImportError, ModuleNotFoundError):
                # Fallback: manually load the module
                gguf_path = os.path.join(custom_nodes_path, "ComfyUI-GGUF")
                if gguf_path not in sys.path:
                    sys.path.insert(0, gguf_path)
                from loader import gguf_sd_loader
            
            # Load with handle_prefix=None to get raw keys
            sd = gguf_sd_loader(file_path, handle_prefix=None)
            logger.info(f"Loaded GGUF using ComfyUI-GGUF ({len(sd)} tensors)")
            return sd, None, None
            
        except Exception as e:
            raise ImportError(
                f"GGUF loading failed: {e}\n"
                "GGUF support requires ComfyUI-GGUF custom node:\n"
                "  https://github.com/city96/ComfyUI-GGUF\n"
                "Install via ComfyUI Manager or git clone into custom_nodes/"
            )
    
    else:
        raise ValueError(f"Unsupported format: {model_format}")


@contextmanager
def init_empty_weights():
    """
    Context manager to create model with empty (meta) weights.
    
    This prevents allocating tensor storage during model instantiation.
    All parameters are created on 'meta' device with no actual memory.
    
    Usage:
        with init_empty_weights():
            model = WanModel(...)
        # model now has meta tensors, no memory used
    """
    old_register_parameter = nn.Module.register_parameter
    old_register_buffer = nn.Module.register_buffer
    
    def register_empty_parameter(module, name, param):
        if param is not None:
            param = nn.Parameter(
                torch.empty(param.shape, device="meta", dtype=param.dtype),
                requires_grad=param.requires_grad,
            )
        old_register_parameter(module, name, param)
    
    def register_empty_buffer(module, name, buffer, persistent=True):
        if buffer is not None:
            buffer = torch.empty(buffer.shape, device="meta", dtype=buffer.dtype)
        old_register_buffer(module, name, buffer, persistent)
    
    try:
        nn.Module.register_parameter = register_empty_parameter
        nn.Module.register_buffer = register_empty_buffer
        yield
    finally:
        nn.Module.register_parameter = old_register_parameter
        nn.Module.register_buffer = old_register_buffer


def get_block_index_from_key(key: str) -> Optional[int]:
    """
    Extract block index from state dict key.
    
    Args:
        key: State dict key like "blocks.15.self_attn.k.weight"
    
    Returns:
        Block index (int) or None if key is not a block weight
    
    Examples:
        >>> get_block_index_from_key("blocks.0.self_attn.k.weight")
        0
        >>> get_block_index_from_key("blocks.15.cross_attn.v.weight")
        15
        >>> get_block_index_from_key("head.modulation")
        None
    """
    if "blocks." not in key:
        return None
    
    try:
        # Handle both "blocks.15.xxx" and "diffusion_model.blocks.15.xxx"
        after_blocks = key.split("blocks.")[1]
        idx_str = after_blocks.split(".")[0]
        return int(idx_str)
    except (IndexError, ValueError):
        return None


def _module_path_exists(model: nn.Module, key: str) -> bool:
    """
    Check if a module path exists in the model.
    
    Unlike checking model.state_dict(), this works for GGMLOps.Linear
    where self.weight = None (which doesn't appear in state_dict).
    
    Args:
        model: Root module
        key: Dot-separated key path (e.g., "blocks.0.self_attn.k.weight")
    
    Returns:
        True if the module path exists (attribute can be None but must exist)
    """
    parts = key.split(".")
    
    try:
        # Navigate to parent module
        module = model
        for part in parts[:-1]:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        
        # Check if final attribute exists
        final_key = parts[-1]
        return hasattr(module, final_key)
    except (AttributeError, IndexError, KeyError, TypeError):
        return False


def assign_weights_with_routing(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    total_blocks: int,
    blocks_to_swap: int,
    main_device: torch.device,
    offload_device: torch.device,
    gguf_reader: Optional[Any] = None,  # Kept for backward compatibility, no longer used
    fp8_optimization: str = "disabled",
    key_prefix: str = "",
) -> Dict[int, str]:
    """
    Assign weights to model with per-block device routing.
    
    This is the core function that prevents VRAM overflow by loading
    swap blocks directly to CPU, never touching GPU.
    
    Supports both regular tensors (safetensors) and GGMLTensor (GGUF).
    GGMLTensor objects are moved to target device while preserving
    their tensor_type and tensor_shape attributes for later dequantization.
    
    Args:
        model: Model instance (can have meta tensors)
        state_dict: State dict loaded to CPU (regular tensors or GGMLTensor)
        total_blocks: Total number of transformer blocks
        blocks_to_swap: Number of blocks to keep on CPU
        main_device: GPU device for non-swap blocks
        offload_device: CPU device for swap blocks
        gguf_reader: DEPRECATED - no longer used, kept for API compatibility
        fp8_optimization: FP8 quantization mode ("disabled", "e4m3fn", "e5m2")
        key_prefix: Prefix to strip from state dict keys
    
    Returns:
        Dict mapping block_index to device string for logging
    """
    # Calculate swap threshold
    # Blocks >= swap_threshold go to CPU
    swap_threshold = total_blocks - blocks_to_swap
    
    # Track which blocks go where
    block_device_map: Dict[int, str] = {}
    
    # Track progress
    total_keys = len(state_dict)
    progress_interval = max(1, total_keys // 10)
    loaded_count = 0
    skipped_count = 0
    skipped_keys = []  # Track first few skipped keys for debugging
    
    # NOTE: We cannot use model.state_dict() for checking key existence when using GGMLOps
    # because GGMLOps.Linear sets self.weight = None, which doesn't appear in state_dict.
    # Instead, we navigate to each module directly and check if the attribute exists.
    
    logger.info(f"Starting weight assignment: {total_keys} keys to process")
    logger.info(f"Blocks to GPU: 0-{swap_threshold-1}, Blocks to CPU: {swap_threshold}-{total_blocks-1}")
    
    for idx, (key, value) in enumerate(state_dict.items()):
        # Log progress periodically
        if idx % progress_interval == 0:
            logger.debug(f"Loading weights: {idx}/{total_keys}")
        
        # Strip prefix if present
        clean_key = key
        if key_prefix and key.startswith(key_prefix):
            clean_key = key[len(key_prefix):]
        
        # Try to set the module tensor - if it fails, the path doesn't exist
        # This is more reliable than _module_path_exists for GGMLOps modules
        try:
            # Test navigation first
            parts = clean_key.split(".")
            test_module = model
            for part in parts[:-1]:
                if part.isdigit():
                    test_module = test_module[int(part)]
                else:
                    test_module = getattr(test_module, part)
            # If we get here, path exists
        except (AttributeError, IndexError, KeyError, TypeError) as e:
            if len(skipped_keys) < 20:  # Store first 20 for debugging
                skipped_keys.append((clean_key, str(e)))
            skipped_count += 1
            continue
        
        # Determine target device for this weight
        block_idx = get_block_index_from_key(clean_key)
        
        if block_idx is not None and block_idx >= swap_threshold:
            # Swap block - load to CPU
            target_device = offload_device
            target_dtype = None  # Preserve original dtype on CPU
            
            if block_idx not in block_device_map:
                block_device_map[block_idx] = "cpu"
        else:
            # Non-swap block or non-block weight - load to GPU
            target_device = main_device
            
            if block_idx is not None and block_idx not in block_device_map:
                block_device_map[block_idx] = str(main_device)
            
            # Apply FP8 optimization if enabled (only for GPU blocks, not GGUF)
            is_ggml_tensor = hasattr(value, 'tensor_type') and value.tensor_type is not None
            if not is_ggml_tensor and fp8_optimization == "e4m3fn" and value.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                target_dtype = torch.float8_e4m3fn
            elif not is_ggml_tensor and fp8_optimization == "e5m2" and value.dtype in [torch.float16, torch.bfloat16, torch.float32]:
                target_dtype = torch.float8_e5m2
            else:
                target_dtype = None  # Preserve original dtype
        
        # Handle tensor movement to target device
        # GGMLTensor from gguf custom node has tensor_type attribute
        # Its .to() method preserves tensor_type and tensor_shape automatically
        is_ggml_tensor = hasattr(value, 'tensor_type') and value.tensor_type is not None
        
        if is_ggml_tensor:
            # GGMLTensor: move to device while preserving quantization info
            # The .to() method is overridden to preserve tensor_type and tensor_shape
            tensor = value.to(device=target_device)
        elif target_dtype is not None and target_dtype != value.dtype:
            # Regular tensor with dtype conversion
            tensor = value.to(device=target_device, dtype=target_dtype)
        else:
            # Regular tensor, preserve dtype
            tensor = value.to(device=target_device)
        
        # Assign to model using set_module_tensor_to_device pattern
        try:
            _set_module_tensor(model, clean_key, tensor)
            loaded_count += 1
        except Exception as e:
            logger.error(f"Failed to assign weight for key '{clean_key}': {e}")
            if len(skipped_keys) < 20:
                skipped_keys.append((clean_key, str(e)))
            skipped_count += 1
    
    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} keys not found in model (may indicate key mismatch)")
        if skipped_keys:
            logger.warning("First skipped keys:")
            for key, error in skipped_keys[:10]:
                logger.warning(f"  {key}: {error}")
    
    logger.info(
        f"Weight assignment complete: {loaded_count}/{total_keys} tensors loaded. "
        f"GPU blocks: 0-{swap_threshold-1}, CPU blocks: {swap_threshold}-{total_blocks-1}"
    )
    
    # Materialize any remaining meta tensors (buffers not in state_dict)
    _materialize_meta_tensors(model, main_device, offload_device)
    
    return block_device_map


def _materialize_meta_tensors(
    model: nn.Module,
    main_device: torch.device,
    fallback_device: torch.device,
) -> None:
    """
    Materialize any remaining meta tensors in the model.
    
    Some model buffers (like RoPE freqs, sinusoidal embeddings) may not be
    in the state_dict and remain as meta tensors after weight assignment.
    This function creates real tensors for them.
    
    Also checks for None weights (GGMLOps.Linear can have weight=None if not loaded).
    
    Args:
        model: Model to check
        main_device: Primary device (GPU)
        fallback_device: Fallback device (CPU)
    """
    meta_count = 0
    none_count = 0
    
    # Check parameters
    for name, param in model.named_parameters():
        if param is None:
            logger.error(f"Parameter is None: {name} - this should not happen!")
            none_count += 1
        elif param.device.type == "meta":
            # Create a real tensor on the fallback device
            logger.warning(f"Meta parameter found: {name}, materializing on {fallback_device}")
            new_param = nn.Parameter(
                torch.zeros(param.shape, dtype=param.dtype, device=fallback_device),
                requires_grad=False
            )
            # Navigate and set
            _set_module_tensor(model, name, new_param)
            meta_count += 1
    
    # Check buffers
    for name, buffer in model.named_buffers():
        if buffer is None:
            logger.warning(f"Buffer is None: {name}, creating zero buffer on {fallback_device}")
            # We can't easily determine the shape, so skip
            none_count += 1
        elif buffer.device.type == "meta":
            # Create a real tensor on the fallback device
            logger.warning(f"Meta buffer found: {name}, materializing on {fallback_device}")
            new_buffer = torch.zeros(buffer.shape, dtype=buffer.dtype, device=fallback_device)
            _set_module_tensor(model, name, new_buffer)
            meta_count += 1
    
    # Check for None weights in Linear modules (GGMLOps.Linear issue)
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is None:
            # This is a problem - weight should have been loaded
            logger.error(f"Module {name} has weight=None after loading! This will cause errors.")
            none_count += 1
    
    if meta_count > 0:
        logger.warning(f"Materialized {meta_count} meta tensors to {fallback_device}")
    
    if none_count > 0:
        logger.error(f"Found {none_count} None weights/params - model may not work correctly!")


def _set_module_tensor(
    model: nn.Module,
    key: str,
    tensor: torch.Tensor,
) -> None:
    """
    Set a tensor in a module by navigating the key path.
    
    For GGMLTensor (GGUF quantized weights), we need special handling
    to preserve the quantization metadata for proper dequantization
    during forward pass.
    
    Args:
        model: Root module
        key: Dot-separated key path (e.g., "blocks.0.self_attn.k.weight")
        tensor: Tensor to set (can be GGMLTensor or regular Tensor)
    """
    parts = key.split(".")
    
    # Navigate to parent module
    module = model
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    
    # Set the final attribute
    final_key = parts[-1]
    
    # Check if this is a GGMLTensor that needs special handling
    is_ggml_tensor = hasattr(tensor, 'tensor_type') and tensor.tensor_type is not None
    
    if hasattr(module, final_key):
        param = getattr(module, final_key)
        if isinstance(param, nn.Parameter):
            # Existing parameter - replace it with new tensor wrapped in Parameter
            setattr(module, final_key, nn.Parameter(tensor, requires_grad=False))
        elif param is None:
            # GGMLOps.Linear initializes with self.weight = None
            # We need to set it as nn.Parameter regardless of tensor type
            # This handles both GGMLTensor and regular tensors (like dequantized biases)
            setattr(module, final_key, nn.Parameter(tensor, requires_grad=False))
        else:
            # Replace buffer or other attribute (not a parameter)
            setattr(module, final_key, tensor)
    else:
        # Attribute doesn't exist yet - create it as Parameter if it looks like a weight/bias
        if final_key in ('weight', 'bias'):
            setattr(module, final_key, nn.Parameter(tensor, requires_grad=False))
        else:
            setattr(module, final_key, tensor)


def create_model_skeleton(
    wan_config: Dict[str, Any],
    dtype: torch.dtype = torch.float16,
    device: str = "meta",
    is_gguf: bool = False,
) -> nn.Module:
    """
    Create WAN model skeleton with empty (meta) weights.
    
    This creates a model structure without allocating any GPU/CPU memory
    for the weights. The actual weights are loaded later via
    assign_weights_with_routing().
    
    Args:
        wan_config: Model configuration from detect_wan_config()
        dtype: Model dtype for inference
        device: Device for skeleton ("meta" for no memory)
        is_gguf: True if loading GGUF model (uses GGMLOps for proper dequantization)
    
    Returns:
        WanModel instance with empty (meta) weights
    """
    from comfy.ldm.wan.model import WanModel, WanAttentionBlock, VaceWanAttentionBlock
    
    variant = wan_config.get("model_variant", "t2v")
    wan_version = wan_config.get("wan_version")
    
    # If version is None, default to 2.2 (safer for modern models)
    if wan_version is None:
        logger.warning("wan_version is None in config! Defaulting to '2.2'")
        wan_version = "2.2"
    
    logger.info(f"create_model_skeleton: variant={variant}, wan_version={wan_version}")
    
    # Determine block class based on variant
    if variant == "vace":
        wan_attn_block_class = VaceWanAttentionBlock
    else:
        wan_attn_block_class = WanAttentionBlock
    
    # Determine model_type for WanModel
    # WAN 2.1 I2V: Uses 'i2v' (has k_img/v_img in cross-attention, img_emb projection)
    # WAN 2.2 I2V: Uses 't2v' (no img_emb, uses larger in_dim=36 instead)
    # WAN 2.1/2.2 T2V: Uses 't2v'
    if variant in ["i2v", "camera", "camera_2.2"]:
        if wan_version == "2.2":
            # WAN 2.2 I2V uses T2V architecture with larger in_dim
            model_type = "t2v"
            logger.info(f"WAN 2.2 I2V: Using model_type='t2v' (no img_emb, in_dim={wan_config.get('in_dim', 36)})")
        else:
            # WAN 2.1 I2V uses I2V architecture with img_emb
            model_type = "i2v"
    else:
        model_type = "t2v"
    
    # Create operations for the model
    # For GGUF models, use GGMLOps which properly dequantizes weights during forward
    if is_gguf:
        try:
            # Add ComfyUI-GGUF folder to path if needed
            gguf_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ComfyUI-GGUF")
            if gguf_folder not in sys.path:
                sys.path.insert(0, gguf_folder)
            from ops import GGMLOps
            operations = GGMLOps
            logger.info("Using GGMLOps for GGUF model (proper dequantization)")
        except ImportError as e:
            logger.warning(f"Could not import GGMLOps ({e}), falling back to manual_cast")
            operations = comfy.ops.manual_cast
    else:
        operations = comfy.ops.manual_cast
    
    with init_empty_weights():
        model = WanModel(
            model_type=model_type,
            patch_size=wan_config.get("patch_size", (1, 2, 2)),
            text_len=wan_config.get("text_len", 512),
            in_dim=wan_config.get("in_dim", 16),
            dim=wan_config.get("dim", 1536),
            ffn_dim=wan_config.get("ffn_dim", 1536 * 4),
            freq_dim=wan_config.get("freq_dim", 256),
            text_dim=wan_config.get("text_dim", 4096),
            out_dim=wan_config.get("out_dim", 16),
            num_heads=wan_config.get("num_heads", 12),
            num_layers=wan_config.get("num_layers", 30),
            window_size=wan_config.get("window_size", (-1, -1)),
            qk_norm=wan_config.get("qk_norm", True),
            cross_attn_norm=wan_config.get("cross_attn_norm", True),
            eps=wan_config.get("eps", 1e-6),
            flf_pos_embed_token_number=wan_config.get("flf_pos_embed_token_number"),
            wan_attn_block_class=wan_attn_block_class,
            device=device,
            dtype=dtype,
            operations=operations,
        )
    
    return model


def create_model_patcher(
    diffusion_model: nn.Module,
    wan_config: Dict[str, Any],
    load_device: torch.device,
    offload_device: torch.device,
    blocks_to_swap: int,
    model_options: Optional[Dict[str, Any]] = None,
    is_gguf: bool = False,
) -> ModelPatcher:
    """
    Create ModelPatcher wrapping the loaded WAN model.
    
    This wraps the diffusion model in the appropriate ComfyUI base model
    class and then in a ModelPatcher for integration with ComfyUI's
    model management system.
    
    Args:
        diffusion_model: Loaded WanModel with weights
        wan_config: Model configuration
        load_device: GPU device for inference
        offload_device: CPU device for offloading
        blocks_to_swap: Number of blocks pre-routed to CPU
        model_options: Optional model options dict
        is_gguf: True if this is a GGUF quantized model
    
    Returns:
        ModelPatcher ready for ComfyUI integration
    """
    from .model_detection import get_model_class_for_config
    
    # Get the appropriate model class
    ModelClass, _ = get_model_class_for_config(wan_config)
    
    # Create a minimal model config object
    model_config = _create_model_config(wan_config)
    
    # Determine if this is an I2V model
    variant = wan_config.get("model_variant", "t2v")
    is_i2v = variant in ["i2v", "camera", "camera_2.2"]
    
    # Create the base model wrapper
    base_model = ModelClass(model_config, image_to_video=is_i2v)
    
    # Assign the diffusion model
    base_model.diffusion_model = diffusion_model
    
    # Create the patcher (use GGUFModelPatcher for GGUF models)
    if is_gguf:
        try:
            # Import GGUFModelPatcher from ComfyUI-GGUF
            # The module is loaded by ComfyUI, so we need to access it properly
            import sys
            import os
            
            # Try direct import from the loaded module
            try:
                # If ComfyUI-GGUF was already loaded, its nodes module is available
                gguf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ComfyUI-GGUF")
                if gguf_path not in sys.path:
                    sys.path.insert(0, gguf_path)
                from nodes import GGUFModelPatcher
            except ImportError:
                # Fallback - shouldn't happen if ComfyUI-GGUF is installed
                raise ImportError("ComfyUI-GGUF not found")
            
            # Create a GGUFModelPatcher directly (same interface as ModelPatcher)
            patcher = GGUFModelPatcher(
                model=base_model,
                load_device=load_device,
                offload_device=offload_device,
            )
            logger.info("Created GGUFModelPatcher for GGUF model")
        except ImportError as e:
            logger.warning(f"Could not import GGUFModelPatcher ({e}), falling back to regular ModelPatcher")
            patcher = ModelPatcher(
                model=base_model,
                load_device=load_device,
                offload_device=offload_device,
            )
    else:
        patcher = ModelPatcher(
            model=base_model,
            load_device=load_device,
            offload_device=offload_device,
        )
    
    # Store routing info for downstream BlockSwap node detection
    patcher.model_options = patcher.model_options or {}
    patcher.model_options["wan22_blockswap_info"] = {
        "blocks_to_swap": blocks_to_swap,
        "total_blocks": wan_config.get("num_layers", 30),
        "pre_routed": True,
        "swap_block_indices": list(range(
            wan_config.get("num_layers", 30) - blocks_to_swap,
            wan_config.get("num_layers", 30)
        )),
    }
    
    if model_options:
        patcher.model_options.update(model_options)
    
    return patcher


def _create_model_config(wan_config: Dict[str, Any]):
    """
    Create a minimal model configuration object for ComfyUI.
    
    Args:
        wan_config: WAN configuration dict
    
    Returns:
        Model config object compatible with WAN21/WAN22 base classes
    """
    from comfy import latent_formats
    
    # Determine WAN version from config
    wan_version = wan_config.get("wan_version", "2.1")
    model_variant = wan_config.get("model_variant", "t2v")
    
    # Select appropriate latent format based on version and variant
    # Note: WAN 2.2 I2V uses WAN 2.1 VAE, only WAN 2.2 T2V uses new VAE
    if wan_version == "2.2" and model_variant == "t2v":
        latent_format_class = latent_formats.Wan22
    else:
        # WAN 2.1, or WAN 2.2 non-T2V variants use WAN 2.1 latent format
        latent_format_class = latent_formats.Wan21
    
    # Filter config to only include valid WanModel parameters
    # These are the parameters accepted by WanModel.__init__()
    valid_unet_keys = {
        "model_type",  # 't2v' or 'i2v'
        "patch_size",
        "text_len",
        "in_dim",
        "dim",
        "ffn_dim",
        "freq_dim",
        "text_dim",
        "out_dim",
        "num_heads",
        "num_layers",
        "window_size",
        "qk_norm",
        "cross_attn_norm",
        "eps",
        "flf_pos_embed_token_number",
        "in_dim_ref_conv",
        # VACE-specific
        "vace_in_dim",
        "vace_layers",
        # Disable auto model creation - we create our own
        "disable_unet_model_creation",
    }
    
    # Build filtered unet_config
    unet_config = {k: v for k, v in wan_config.items() if k in valid_unet_keys}
    
    # Derive model_type based on variant AND version
    # WAN 2.1 I2V: Uses 'i2v' (has k_img/v_img in cross-attention, img_emb projection)
    # WAN 2.2 I2V: Uses 't2v' (no img_emb, uses larger in_dim=36 instead)
    variant = wan_config.get("model_variant", "t2v")
    version = wan_config.get("wan_version", "2.1")
    
    if variant in ["i2v", "camera", "camera_2.2", "vace"]:
        if version == "2.2" and variant == "i2v":
            # WAN 2.2 I2V uses T2V architecture with larger in_dim
            unet_config["model_type"] = "t2v"
        else:
            # WAN 2.1 I2V and other variants use I2V architecture
            unet_config["model_type"] = "i2v"
    else:
        unet_config["model_type"] = "t2v"
    
    # Disable auto model creation since we create the model ourselves
    unet_config["disable_unet_model_creation"] = True
    
    # Create a config-like object that matches ComfyUI's expected interface
    class WANModelConfig:
        def __init__(self, config: Dict[str, Any], latent_fmt):
            self.unet_config = config
            self.sampling_settings = {
                "sigma_max": 1.0,
                "sigma_min": 0.0,
            }
            self.latent_format = latent_fmt()
            self.supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]
            self.memory_usage_factor = 1.0
            
            # Required by ComfyUI's BaseModel.__init__
            self.manual_cast_dtype = None
            self.custom_operations = None
            self.optimizations = {}
            
        def get_model(self, *args, **kwargs):
            # This won't be called since we create the model ourselves
            pass
    
    return WANModelConfig(unet_config, latent_format_class)


def verify_block_devices(
    model: nn.Module,
    total_blocks: int,
    blocks_to_swap: int,
) -> bool:
    """
    Verify that blocks are on the correct devices after loading.
    
    Args:
        model: Loaded model
        total_blocks: Total number of blocks
        blocks_to_swap: Number of blocks that should be on CPU
    
    Returns:
        True if all blocks are on correct devices
    """
    swap_threshold = total_blocks - blocks_to_swap
    all_correct = True
    
    for i, block in enumerate(model.blocks):
        # Check first parameter of block
        for param in block.parameters():
            device = param.device
            expected_gpu = i < swap_threshold
            is_gpu = device.type == "cuda"
            
            if expected_gpu != is_gpu:
                logger.warning(
                    f"Block {i}: expected {'GPU' if expected_gpu else 'CPU'}, "
                    f"got {device}"
                )
                all_correct = False
            break  # Only check first param per block
    
    return all_correct
