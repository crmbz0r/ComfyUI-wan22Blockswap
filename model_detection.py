"""
WAN model detection utilities.

Detects WAN version (2.1/2.2), variant (t2v/i2v/vace/etc.),
and model parameters from state dict keys and shapes.
"""

import torch
import logging
from typing import Dict, Any, Optional, Set, Tuple

logger = logging.getLogger("WAN22BlockSwapLoader.Detection")


def _get_tensor_shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    """
    Safely get tensor shape, handling GGMLTensor from GGUF models.
    
    GGMLTensor stores the logical (dequantized) shape in `tensor_shape`,
    while the actual data shape is the packed byte representation.
    
    Args:
        tensor: Regular tensor or GGMLTensor
    
    Returns:
        Tuple of shape dimensions
    """
    # GGMLTensor from gguf custom node stores logical shape in tensor_shape
    if hasattr(tensor, 'tensor_shape'):
        shape = tensor.tensor_shape
        # tensor_shape can be a torch.Size or tuple
        if isinstance(shape, torch.Size):
            return tuple(shape)
        return tuple(shape)
    
    # Regular tensor - use .shape directly
    return tuple(tensor.shape)


# Key patterns for WAN detection (based on ComfyUI's model_detection.py)
WAN_DETECTION_KEYS = {
    "wan_base": "head.modulation",  # Present in all WAN models
    "wan_blocks": "blocks.0.self_attn.k.weight",  # Transformer blocks
}

# Variant detection keys (order matters - more specific first)
VARIANT_DETECTION_KEYS = [
    ("vace", "vace_patch_embedding.weight"),
    ("camera_2.2", "control_adapter.conv.weight"),  # Camera 2.2 has no img_emb
    ("camera", "img_emb.proj.0.bias"),  # Camera 2.1 has img_emb
    ("s2v", "casual_audio_encoder.encoder.final_linear.weight"),
    ("humo", "audio_proj.audio_proj_glob_1.layer.bias"),
    ("animate", "face_adapter.fuser_blocks.0.k_norm.weight"),
    ("i2v", "img_emb.proj.0.bias"),  # Must be after camera check
]

# Model dimension to config mapping
DIM_TO_CONFIG = {
    1536: {"model_size": "1.3B", "num_layers": 30, "num_heads": 12},
    2048: {"model_size": "5B", "num_layers": 30, "num_heads": 16},
    3072: {"model_size": "5B-alt", "num_layers": 30, "num_heads": 24},
    4096: {"model_size": "14B", "num_layers": 40, "num_heads": 32},
    5120: {"model_size": "14B-alt", "num_layers": 40, "num_heads": 40},
}


def detect_wan_config(
    state_dict: Dict[str, torch.Tensor],
    wan_version: str = "auto",
    model_variant: str = "auto",
    key_prefix: str = "",
) -> Dict[str, Any]:
    """
    Detect WAN model configuration from state dict.
    
    Analyzes state dict keys and tensor shapes to determine:
    - WAN version (2.1 or 2.2)
    - Model variant (t2v, i2v, vace, camera, s2v, humo, animate)
    - Model size and architecture parameters
    
    Args:
        state_dict: Model state dict
        wan_version: "auto" to detect, or "2.1"/"2.2" to override
        model_variant: "auto" to detect, or specific variant
        key_prefix: Prefix for state dict keys (e.g., "diffusion_model.")
    
    Returns:
        Configuration dict with model parameters
    
    Raises:
        ValueError: If state dict is not a WAN model
    """
    config: Dict[str, Any] = {}
    state_dict_keys: Set[str] = set(state_dict.keys())
    
    # Verify this is a WAN model
    base_key = f"{key_prefix}{WAN_DETECTION_KEYS['wan_base']}"
    has_wan_keys = any(base_key in key for key in state_dict_keys)
    if not has_wan_keys:
        raise ValueError(
            f"State dict does not appear to be a WAN model. "
            f"Missing key pattern: {base_key}"
        )
    
    # Detect model dimensions from modulation weight
    modulation_key = f"{key_prefix}head.modulation"
    dim = _detect_model_dim(state_dict, key_prefix)
    config["dim"] = dim
    
    # Get config from dimension lookup
    if dim in DIM_TO_CONFIG:
        config.update(DIM_TO_CONFIG[dim])
    else:
        logger.warning(f"Unknown model dimension: {dim}. Detecting from state dict.")
        config["num_layers"] = _count_blocks(state_dict_keys, key_prefix)
        config["num_heads"] = dim // 128  # Estimate
        config["model_size"] = "unknown"
    
    # Verify/override num_layers from actual block count
    actual_layers = _count_blocks(state_dict_keys, key_prefix)
    if actual_layers != config.get("num_layers", 0):
        logger.info(
            f"Block count mismatch: config={config.get('num_layers')}, "
            f"actual={actual_layers}. Using actual."
        )
        config["num_layers"] = actual_layers
    
    # Detect variant
    if model_variant == "auto":
        config["model_variant"] = _detect_variant(state_dict_keys, key_prefix, state_dict)
    else:
        config["model_variant"] = model_variant
    
    # Detect/set WAN version
    if wan_version == "auto":
        config["wan_version"] = _detect_wan_version(state_dict_keys, config, key_prefix, state_dict)
    else:
        config["wan_version"] = wan_version
    
    # Set derived parameters
    config["model_type"] = config["model_variant"]
    if config["model_variant"] in ["camera", "camera_2.2"]:
        config["model_type"] = "i2v"  # Camera variants are I2V-based
    
    # Get FFN dim from state dict
    ffn_key = f"{key_prefix}blocks.0.ffn.0.weight"
    if ffn_key in state_dict:
        config["ffn_dim"] = _get_tensor_shape(state_dict[ffn_key])[0]
    else:
        config["ffn_dim"] = dim * 4  # Default ratio
    
    # Get in_dim from patch embedding
    patch_key = f"{key_prefix}patch_embedding.weight"
    if patch_key in state_dict:
        config["in_dim"] = _get_tensor_shape(state_dict[patch_key])[1]
    else:
        config["in_dim"] = 16  # Default
    
    # Get out_dim from head weight
    head_key = f"{key_prefix}head.head.weight"
    if head_key in state_dict:
        config["out_dim"] = _get_tensor_shape(state_dict[head_key])[0] // 4
    else:
        config["out_dim"] = 16  # Default
    
    # Standard parameters
    config["text_len"] = 512
    config["freq_dim"] = 256
    config["text_dim"] = 4096
    config["patch_size"] = (1, 2, 2)
    config["window_size"] = (-1, -1)
    config["qk_norm"] = True
    config["cross_attn_norm"] = True
    config["eps"] = 1e-6
    
    # VACE-specific parameters
    if config["model_variant"] == "vace":
        vace_key = f"{key_prefix}vace_patch_embedding.weight"
        if vace_key in state_dict:
            config["vace_in_dim"] = _get_tensor_shape(state_dict[vace_key])[1]
        config["vace_layers"] = _count_blocks(
            state_dict_keys, key_prefix, block_pattern="vace_blocks."
        )
    
    # FLF position embedding
    flf_key = f"{key_prefix}img_emb.emb_pos"
    if flf_key in state_dict:
        config["flf_pos_embed_token_number"] = _get_tensor_shape(state_dict[flf_key])[1]
    
    logger.info(
        f"Detected WAN config: version={config['wan_version']}, "
        f"variant={config['model_variant']}, size={config.get('model_size', 'unknown')}, "
        f"blocks={config['num_layers']}, dim={config['dim']}"
    )
    
    return config


def _detect_model_dim(
    state_dict: Dict[str, torch.Tensor],
    key_prefix: str = "",
) -> int:
    """
    Detect model hidden dimension from weights.
    
    Args:
        state_dict: Model state dict
        key_prefix: Prefix for state dict keys
    
    Returns:
        Hidden dimension (int)
    
    Raises:
        ValueError: If dimension cannot be detected
    """
    # Try head.modulation first (most reliable)
    modulation_key = f"{key_prefix}head.modulation"
    for key, tensor in state_dict.items():
        if modulation_key in key:
            return _get_tensor_shape(tensor)[-1]
    
    # Try self-attention weights
    attn_keys = [
        f"{key_prefix}blocks.0.self_attn.q.weight",
        f"{key_prefix}blocks.0.self_attn.k.weight",
    ]
    for attn_key in attn_keys:
        if attn_key in state_dict:
            return _get_tensor_shape(state_dict[attn_key])[0]
    
    raise ValueError("Could not detect model dimension from state dict")


def _count_blocks(
    state_dict_keys: Set[str],
    key_prefix: str = "",
    block_pattern: str = "blocks.",
) -> int:
    """
    Count number of transformer blocks.
    
    Args:
        state_dict_keys: Set of state dict keys
        key_prefix: Prefix for keys
        block_pattern: Pattern to match blocks (e.g., "blocks." or "vace_blocks.")
    
    Returns:
        Number of blocks (int)
    """
    full_pattern = f"{key_prefix}{block_pattern}"
    max_block = -1
    
    for key in state_dict_keys:
        if full_pattern in key:
            try:
                # Extract index from pattern like "blocks.15.xxx"
                after_pattern = key.split(full_pattern)[1]
                idx_str = after_pattern.split(".")[0]
                idx = int(idx_str)
                max_block = max(max_block, idx)
            except (IndexError, ValueError):
                pass
    
    return max_block + 1 if max_block >= 0 else 0


def _detect_variant(
    state_dict_keys: Set[str],
    key_prefix: str = "",
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> str:
    """
    Detect WAN model variant from state dict keys and tensor shapes.
    
    The detection logic follows this order of specificity:
    1. VACE (has vace_patch_embedding)
    2. Camera 2.2 (has control_adapter but no img_emb)  
    3. Camera 2.1 (has both control_adapter and img_emb)
    4. S2V (has casual_audio_encoder)
    5. HuMo (has audio_proj)
    6. Animate (has face_adapter)
    7. I2V vs T2V (distinguished by in_dim or img_emb keys)
    
    Args:
        state_dict_keys: Set of state dict keys
        key_prefix: Prefix for keys
        state_dict: Optional state dict for checking tensor shapes
    
    Returns:
        Variant string: "t2v", "i2v", "vace", "camera", "camera_2.2", "s2v", "humo", "animate"
    """
    # Check for VACE first
    vace_key = f"{key_prefix}vace_patch_embedding.weight"
    if any(vace_key in key for key in state_dict_keys):
        logger.info("Detected variant: vace")
        return "vace"
    
    # Check for control_adapter (camera variants)
    control_key = f"{key_prefix}control_adapter.conv.weight"
    has_control = any(control_key in key for key in state_dict_keys)
    
    # Check for img_emb (I2V indicator in WAN 2.1)
    img_emb_key = f"{key_prefix}img_emb.proj.0.bias"
    has_img_emb = any(img_emb_key in key for key in state_dict_keys)
    
    if has_control:
        if has_img_emb:
            logger.info("Detected variant: camera (2.1)")
            return "camera"
        else:
            logger.info("Detected variant: camera_2.2")
            return "camera_2.2"
    
    # Check for S2V
    s2v_key = f"{key_prefix}casual_audio_encoder.encoder.final_linear.weight"
    if any(s2v_key in key for key in state_dict_keys):
        logger.info("Detected variant: s2v")
        return "s2v"
    
    # Check for HuMo
    humo_key = f"{key_prefix}audio_proj.audio_proj_glob_1.layer.bias"
    if any(humo_key in key for key in state_dict_keys):
        logger.info("Detected variant: humo")
        return "humo"
    
    # Check for Animate
    animate_key = f"{key_prefix}face_adapter.fuser_blocks.0.k_norm.weight"
    if any(animate_key in key for key in state_dict_keys):
        logger.info("Detected variant: animate")
        return "animate"
    
    # Now distinguish between I2V and T2V
    # WAN 2.1 I2V has img_emb keys
    if has_img_emb:
        logger.info("Detected variant: i2v (has img_emb)")
        return "i2v"
    
    # WAN 2.2 I2V doesn't have img_emb but has larger in_dim
    # T2V has in_dim=16, I2V has in_dim=36
    if state_dict is not None:
        patch_key = f"{key_prefix}patch_embedding.weight"
        if patch_key in state_dict:
            in_dim = _get_tensor_shape(state_dict[patch_key])[1]
            if in_dim > 16:
                # in_dim > 16 indicates I2V (image conditioning channels)
                # 36 = 16 (latent) + 20 (image conditioning)
                logger.info(f"Detected variant: i2v (in_dim={in_dim} > 16)")
                return "i2v"
    
    # Default to t2v (text-to-video)
    logger.info("No variant-specific keys found, assuming t2v")
    return "t2v"


def _detect_wan_version(
    state_dict_keys: Set[str],
    config: Dict[str, Any],
    key_prefix: str = "",
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> str:
    """
    Detect WAN version (2.1 or 2.2).
    
    Detection method based on GGUF metadata analysis:
    - WAN 2.1: Uses F32 for bias tensors
    - WAN 2.2: Uses F16 for bias tensors
    - WAN 2.1 I2V: Has k_img/v_img keys in cross_attn (tensor_count=1303)
    - WAN 2.2 I2V: No k_img/v_img, uses larger in_dim (tensor_count=1095)
    - WAN 2.2 specific variants: s2v, animate, camera_2.2
    
    Args:
        state_dict_keys: Set of state dict keys
        config: Current configuration dict
        key_prefix: Prefix for keys
        state_dict: Optional state dict for dtype checking
    
    Returns:
        Version string: "2.1" or "2.2"
    """
    variant = config.get("model_variant", "t2v")
    
    # WAN 2.2 specific variants
    if variant in ["s2v", "animate", "camera_2.2"]:
        return "2.2"
    
    # Check for I2V - WAN 2.2 I2V doesn't have img_emb or k_img keys
    img_emb_key = f"{key_prefix}img_emb.proj.0.bias"
    k_img_key = f"{key_prefix}blocks.0.cross_attn.k_img.weight"
    has_img_emb = any(img_emb_key in key for key in state_dict_keys)
    has_k_img = any(k_img_key in key for key in state_dict_keys)
    
    if variant == "i2v":
        if has_img_emb or has_k_img:
            # WAN 2.1 I2V has img_emb and k_img/v_img attention keys
            return "2.1"
        else:
            # WAN 2.2 I2V uses larger in_dim without img_emb or k_img
            return "2.2"
    
    # For T2V and other variants, check bias tensor dtype
    # WAN 2.1 uses F32 for biases, WAN 2.2 uses F16
    if state_dict is not None:
        bias_key = f"{key_prefix}patch_embedding.bias"
        if bias_key in state_dict:
            tensor = state_dict[bias_key]
            # Handle GGMLTensor or regular tensor
            if hasattr(tensor, 'dtype'):
                dtype = tensor.dtype
            elif hasattr(tensor, 'tensor_type'):
                # GGMLTensor - check the tensor_type string
                tensor_type = str(getattr(tensor, 'tensor_type', '')).upper()
                if 'F32' in tensor_type or 'FLOAT32' in tensor_type:
                    return "2.1"
                elif 'F16' in tensor_type or 'FLOAT16' in tensor_type:
                    return "2.2"
            else:
                dtype = None
            
            if dtype is not None:
                if dtype == torch.float32:
                    return "2.1"
                elif dtype == torch.float16:
                    return "2.2"
    
    # Default to 2.1 for unknown cases
    return "2.1"


def get_model_class_for_config(config: Dict[str, Any]):
    """
    Get the appropriate ComfyUI model class for the detected config.
    
    Args:
        config: Configuration dict from detect_wan_config()
    
    Returns:
        Tuple of (ModelClass, unet_model_class)
    """
    from comfy.model_base import WAN21, WAN22, WAN21_Vace, WAN21_Camera, WAN22_S2V, WAN22_Animate
    from comfy.ldm.wan import model as wan_model
    
    variant = config.get("model_variant", "t2v")
    version = config.get("wan_version", "2.1")
    
    # Map variant to model class
    if variant == "vace":
        return WAN21_Vace, wan_model.VaceWanModel
    elif variant in ["camera", "camera_2.2"]:
        return WAN21_Camera, wan_model.CameraWanModel
    elif variant == "s2v":
        return WAN22_S2V, wan_model.WanModel_S2V
    elif variant == "animate":
        return WAN22_Animate, None  # Uses model_animate module
    elif version == "2.2":
        return WAN22, wan_model.WanModel
    else:
        return WAN21, wan_model.WanModel
