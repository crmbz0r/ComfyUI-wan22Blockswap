"""
GGUF model loading for WAN models.

This module provides functions to load GGUF-quantized WAN models
with support for streaming tensor access and deferred dequantization.

GGUF (GPT-Generated Unified Format) is a format for storing quantized
model weights that enables significant memory savings while maintaining
model quality.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from .gguf_utils import GGUFParameter

logger = logging.getLogger("WAN22BlockSwapLoader.GGUF")

# Try to import gguf package
try:
    import gguf
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    gguf = None

# GGUF tensor type constants
GGUF_TYPE_F32 = 0
GGUF_TYPE_F16 = 1


def load_gguf(file_path: str) -> Tuple[Dict[str, torch.Tensor], Any]:
    """
    Load GGUF model file with eager data loading.
    
    This function loads a GGUF file and creates a state dict with
    GGUFParameter tensors containing the actual quantized data.
    Dequantization happens on-demand during forward pass via the
    GGUFParameter's __torch_function__ hook.
    
    Unlike ComfyUI-GGUF's GGMLTensor, our GGUFParameter uses standard
    torch.Tensor._make_subclass which works correctly with block swapping
    operations (.to(), .clone(), .detach() all work as expected).
    
    Args:
        file_path: Path to GGUF model file
    
    Returns:
        Tuple of (state_dict with GGUFParameters, GGUF reader)
    
    Raises:
        ImportError: If gguf package is not installed
        FileNotFoundError: If file doesn't exist
    """
    if not GGUF_AVAILABLE:
        raise ImportError(
            "GGUF support requires the 'gguf' package. "
            "Install with: pip install gguf"
        )
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"GGUF file not found: {file_path}")
    
    logger.info(f"Loading GGUF file: {file_path}")
    reader = gguf.GGUFReader(str(file_path))
    
    # Parse metadata for debugging
    metadata = _parse_metadata(reader)
    logger.debug(f"GGUF metadata: {len(metadata)} fields")
    
    # Create state dict with actual tensor data (eager loading)
    state_dict: Dict[str, torch.Tensor] = {}
    quantized_count = 0
    float_count = 0
    
    for tensor_info in reader.tensors:
        name = tensor_info.name
        quant_type = tensor_info.tensor_type
        shape = tuple(tensor_info.shape)
        
        # Load actual tensor data from GGUF file
        # tensor_info.data is a numpy array with the raw data
        try:
            import numpy as np
            raw_data = np.array(tensor_info.data)
            tensor_data = torch.from_numpy(raw_data.copy())  # Copy to own the memory
        except Exception as e:
            logger.warning(f"Failed to load tensor {name}: {e}")
            continue
        
        # Check if tensor is quantized
        is_quantized = quant_type not in [GGUF_TYPE_F32, GGUF_TYPE_F16]
        
        if is_quantized:
            # Create GGUFParameter with actual quantized data
            # The data stays quantized until used in operations
            param = GGUFParameter(tensor_data, quant_type=quant_type)
            param.quant_shape = shape  # Store original shape
            param.gguf_shape = shape   # Also store as gguf_shape for compatibility
            quantized_count += 1
        else:
            # Non-quantized tensor - convert to proper dtype
            if quant_type == GGUF_TYPE_F32:
                tensor_data = tensor_data.to(dtype=torch.float32)
            else:  # F16
                tensor_data = tensor_data.to(dtype=torch.float16)
            param = tensor_data
            float_count += 1
        
        # Convert GGUF key format to state dict format
        sd_key = _gguf_key_to_state_dict_key(name)
        state_dict[sd_key] = param
    
    logger.info(
        f"Loaded GGUF: {len(state_dict)} tensors "
        f"({quantized_count} quantized, {float_count} float)"
    )
    
    return state_dict, reader


def _parse_metadata(reader: Any) -> Dict[str, Any]:
    """
    Parse metadata from GGUF reader.
    
    Args:
        reader: GGUF reader instance
    
    Returns:
        Dict of metadata fields
    """
    metadata = {}
    
    try:
        for key in reader.fields:
            try:
                field = reader.fields[key]
                if hasattr(field, "parts"):
                    # Multi-part field
                    metadata[key] = [p.tolist() if hasattr(p, "tolist") else p 
                                    for p in field.parts]
                elif hasattr(field, "data"):
                    data = field.data
                    if hasattr(data, "tolist"):
                        metadata[key] = data.tolist()
                    else:
                        metadata[key] = data
            except Exception:
                pass  # Skip fields that can't be parsed
    except Exception:
        pass
    
    return metadata


def _gguf_key_to_state_dict_key(gguf_key: str) -> str:
    """
    Convert GGUF tensor name to ComfyUI state dict key.
    
    GGUF uses underscores where PyTorch uses dots for module paths.
    
    Args:
        gguf_key: GGUF tensor name (e.g., "model_blocks_0_self_attn_k_weight")
    
    Returns:
        State dict key (e.g., "blocks.0.self_attn.k.weight")
    """
    # Common prefixes to strip
    prefixes_to_strip = ["model.", "diffusion_model."]
    
    key = gguf_key
    for prefix in prefixes_to_strip:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    
    # GGUF typically uses the same format as state dict
    # But may need conversion for specific patterns
    return key


def _get_dequantized_shape(
    quantized_shape: tuple,
    quant_type: int,
) -> tuple:
    """
    Calculate the shape of dequantized tensor.
    
    For most quantization types, the shape doesn't change during
    dequantization - only the dtype changes from packed bytes to float.
    
    Args:
        quantized_shape: Shape of quantized tensor
        quant_type: GGUF quantization type
    
    Returns:
        Shape after dequantization
    """
    # For most cases, dequantized shape is same as quantized shape
    # The internal representation may be packed, but the logical shape
    # is preserved during quantization
    return quantized_shape


def get_tensor_from_reader(
    reader: Any,
    key: str,
) -> Tuple[Any, int]:
    """
    Get tensor data and quant type from GGUF reader.
    
    Args:
        reader: GGUF reader instance
        key: Tensor key to look for
    
    Returns:
        Tuple of (tensor_info, quant_type)
    
    Raises:
        KeyError: If tensor not found
    """
    # Try exact match first
    for tensor_info in reader.tensors:
        if tensor_info.name == key:
            return tensor_info, tensor_info.tensor_type
    
    # Try with/without common prefixes
    prefixes = ["", "model.", "diffusion_model."]
    for prefix in prefixes:
        search_key = prefix + key
        for tensor_info in reader.tensors:
            if tensor_info.name == search_key:
                return tensor_info, tensor_info.tensor_type
    
    raise KeyError(f"GGUF tensor not found: {key}")


def load_tensor_to_device(
    reader: Any,
    key: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Load a specific tensor from GGUF reader directly to device.
    
    Args:
        reader: GGUF reader instance
        key: Tensor key
        device: Target device
    
    Returns:
        Tensor on target device (GGUFParameter if quantized)
    """
    tensor_info, quant_type = get_tensor_from_reader(reader, key)
    
    is_quantized = quant_type not in [GGUF_TYPE_F32, GGUF_TYPE_F16]
    
    if is_quantized:
        # Load as GGUFParameter
        import numpy as np
        data = torch.from_numpy(tensor_info.data.copy())
        data = data.to(device)
        return GGUFParameter(data, quant_type=quant_type)
    else:
        # Load as regular tensor
        dtype = torch.float32 if quant_type == GGUF_TYPE_F32 else torch.float16
        data = torch.from_numpy(tensor_info.data.copy())
        return data.to(device=device, dtype=dtype)
