"""
GGUF utility functions for tensor handling.

Provides GGUFParameter class and dequantization functions for
on-demand tensor dequantization during forward pass.
"""

import torch
import logging
from typing import Optional, Tuple, Dict, Callable
import threading

logger = logging.getLogger("WAN22BlockSwapLoader.GGUF")

# Thread-local recursion guard for __torch_function__
_local = threading.local()

def _is_in_torch_function():
    """Check if we're already inside __torch_function__ to prevent recursion."""
    return getattr(_local, 'in_torch_function', False)

def _set_in_torch_function(value: bool):
    """Set the recursion guard flag."""
    _local.in_torch_function = value


class GGUFParameter(torch.nn.Parameter):
    """
    Parameter wrapper for GGUF quantized tensors.
    
    This class stores quantized tensor data and provides on-demand
    dequantization during forward pass. The tensor remains in
    quantized form for storage efficiency.
    
    Attributes:
        quant_type: GGUF quantization type (Q4_K, Q5_K, Q6_K, Q8_0, etc.)
        quant_shape: Original tensor shape (may differ from data shape)
        gguf_shape: Shape as stored in GGUF file
    
    Example:
        >>> param = GGUFParameter(quantized_data, quant_type=12)  # Q4_K
        >>> # During forward pass, dequantization happens automatically
        >>> output = layer(input)  # param is dequantized when used
    """
    
    def __new__(
        cls,
        data: torch.Tensor,
        requires_grad: bool = False,
        quant_type: Optional[int] = None,
    ):
        """
        Create new GGUFParameter.
        
        Args:
            data: Tensor data (quantized bytes or regular tensor)
            requires_grad: Whether to track gradients (usually False)
            quant_type: GGUF quantization type (None for regular tensors)
        
        Returns:
            GGUFParameter instance
        """
        instance = torch.Tensor._make_subclass(cls, data, requires_grad)
        instance.quant_type = quant_type
        instance.quant_shape = None  # Set by loader if needed
        instance.gguf_shape = None  # Original GGUF shape
        return instance
    
    def __repr__(self) -> str:
        """String representation showing quantization info."""
        quant_name = QUANT_TYPE_NAMES.get(self.quant_type, f"type_{self.quant_type}")
        return (
            f"GGUFParameter(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device}, quant_type={quant_name})"
        )
    
    def __reduce_ex__(self, protocol):
        """Support pickling for model saving/loading."""
        return (
            self.__class__,
            (torch.Tensor(self), False, self.quant_type),
        )
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Custom torch function dispatch.
        
        When this tensor is used in operations, it may need to be
        dequantized first. This happens automatically during
        mathematical operations.
        """
        kwargs = kwargs or {}
        
        # Recursion guard - if we're already in __torch_function__, use default behavior
        if _is_in_torch_function():
            return super().__torch_function__(func, types, args, kwargs)
        
        # Set recursion guard
        _set_in_torch_function(True)
        try:
            # For most operations, we need to dequantize first
            new_args = []
            for arg in args:
                if isinstance(arg, GGUFParameter) and getattr(arg, "quant_type", None) is not None:
                    # Dequantize for the operation
                    new_args.append(dequantize_gguf_tensor(arg))
                else:
                    new_args.append(arg)
            
            return func(*new_args, **kwargs)
        finally:
            # Clear recursion guard
            _set_in_torch_function(False)


# GGUF quantization type constants and names
QUANT_TYPE_NAMES: Dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "BF16",
}

# Block sizes for different quantization types
QUANT_BLOCK_SIZES: Dict[int, int] = {
    2: 32,    # Q4_0
    3: 32,    # Q4_1
    6: 32,    # Q5_0
    7: 32,    # Q5_1
    8: 32,    # Q8_0
    9: 32,    # Q8_1
    10: 256,  # Q2_K
    11: 256,  # Q3_K
    12: 256,  # Q4_K
    13: 256,  # Q5_K
    14: 256,  # Q6_K
    15: 256,  # Q8_K
}


def dequantize_gguf_tensor(tensor: GGUFParameter) -> torch.Tensor:
    """
    Dequantize a GGUF tensor to full precision.
    
    This function converts quantized tensor data back to float16/float32
    for computation. The dequantization method depends on the quantization
    type stored in the tensor.
    
    Args:
        tensor: GGUFParameter with quantized data
    
    Returns:
        Dequantized tensor (typically float16)
    
    Raises:
        ValueError: If quantization type is not supported
    """
    if not hasattr(tensor, "quant_type") or tensor.quant_type is None:
        # Already a regular tensor
        return tensor
    
    quant_type = tensor.quant_type
    # Access device directly from underlying tensor to avoid __torch_function__ recursion
    device = torch.Tensor.device.__get__(tensor)
    
    # For F32 and F16, no dequantization needed - just cast
    if quant_type == 0:  # F32
        # Return as regular tensor to avoid further GGUFParameter dispatch
        result = tensor.data.to(dtype=torch.float32)
        return result
    elif quant_type == 1:  # F16
        result = tensor.data.to(dtype=torch.float16)
        return result
    elif quant_type == 29:  # BF16
        result = tensor.data.to(dtype=torch.bfloat16)
        return result
    
    # Try to use the gguf package's dequantization if available
    try:
        from gguf.quants import dequantize as gguf_dequantize
        
        # Convert to numpy for gguf dequantization
        # Use .data to get underlying tensor and avoid __torch_function__ dispatch
        np_data = tensor.data.detach().cpu().numpy()
        result = gguf_dequantize(np_data, quant_type)
        return torch.from_numpy(result).to(device=device, dtype=torch.float16)
    
    except ImportError:
        # Fallback: try ComfyUI's GGUF implementation
        try:
            from comfy.ops import dequantize_tensor
            return dequantize_tensor(tensor.data)
        except (ImportError, AttributeError):
            pass
    
    except Exception as e:
        logger.warning(f"GGUF dequantization failed: {e}")
    
    # If all else fails, raise an error
    quant_name = QUANT_TYPE_NAMES.get(quant_type, f"type_{quant_type}")
    raise ValueError(
        f"Cannot dequantize GGUF tensor with type {quant_name}. "
        f"Ensure the 'gguf' package is installed: pip install gguf"
    )


def get_quant_type_name(quant_type: int) -> str:
    """
    Get human-readable name for a quantization type.
    
    Args:
        quant_type: GGUF quantization type number
    
    Returns:
        Name string (e.g., "Q4_K", "Q8_0")
    """
    return QUANT_TYPE_NAMES.get(quant_type, f"UNKNOWN_{quant_type}")


def get_block_size(quant_type: int) -> int:
    """
    Get the block size for a quantization type.
    
    Args:
        quant_type: GGUF quantization type number
    
    Returns:
        Block size (number of values per quantization block)
    """
    return QUANT_BLOCK_SIZES.get(quant_type, 32)


def is_quantized(tensor: torch.Tensor) -> bool:
    """
    Check if a tensor is a quantized GGUFParameter.
    
    Args:
        tensor: Tensor to check
    
    Returns:
        True if tensor is quantized GGUFParameter
    """
    if not isinstance(tensor, GGUFParameter):
        return False
    return tensor.quant_type is not None and tensor.quant_type > 1


def estimate_memory_reduction(quant_type: int) -> float:
    """
    Estimate memory reduction factor for a quantization type.
    
    Returns the approximate ratio of quantized size to float16 size.
    Lower values mean more compression.
    
    Args:
        quant_type: GGUF quantization type
    
    Returns:
        Memory reduction factor (e.g., 0.25 for 4-bit quant)
    """
    reductions = {
        0: 2.0,    # F32 is larger than F16
        1: 1.0,    # F16 is baseline
        2: 0.25,   # Q4_0: 4 bits per value
        3: 0.28,   # Q4_1: 4 bits + overhead
        6: 0.31,   # Q5_0: 5 bits per value
        7: 0.34,   # Q5_1: 5 bits + overhead
        8: 0.50,   # Q8_0: 8 bits per value
        9: 0.53,   # Q8_1: 8 bits + overhead
        10: 0.19,  # Q2_K: ~2.5 bits
        11: 0.22,  # Q3_K: ~3.5 bits
        12: 0.28,  # Q4_K: ~4.5 bits
        13: 0.34,  # Q5_K: ~5.5 bits
        14: 0.41,  # Q6_K: ~6.5 bits
        15: 0.53,  # Q8_K: ~8.5 bits
    }
    return reductions.get(quant_type, 0.5)
