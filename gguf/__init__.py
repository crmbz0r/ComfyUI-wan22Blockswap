"""
GGUF loading utilities for WAN22BlockSwapLoader.

Provides support for loading GGUF-quantized WAN models with
deferred dequantization at forward pass time.
"""

from .gguf import load_gguf
from .gguf_utils import GGUFParameter, dequantize_gguf_tensor

__all__ = ["load_gguf", "GGUFParameter", "dequantize_gguf_tensor"]
