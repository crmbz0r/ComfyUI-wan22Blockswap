"""
GGUF loading utilities for WAN22BlockSwapLoader.

Provides support for loading GGUF-quantized WAN models with
deferred dequantization at forward pass time.

This module provides a safer GGUFParameter class (based on WanVideoWrapper)
that works correctly with block swapping, unlike ComfyUI-GGUF's GGMLTensor
which can cause CUDA corruption after repeated block movements.
"""

from .gguf import load_gguf, GGUF_AVAILABLE
from .gguf_utils import GGUFParameter, dequantize_gguf_tensor

__all__ = ["load_gguf", "GGUF_AVAILABLE", "GGUFParameter", "dequantize_gguf_tensor"]
