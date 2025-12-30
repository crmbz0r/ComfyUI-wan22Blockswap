"""Utility functions for WAN 2.2 BlockSwap operations.

This module contains helper functions for logging, device management,
memory cleanup, and other utility operations used throughout the
block swapping system.
"""

import torch
import gc
from typing import Any, Optional


def log_debug(message: str, debug_enabled: bool = False) -> None:
    """
    Log debug messages if debug mode is enabled.

    Args:
        message: The debug message to log
        debug_enabled: Whether debug logging is enabled
    """
    if debug_enabled:
        print(f"[BlockSwap] {message}")


def sync_gpu(debug_enabled: bool = False) -> None:
    """
    Synchronize GPU operations before block operations.

    Args:
        debug_enabled: Whether to log debug messages
    """
    if debug_enabled:
        log_debug("Phase 1: Synchronizing GPU...", debug_enabled)

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            if debug_enabled:
                log_debug("GPU synchronized", debug_enabled)
    except Exception as e:
        if debug_enabled:
            log_debug(f"GPU sync failed: {str(e)[:80]}", debug_enabled)


def clear_device_caches(debug_enabled: bool = False) -> None:
    """
    Clear caches for all available devices.

    Args:
        debug_enabled: Whether to log debug messages
    """
    if debug_enabled:
        log_debug("Phase 6: Clearing device caches", debug_enabled)

    # Clear CUDA cache
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if debug_enabled:
                log_debug("CUDA cache cleared and synchronized", debug_enabled)
    except Exception as e:
        if debug_enabled:
            log_debug(f"CUDA cache clear failed: {str(e)[:80]}", debug_enabled)

    # Clear XPU cache (Intel GPUs)
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
            if debug_enabled:
                log_debug("XPU cache cleared", debug_enabled)
    except Exception:
        pass

    # Clear MPS cache (Apple Silicon)
    try:
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
            if debug_enabled:
                log_debug("MPS cache cleared", debug_enabled)
    except Exception:
        pass


def get_model_device(model: Any) -> torch.device:
    """
    Get the primary device of a model.

    Args:
        model: The model to check

    Returns:
        torch.device: The primary device of the model
    """
    try:
        # Try to get device from first parameter
        first_param = next(model.parameters())
        return first_param.device
    except (StopIteration, AttributeError):
        # Fallback to CUDA if available
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_device_availability() -> bool:
    """
    Validate that required devices are available.

    Returns:
        bool: True if required devices are available
    """
    try:
        if not torch.cuda.is_available():
            print("[BlockSwap] Warning: CUDA not available, using CPU only")
            return False
        return True
    except Exception:
        return False


def format_memory_usage() -> str:
    """
    Format current memory usage for logging.

    Returns:
        str: Formatted memory usage string
    """
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            return f"Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
        else:
            return "CUDA not available"
    except Exception:
        return "Memory info unavailable"


def safe_model_operation(operation_func, *args, **kwargs):
    """
    Execute a model operation with error handling.

    Args:
        operation_func: The function to execute
        *args: Arguments for the operation function
        **kwargs: Keyword arguments for the operation function

    Returns:
        Result of the operation or None if failed
    """
    try:
        return operation_func(*args, **kwargs)
    except (RuntimeError, torch.cuda.OutOfMemoryError, Exception) as e:
        print(f"[BlockSwap] Operation failed: {str(e)[:100]}")
        return None


def cleanup_memory() -> None:
    """
    Perform comprehensive memory cleanup.
    """
    gc.collect()
    gc.collect()

    try:
        import comfy.model_management as mm
        mm.soft_empty_cache()
    except Exception:
        pass

    clear_device_caches()
