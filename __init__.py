"""Module initialization for ComfyUI-Wan22Blockswap.

This module imports and exports all the components of the block swapping
system, making them available when the package is imported. It provides
a clean interface for accessing the functionality.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .blockswap_looper import WAN22BlockSwapLooper
from .config import BlockSwapConfig
from .block_manager import BlockManager, BlockSwapTracker
from .callbacks import lazy_load_callback, cleanup_callback
from .utils import log_debug, sync_gpu, clear_device_caches

# Export all public components for easy access
__all__ = [
    "NODE_CLASS_MAPPINGS",           # ComfyUI node registration mappings
    "NODE_DISPLAY_NAME_MAPPINGS",    # ComfyUI node display names
    "WAN22BlockSwapLooper",          # Specialized looper node for multi-loop integration
    "BlockSwapConfig",               # Configuration and input type definitions
    "BlockManager",                  # Core block swapping operations
    "BlockSwapTracker",              # State tracking for cleanup operations
    "lazy_load_callback",            # ON_LOAD callback for lazy loading
    "cleanup_callback",              # ON_CLEANUP callback for cleanup
    "log_debug",                     # Debug logging utility
    "sync_gpu",                      # GPU synchronization utility
    "clear_device_caches"            # Device cache clearing utility
]
