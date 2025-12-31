"""Module initialization for ComfyUI-Wan22Blockswap.

This module imports and exports all the components of the block swapping
system, making them available when the package is imported. It provides
a clean interface for accessing the functionality.

Includes:
- wan22BlockSwap: Dynamic block swapping via ON_LOAD callbacks
- WAN22BlockSwapLoader: Loader with integrated pre-routing (prevents VRAM spikes)
- WAN22BlockSwapLooper: Multi-loop integration node
"""

from .nodes import NODE_CLASS_MAPPINGS as _NODES_MAPPINGS
from .nodes import NODE_DISPLAY_NAME_MAPPINGS as _NODES_DISPLAY_MAPPINGS
from .blockswap_loader import (
    NODE_CLASS_MAPPINGS as _LOADER_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _LOADER_DISPLAY_MAPPINGS,
    WAN22BlockSwapLoader,
)
from .blockswap_looper import WAN22BlockSwapLooper
from .config import BlockSwapConfig

# Merge node mappings from all modules
NODE_CLASS_MAPPINGS = {**_NODES_MAPPINGS, **_LOADER_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**_NODES_DISPLAY_MAPPINGS, **_LOADER_DISPLAY_MAPPINGS}
from .block_manager import BlockManager, BlockSwapTracker
from .callbacks import lazy_load_callback, cleanup_callback
from .utils import log_debug, sync_gpu, clear_device_caches
from .model_tracker import (
    BlockSwapModelTracker,
    CleanupMode,
    CleanupDecision,
    ModelPrepState,
    SessionState,
)
from .looper_helpers import (
    prepare_model_for_loop,
    cleanup_loop_blockswap,
    validate_tensor_consistency,
    reset_model_blockswap_state,
    start_blockswap_session,
    end_blockswap_session,
    update_session_loop_state,
)

# Export all public components for easy access
__all__ = [
    "NODE_CLASS_MAPPINGS",           # ComfyUI node registration mappings
    "NODE_DISPLAY_NAME_MAPPINGS",    # ComfyUI node display names
    "WAN22BlockSwapLoader",          # NEW: Loader with integrated pre-routing
    "WAN22BlockSwapLooper",          # Specialized looper node for multi-loop integration
    "BlockSwapConfig",               # Configuration and input type definitions
    "BlockManager",                  # Core block swapping operations
    "BlockSwapTracker",              # State tracking for cleanup operations
    "lazy_load_callback",            # ON_LOAD callback for lazy loading
    "cleanup_callback",              # ON_CLEANUP callback for cleanup
    "log_debug",                     # Debug logging utility
    "sync_gpu",                      # GPU synchronization utility
    "clear_device_caches",           # Device cache clearing utility
    # Model tracker components for smart cleanup
    "BlockSwapModelTracker",         # Singleton tracker for model identity across loops
    "CleanupMode",                   # Enum for cleanup mode configuration
    "CleanupDecision",               # Enum for cleanup decision results
    "ModelPrepState",                # Dataclass for model preparation state
    "SessionState",                  # Dataclass for session state tracking
    # Looper helper functions
    "prepare_model_for_loop",        # Prepare model for a single loop iteration
    "cleanup_loop_blockswap",        # Clean up BlockSwap state between loops
    "validate_tensor_consistency",   # Validate tensor device/dtype consistency
    "reset_model_blockswap_state",   # Reset BlockSwap state on a model
    "start_blockswap_session",       # Start a new tracking session
    "end_blockswap_session",         # End a tracking session
    "update_session_loop_state",     # Update session loop progress
]
