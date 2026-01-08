"""Module initialization for ComfyUI_Wan22Blockswap.

This module imports and exports all the components of the block swapping
system, making them available when the package is imported. It provides
a clean interface for accessing the functionality.

ACTIVE NODES (working and tested):
- WANModelLoader: Simple all-in-one WAN model loader (no BlockSwap)
- WAN22BlockSwapPatcher: Apply BlockSwap to any loaded model
- WAN22BlockSwapComboPatcher: Apply BlockSwap to HIGH+LOW model pair with auto-switch
- WAN22BlockSwapCleanup: Clean up BlockSwap state after sampling
- WAN22BlockSwapReposition: Re-position blocks for next sampling run
- WAN22FullCleanup: Aggressive cleanup at end of workflow
- WAN22TiledVAEDecode: High-quality tiled VAE decode for WAN video models

DEPRECATED NODES (commented out, kept for reference):
- wan22BlockSwap: Old callback-based approach
- WANBlockSwapWrapper: Old universal wrapper
- WAN22BlockSwapLoader: Old loader with integrated pre-routing
- WAN22BlockSwapLooperModels: Old looper integration
- WAN22BlockSwapSequencer: Old sequencer for LoRA
- WANBlockSwapMetaLoader: Old meta loader
"""

# ============================================================
# DEPRECATED: Old callback-based nodes (kept for reference)
# ============================================================
# from .nodes import NODE_CLASS_MAPPINGS as _NODES_MAPPINGS
# from .nodes import NODE_DISPLAY_NAME_MAPPINGS as _NODES_DISPLAY_MAPPINGS
# from .blockswap_loader import (
#     NODE_CLASS_MAPPINGS as _LOADER_MAPPINGS,
#     NODE_DISPLAY_NAME_MAPPINGS as _LOADER_DISPLAY_MAPPINGS,
#     WAN22BlockSwapLoader,
# )
# from .blockswap_looper import (
#     NODE_CLASS_MAPPINGS as _LOOPER_MAPPINGS,
#     NODE_DISPLAY_NAME_MAPPINGS as _LOOPER_DISPLAY_MAPPINGS,
#     WAN22BlockSwapLooperModels,
#     WAN22BlockSwapSequencer,
# )
# from .blockswap_meta_loader import (
#     NODE_CLASS_MAPPINGS as _META_LOADER_MAPPINGS,
#     NODE_DISPLAY_NAME_MAPPINGS as _META_LOADER_DISPLAY_MAPPINGS,
#     WANBlockSwapMetaLoader,
# )

# ============================================================
# ACTIVE: Current working nodes
# ============================================================
from .wan_loader import (
    NODE_CLASS_MAPPINGS as _WAN_LOADER_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _WAN_LOADER_DISPLAY_MAPPINGS,
    WANModelLoader,
)
from .blockswap_forward import (
    NODE_CLASS_MAPPINGS as _FORWARD_PATCHER_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _FORWARD_PATCHER_DISPLAY_MAPPINGS,
    WAN22BlockSwapPatcher,
    WAN22BlockSwapComboPatcher,
    WAN22BlockSwapCleanup,
    WAN22BlockSwapReposition,
    WAN22FullCleanup,
    BlockSwapForwardPatcher,
)
from .vae_decode import (
    NODE_CLASS_MAPPINGS as _VAE_DECODE_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _VAE_DECODE_DISPLAY_MAPPINGS,
    WAN22TiledVAEDecode,
)
from .config import BlockSwapConfig

# Merge node mappings from active modules only
NODE_CLASS_MAPPINGS = {**_WAN_LOADER_MAPPINGS, **_FORWARD_PATCHER_MAPPINGS, **_VAE_DECODE_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**_WAN_LOADER_DISPLAY_MAPPINGS, **_FORWARD_PATCHER_DISPLAY_MAPPINGS, **_VAE_DECODE_DISPLAY_MAPPINGS}

# Register JavaScript directory for frontend UI extensions
WEB_DIRECTORY = "./js"

# Core utilities (still used by active nodes)
from .block_manager import BlockManager, BlockSwapTracker
from .callbacks import lazy_load_callback, cleanup_callback
from .utils import log_debug, sync_gpu, clear_device_caches

# ============================================================
# DEPRECATED: Model tracker and looper helpers (kept for reference)
# These were used by the old callback-based looper integration
# ============================================================
# from .model_tracker import (
#     BlockSwapModelTracker,
#     CleanupMode,
#     CleanupDecision,
#     ModelPrepState,
#     SessionState,
# )
# from .looper_helpers import (
#     prepare_model_for_loop,
#     cleanup_loop_blockswap,
#     validate_tensor_consistency,
#     reset_model_blockswap_state,
#     start_blockswap_session,
#     end_blockswap_session,
#     update_session_loop_state,
# )

# Export all public components for easy access
__all__ = [
    # ComfyUI node registration
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    
    # ACTIVE NODES
    "WANModelLoader",                # Simple all-in-one WAN model loader
    "WAN22BlockSwapPatcher",         # Forward patcher for any model
    "WAN22BlockSwapComboPatcher",    # Combo patcher for high+low models in one
    "WAN22BlockSwapCleanup",         # Cleanup node to free memory
    "WAN22BlockSwapReposition",      # Reposition blocks after cleanup
    "WAN22FullCleanup",              # Aggressive cleanup at end of workflow
    "WAN22TiledVAEDecode",           # High-quality tiled VAE decode for WAN video
    
    # Core components
    "BlockSwapForwardPatcher",       # Core forward patching logic
    "BlockSwapConfig",               # Configuration and input type definitions
    "BlockManager",                  # Core block swapping operations
    "BlockSwapTracker",              # State tracking for cleanup operations
    
    # Utilities
    "lazy_load_callback",            # ON_LOAD callback for lazy loading
    "cleanup_callback",              # ON_CLEANUP callback for cleanup
    "log_debug",                     # Debug logging utility
    "sync_gpu",                      # GPU synchronization utility
    "clear_device_caches",           # Device cache clearing utility
    
    # DEPRECATED (commented out but listed for reference):
    # "WAN22BlockSwapLoader",        # Old loader with integrated pre-routing
    # "WAN22BlockSwapLooperModels",  # Old looper for high/low model pairs
    # "WAN22BlockSwapSequencer",     # Old looper for LoRA sequences
    # "WANBlockSwapMetaLoader",      # Old meta loader
    # "BlockSwapModelTracker",       # Old model tracker
    # "CleanupMode", "CleanupDecision", "ModelPrepState", "SessionState",
    # "prepare_model_for_loop", "cleanup_loop_blockswap", etc.
]
