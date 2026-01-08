"""
BlockSwap Forward Patcher for WAN models.

This module provides a node that patches any loaded WAN model's forward method
to enable block swapping during inference. This allows the model to work with
any standard ComfyUI KSampler while keeping VRAM usage low.

The approach is based on WanVideoWrapper's block swapping implementation:
- Blocks 0 to (N - blocks_to_swap - 1) stay on GPU
- Blocks (N - blocks_to_swap) to (N-1) are swapped CPU <-> GPU during forward
- Uses non_blocking transfers and optional prefetching for performance
"""

import torch
import torch.nn as nn
import gc
import time
import types
import logging
import warnings
import sys
import io
from typing import Optional, Dict, Any, Tuple
from contextlib import nullcontext, contextmanager

import comfy.model_management as mm

# Import IO.ANY for universal input types
try:
    from comfy.comfy_types import IO
    ANY_TYPE = IO.ANY
except (ImportError, AttributeError):
    # Fallback for older ComfyUI versions
    ANY_TYPE = "*"

logger = logging.getLogger("ComfyUI_Wan22Blockswap")


# ============================================================================
# CUDA Health Check Functions
# ============================================================================

_cuda_corruption_detected = False


def reset_cuda_corruption_flag():
    """Reset the CUDA corruption flag (call at start of new generation)."""
    global _cuda_corruption_detected
    _cuda_corruption_detected = False


def is_cuda_healthy() -> bool:
    """
    Check if CUDA is in a healthy state by attempting basic operations.
    
    Returns True if CUDA operations work, False if corrupted.
    """
    global _cuda_corruption_detected
    
    if _cuda_corruption_detected:
        return False
    
    if not torch.cuda.is_available():
        return True  # No CUDA, so technically "healthy"
    
    try:
        # Try a simple CUDA operation
        device = torch.device("cuda", 0)
        
        # Small test tensor allocation
        test = torch.zeros(16, device=device)
        test = test + 1
        del test
        
        # Sync to catch any delayed errors
        torch.cuda.synchronize()
        
        return True
    except RuntimeError as e:
        if "CUDA" in str(e) or "cuda" in str(e):
            _cuda_corruption_detected = True
            logger.error(f"CUDA corruption detected: {e}")
            return False
        raise


def check_cuda_or_raise(context: str = ""):
    """
    Check CUDA health and raise with helpful message if corrupted.
    
    Args:
        context: Description of what operation was attempted
    """
    if not is_cuda_healthy():
        msg = "CUDA state is corrupted. Please restart ComfyUI to continue."
        if context:
            msg = f"CUDA corruption detected during {context}. {msg}"
        raise RuntimeError(msg)


class FilteredWriter:
    """Wrapper for stdout/stderr that filters out specific messages."""
    def __init__(self, original, filter_patterns):
        self.original = original
        self.filter_patterns = filter_patterns
    
    def write(self, text):
        # Check if any filter pattern matches
        text_lower = text.lower()
        for pattern in self.filter_patterns:
            if pattern in text_lower:
                return  # Suppress this message
        self.original.write(text)
    
    def flush(self):
        self.original.flush()
    
    def __getattr__(self, name):
        return getattr(self.original, name)


# Global state for persistent warning filter
_original_stdout = None
_original_stderr = None
_filter_installed = False


def install_unpin_warning_filter():
    """
    Install a persistent filter for unpin warnings.
    
    This is used by the Combo Patcher to suppress warnings throughout the
    entire sampling process, not just during our callbacks.
    """
    global _original_stdout, _original_stderr, _filter_installed
    
    if _filter_installed:
        return  # Already installed
    
    filter_patterns = ["unpin", "not pinned", "pin error"]
    
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    
    # Only wrap if not already a FilteredWriter
    if not isinstance(sys.stdout, FilteredWriter):
        sys.stdout = FilteredWriter(_original_stdout, filter_patterns)
    if not isinstance(sys.stderr, FilteredWriter):
        sys.stderr = FilteredWriter(_original_stderr, filter_patterns)
    
    _filter_installed = True
    logger.debug("Unpin warning filter installed")


def uninstall_unpin_warning_filter():
    """Remove the persistent unpin warning filter."""
    global _original_stdout, _original_stderr, _filter_installed
    
    if not _filter_installed:
        return
    
    if _original_stdout is not None:
        sys.stdout = _original_stdout
    if _original_stderr is not None:
        sys.stderr = _original_stderr
    
    _original_stdout = None
    _original_stderr = None
    _filter_installed = False
    logger.debug("Unpin warning filter uninstalled")


@contextmanager
def suppress_unpin_warnings():
    """
    Context manager to suppress 'Tried to unpin tensor not pinned by ComfyUI' warnings.
    
    These warnings occur when we move tensors that weren't originally pinned by ComfyUI's
    memory management. They're harmless but spam the console during block swap operations.
    """
    filter_patterns = ["unpin", "not pinned", "pin error"]
    
    # Capture stdout and stderr since ComfyUI prints these directly
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = FilteredWriter(old_stdout, filter_patterns)
    sys.stderr = FilteredWriter(old_stderr, filter_patterns)
    
    # Also filter warnings module
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*unpin.*")
        warnings.filterwarnings("ignore", message=".*not pinned.*")
        
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class BlockSwapForwardPatcher:
    """
    Patches a WAN model's forward method to enable block swapping during inference.
    
    This class wraps the model's forward_orig method and handles:
    1. Pre-positioning blocks (GPU vs CPU) before sampling starts
    2. Moving blocks to GPU just before they're needed
    3. Moving blocks back to CPU after they're done
    4. Optional prefetching of next block(s) for overlap
    """
    
    def __init__(
        self,
        diffusion_model: nn.Module,
        blocks_to_swap: int = 20,
        main_device: torch.device = None,
        offload_device: torch.device = None,
        use_non_blocking: bool = True,
        prefetch_blocks: int = 0,
        offload_img_emb: bool = False,
        offload_txt_emb: bool = False,
        debug: bool = False,
        model_patcher = None,
    ):
        """
        Initialize the BlockSwap patcher.
        
        Args:
            diffusion_model: The WAN diffusion model (WanModel instance)
            blocks_to_swap: Number of blocks to swap (from the end)
            main_device: GPU device for computation
            offload_device: CPU device for offloading
            use_non_blocking: Use non_blocking transfers for overlap
            prefetch_blocks: Number of blocks to prefetch ahead
            offload_img_emb: Whether to offload img_emb to CPU
            offload_txt_emb: Whether to offload text embeddings to CPU
            debug: Enable timing debug output
            model_patcher: The ComfyUI ModelPatcher wrapping this model (for cleanup)
        """
        self.diffusion_model = diffusion_model
        self.model_patcher = model_patcher
        self.blocks_to_swap = blocks_to_swap
        self.main_device = main_device or torch.device("cuda:0")
        self.offload_device = offload_device or torch.device("cpu")
        self.use_non_blocking = use_non_blocking
        self.prefetch_blocks = prefetch_blocks
        self.offload_img_emb = offload_img_emb
        self.offload_txt_emb = offload_txt_emb
        self.debug = debug
        
        # Store original forward for restoration
        self._original_forward_orig = None
        self._is_patched = False
        
        # Get blocks from the model
        self.blocks = None
        if hasattr(diffusion_model, 'blocks'):
            self.blocks = diffusion_model.blocks
        elif hasattr(diffusion_model, 'diffusion_model') and hasattr(diffusion_model.diffusion_model, 'blocks'):
            self.blocks = diffusion_model.diffusion_model.blocks
            
        if self.blocks is None:
            raise ValueError("Could not find 'blocks' attribute on diffusion model")
        
        self.num_blocks = len(self.blocks)
        self.swap_start_idx = max(0, self.num_blocks - blocks_to_swap)
        
        logger.info(f"BlockSwapForwardPatcher initialized:")
        logger.info(f"  Total blocks: {self.num_blocks}")
        logger.info(f"  Blocks to swap: {blocks_to_swap}")
        logger.info(f"  Swap starts at block: {self.swap_start_idx}")
        logger.info(f"  GPU blocks: 0-{self.swap_start_idx - 1}")
        logger.info(f"  CPU blocks: {self.swap_start_idx}-{self.num_blocks - 1}")
        
        # Track loaded size for ComfyUI compatibility
        self._reported_loaded_size = 0
        
        # Detect if this is a GGUF model and what type
        self._gguf_type = self._detect_gguf_type()
        self._is_gguf_model = self._gguf_type is not None
        
        if self._gguf_type == "comfyui-gguf":
            # ComfyUI-GGUF uses GGMLTensor with problematic custom .to()/.clone()/.detach()
            logger.warning("=" * 70)
            logger.warning("WARNING: ComfyUI-GGUF model detected!")
            logger.warning("ComfyUI-GGUF's GGMLTensor has custom methods that may cause")
            logger.warning("CUDA corruption after multiple generations with block swap.")
            logger.warning("")
            logger.warning("If you experience 'CUDA error: invalid argument' after 2-3 runs:")
            logger.warning("  1. Restart ComfyUI to reset CUDA state")
            logger.warning("  2. Consider using WanVideoWrapper's GGUF loader instead,")
            logger.warning("     which has a more stable GGUF implementation")
            logger.warning("=" * 70)
        elif self._gguf_type == "wanvideowrapper":
            logger.info("  Model type: WanVideoWrapper GGUF (safe GGUFParameter)")
        elif self._is_gguf_model:
            logger.info("  Model type: Unknown GGUF (using cautious handling)")
    
    def _detect_gguf_type(self) -> Optional[str]:
        """
        Detect if the model contains GGUF tensors and identify the source.
        
        Returns:
            - "comfyui-gguf": ComfyUI-GGUF package (problematic GGMLTensor)
            - "wanvideowrapper": WanVideoWrapper's GGUF or this package's GGUF (safer GGUFParameter)
            - "unknown": Some other GGUF implementation
            - None: Not a GGUF model
        """
        try:
            for param in self.diffusion_model.parameters():
                param_type = type(param).__name__
                param_module = type(param).__module__ or ""
                
                # Check for ComfyUI-GGUF's GGMLTensor
                # Key identifiers: tensor_type, tensor_shape, ggml_* methods, patches_uuid
                if param_type == 'GGMLTensor' or 'GGMLTensor' in param_type:
                    # Check for the problematic custom methods
                    has_custom_clone = hasattr(param, 'clone') and getattr(type(param).clone, '__func__', None) is not getattr(torch.Tensor.clone, '__func__', object())
                    if hasattr(param, 'tensor_type') or hasattr(param, 'patches_uuid'):
                        logger.debug(f"Detected ComfyUI-GGUF GGMLTensor in param")
                        return "comfyui-gguf"
                
                # Check for WanVideoWrapper's GGUFParameter or this package's GGUFParameter
                # Key identifiers: quant_type, quant_shape (ours uses quant_shape not as_tensor)
                if param_type == 'GGUFParameter' or 'GGUFParameter' in param_type:
                    if hasattr(param, 'quant_type'):
                        # Either WanVideoWrapper or our own - both are safe
                        logger.debug(f"Detected safe GGUFParameter in param (module: {param_module})")
                        return "wanvideowrapper"
                
                # Generic GGUF detection
                if 'GGML' in param_type or 'GGUF' in param_type:
                    return "unknown"
                if hasattr(param, 'tensor_type') or hasattr(param, 'quant_type'):
                    return "unknown"
                    
        except Exception as e:
            logger.debug(f"Exception during GGUF detection: {e}")
        
        return None
    
    def _safe_block_to_device(self, block: nn.Module, device: torch.device, sync: bool = True):
        """
        Move a block to a device safely, handling GGUF tensors carefully.
        
        For ComfyUI-GGUF models (GGMLTensor), we add extra precautions:
        - Full synchronization before and after
        - Always blocking transfers
        - Extra error handling
        
        For WanVideoWrapper GGUF (GGUFParameter), standard handling is fine.
        """
        try:
            # For ComfyUI-GGUF, be extra careful
            if self._gguf_type == "comfyui-gguf":
                # Full sync before any movement
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Move the block with blocking transfer
                block.to(device, non_blocking=False)
                
                # Full sync after to ensure completion
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
            else:
                # Standard path for non-GGUF or WanVideoWrapper GGUF
                if sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Move the block
                use_non_blocking = self.use_non_blocking and not self._is_gguf_model
                block.to(device, non_blocking=use_non_blocking)
                
                if sync and torch.cuda.is_available():
                    torch.cuda.synchronize()
                
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "cuda" in error_msg or "invalid argument" in error_msg:
                logger.error(f"CUDA error during block movement: {e}")
                if self._gguf_type == "comfyui-gguf":
                    logger.error("This is likely due to ComfyUI-GGUF's GGMLTensor behavior.")
                    logger.error("Please restart ComfyUI to reset CUDA state.")
                else:
                    logger.error("CUDA state may be corrupted - restart ComfyUI recommended")
                raise
            raise
    
    def calculate_model_size(self) -> int:
        """Calculate total model size in bytes for ComfyUI compatibility."""
        total_size = 0
        for param in self.diffusion_model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size
    
    def calculate_loaded_size(self) -> int:
        """Calculate size of model currently on GPU."""
        loaded_size = 0
        for param in self.diffusion_model.parameters():
            if param.device.type == 'cuda':
                loaded_size += param.numel() * param.element_size()
        return loaded_size
    
    def _reset_cuda_if_needed(self):
        """Aggressively reset CUDA if there are any pending errors."""
        if not torch.cuda.is_available():
            return
        
        try:
            # Test if CUDA is healthy with a simple operation
            torch.cuda.synchronize()
            _ = torch.zeros(1, device='cuda')
        except Exception as e:
            logger.warning(f"CUDA appears unhealthy, attempting reset: {e}")
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                gc.collect()
                # Force a new context
                torch.cuda.set_device(0)
                torch.cuda.synchronize()
            except Exception as e2:
                logger.error(f"CUDA reset failed: {e2}")
                raise RuntimeError(
                    "CUDA is in an invalid state from a previous error. "
                    "Please restart ComfyUI to reset the GPU state."
                ) from e2
    
    def pre_position_blocks(self):
        """
        Pre-position blocks on their initial devices before sampling.
        
        Blocks 0 to swap_start_idx-1: GPU
        Blocks swap_start_idx to num_blocks-1: CPU
        """
        logger.info("Pre-positioning blocks for block swap...")
        
        # For ComfyUI-GGUF models, check CUDA health first
        if self._gguf_type == "comfyui-gguf":
            check_cuda_or_raise("pre_position_blocks start (ComfyUI-GGUF model)")
        
        # CRITICAL: Reset CUDA first to catch any lingering errors
        self._reset_cuda_if_needed()
        
        # AGGRESSIVE CUDA reset to avoid issues from previous runs
        # This is critical for preventing CUDA errors on consecutive runs
        if torch.cuda.is_available():
            try:
                # Wait for all async operations to complete
                torch.cuda.synchronize()
            except RuntimeError as e:
                logger.warning(f"CUDA sync warning (attempting reset): {e}")
                if self._gguf_type == "comfyui-gguf":
                    raise RuntimeError(
                        "CUDA sync failed with ComfyUI-GGUF model. "
                        "This is a known issue with GGMLTensor block swapping. "
                        "Please restart ComfyUI to reset the CUDA state."
                    ) from e
                try:
                    # Try to reset CUDA context
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.synchronize()
                except Exception as e2:
                    logger.warning(f"CUDA reset also failed: {e2}")
            
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"CUDA empty_cache warning: {e}")
        
        # Soft cleanup - DON'T use mm.unload_all_models() as it corrupts CUDA state for other models
        mm.soft_empty_cache()
        gc.collect()
        
        # Move non-block parameters to GPU (patch_embedding, head, etc.)
        # Suppress "unpin tensor" warnings during transfers
        # Always use blocking transfers for GGUF safety
        with suppress_unpin_warnings():
            for name, param in self.diffusion_model.named_parameters():
                if "blocks." not in name:
                    # Keep non-block params on GPU
                    if param.device != self.main_device:
                        param.data = param.data.to(self.main_device, non_blocking=False)
                elif self.offload_txt_emb and "txt_emb" in name:
                    param.data = param.data.to(self.offload_device, non_blocking=False)
                elif self.offload_img_emb and "img_emb" in name:
                    param.data = param.data.to(self.offload_device, non_blocking=False)
            
            # Position blocks - use _safe_block_to_device for proper GGUF handling
            for b, block in enumerate(self.blocks):
                if b < self.swap_start_idx:
                    # Keep on GPU
                    self._safe_block_to_device(block, self.main_device, sync=False)
                else:
                    # Move to CPU
                    self._safe_block_to_device(block, self.offload_device, sync=False)
        
        # Sync to ensure all transfers complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Calculate and store the loaded size for ComfyUI
        self._reported_loaded_size = self.calculate_loaded_size()
        
        logger.info(f"Block positioning complete: {self.swap_start_idx} on GPU, {self.num_blocks - self.swap_start_idx} on CPU")
        logger.info(f"Reported loaded size to ComfyUI: {self._reported_loaded_size / (1024*1024):.2f} MB")
    
    def cleanup(self, move_to_cpu: bool = True, force_contiguous: bool = False):
        """
        Clean up after sampling - move all blocks to CPU and free memory.
        
        Args:
            move_to_cpu: If True, move all blocks to CPU. If False, just clear caches.
            force_contiguous: DISABLED - was breaking tensor references. Kept for API compat.
        """
        logger.info("BlockSwap cleanup starting...")
        
        if move_to_cpu:
            # Sync first to ensure all async transfers are complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Move ALL blocks to CPU with BLOCKING transfer
            # Using non_blocking=False ensures memory is fully transferred before continuing
            # Suppress "unpin tensor" warnings during transfers
            with suppress_unpin_warnings():
                for b, block in enumerate(self.blocks):
                    self._safe_block_to_device(block, self.offload_device, sync=False)
                
                # Move non-block params to CPU too
                for name, param in self.diffusion_model.named_parameters():
                    if "blocks." not in name:
                        param.data = param.data.to(self.offload_device, non_blocking=False)
            
            # Final sync
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Clear CUDA caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Garbage collection
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # For ComfyUI-GGUF models, verify CUDA health after cleanup
        if self._gguf_type == "comfyui-gguf":
            if not is_cuda_healthy():
                logger.warning("CUDA health check failed after cleanup - state may be corrupted")
        
        logger.info("BlockSwap cleanup complete")
    
    def patch_forward(self):
        """
        Patch the model's forward_orig method to include block swapping.
        """
        if self._is_patched:
            logger.warning("Model already patched, skipping")
            return
        
        # Store original forward
        self._original_forward_orig = self.diffusion_model.forward_orig
        
        # Create the patched forward
        patcher = self  # Capture self for closure
        original_forward = self._original_forward_orig
        
        def patched_forward_orig(
            x,
            timestep,
            context,
            clip_fea=None,
            freqs=None,
            transformer_options={},
            **kwargs
        ):
            """
            Patched forward_orig that includes block swapping.
            
            This replaces the model's forward_orig and handles:
            1. All pre-block operations (embeddings, etc.)
            2. Block-by-block processing with GPU<->CPU swapping
            3. All post-block operations (head, etc.)
            """
            # Get the model reference
            model = patcher.diffusion_model
            
            # === PRE-BLOCK OPERATIONS ===
            # These are copied/adapted from WanModel.forward_orig
            
            # Handle timestep
            if isinstance(timestep, list):
                timestep = timestep[0].to(x[0].device)
            
            # Process inputs (this varies by model, so we do minimal processing)
            # The actual embedding happens inside the original forward
            
            # === BLOCK LOOP WITH SWAPPING ===
            # We need to intercept the block loop specifically
            # Since we can't easily split forward_orig, we'll use a different approach:
            # Wrap each block's forward method instead
            
            # For now, let's call the original forward but with blocks that auto-swap
            # This is handled by the block wrappers we set up
            
            return original_forward(
                x, timestep, context,
                clip_fea=clip_fea,
                freqs=freqs,
                transformer_options=transformer_options,
                **kwargs
            )
        
        # Instead of patching forward_orig directly (complex due to WanModel structure),
        # we'll wrap each block's forward method
        self._wrap_blocks()
        
        self._is_patched = True
        logger.info("Forward method patched for block swapping")
    
    def _wrap_blocks(self):
        """
        Wrap each block's forward method to handle GPU<->CPU swapping.
        """
        patcher = self
        
        wrapped_count = 0
        skipped_count = 0
        
        for b, block in enumerate(self.blocks):
            if b >= self.swap_start_idx:
                # Check if this block is ALREADY wrapped - if so, SKIP re-wrapping
                if hasattr(block, '_blockswap_wrapped') and block._blockswap_wrapped:
                    if hasattr(block, '_original_forward'):
                        # Already properly wrapped - don't double-wrap!
                        logger.debug(f"Block {b} already wrapped, skipping")
                        skipped_count += 1
                        continue
                    else:
                        # Marked as wrapped but no original - this is a bug state
                        # Try to unwrap by assuming current forward is usable
                        logger.warning(f"Block {b} marked wrapped but no _original_forward - resetting")
                        block._blockswap_wrapped = False
                
                # Not wrapped yet - wrap it
                original_block_forward = block.forward
                block_idx = b
                
                def make_wrapped_forward(orig_forward, idx):
                    def wrapped_forward(*args, **kwargs):
                        block = patcher.blocks[idx]
                        
                        # Check CUDA health before doing anything
                        check_cuda_or_raise(f"block {idx} forward")
                        
                        # Check if block is already on GPU - skip transfer if so
                        first_param = next(block.parameters(), None)
                        already_on_gpu = first_param is not None and first_param.device.type == 'cuda'
                        
                        # === MOVE TO GPU (only if needed) ===
                        if patcher.debug:
                            transfer_start = time.perf_counter()
                        
                        if not already_on_gpu:
                            try:
                                # Sync before moving to ensure clean state
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                
                                # Move this block to GPU (suppress "unpin tensor" warnings)
                                # Use blocking transfer for GGUF safety
                                with suppress_unpin_warnings():
                                    block.to(patcher.main_device, non_blocking=False)
                                
                                # Sync after moving to GPU
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                            except RuntimeError as e:
                                if "CUDA" in str(e) or "cuda" in str(e):
                                    global _cuda_corruption_detected
                                    _cuda_corruption_detected = True
                                    logger.error(f"CUDA error moving block {idx} to GPU: {e}")
                                raise
                        
                        if patcher.debug:
                            transfer_end = time.perf_counter()
                            transfer_time = transfer_end - transfer_start
                            compute_start = time.perf_counter()
                        
                        # === COMPUTE ===
                        result = orig_forward(*args, **kwargs)
                        
                        # Sync after compute to ensure result is ready before moving block
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        
                        if patcher.debug:
                            compute_end = time.perf_counter()
                            compute_time = compute_end - compute_start
                            to_cpu_start = time.perf_counter()
                        
                        # === MOVE BACK TO CPU (only if we moved it, to maintain consistent state) ===
                        if not already_on_gpu:
                            try:
                                with suppress_unpin_warnings():
                                    block.to(patcher.offload_device, non_blocking=False)
                                
                                # Sync after moving to CPU
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                            except RuntimeError as e:
                                if "CUDA" in str(e) or "cuda" in str(e):
                                    _cuda_corruption_detected = True
                                    logger.error(f"CUDA error moving block {idx} to CPU: {e}")
                                raise
                        
                        if patcher.debug:
                            to_cpu_end = time.perf_counter()
                            to_cpu_time = to_cpu_end - to_cpu_start
                            if already_on_gpu:
                                logger.info(f"Block {idx}: [GPU-RESIDENT] compute_time={compute_time:.4f}s")
                            else:
                                logger.info(f"Block {idx}: transfer_time={transfer_time:.4f}s, compute_time={compute_time:.4f}s, to_cpu_transfer_time={to_cpu_time:.4f}s")
                        
                        return result
                    
                    return wrapped_forward
                
                # Apply the wrapper
                block.forward = make_wrapped_forward(original_block_forward, block_idx)
                block._original_forward = original_block_forward
                block._blockswap_wrapped = True
                wrapped_count += 1
        
        if wrapped_count > 0:
            logger.info(f"Wrapped {wrapped_count} blocks for swapping (skipped {skipped_count} already wrapped)")
        else:
            logger.info(f"All {skipped_count} blocks already wrapped, no new wrapping needed")
    
    def unpatch(self):
        """
        Restore original forward methods and model patcher methods.
        """
        if not self._is_patched:
            return
        
        # Restore block forwards
        for block in self.blocks:
            if hasattr(block, '_original_forward'):
                block.forward = block._original_forward
                delattr(block, '_original_forward')
            if hasattr(block, '_blockswap_wrapped'):
                delattr(block, '_blockswap_wrapped')
        
        # Restore main forward if we patched it
        if self._original_forward_orig is not None:
            self.diffusion_model.forward_orig = self._original_forward_orig
            self._original_forward_orig = None
        
        # Restore model patcher methods
        if self.model_patcher is not None:
            if hasattr(self.model_patcher, '_original_partially_load'):
                self.model_patcher.partially_load = self.model_patcher._original_partially_load
                delattr(self.model_patcher, '_original_partially_load')
            if hasattr(self.model_patcher, '_original_partially_unload'):
                self.model_patcher.partially_unload = self.model_patcher._original_partially_unload
                delattr(self.model_patcher, '_original_partially_unload')
            if hasattr(self.model_patcher, '_original_model_patches_to'):
                self.model_patcher.model_patches_to = self.model_patcher._original_model_patches_to
                delattr(self.model_patcher, '_original_model_patches_to')
            if hasattr(self.model_patcher, '_original_loaded_size'):
                self.model_patcher.loaded_size = self.model_patcher._original_loaded_size
                delattr(self.model_patcher, '_original_loaded_size')
            if hasattr(self.model_patcher, '_original_model_size'):
                self.model_patcher.model_size = self.model_patcher._original_model_size
                delattr(self.model_patcher, '_original_model_size')
            if hasattr(self.model_patcher, '_blockswap_patcher'):
                delattr(self.model_patcher, '_blockswap_patcher')
        
        self._is_patched = False
        logger.info("Block swap patches removed")


class WAN22BlockSwapPatcher:
    """
    ComfyUI node that patches a WAN model for block swapping during inference.
    
    This node takes any loaded WAN model and modifies it to use block swapping,
    allowing it to work with any standard KSampler while keeping VRAM usage low.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "blocks_to_swap": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 40,
                    "step": 1,
                    "tooltip": "Number of blocks to swap between GPU and CPU. Higher = less VRAM but slower."
                }),
            },
            "optional": {
                "use_non_blocking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use async transfers. Faster but causes shared memory buildup. Set to FALSE to avoid shared memory issues."
                }),
                "prefetch_blocks": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Number of blocks to prefetch ahead (experimental)."
                }),
                "offload_img_emb": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload image embeddings to CPU."
                }),
                "offload_txt_emb": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload text embeddings to CPU."
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print timing info for each block."
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch_model"
    CATEGORY = "WAN22/BlockSwap"
    DESCRIPTION = "Patches a WAN model for block swapping during inference. Use after loading and applying LoRAs."
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute to ensure blocks are properly positioned
        import time
        return time.time()
    
    def patch_model(
        self,
        model,
        blocks_to_swap: int = 20,
        use_non_blocking: bool = True,
        prefetch_blocks: int = 0,
        offload_img_emb: bool = False,
        offload_txt_emb: bool = False,
        debug: bool = False,
    ):
        """
        Patch the model for block swapping.
        """
        logger.info("=" * 60)
        logger.info("WAN22 BlockSwap Patcher")
        logger.info("=" * 60)
        
        # CRITICAL: Reset CUDA state before doing ANYTHING
        # This prevents "invalid argument" errors on consecutive runs
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Pre-patch CUDA reset warning: {e}")
        gc.collect()
        
        # Get the diffusion model from the patcher
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            diffusion_model = model.model.diffusion_model
        else:
            raise ValueError("Could not find diffusion_model in the model patcher")
        
        # Check if there's an EXISTING patcher from a previous run - clean it up first
        if hasattr(model, '_blockswap_patcher'):
            logger.info("Found existing BlockSwap patcher from previous run - cleaning up first")
            old_patcher = model._blockswap_patcher
            try:
                # Unpatch the old patcher to restore original forward methods
                old_patcher.unpatch()
                logger.info("Old patcher unpatched successfully")
            except Exception as e:
                logger.warning(f"Error unpatching old patcher: {e}")
            
            # Restore overridden methods on model
            if hasattr(model, '_original_loaded_size'):
                model.loaded_size = model._original_loaded_size
                delattr(model, '_original_loaded_size')
            if hasattr(model, '_original_model_size'):
                model.model_size = model._original_model_size
                delattr(model, '_original_model_size')
            if hasattr(model, '_original_partially_load'):
                model.partially_load = model._original_partially_load
                delattr(model, '_original_partially_load')
            if hasattr(model, '_original_partially_unload'):
                model.partially_unload = model._original_partially_unload
                delattr(model, '_original_partially_unload')
            
            delattr(model, '_blockswap_patcher')
            logger.info("Old patcher reference removed")
            
            # Extra CUDA reset after cleanup
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except:
                    pass
            gc.collect()
        
        # Check if it's a WAN model
        if not hasattr(diffusion_model, 'blocks'):
            raise ValueError("Model does not have 'blocks' attribute - is this a WAN model?")
        
        num_blocks = len(diffusion_model.blocks)
        logger.info(f"Found WAN model with {num_blocks} blocks")
        
        # Clamp blocks_to_swap
        blocks_to_swap = max(0, min(blocks_to_swap, num_blocks))
        
        # Get devices
        main_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        offload_device = torch.device("cpu")
        
        # Create the patcher
        patcher = BlockSwapForwardPatcher(
            diffusion_model=diffusion_model,
            blocks_to_swap=blocks_to_swap,
            main_device=main_device,
            offload_device=offload_device,
            use_non_blocking=use_non_blocking,
            prefetch_blocks=prefetch_blocks,
            offload_img_emb=offload_img_emb,
            offload_txt_emb=offload_txt_emb,
            debug=debug,
            model_patcher=model,  # Pass the model patcher for cleanup
        )
        
        # Pre-position blocks
        patcher.pre_position_blocks()
        
        # Patch the forward methods
        patcher.patch_forward()
        
        # Store patcher reference on model for later cleanup
        model._blockswap_patcher = patcher
        
        # Calculate the actual loaded size for accurate reporting
        loaded_size_bytes = patcher._reported_loaded_size
        total_size = patcher.calculate_model_size()
        
        logger.info(f"Model sizes - Total: {total_size / (1024*1024):.2f} MB, Loaded on GPU: {loaded_size_bytes / (1024*1024):.2f} MB")
        
        # Prevent ComfyUI from re-loading/moving our model
        # We override key methods to make ComfyUI think the model is already fully loaded
        
        # Override loaded_size to report our actual loaded size
        original_loaded_size = model.loaded_size
        def blockswap_loaded_size(self_model):
            """Report our actual loaded size to ComfyUI."""
            return loaded_size_bytes
        model.loaded_size = types.MethodType(blockswap_loaded_size, model)
        model._original_loaded_size = original_loaded_size
        
        # Override model_size to match loaded_size (trick ComfyUI into thinking it's fully loaded)
        original_model_size = model.model_size
        def blockswap_model_size(self_model):
            """Report model as 'fully loaded' to prevent ComfyUI from loading more."""
            return loaded_size_bytes  # Same as loaded_size = fully loaded
        model.model_size = types.MethodType(blockswap_model_size, model)
        model._original_model_size = original_model_size
        
        # Override partially_load to be a no-op
        original_partially_load = model.partially_load
        def blockswap_aware_partially_load(self_model, device, extra_memory=0, force_patch_weights=False):
            """
            Modified partially_load that doesn't move blocks we're managing.
            This prevents ComfyUI from undoing our block positioning.
            """
            logger.info("BlockSwap-aware partially_load called - returning already loaded size")
            # Return 0 to indicate no additional memory was loaded
            return 0
        model.partially_load = types.MethodType(blockswap_aware_partially_load, model)
        model._original_partially_load = original_partially_load
        
        # Override partially_unload to prevent ComfyUI from unloading our positioned blocks
        if hasattr(model, 'partially_unload'):
            original_partially_unload = model.partially_unload
            def blockswap_aware_partially_unload(self_model, device, memory_to_free):
                """Prevent ComfyUI from unloading our blocks mid-sampling."""
                logger.info("BlockSwap-aware partially_unload called - skipping")
                return 0  # Report nothing was unloaded
            model.partially_unload = types.MethodType(blockswap_aware_partially_unload, model)
            model._original_partially_unload = original_partially_unload
        
        # Also prevent model.model_patches_to from moving the diffusion model
        if hasattr(model, 'model_patches_to'):
            original_patches_to = model.model_patches_to
            def blockswap_aware_patches_to(self_model, device=None, dtype=None, force_patch_weights=False):
                """Skip patches_to for diffusion model, we've already positioned it."""
                logger.info("BlockSwap-aware model_patches_to called - skipping")
                return
            model.model_patches_to = types.MethodType(blockswap_aware_patches_to, model)
            model._original_model_patches_to = original_patches_to
        
        # Remove from ComfyUI's loaded models to prevent interference
        try:
            for loaded_model in list(mm.current_loaded_models):
                if hasattr(loaded_model, 'model') and loaded_model.model is model:
                    mm.current_loaded_models.remove(loaded_model)
                    logger.info("Removed model from ComfyUI's loaded models cache")
                    break
        except Exception as e:
            logger.warning(f"Could not remove model from cache: {e}")
        
        logger.info("=" * 60)
        logger.info(f"Block swap patching complete!")
        logger.info(f"  GPU blocks: 0-{patcher.swap_start_idx - 1}")
        logger.info(f"  Swapped blocks: {patcher.swap_start_idx}-{num_blocks - 1}")
        logger.info("=" * 60)
        
        return (model,)


class WAN22BlockSwapComboPatcher:
    """
    ComfyUI node that patches BOTH high and low noise WAN models for block swapping.
    
    This is designed for use with combo KSamplers like WanVideoLooper or Integrated KSampler
    that take both models as inputs. 
    
    The problem: When using two separate patchers, ComfyUI executes BOTH patcher nodes
    before the KSampler runs, loading both models' blocks to GPU simultaneously.
    
    The solution: This combo patcher:
    1. Takes BOTH models as input (single node execution)
    2. Positions HIGH noise blocks on GPU (ready for sampling)
    3. Keeps LOW noise blocks ALL on CPU (waiting)
    4. Uses ON_CLEANUP callback to switch models when high noise completes
    
    This ensures only ONE model's blocks are on GPU at a time!
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high": ("MODEL", {
                    "tooltip": "The high noise WAN model (runs first in MoE sampling)."
                }),
                "model_low": ("MODEL", {
                    "tooltip": "The low noise WAN model (runs second in MoE sampling)."
                }),
                "blocks_to_swap": ("INT", {
                    "default": 20,
                    "min": 0,
                    "max": 40,
                    "step": 1,
                    "tooltip": "Number of blocks to swap between GPU and CPU during inference. Higher = less VRAM but slower."
                }),
            },
            "optional": {
                "use_non_blocking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use non-blocking transfers. Set to False to avoid pinned memory issues."
                }),
                "offload_img_emb": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload image embeddings to CPU."
                }),
                "offload_txt_emb": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload text embeddings to CPU."
                }),
                "vace_blocks_to_swap": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 40,
                    "step": 1,
                    "tooltip": "Number of VACE blocks to swap (for VACE models, 0 to disable)."
                }),
                "prefetch_blocks": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Number of blocks to prefetch ahead (experimental)."
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print timing info for each block swap operation."
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("model_high", "model_low")
    FUNCTION = "patch_models"
    CATEGORY = "WAN22/BlockSwap"
    DESCRIPTION = "Patches BOTH high and low noise WAN models for block swapping. Use with combo KSamplers (Integrated, Looper, etc.)."
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute to ensure blocks are properly positioned
        return time.time()
    
    def _reset_cuda_error_state(self):
        """Reset CUDA error state before operations that might fail due to previous errors."""
        if torch.cuda.is_available():
            try:
                # Synchronize to flush any pending operations
                torch.cuda.synchronize()
                # Clear any pending CUDA errors by calling a safe operation
                torch.cuda.current_device()
                # Empty cache to release any corrupted allocations
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error resetting CUDA state: {e}")
                # Try a more aggressive reset
                try:
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.empty_cache()
                except:
                    pass
    
    def _cleanup_existing_patcher(self, model, model_name):
        """Clean up any existing BlockSwap patcher on a model."""
        
        # First, reset CUDA error state in case previous run left it corrupted
        self._reset_cuda_error_state()
        
        if hasattr(model, '_blockswap_patcher'):
            logger.info(f"Found existing BlockSwap patcher on {model_name} - cleaning up")
            old_patcher = model._blockswap_patcher
            
            # CRITICAL: Move ALL blocks to CPU first to reset state
            # Use the patcher's safe method if available for GGUF compatibility
            try:
                with suppress_unpin_warnings():
                    if hasattr(old_patcher, 'blocks') and old_patcher.blocks is not None:
                        logger.info(f"  Moving all blocks to CPU for {model_name}...")
                        has_safe_method = hasattr(old_patcher, '_safe_block_to_device')
                        for idx, block in enumerate(old_patcher.blocks):
                            try:
                                # Reset CUDA before each block move to handle async errors
                                if idx % 5 == 0:  # Every 5 blocks
                                    self._reset_cuda_error_state()
                                # Use safe block movement if available
                                if has_safe_method:
                                    old_patcher._safe_block_to_device(block, torch.device("cpu"), idx)
                                else:
                                    block.to(torch.device("cpu"), non_blocking=False)
                            except Exception as e:
                                logger.warning(f"  Block {idx} move failed: {e}")
                                # Try to recover and continue
                                self._reset_cuda_error_state()
                        # Sync after all moves
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"Error moving blocks to CPU on {model_name}: {e}")
            
            # Now unpatch the forward method
            try:
                old_patcher.unpatch()
            except Exception as e:
                logger.warning(f"Error unpatching old patcher on {model_name}: {e}")
            
            # Restore overridden methods
            for attr in ['_original_loaded_size', '_original_model_size', '_original_partially_load', 
                         '_original_partially_unload', '_original_model_patches_to']:
                if hasattr(model, attr):
                    original_name = attr.replace('_original_', '')
                    setattr(model, original_name, getattr(model, attr))
                    delattr(model, attr)
            
            if hasattr(model, '_blockswap_patcher'):
                delattr(model, '_blockswap_patcher')
            if hasattr(model, '_combo_patcher_role'):
                delattr(model, '_combo_patcher_role')
        
        # Also check if model has blocks in inconsistent state even without patcher
        # This handles cases where the patcher attribute was lost but blocks are mixed
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
                diffusion_model = model.model.diffusion_model
                if hasattr(diffusion_model, 'blocks'):
                    # Check first block's device - if on GPU, move all to CPU for clean start
                    first_block = diffusion_model.blocks[0]
                    first_param = next(first_block.parameters(), None)
                    if first_param is not None and first_param.device.type == 'cuda':
                        logger.info(f"  {model_name}: Found GPU blocks without patcher - resetting to CPU")
                        with suppress_unpin_warnings():
                            for block in diffusion_model.blocks:
                                block.to(torch.device("cpu"), non_blocking=False)
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
        except Exception as e:
            logger.debug(f"Block device check failed for {model_name}: {e}")
    
    def _create_patcher(self, model, blocks_to_swap, use_non_blocking, prefetch_blocks,
                        offload_img_emb, offload_txt_emb, debug):
        """Create a BlockSwapForwardPatcher for a model WITHOUT positioning blocks."""
        
        # Get the diffusion model
        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            diffusion_model = model.model.diffusion_model
        else:
            raise ValueError("Could not find diffusion_model in the model patcher")
        
        if not hasattr(diffusion_model, 'blocks'):
            raise ValueError("Model does not have 'blocks' attribute - is this a WAN model?")
        
        num_blocks = len(diffusion_model.blocks)
        blocks_to_swap = max(0, min(blocks_to_swap, num_blocks))
        
        main_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        offload_device = torch.device("cpu")
        
        # Create the patcher
        patcher = BlockSwapForwardPatcher(
            diffusion_model=diffusion_model,
            blocks_to_swap=blocks_to_swap,
            main_device=main_device,
            offload_device=offload_device,
            use_non_blocking=use_non_blocking,
            prefetch_blocks=prefetch_blocks,
            offload_img_emb=offload_img_emb,
            offload_txt_emb=offload_txt_emb,
            debug=debug,
            model_patcher=model,
        )
        
        return patcher
    
    def _move_all_blocks_to_cpu(self, patcher, model_name):
        """Move ALL blocks of a model to CPU using safe block movement."""
        logger.info(f"Moving ALL blocks to CPU for {model_name}...")
        with suppress_unpin_warnings():
            for idx, block in enumerate(patcher.blocks):
                patcher._safe_block_to_device(block, patcher.offload_device, idx)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info(f"  {model_name}: All {len(patcher.blocks)} blocks now on CPU")
    
    def _override_model_methods(self, model, loaded_size_bytes):
        """Override ComfyUI model management methods to prevent interference."""
        
        def blockswap_loaded_size(self_model):
            return loaded_size_bytes
        model.loaded_size = types.MethodType(blockswap_loaded_size, model)
        
        def blockswap_model_size(self_model):
            return loaded_size_bytes
        model.model_size = types.MethodType(blockswap_model_size, model)
        
        def blockswap_aware_partially_load(self_model, device, extra_memory=0, force_patch_weights=False):
            return 0
        model.partially_load = types.MethodType(blockswap_aware_partially_load, model)
        
        if hasattr(model, 'partially_unload'):
            def blockswap_aware_partially_unload(self_model, device, memory_to_free):
                return 0
            model.partially_unload = types.MethodType(blockswap_aware_partially_unload, model)
    
    def patch_models(
        self,
        model_high,
        model_low,
        blocks_to_swap: int = 20,
        use_non_blocking: bool = False,
        offload_img_emb: bool = False,
        offload_txt_emb: bool = False,
        vace_blocks_to_swap: int = 0,
        prefetch_blocks: int = 1,
        debug: bool = False,
    ):
        """
        Patch both models for block swapping with proper sequencing.
        
        HIGH noise model: blocks positioned on GPU, ready to sample
        LOW noise model: ALL blocks on CPU, will be positioned when high finishes
        """
        # Reset the CUDA corruption flag at the start of each new generation
        reset_cuda_corruption_flag()
        
        # Check CUDA health before we start
        try:
            check_cuda_or_raise("patch_models start")
        except RuntimeError as e:
            logger.error(str(e))
            raise
        
        # Install persistent warning filter for the entire sampling process
        # This suppresses "Tried to unpin tensor not pinned by ComfyUI" spam
        install_unpin_warning_filter()
        
        logger.info("=" * 60)
        logger.info("WAN22 BlockSwap COMBO Patcher")
        logger.info("  For use with Integrated KSampler / WanVideoLooper")
        logger.info("=" * 60)
        
        # ========================================
        # Pre-cleanup: Ensure CUDA is in a good state
        # ========================================
        # This is critical when queueing multiple runs - the previous run
        # may have left CUDA in a bad state
        if torch.cuda.is_available():
            try:
                # Synchronize any pending operations
                torch.cuda.synchronize()
                
                # Clear any cached memory
                torch.cuda.empty_cache()
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                
                # Verify CUDA is responsive with a small test allocation
                test_tensor = torch.zeros(1, device='cuda')
                del test_tensor
                torch.cuda.synchronize()
                
            except RuntimeError as e:
                # CUDA is in a bad state - try to recover
                logger.warning(f"CUDA state issue detected: {e}")
                logger.warning("Attempting CUDA recovery...")
                try:
                    # Force empty cache
                    torch.cuda.empty_cache()
                    # Try synchronization again
                    torch.cuda.synchronize()
                except Exception as e2:
                    logger.error(f"CUDA recovery failed: {e2}")
                    logger.error("You may need to restart ComfyUI to clear CUDA errors")
        
        gc.collect()
        
        # Clean up any existing patchers from previous runs
        self._cleanup_existing_patcher(model_high, "model_high")
        self._cleanup_existing_patcher(model_low, "model_low")
        
        # Extra cleanup after removing old patchers
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except:
                pass
        gc.collect()
        
        # Get block counts for logging
        num_blocks_high = len(model_high.model.diffusion_model.blocks)
        num_blocks_low = len(model_low.model.diffusion_model.blocks)
        
        # CRITICAL CHECK: Ensure HIGH and LOW don't share the same diffusion_model
        # If they do, wrapping one will affect the other, causing double-wrapping
        if model_high.model.diffusion_model is model_low.model.diffusion_model:
            logger.error("=" * 60)
            logger.error("ERROR: HIGH and LOW models share the same diffusion_model!")
            logger.error("This will cause double-wrapping and duplicate block swaps.")
            logger.error("Each model must have its own diffusion_model instance.")
            logger.error("=" * 60)
            raise ValueError("HIGH and LOW models must have separate diffusion_model instances")
        
        # Also check if blocks are the same objects
        blocks_high = model_high.model.diffusion_model.blocks
        blocks_low = model_low.model.diffusion_model.blocks
        if blocks_high is blocks_low:
            logger.error("=" * 60)
            logger.error("ERROR: HIGH and LOW models share the same blocks list!")
            logger.error("This will cause double-wrapping and duplicate block swaps.")
            logger.error("=" * 60)
            raise ValueError("HIGH and LOW models must have separate block instances")
        
        logger.info(f"High noise model: {num_blocks_high} blocks")
        logger.info(f"Low noise model: {num_blocks_low} blocks")
        logger.info(f"Blocks to swap: {blocks_to_swap}")
        
        # ========================================
        # Step 1: Create patchers for BOTH models
        # ========================================
        logger.info("")
        logger.info("Creating BlockSwap patchers...")
        
        patcher_high = self._create_patcher(
            model_high, blocks_to_swap, use_non_blocking, prefetch_blocks,
            offload_img_emb, offload_txt_emb, debug
        )
        
        patcher_low = self._create_patcher(
            model_low, blocks_to_swap, use_non_blocking, prefetch_blocks,
            offload_img_emb, offload_txt_emb, debug
        )
        
        # ========================================
        # Step 2: Move LOW noise model ALL to CPU
        # ========================================
        logger.info("")
        logger.info("Step 1: Moving LOW noise model entirely to CPU...")
        self._move_all_blocks_to_cpu(patcher_low, "model_low")
        
        # ========================================
        # Step 3: Position HIGH noise blocks (GPU/CPU split)
        # ========================================
        logger.info("")
        logger.info("Step 2: Positioning HIGH noise model blocks...")
        patcher_high.pre_position_blocks()
        
        # ========================================
        # Step 4: Patch forward methods for BOTH models
        # ========================================
        logger.info("")
        logger.info("Step 3: Patching forward methods...")
        patcher_high.patch_forward()
        patcher_low.patch_forward()
        
        # Store patcher references
        model_high._blockswap_patcher = patcher_high
        model_low._blockswap_patcher = patcher_low
        
        # Cross-reference for the callback
        patcher_high._paired_patcher = patcher_low
        patcher_low._paired_patcher = patcher_high
        patcher_high._paired_model = model_low
        patcher_low._paired_model = model_high
        
        # ========================================
        # Step 5: Override ComfyUI model methods
        # ========================================
        # High model reports its loaded size, low model reports 0 (all on CPU)
        self._override_model_methods(model_high, patcher_high._reported_loaded_size)
        self._override_model_methods(model_low, 0)  # All on CPU = 0 loaded
        
        # Mark models with their roles
        model_high._combo_patcher_role = "high"
        model_low._combo_patcher_role = "low"
        
        # ========================================
        # Step 6: Add callback for model switching
        # ========================================
        logger.info("")
        logger.info("Step 4: Setting up model switching callback...")
        
        # Create the switching callback
        def on_high_noise_cleanup(model_instance):
            """
            Called when high noise model finishes sampling (ON_CLEANUP).
            This moves high noise to CPU and positions low noise for sampling.
            """
            logger.info("")
            logger.info("=" * 60)
            logger.info("HIGH NOISE COMPLETE - Switching to LOW noise model")
            logger.info("=" * 60)
            
            # Get the patchers
            high_patcher = getattr(model_high, '_blockswap_patcher', None)
            low_patcher = getattr(model_low, '_blockswap_patcher', None)
            
            if high_patcher is None or low_patcher is None:
                logger.warning("Could not find patchers for model switching!")
                return
            
            # Step A: Move HIGH noise blocks to CPU
            logger.info("Moving HIGH noise blocks to CPU...")
            try:
                with suppress_unpin_warnings():
                    high_patcher.cleanup(move_to_cpu=True)
            except Exception as e:
                logger.warning(f"Error moving high noise to CPU: {e}")
            
            # Step B: Clear CUDA cache
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"CUDA cache clear warning: {e}")
            gc.collect()
            
            # Step C: Position LOW noise blocks for sampling
            logger.info("Positioning LOW noise blocks for sampling...")
            try:
                low_patcher.pre_position_blocks()
            except Exception as e:
                logger.warning(f"Error positioning low noise: {e}")
            
            # Report VRAM status
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"VRAM after switch - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            
            logger.info("Model switch complete - LOW noise ready for sampling")
            logger.info("=" * 60)
        
        # Add the callback to high noise model
        try:
            from comfy.patcher_extension import CallbacksMP
            model_high.add_callback(CallbacksMP.ON_CLEANUP, on_high_noise_cleanup)
            logger.info("Added ON_CLEANUP callback to HIGH noise model")
            logger.info("  -> When high noise sampling finishes, low noise will be positioned")
            
            # Add cleanup callback to low noise model for proper cleanup
            def on_low_noise_cleanup(model_instance):
                """
                Called when low noise model finishes sampling (ON_CLEANUP).
                This cleans up BOTH models to prevent CUDA errors when ComfyUI
                tries to unload models (e.g., for VAE decode or next iteration).
                """
                logger.info("")
                logger.info("=" * 60)
                logger.info("LOW NOISE COMPLETE - Cleaning up BOTH models")
                logger.info("=" * 60)
                
                # Get both patchers
                high_patcher = getattr(model_high, '_blockswap_patcher', None)
                low_patcher = getattr(model_low, '_blockswap_patcher', None)
                
                # CRITICAL: Move ALL blocks of BOTH models to CPU
                # This ensures models are in a consistent state before ComfyUI
                # tries to unload them (e.g., for VAE decode)
                
                # Sync CUDA first
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception as e:
                        logger.warning(f"CUDA sync warning: {e}")
                
                # Clean up LOW noise model (was just active) - use safe method if available
                if low_patcher is not None:
                    logger.info("Moving LOW noise blocks to CPU...")
                    try:
                        with suppress_unpin_warnings():
                            for idx, block in enumerate(low_patcher.blocks):
                                low_patcher._safe_block_to_device(block, low_patcher.offload_device, idx)
                            # Move non-block params too
                            for name, param in low_patcher.diffusion_model.named_parameters():
                                if "blocks." not in name:
                                    param.data = param.data.to(low_patcher.offload_device, non_blocking=False)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    except Exception as e:
                        logger.warning(f"Error cleaning up low noise: {e}")
                
                # Clean up HIGH noise model (was moved to CPU earlier, but ensure it's fully there)
                if high_patcher is not None:
                    logger.info("Ensuring HIGH noise blocks are on CPU...")
                    try:
                        with suppress_unpin_warnings():
                            for idx, block in enumerate(high_patcher.blocks):
                                high_patcher._safe_block_to_device(block, high_patcher.offload_device, idx)
                            for name, param in high_patcher.diffusion_model.named_parameters():
                                if "blocks." not in name:
                                    param.data = param.data.to(high_patcher.offload_device, non_blocking=False)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    except Exception as e:
                        logger.warning(f"Error cleaning up high noise: {e}")
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception as e:
                        logger.warning(f"CUDA cache clear warning: {e}")
                gc.collect()
                
                # Uninstall warning filter
                uninstall_unpin_warning_filter()
                
                # Report VRAM status
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    logger.info(f"VRAM after cleanup - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
                
                # Reset CUDA corruption flag for next iteration
                reset_cuda_corruption_flag()
                
                logger.info("Both models cleaned up - ready for next iteration or VAE decode")
                logger.info("=" * 60)
            
            model_low.add_callback(CallbacksMP.ON_CLEANUP, on_low_noise_cleanup)
            logger.info("Added ON_CLEANUP callback to LOW noise model")
            logger.info("  -> Will clean up BOTH models after sampling completes")
            
        except Exception as e:
            logger.error(f"CRITICAL: Could not add cleanup callback: {e}")
            logger.error("Model switching will NOT work automatically!")
            logger.error("Consider using separate KSamplers instead of combo KSampler.")
        
        # ========================================
        # Final: Ensure CUDA is in clean state for subsequent operations
        # ========================================
        # This is CRITICAL - without this, the CLIP model loading may fail
        # with "CUDA error: invalid argument" because the GPU state is corrupted
        # from all our block movements
        if torch.cuda.is_available():
            try:
                # Full synchronization to flush all pending operations
                torch.cuda.synchronize()
                
                # Empty cache to defragment memory
                torch.cuda.empty_cache()
                
                # Reset memory stats (helps with some edge cases)
                torch.cuda.reset_peak_memory_stats()
                
                # One more sync to be safe
                torch.cuda.synchronize()
                
                # Test allocation to verify GPU is responsive
                # Use a reasonable size (1MB) to catch fragmentation issues
                test_tensor = torch.zeros(256, 1024, device='cuda', dtype=torch.float32)  # 1MB
                del test_tensor
                torch.cuda.synchronize()
                
                # Small delay to let CUDA driver fully stabilize
                # This helps when many tensor movements have occurred
                time.sleep(0.1)
                
                # Final sync and cache clear
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.warning(f"Post-setup CUDA cleanup warning: {e}")
                # Try to recover
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except:
                    pass
        
        # Force garbage collection
        gc.collect()
        
        # ========================================
        # Report status
        # ========================================
        logger.info("")
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"VRAM - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("Combo Patcher Setup Complete!")
        logger.info("  [OK] HIGH noise: Blocks positioned, ready to sample")
        logger.info("  [OK] LOW noise: All blocks on CPU, waiting")
        logger.info("  [OK] Callbacks: HIGH->LOW switch + full cleanup registered")
        logger.info("  [OK] After each iteration: Both models cleaned to CPU")
        logger.info("  [OK] Unpin warnings: Suppressed for clean output")
        
        # Detect GGUF models and warn using our patcher instances
        high_is_gguf = patcher_high._is_gguf_model if patcher_high else False
        low_is_gguf = patcher_low._is_gguf_model if patcher_low else False
        if high_is_gguf or low_is_gguf:
            high_type = patcher_high._gguf_type if patcher_high else None
            low_type = patcher_low._gguf_type if patcher_low else None
            if high_type == "comfyui-gguf" or low_type == "comfyui-gguf":
                logger.info("  [!] ComfyUI-GGUF detected: Using extra-safe block movement mode")
                logger.info("      If CUDA errors occur after 2-3 runs, restart ComfyUI")
            else:
                logger.info("  [!] GGUF model detected: Using safe block movement mode")
        
        logger.info("=" * 60)
        
        # Final CUDA health check
        if not is_cuda_healthy():
            logger.warning("CUDA may be unstable after setup - if errors occur, restart ComfyUI")
        
        return (model_high, model_low)


class WAN22BlockSwapCleanup:
    """
    ComfyUI node that cleans up memory after sampling with BlockSwap.
    
    Place this node after your KSampler to free VRAM and RAM.
    Can be used between high/low noise runs to free memory.
    
    For Integrated KSampler: Connect latent output
    For WanVideoLooper: Connect images output (since looper outputs images directly)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                # Either latent or images can be connected to trigger execution order
                "latent": ("LATENT", {"tooltip": "Connect latent output from Integrated KSampler"}),
                "images": ("IMAGE", {"tooltip": "Connect images output from WanVideoLooper"}),
                "move_to_cpu": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Move all model blocks to CPU to free VRAM."
                }),
                "unpatch": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Remove all BlockSwap patches (use at end of workflow)."
                }),
                "clear_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clear CUDA cache and run garbage collection."
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "LATENT", "IMAGE")
    RETURN_NAMES = ("model", "latent", "images")
    FUNCTION = "cleanup"
    CATEGORY = "WAN22/BlockSwap"
    DESCRIPTION = "Cleans up memory after BlockSwap sampling. Connect latent (Integrated KSampler) or images (WanVideoLooper)."
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute cleanup
        import time
        return time.time()
    
    def cleanup(
        self,
        model,
        latent=None,
        images=None,
        move_to_cpu: bool = True,
        unpatch: bool = False,
        clear_cache: bool = True,
    ):
        """
        Clean up memory after sampling.
        
        Args:
            model: The model to clean up
            latent: Optional latent pass-through (from Integrated KSampler)
            images: Optional image pass-through (from WanVideoLooper)
            move_to_cpu: Move all blocks to CPU
            unpatch: Remove BlockSwap patches
            clear_cache: Clear CUDA cache and run GC
        """
        logger.info("=" * 60)
        logger.info("WAN22 BlockSwap Cleanup")
        logger.info("=" * 60)
        
        # Sync CUDA first to avoid async errors
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"CUDA sync warning: {e}")
        
        # Check if model has a blockswap patcher
        if hasattr(model, '_blockswap_patcher'):
            patcher = model._blockswap_patcher
            
            if move_to_cpu:
                try:
                    patcher.cleanup(move_to_cpu=True)
                except Exception as e:
                    logger.warning(f"Cleanup warning: {e}")
            
            if unpatch:
                try:
                    patcher.unpatch()
                    logger.info("BlockSwap patches removed from model")
                except Exception as e:
                    logger.warning(f"Unpatch warning: {e}")
        else:
            logger.info("Model does not have BlockSwap patcher attached")
        
        if clear_cache:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"CUDA cleanup warning: {e}")
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            logger.info("Cleared CUDA cache and ran garbage collection")
        
        # Report memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"CUDA memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
        logger.info("=" * 60)
        
        return (model, latent, images)


class WAN22BlockSwapReposition:
    """
    ComfyUI node that re-positions blocks for a new sampling run.
    
    Use this between high/low noise KSamplers if you cleaned up in between.
    This moves blocks back to their correct GPU/CPU positions.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),  # Pass-through to ensure execution order
            },
        }
    
    RETURN_TYPES = ("MODEL", "LATENT")
    RETURN_NAMES = ("model", "latent")
    FUNCTION = "reposition"
    CATEGORY = "WAN22/BlockSwap"
    DESCRIPTION = "Re-positions blocks for a new sampling run after cleanup."
    
    def reposition(self, model, latent):
        """
        Re-position blocks for a new sampling run.
        """
        logger.info("=" * 60)
        logger.info("WAN22 BlockSwap Reposition")
        logger.info("=" * 60)
        
        if hasattr(model, '_blockswap_patcher'):
            patcher = model._blockswap_patcher
            patcher.pre_position_blocks()
            logger.info("Blocks repositioned for new sampling run")
        else:
            logger.warning("Model does not have BlockSwap patcher attached - nothing to reposition")
        
        logger.info("=" * 60)
        
        return (model, latent)


class WAN22FullCleanup:
    """
    ComfyUI node that performs aggressive memory cleanup at the END of a workflow.
    
    This mimics the behavior of ComfyUI's "Free Model and Node Cache" button.
    
    IMPORTANT: This should be placed at the VERY END of your workflow, after all
    KSamplers and other processing. It will unload all models from GPU memory
    AND set a flag to clear the node cache after the workflow completes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                # Use ANY_TYPE (IO.ANY) which properly accepts any input type
                # Connect any output here (Filenames, IMAGE, LATENT, etc.) to trigger cleanup at end of workflow
                # NOTE: Do NOT connect MODEL outputs here - it can cause CUDA issues
                "any_input": (ANY_TYPE,),
                "unload_models": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload all models from GPU (like the 'Free Model' button). This frees VRAM but models will reload on next run."
                }),
                "free_memory": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Set flag to clear node cache after workflow completes (like the 'Free Model and Node Cache' button). This frees RAM."
                }),
                "clear_cuda_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clear PyTorch CUDA cache."
                }),
                "run_gc": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Run Python garbage collection."
                }),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "full_cleanup"
    CATEGORY = "WAN22/BlockSwap"
    DESCRIPTION = "Aggressive memory cleanup at the END of a workflow. Mimics 'Free Model and Node Cache' button behavior."
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Accept any input type
        return True
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Always re-execute cleanup
        return time.time()
    
    def full_cleanup(
        self,
        any_input=None,
        unload_models: bool = True,
        free_memory: bool = True,
        clear_cuda_cache: bool = True,
        run_gc: bool = True,
    ):
        """
        Perform aggressive memory cleanup.
        
        This is designed to run at the END of a workflow and mimics the
        'Free Model and Node Cache' button in ComfyUI's UI.
        """
        logger.info("=" * 60)
        logger.info("WAN22 Full Cleanup (End of Workflow)")
        logger.info("=" * 60)
        
        # Report initial memory
        if torch.cuda.is_available():
            allocated_before = torch.cuda.memory_allocated() / (1024**3)
            reserved_before = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"BEFORE - Allocated: {allocated_before:.2f} GB, Reserved: {reserved_before:.2f} GB")
        
        # Step 0: Set flags on prompt_queue to trigger cleanup AFTER workflow completes
        # This is exactly what the "Free Model and Node Cache" button does
        try:
            from server import PromptServer
            if hasattr(PromptServer, 'instance') and PromptServer.instance is not None:
                if hasattr(PromptServer.instance, 'prompt_queue'):
                    if unload_models:
                        PromptServer.instance.prompt_queue.set_flag("unload_models", True)
                        logger.info("Set 'unload_models' flag on prompt_queue")
                    if free_memory:
                        PromptServer.instance.prompt_queue.set_flag("free_memory", True)
                        logger.info("Set 'free_memory' flag on prompt_queue (node cache will clear after workflow)")
        except Exception as e:
            logger.warning(f"Could not set prompt_queue flags: {e}")
        
        # Step 1: Sync CUDA to ensure all operations complete
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"CUDA sync warning: {e}")
        
        # Step 2: Run garbage collection first
        if run_gc:
            gc.collect()
            logger.info("Ran Python garbage collection")
        
        # Step 3: Clear CUDA cache first time
        if clear_cuda_cache and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache (pre-unload)")
            except Exception as e:
                logger.warning(f"CUDA cache clear warning: {e}")
        
        # Step 4: Unload all models (this is what the button does)
        if unload_models:
            try:
                mm.unload_all_models()
                logger.info("Unloaded all models from GPU (mm.unload_all_models)")
            except Exception as e:
                logger.warning(f"Model unload warning: {e}")
        
        # Step 5: Clear CUDA cache again after unloading
        if clear_cuda_cache and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache (post-unload)")
            except Exception as e:
                logger.warning(f"CUDA cache clear warning: {e}")
        
        # Step 6: Sync and final cleanup
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as e:
                logger.warning(f"Final CUDA sync warning: {e}")
        
        # Step 7: Run gc again to catch freed references
        if run_gc:
            gc.collect()
            logger.info("Ran final garbage collection")
        
        # Step 8: soft_empty_cache for any ComfyUI managed memory
        try:
            mm.soft_empty_cache()
            logger.info("Called mm.soft_empty_cache()")
        except Exception as e:
            logger.warning(f"soft_empty_cache warning: {e}")
        
        # Report final memory
        if torch.cuda.is_available():
            allocated_after = torch.cuda.memory_allocated() / (1024**3)
            reserved_after = torch.cuda.memory_reserved() / (1024**3)
            freed_allocated = allocated_before - allocated_after
            freed_reserved = reserved_before - reserved_after
            logger.info(f"AFTER - Allocated: {allocated_after:.2f} GB, Reserved: {reserved_after:.2f} GB")
            logger.info(f"FREED - Allocated: {freed_allocated:.2f} GB, Reserved: {freed_reserved:.2f} GB")
        
        logger.info("=" * 60)
        logger.info("Full cleanup complete!")
        if free_memory:
            logger.info("Node cache will be cleared AFTER this workflow completes.")
        logger.info("=" * 60)
        
        return ()


# Node registration
NODE_CLASS_MAPPINGS = {
    "WAN22BlockSwapPatcher": WAN22BlockSwapPatcher,
    "WAN22BlockSwapComboPatcher": WAN22BlockSwapComboPatcher,
    "WAN22BlockSwapCleanup": WAN22BlockSwapCleanup,
    "WAN22BlockSwapReposition": WAN22BlockSwapReposition,
    "WAN22FullCleanup": WAN22FullCleanup,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WAN22BlockSwapPatcher": "WAN 2.2 BlockSwap Patcher",
    "WAN22BlockSwapComboPatcher": "WAN 2.2 BlockSwap Combo Patcher",
    "WAN22BlockSwapCleanup": "WAN 2.2 BlockSwap Cleanup",
    "WAN22BlockSwapReposition": "WAN 2.2 BlockSwap Reposition",
    "WAN22FullCleanup": "WAN 2.2 Full Cleanup (End)",
}
