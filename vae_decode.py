"""
WAN Video Tiled VAE Decode for ComfyUI_Wan22Blockswap.

This module provides a high-quality tiled VAE decode node specifically
designed for WAN video models. Unlike standard tiled VAE decoders that
tile in all dimensions (including temporal), this implementation:

1. Tiles ONLY spatially (H, W dimensions)
2. Processes ALL frames together per tile (preserves temporal coherence)
3. Uses linear feathering with min-based 2D masks for smooth blending
4. Supports the causal temporal processing of WAN VAE architecture

This approach prevents flickering, color inconsistencies, and tile boundary
artifacts that occur with standard tiled decoders on video latents.

Based on research from WanVideoWrapper's tiled VAE implementation.
"""

import torch
import gc
import logging
from typing import Tuple, Optional

import comfy.model_management as mm
import comfy.utils

logger = logging.getLogger("ComfyUI_Wan22Blockswap")


def _build_1d_mask(length: int, left_bound: bool, right_bound: bool, border_width: int) -> torch.Tensor:
    """
    Build a 1D feathering mask for tile blending.
    
    Args:
        length: Total length of the mask
        left_bound: If True, this is the leftmost tile (no left feathering)
        right_bound: If True, this is the rightmost tile (no right feathering)
        border_width: Width of the feathering region in pixels
    
    Returns:
        1D tensor with linear ramp at non-boundary edges
    """
    x = torch.ones((length,), dtype=torch.float32)
    
    if border_width <= 0:
        return x
    
    # Left edge feathering (if not at boundary)
    if not left_bound:
        ramp = (torch.arange(border_width, dtype=torch.float32) + 1) / border_width
        x[:border_width] = ramp
    
    # Right edge feathering (if not at boundary)
    if not right_bound:
        ramp = (torch.arange(border_width, dtype=torch.float32) + 1) / border_width
        x[-border_width:] = torch.flip(ramp, dims=(0,))
    
    return x


def _build_2d_mask(
    height: int, 
    width: int, 
    is_bound: Tuple[bool, bool, bool, bool],
    border_width: Tuple[int, int]
) -> torch.Tensor:
    """
    Build a 2D feathering mask for tile blending.
    
    Uses the minimum of H and W masks for clean corner handling.
    
    Args:
        height: Height of the tile in pixels
        width: Width of the tile in pixels
        is_bound: Tuple of (top_bound, bottom_bound, left_bound, right_bound)
        border_width: Tuple of (h_border, w_border) in pixels
    
    Returns:
        Tensor of shape (1, 1, 1, H, W) for broadcasting with video tensors
    """
    top_bound, bottom_bound, left_bound, right_bound = is_bound
    h_border, w_border = border_width
    
    # Build 1D masks
    h_mask = _build_1d_mask(height, top_bound, bottom_bound, h_border)
    w_mask = _build_1d_mask(width, left_bound, right_bound, w_border)
    
    # Expand to 2D: h_mask becomes (H, W), w_mask becomes (H, W)
    h_2d = h_mask.unsqueeze(1).expand(height, width)  # (H, 1) -> (H, W)
    w_2d = w_mask.unsqueeze(0).expand(height, width)  # (1, W) -> (H, W)
    
    # Take minimum for proper corner blending
    mask_2d = torch.minimum(h_2d, w_2d)
    
    # Reshape to (1, 1, 1, H, W) for broadcasting with (B, C, T, H, W) tensors
    return mask_2d.reshape(1, 1, 1, height, width)


def tiled_decode_wan(
    vae,
    samples: torch.Tensor,
    tile_size: Tuple[int, int] = (34, 34),
    tile_stride: Tuple[int, int] = (18, 16),
    upscale_factor: int = 8,
    clear_cache_between_tiles: bool = False,
) -> torch.Tensor:
    """
    Decode video latents using spatial-only tiling with temporal coherence.
    
    This function tiles only in spatial dimensions (H, W) while processing
    all temporal frames together, preserving the VAE's causal temporal
    processing and avoiding frame flickering.
    
    Args:
        vae: ComfyUI VAE object
        samples: Latent tensor of shape (B, C, T, H, W) - 5D video latent
        tile_size: (height, width) of each tile in latent space
        tile_stride: (height_stride, width_stride) - step between tiles
        upscale_factor: VAE upscaling factor (typically 8 for WAN)
        clear_cache_between_tiles: If True, clear CUDA cache between tiles
    
    Returns:
        Decoded tensor of shape (B, C, T, H*upscale_factor, W*upscale_factor)
    """
    device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    
    # Handle both 4D (image) and 5D (video) latents
    if samples.dim() == 4:
        # Image latent: (B, C, H, W) -> (B, C, 1, H, W)
        samples = samples.unsqueeze(2)
        is_video = False
    else:
        is_video = True
    
    B, C, T, H, W = samples.shape
    size_h, size_w = tile_size
    stride_h, stride_w = tile_stride
    
    # Calculate output dimensions
    out_h = H * upscale_factor
    out_w = W * upscale_factor
    
    # Calculate border widths for mask blending (in pixel space)
    h_overlap = size_h - stride_h
    w_overlap = size_w - stride_w
    h_border_pixels = h_overlap * upscale_factor
    w_border_pixels = w_overlap * upscale_factor
    
    logger.info(f"[WAN22 Tiled VAE] Input: {B}x{C}x{T}x{H}x{W}, Output: {out_h}x{out_w}")
    logger.info(f"[WAN22 Tiled VAE] Tile: {size_h}x{size_w}, Stride: {stride_h}x{stride_w}")
    logger.info(f"[WAN22 Tiled VAE] Overlap: {h_overlap}x{w_overlap} latent, {h_border_pixels}x{w_border_pixels} pixels")
    
    # Generate tile positions
    tasks = []
    for h in range(0, H, stride_h):
        # Skip if previous tile already covers this area
        if h > 0 and (h - stride_h + size_h >= H):
            continue
        for w in range(0, W, stride_w):
            # Skip if previous tile already covers this area
            if w > 0 and (w - stride_w + size_w >= W):
                continue
            
            h_end = min(h + size_h, H)
            w_end = min(w + size_w, W)
            tasks.append((h, h_end, w, w_end))
    
    logger.info(f"[WAN22 Tiled VAE] Processing {len(tasks)} tiles...")
    
    # Allocate accumulation buffers on CPU to save VRAM
    # We'll accumulate in float32 for precision
    values = torch.zeros((B, 3, T, out_h, out_w), dtype=torch.float32, device='cpu')
    weight = torch.zeros((1, 1, 1, out_h, out_w), dtype=torch.float32, device='cpu')
    
    # Process each tile
    pbar = comfy.utils.ProgressBar(len(tasks))
    
    for idx, (h_start, h_end, w_start, w_end) in enumerate(tasks):
        # Extract tile (all frames at once for temporal coherence)
        tile_latent = samples[:, :, :, h_start:h_end, w_start:w_end]
        
        # Move tile to computation device
        tile_latent = tile_latent.to(device)
        
        # Decode tile through VAE
        # ComfyUI VAE decode expects (B, C, H, W) for images or handles 5D for video
        try:
            if is_video:
                # For video VAE, we need to handle the temporal dimension
                # Most WAN VAEs expect 5D input directly
                decoded = vae.decode(tile_latent)
            else:
                # For image, squeeze and decode
                decoded = vae.decode(tile_latent.squeeze(2))
                decoded = decoded.unsqueeze(2)
        except Exception as e:
            logger.warning(f"[WAN22 Tiled VAE] Direct decode failed: {e}, trying frame-by-frame")
            # Fallback: decode frame by frame
            decoded_frames = []
            for t in range(T):
                frame_latent = tile_latent[:, :, t:t+1, :, :]
                # Squeeze temporal dim for standard decode
                frame_decoded = vae.decode(frame_latent.squeeze(2))
                if frame_decoded.dim() == 4:
                    frame_decoded = frame_decoded.unsqueeze(2)
                decoded_frames.append(frame_decoded)
            decoded = torch.cat(decoded_frames, dim=2)
        
        # Move decoded tile to CPU for accumulation
        decoded = decoded.to('cpu').float()
        
        # Handle output format - ensure (B, C, T, H, W)
        if decoded.dim() == 4:
            # (B, C, H, W) -> (B, C, 1, H, W)
            decoded = decoded.unsqueeze(2)
        
        # Get actual decoded tile dimensions
        _, _, _, tile_out_h, tile_out_w = decoded.shape
        
        # Calculate pixel-space positions
        h_start_px = h_start * upscale_factor
        w_start_px = w_start * upscale_factor
        h_end_px = h_start_px + tile_out_h
        w_end_px = w_start_px + tile_out_w
        
        # Determine boundary status
        is_top = (h_start == 0)
        is_bottom = (h_end >= H)
        is_left = (w_start == 0)
        is_right = (w_end >= W)
        
        # Build blending mask
        mask = _build_2d_mask(
            tile_out_h, tile_out_w,
            is_bound=(is_top, is_bottom, is_left, is_right),
            border_width=(h_border_pixels, w_border_pixels)
        )
        
        # Accumulate weighted values
        values[:, :, :, h_start_px:h_end_px, w_start_px:w_end_px] += decoded * mask
        weight[:, :, :, h_start_px:h_end_px, w_start_px:w_end_px] += mask
        
        # Optional cache clearing for very high resolutions
        if clear_cache_between_tiles:
            del decoded, tile_latent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        pbar.update(1)
    
    # Normalize by accumulated weights
    # Avoid division by zero
    weight = torch.clamp(weight, min=1e-8)
    result = values / weight
    
    # Clamp to valid range
    result = result.clamp(-1, 1)
    
    # If input was 4D (image), squeeze back
    if not is_video:
        result = result.squeeze(2)
    
    logger.info(f"[WAN22 Tiled VAE] Decode complete, output shape: {tuple(result.shape)}")
    
    return result


class WAN22TiledVAEDecode:
    """
    High-quality tiled VAE decode for WAN video models.
    
    This node provides spatial-only tiling for video latent decoding,
    preserving temporal coherence by processing all frames together
    per tile. Uses linear feathering with smooth blending to avoid
    tile boundary artifacts.
    
    Recommended for high-resolution video generation where standard
    VAE decode would run out of VRAM.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tile_height": ("INT", {
                    "default": 34, 
                    "min": 8, 
                    "max": 128, 
                    "step": 2,
                    "tooltip": "Tile height in latent space (pixels = tile * 8)"
                }),
                "tile_width": ("INT", {
                    "default": 34, 
                    "min": 8, 
                    "max": 128, 
                    "step": 2,
                    "tooltip": "Tile width in latent space (pixels = tile * 8)"
                }),
                "stride_height": ("INT", {
                    "default": 18, 
                    "min": 4, 
                    "max": 64, 
                    "step": 2,
                    "tooltip": "Vertical step between tiles. Lower = more overlap = smoother blending"
                }),
                "stride_width": ("INT", {
                    "default": 16, 
                    "min": 4, 
                    "max": 64, 
                    "step": 2,
                    "tooltip": "Horizontal step between tiles. Lower = more overlap = smoother blending"
                }),
            },
            "optional": {
                "clear_cache": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Clear CUDA cache between tiles. Slower but uses less peak VRAM."
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "WAN22/VAE"
    DESCRIPTION = (
        "High-quality tiled VAE decode for WAN video models. "
        "Tiles spatially only (not temporally) to preserve frame coherence. "
        "Uses smooth linear blending to avoid tile artifacts. "
        "Ideal for high-resolution videos that exceed VRAM limits."
    )
    
    def decode(
        self, 
        samples, 
        vae, 
        tile_height: int = 34, 
        tile_width: int = 34,
        stride_height: int = 18,
        stride_width: int = 16,
        clear_cache: bool = False
    ):
        """
        Decode video latents using spatial tiling with temporal coherence.
        """
        latent = samples["samples"]
        
        # Decode using our tiled approach
        decoded = tiled_decode_wan(
            vae=vae,
            samples=latent,
            tile_size=(tile_height, tile_width),
            tile_stride=(stride_height, stride_width),
            upscale_factor=8,  # WAN VAE uses 8x upscaling
            clear_cache_between_tiles=clear_cache,
        )
        
        # Convert from (B, C, T, H, W) or (B, C, H, W) to ComfyUI image format
        # ComfyUI expects (N, H, W, C) with values in [0, 1]
        
        if decoded.dim() == 5:
            # Video: (B, C, T, H, W) -> (B*T, H, W, C)
            B, C, T, H, W = decoded.shape
            # Rearrange: (B, C, T, H, W) -> (B, T, H, W, C) -> (B*T, H, W, C)
            decoded = decoded.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
            decoded = decoded.reshape(B * T, H, W, C)
        else:
            # Image: (B, C, H, W) -> (B, H, W, C)
            decoded = decoded.permute(0, 2, 3, 1)
        
        # Convert from [-1, 1] to [0, 1]
        decoded = (decoded + 1.0) / 2.0
        decoded = decoded.clamp(0, 1)
        
        return (decoded,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "WAN22TiledVAEDecode": WAN22TiledVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WAN22TiledVAEDecode": "WAN22 Tiled VAE Decode",
}
