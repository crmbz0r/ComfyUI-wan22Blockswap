"""Configuration module for WAN 2.2 BlockSwap node parameters and validation."""

from typing import Dict, Any


class BlockSwapConfig:
    """Configuration class for WAN 2.2 BlockSwap parameters.

    This class handles all the input parameter definitions, validation,
    and model variant detection for the block swapping functionality.
    """

    @staticmethod
    def get_input_types() -> Dict[str, Any]:
        """
        Define input types for the WanVideo22BlockSwap node.

        Returns:
            Dict containing required and optional parameters with their types,
            defaults, and validation rules.
        """
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "ComfyUI native WAN model "
                        "(WAN 2.1/2.2, possibly 2.6 when released, etc.) - Supports both native and GGUF models"
                    },
                ),
                "blocks_to_swap": (
                    "INT",
                    {
                        "default": 20,
                        "min": 0,
                        "max": 48,
                        "step": 1,
                        "tooltip": "Number of transformer blocks to swap to CPU. "
                        "1.3B/5B models: 30 blocks, "
                        "14B model: 40 blocks, "
                        "LongCat: 48 blocks",
                    },
                ),
                "offload_txt_emb": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Offload text_embedding to CPU. "
                        "Reduces VRAM by ~500MB but may impact performance",
                    },
                ),
                "offload_img_emb": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Offload img_emb to CPU (I2V models only). "
                        "Reduces VRAM by ~200MB but may impact performance",
                    },
                ),
            },
            "optional": {
                "use_non_blocking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Use non-blocking memory transfers. "
                        "Faster but reserves more RAM (~5-10% additional)",
                    },
                ),
                "vace_blocks_to_swap": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 15,
                        "step": 1,
                        "tooltip": "VACE model blocks to swap (0 = auto, "
                        "1-15 = specific count). "
                        "VACE model has 15 blocks total",
                    },
                ),
                "prefetch_blocks": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 40,
                        "step": 1,
                        "tooltip": "Prefetch N blocks ahead for performance. "
                        "Value of 1 usually sufficient to offset "
                        "speed loss from swapping. "
                        "Use debug mode to find optimal value",
                    },
                ),
                "block_swap_debug": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable debug logging to monitor block swap "
                        "performance and memory usage",
                    },
                ),
            },
        }

    @staticmethod
    def validate_model_type(model_type_str: str) -> bool:
        """
        Validate if the model is a supported WAN model type.

        Args:
            model_type_str: String representation of model type

        Returns:
            bool: True if model is supported, False otherwise
        """
        return 'wan' in model_type_str.lower() or 'gguf' in model_type_str.lower()

    @staticmethod
    def get_model_variant_info(blocks_to_swap: int) -> Dict[str, Any]:
        """
        Get model variant information based on blocks to swap.

        Args:
            blocks_to_swap: Number of blocks to swap

        Returns:
            Dict containing model variant information
        """
        if blocks_to_swap <= 30:
            return {
                "variant": "1.3B/5B",
                "max_blocks": 30,
                "description": "Standard WAN model variant"
            }
        elif blocks_to_swap <= 40:
            return {
                "variant": "14B",
                "max_blocks": 40,
                "description": "Large WAN model variant"
            }
        else:
            return {
                "variant": "LongCat",
                "max_blocks": 48,
                "description": "LongCat WAN model variant"
            }

    @staticmethod
    def validate_parameters(
        blocks_to_swap: int,
        vace_blocks_to_swap: int,
        prefetch_blocks: int
    ) -> Dict[str, Any]:
        """
        Validate parameter combinations and return warnings.

        Args:
            blocks_to_swap: Number of transformer blocks to swap
            vace_blocks_to_swap: Number of VACE blocks to swap
            prefetch_blocks: Number of blocks to prefetch

        Returns:
            Dict containing validation results and warnings
        """
        warnings = []

        # Validate blocks_to_swap
        if blocks_to_swap > 48:
            warnings.append("blocks_to_swap exceeds maximum recommended value of 48")

        # Validate VACE blocks
        if vace_blocks_to_swap > 15:
            warnings.append("vace_blocks_to_swap exceeds maximum of 15")

        # Validate prefetch blocks
        if prefetch_blocks > 40:
            warnings.append("prefetch_blocks exceeds maximum recommended value of 40")

        # Check for potential performance issues
        if prefetch_blocks > blocks_to_swap:
            warnings.append("prefetch_blocks should typically be less than blocks_to_swap")

        return {
            "is_valid": len(warnings) == 0,
            "warnings": warnings,
            "model_info": BlockSwapConfig.get_model_variant_info(blocks_to_swap)
        }
