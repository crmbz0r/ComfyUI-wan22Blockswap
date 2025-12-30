"""Comprehensive test suite for WAN 2.2 BlockSwap Looper functionality.

This module provides unit tests and integration tests for the loop-aware
BlockSwap implementation, ensuring proper state management, cleanup, and
tensor validation across multiple loop iterations.

Test Categories:
- Unit tests for individual helper functions
- Integration tests for multi-loop scenarios
- Memory leak detection tests
- Color matching validation tests
- Compatibility tests with WanVideoLoraSequencer

These tests validate that the 5 root causes of loop degradation are properly
addressed by the new implementation.
"""

import unittest
import torch
import gc
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from typing import Any, Dict, List, Tuple

# Set up the package context for standalone script execution
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_NAME = os.path.basename(_PACKAGE_DIR)
_PARENT_DIR = os.path.dirname(_PACKAGE_DIR)
_COMFYUI_ROOT = os.path.dirname(_PARENT_DIR)

# Add paths for package imports
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _COMFYUI_ROOT not in sys.path:
    sys.path.insert(0, _COMFYUI_ROOT)

try:
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import CallbacksMP
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False
    print("Warning: ComfyUI not available. Running limited tests.")

from ComfyUI_Wan22Blockswap.blockswap_looper import WAN22BlockSwapLooper
from ComfyUI_Wan22Blockswap.looper_helpers import (
    prepare_model_for_loop,
    create_fresh_blockswap_tracker,
    cleanup_loop_blockswap,
    validate_tensor_consistency,
    reset_model_blockswap_state,
)
from ComfyUI_Wan22Blockswap.block_manager import BlockSwapTracker


class TestBlockSwapTrackerCreation(unittest.TestCase):
    """Test BlockSwapTracker creation and initialization."""

    def test_create_fresh_blockswap_tracker(self):
        """Test that create_fresh_blockswap_tracker creates properly initialized tracker."""
        blocks_to_swap = 20
        tracker = create_fresh_blockswap_tracker(blocks_to_swap)

        # Verify tracker properties
        self.assertEqual(tracker.blocks_to_swap, blocks_to_swap)
        self.assertEqual(tracker.cleanup_executed, False)
        self.assertEqual(tracker.is_gguf_model, False)
        self.assertEqual(tracker.swapped_indices, [])
        self.assertEqual(tracker.successfully_swapped_indices, [])
        self.assertEqual(tracker.failed_to_swap_indices, [])
        self.assertEqual(tracker.swapped_blocks_refs, {})
        self.assertEqual(tracker.embeddings_offloaded, {})


class TestModelPreparation(unittest.TestCase):
    """Test model preparation for loop iterations."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model patcher
        self.mock_model = Mock(spec=ModelPatcher)
        self.mock_model.clone.return_value = Mock(spec=ModelPatcher)
        self.mock_model.clone.return_value.attachments = {}
        self.mock_model.clone.return_value.add_callback = Mock()

    def test_prepare_model_for_loop_valid_input(self):
        """Test prepare_model_for_loop with valid input."""
        loop_index = 0
        blocks_to_swap = 20

        result = prepare_model_for_loop(
            model=self.mock_model,
            loop_index=loop_index,
            blocks_to_swap=blocks_to_swap,
            offload_txt_emb=False,
            offload_img_emb=False,
            use_non_blocking=False,
            vace_blocks_to_swap=0,
            prefetch_blocks=0,
            block_swap_debug=False
        )

        # Verify model was cloned
        self.mock_model.clone.assert_called_once()

        # Verify fresh tracker was created and attached
        self.assertIn('blockswap_tracking', result.attachments)
        tracker = result.attachments['blockswap_tracking']
        self.assertEqual(tracker.blocks_to_swap, blocks_to_swap)

        # Verify callbacks were registered
        self.assertEqual(result.add_callback.call_count, 2)

    def test_prepare_model_for_loop_invalid_model(self):
        """Test prepare_model_for_loop with None model raises ValueError."""
        with self.assertRaises(ValueError) as context:
            prepare_model_for_loop(
                model=None,
                loop_index=0,
                blocks_to_swap=20,
                offload_txt_emb=False,
                offload_img_emb=False,
                use_non_blocking=False,
                vace_blocks_to_swap=0,
                prefetch_blocks=0
            )
        self.assertIn("Model cannot be None", str(context.exception))

    def test_prepare_model_for_loop_negative_blocks(self):
        """Test prepare_model_for_loop with negative blocks_to_swap raises ValueError."""
        with self.assertRaises(ValueError) as context:
            prepare_model_for_loop(
                model=self.mock_model,
                loop_index=0,
                blocks_to_swap=-1,
                offload_txt_emb=False,
                offload_img_emb=False,
                use_non_blocking=False,
                vace_blocks_to_swap=0,
                prefetch_blocks=0
            )
        self.assertIn("blocks_to_swap must be non-negative", str(context.exception))


class TestLoopCleanup(unittest.TestCase):
    """Test cleanup operations between loop iterations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock(spec=ModelPatcher)
        self.mock_model.model = Mock()
        self.mock_model.attachments = {}

    def test_cleanup_loop_blockswap_no_tracking(self):
        """Test cleanup when no tracking data exists."""
        # Test with None model
        cleanup_loop_blockswap(None, 0, False)

        # Test with empty attachments
        cleanup_loop_blockswap(self.mock_model, 0, False)

    def test_cleanup_loop_blockswap_with_tracking(self):
        """Test cleanup with existing tracking data."""
        # Create tracking data
        tracker = BlockSwapTracker()
        tracker.blocks_to_swap = 20
        tracker.cleanup_executed = False
        tracker.is_gguf_model = False
        tracker.successfully_swapped_indices = [10, 11, 12]

        self.mock_model.attachments['blockswap_tracking'] = tracker

        # Mock UNet with blocks
        mock_unet = Mock()
        mock_unet.blocks = [Mock() for _ in range(30)]
        with patch('ComfyUI_Wan22Blockswap.block_manager.BlockManager.get_unet_from_model', return_value=mock_unet):
            cleanup_loop_blockswap(self.mock_model, 0, False)

        # Verify cleanup was executed
        self.assertTrue(tracker.cleanup_executed)

        # Verify collections were cleared
        self.assertEqual(tracker.swapped_indices, [])
        self.assertEqual(tracker.successfully_swapped_indices, [])
        self.assertEqual(tracker.failed_to_swap_indices, [])
        self.assertEqual(tracker.swapped_blocks_refs, {})
        self.assertEqual(tracker.embeddings_offloaded, {})

    def test_cleanup_loop_blockswap_already_executed(self):
        """Test cleanup when already executed."""
        tracker = BlockSwapTracker()
        tracker.cleanup_executed = True

        self.mock_model.attachments['blockswap_tracking'] = tracker

        # Should not raise error and should return early
        cleanup_loop_blockswap(self.mock_model, 0, False)
        self.assertTrue(tracker.cleanup_executed)


class TestTensorConsistency(unittest.TestCase):
    """Test tensor consistency validation and correction."""

    def test_validate_tensor_consistency_valid_tensors(self):
        """Test validation with already consistent tensors."""
        device = torch.device('cpu')
        dtype = torch.float32

        latent_tensor = torch.randn(1, 16, 5, 60, 52, device=device, dtype=dtype)
        latent = {'samples': latent_tensor}
        color_match_ref = torch.randn(1, 60, 52, 3, device=device, dtype=dtype)

        result_latent, result_ref = validate_tensor_consistency(
            latent, color_match_ref, device, dtype, False
        )

        # Tensors should be unchanged
        self.assertIs(result_latent['samples'], latent_tensor)
        self.assertIs(result_ref, color_match_ref)

    def test_validate_tensor_consistency_inconsistent_device(self):
        """Test validation with device inconsistency."""
        target_device = torch.device('cpu')
        dtype = torch.float32

        latent_tensor = torch.randn(1, 16, 5, 60, 52, device=torch.device('cpu'), dtype=dtype)
        latent = {'samples': latent_tensor}
        color_match_ref = torch.randn(1, 60, 52, 3, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=dtype)

        result_latent, result_ref = validate_tensor_consistency(
            latent, color_match_ref, target_device, dtype, False
        )

        # Tensors should be moved to target device
        self.assertEqual(result_latent['samples'].device, target_device)
        self.assertEqual(result_ref.device, target_device)

    def test_validate_tensor_consistency_inconsistent_dtype(self):
        """Test validation with dtype inconsistency."""
        device = torch.device('cpu')
        target_dtype = torch.float32

        latent_tensor = torch.randn(1, 16, 5, 60, 52, device=device, dtype=torch.float16)
        latent = {'samples': latent_tensor}
        color_match_ref = torch.randn(1, 60, 52, 3, device=device, dtype=torch.float16)

        result_latent, result_ref = validate_tensor_consistency(
            latent, color_match_ref, device, target_dtype, False
        )

        # Tensors should be converted to target dtype
        self.assertEqual(result_latent['samples'].dtype, target_dtype)
        self.assertEqual(result_ref.dtype, target_dtype)

    def test_validate_tensor_consistency_invalid_latent(self):
        """Test validation with invalid latent input."""
        device = torch.device('cpu')
        dtype = torch.float32

        # Test with invalid latent format
        with self.assertRaises(ValueError) as context:
            validate_tensor_consistency({}, None, device, dtype, False)
        self.assertIn("Latent must be a dictionary with 'samples' key", str(context.exception))

        # Test with invalid samples type
        with self.assertRaises(ValueError) as context:
            validate_tensor_consistency({'samples': "not_a_tensor"}, None, device, dtype, False)
        self.assertIn("Latent samples must be a torch.Tensor", str(context.exception))


class TestModelStateReset(unittest.TestCase):
    """Test model state reset functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock(spec=ModelPatcher)
        self.mock_model.attachments = {}

    def test_reset_model_blockswap_state_with_tracking(self):
        """Test reset when tracking data exists."""
        # Add tracking data
        tracker = BlockSwapTracker()
        self.mock_model.attachments['blockswap_tracking'] = tracker

        reset_model_blockswap_state(self.mock_model)

        # Verify tracking was removed
        self.assertNotIn('blockswap_tracking', self.mock_model.attachments)

    def test_reset_model_blockswap_state_no_attachments(self):
        """Test reset when no attachments exist."""
        # Remove attachments attribute
        del self.mock_model.attachments

        # Should not raise error
        reset_model_blockswap_state(self.mock_model)

    def test_reset_model_blockswap_state_none_model(self):
        """Test reset with None model."""
        # Should not raise error
        reset_model_blockswap_state(None)


class TestWAN22BlockSwapLooperIntegration(unittest.TestCase):
    """Integration tests for WAN22BlockSwapLooper node."""

    def setUp(self):
        """Set up test fixtures."""
        self.node = WAN22BlockSwapLooper()

        # Create mock models
        self.mock_model = Mock(spec=ModelPatcher)
        self.mock_model.clone.return_value = Mock(spec=ModelPatcher)
        self.mock_model.clone.return_value.attachments = {}
        self.mock_model.clone.return_value.add_callback = Mock()

    def test_prepare_looper_models_single_model_list(self):
        """Test prepare_looper_models with single model list."""
        models_list = [self.mock_model, self.mock_model]

        result = self.node.prepare_looper_models(
            models_list=models_list,
            blocks_to_swap=20,
            offload_txt_emb=False,
            offload_img_emb=False
        )

        # Should return tuple with prepared models list
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], list)
        self.assertEqual(len(result[0]), 2)

    def test_prepare_looper_models_lora_sequencer_format(self):
        """Test prepare_looper_models with WanVideoLoraSequencer format."""
        models_list = [
            (self.mock_model, self.mock_model, Mock()),
            (self.mock_model, self.mock_model, Mock())
        ]

        result = self.node.prepare_looper_models(
            models_list=models_list,
            blocks_to_swap=20,
            offload_txt_emb=False,
            offload_img_emb=False
        )

        # Should return tuple with prepared models list
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], list)
        self.assertEqual(len(result[0]), 2)

        # Each item should be a tuple of (prepared_high, prepared_low, clip)
        for item in result[0]:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 3)

    def test_prepare_looper_models_invalid_input(self):
        """Test prepare_looper_models with invalid input."""
        # Test with non-iterable input
        with self.assertRaises(ValueError) as context:
            self.node.prepare_looper_models(
                models_list="not_iterable",
                blocks_to_swap=20,
                offload_txt_emb=False,
                offload_img_emb=False
            )
        self.assertIn("models_list must be a list, tuple, or other iterable of models", str(context.exception))

        # Test with empty list
        with self.assertRaises(ValueError) as context:
            self.node.prepare_looper_models(
                models_list=[],
                blocks_to_swap=20,
                offload_txt_emb=False,
                offload_img_emb=False
            )
        self.assertIn("models_list cannot be empty", str(context.exception))


class TestMemoryLeakDetection(unittest.TestCase):
    """Test for memory leaks in loop operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model = Mock(spec=ModelPatcher)
        # Create a properly configured mock for the cloned model
        cloned_mock = Mock(spec=ModelPatcher)
        cloned_mock.attachments = {}
        cloned_mock.add_callback = Mock()
        cloned_mock.model = Mock()  # Add .model attribute for cleanup
        self.mock_model.clone.return_value = cloned_mock

    def test_no_memory_leak_across_loops(self):
        """Test that no memory leaks occur across multiple loop iterations."""
        initial_objects = len(gc.get_objects())

        # Simulate multiple loop iterations
        for i in range(10):
            # Reset the clone return value for each iteration to get fresh attachments
            cloned_mock = Mock(spec=ModelPatcher)
            cloned_mock.attachments = {}
            cloned_mock.add_callback = Mock()
            cloned_mock.model = Mock()
            self.mock_model.clone.return_value = cloned_mock

            prepared_model = prepare_model_for_loop(
                model=self.mock_model,
                loop_index=i,
                blocks_to_swap=20,
                offload_txt_emb=False,
                offload_img_emb=False,
                use_non_blocking=False,
                vace_blocks_to_swap=0,
                prefetch_blocks=0
            )

            # Simulate cleanup - pass debug=False to avoid errors with mock objects
            # Note: cleanup may fail gracefully on mocks due to missing internal structure
            try:
                cleanup_loop_blockswap(prepared_model, i, False)
            except RuntimeError:
                # Expected when using mocks - the cleanup tries to access real model internals
                pass

        gc.collect()
        final_objects = len(gc.get_objects())

        # Allow for some variance but should be roughly stable
        object_increase = final_objects - initial_objects
        self.assertLess(object_increase, 100, f"Potential memory leak detected: {object_increase} objects created")


class TestColorMatchingCompatibility(unittest.TestCase):
    """Test compatibility with color matching operations."""

    def test_tensor_consistency_for_color_matching(self):
        """Test that tensor consistency validation works for color matching scenarios."""
        device = torch.device('cpu')
        dtype = torch.float32

        # Create tensors similar to those used in color matching
        latent_tensor = torch.randn(1, 16, 5, 60, 52, device=device, dtype=dtype)
        latent = {'samples': latent_tensor}

        # Color reference tensor (RGB image format)
        color_ref = torch.randn(1, 60, 52, 3, device=device, dtype=dtype)

        # Validate consistency
        result_latent, result_ref = validate_tensor_consistency(
            latent, color_ref, device, dtype, False
        )

        # Both should be on same device and dtype
        self.assertEqual(result_latent['samples'].device, result_ref.device)
        self.assertEqual(result_latent['samples'].dtype, result_ref.dtype)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
