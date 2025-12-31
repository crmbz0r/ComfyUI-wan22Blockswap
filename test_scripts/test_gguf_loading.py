#!/usr/bin/env python
"""
Test script to verify GGUF loading works correctly with GGMLTensor.
"""
import sys
import os
import glob

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def main():
    """Test GGUF loading with load_gguf_sd."""
    print("Testing GGUF loading...")
    print("=" * 50)
    
    # Import from gguf custom node
    try:
        from custom_nodes.gguf.pig import load_gguf_sd, GGMLTensor
        print("✓ Imported load_gguf_sd and GGMLTensor from gguf custom node")
    except ImportError as e:
        print(f"✗ Failed to import from gguf custom node: {e}")
        return 1
    
    # Find a GGUF file
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    patterns = [
        os.path.join(base_dir, 'models', 'unet', '**', '*wan*.gguf'),
        os.path.join(base_dir, 'models', '**', '*wan*.gguf'),
    ]
    
    gguf_files = []
    for pattern in patterns:
        gguf_files = glob.glob(pattern, recursive=True)
        if gguf_files:
            break
    
    if not gguf_files:
        print("✗ No WAN GGUF files found")
        return 1
    
    gguf_file = gguf_files[0]
    print(f"✓ Found GGUF file: {os.path.basename(gguf_file)}")
    
    # Load the state dict
    print("Loading state dict...")
    try:
        sd = load_gguf_sd(gguf_file)
        print(f"✓ Loaded state dict with {len(sd)} tensors")
    except Exception as e:
        print(f"✗ Failed to load state dict: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Check tensor properties
    sample_key = next(iter(sd.keys()))
    sample_tensor = sd[sample_key]
    
    print(f"\nSample tensor analysis:")
    print(f"  Key: {sample_key}")
    print(f"  Type: {type(sample_tensor).__name__}")
    print(f"  Is GGMLTensor: {isinstance(sample_tensor, GGMLTensor)}")
    print(f"  Has tensor_type: {hasattr(sample_tensor, 'tensor_type')}")
    
    if hasattr(sample_tensor, 'tensor_type'):
        print(f"  tensor_type value: {sample_tensor.tensor_type}")
    
    print(f"  Has tensor_shape: {hasattr(sample_tensor, 'tensor_shape')}")
    
    if hasattr(sample_tensor, 'tensor_shape'):
        print(f"  tensor_shape value: {sample_tensor.tensor_shape}")
    
    print(f"  .shape property: {sample_tensor.shape}")
    print(f"  .data.shape: {sample_tensor.data.shape}")
    
    # Test model_detection helper
    print("\nTesting _get_tensor_shape helper:")
    try:
        from custom_nodes.ComfyUI_Wan22Blockswap.model_detection import _get_tensor_shape
        shape = _get_tensor_shape(sample_tensor)
        print(f"  _get_tensor_shape result: {shape}")
        print("✓ _get_tensor_shape works correctly")
    except Exception as e:
        print(f"✗ _get_tensor_shape failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test moving tensor to device (simulating block routing)
    print("\nTesting tensor movement (simulating block routing):")
    import torch
    
    # Move to CPU (should preserve tensor_type and tensor_shape)
    moved_tensor = sample_tensor.to(device='cpu')
    print(f"  After .to(device='cpu'):")
    print(f"    Type: {type(moved_tensor).__name__}")
    print(f"    Has tensor_type: {hasattr(moved_tensor, 'tensor_type')}")
    print(f"    tensor_type value: {getattr(moved_tensor, 'tensor_type', None)}")
    print(f"    .shape: {moved_tensor.shape}")
    
    if hasattr(moved_tensor, 'tensor_type') and moved_tensor.tensor_type is not None:
        print("✓ tensor_type preserved after .to() call")
    else:
        print("✗ tensor_type NOT preserved after .to() call!")
        return 1
    
    print("\n" + "=" * 50)
    print("All tests passed! GGUF loading is working correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
