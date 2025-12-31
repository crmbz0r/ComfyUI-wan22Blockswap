#!/usr/bin/env python
"""
Test script to verify GGUF model detection works correctly.
Tests both I2V and T2V variant detection based on in_dim.
"""
import sys
import os
import glob

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def main():
    """Test GGUF model detection."""
    print("Testing GGUF Model Detection...")
    print("=" * 60)
    
    # Import from gguf custom node
    try:
        from custom_nodes.gguf.pig import load_gguf_sd
        print("✓ Imported load_gguf_sd from gguf custom node")
    except ImportError as e:
        print(f"✗ Failed to import from gguf custom node: {e}")
        return 1
    
    # Import our model detection
    try:
        from custom_nodes.ComfyUI_Wan22Blockswap.model_detection import detect_wan_config
        print("✓ Imported detect_wan_config from model_detection")
    except ImportError as e:
        print(f"✗ Failed to import model_detection: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Find GGUF files
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
    
    # Test each file
    success_count = 0
    for gguf_file in gguf_files[:3]:  # Test up to 3 files
        print(f"\n{'─' * 60}")
        filename = os.path.basename(gguf_file)
        print(f"Testing: {filename}")
        
        # Load the state dict
        try:
            sd = load_gguf_sd(gguf_file)
            print(f"  ✓ Loaded {len(sd)} tensors")
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue
        
        # Test model detection
        try:
            config = detect_wan_config(sd)
            print(f"  ✓ Detection succeeded!")
            print(f"    - Version: {config.get('wan_version', '?')}")
            print(f"    - Variant: {config.get('model_variant', '?')}")
            print(f"    - in_dim: {config.get('in_dim', '?')}")
            print(f"    - Layers: {config.get('num_layers', '?')}")
            print(f"    - Dim: {config.get('dim', '?')}")
            
            # Show detection method details
            has_k_img = any('k_img' in k for k in sd.keys())
            has_img_emb = any('img_emb' in k for k in sd.keys())
            bias_key = 'patch_embedding.bias'
            bias_dtype = "unknown"
            if bias_key in sd:
                t = sd[bias_key]
                if hasattr(t, 'dtype'):
                    bias_dtype = str(t.dtype)
                elif hasattr(t, 'tensor_type'):
                    bias_dtype = str(t.tensor_type)
            print(f"    - has_k_img: {has_k_img}")
            print(f"    - has_img_emb: {has_img_emb}")
            print(f"    - bias_dtype: {bias_dtype}")
            
            # Validate detection based on in_dim (the ground truth)
            in_dim = config.get('in_dim', 16)
            variant = config.get('model_variant', 't2v')
            
            if (in_dim > 16 and variant == 'i2v') or (in_dim == 16 and variant == 't2v'):
                print(f"  ✓ Variant detection correct based on in_dim!")
                success_count += 1
            else:
                print(f"  ⚠ Variant/in_dim mismatch! in_dim={in_dim}, variant={variant}")
                
        except Exception as e:
            print(f"  ✗ Detection failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'=' * 60}")
    print(f"Tests completed: {success_count}/{min(len(gguf_files), 3)} passed")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
