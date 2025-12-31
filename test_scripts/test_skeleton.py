#!/usr/bin/env python
"""Test WAN 2.2 I2V model skeleton creation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import torch

def main():
    print("Testing WAN 2.2 I2V model skeleton...")
    
    # Simulate WAN 2.2 I2V config
    config = {
        'dim': 5120,
        'num_layers': 40,
        'num_heads': 40,
        'model_variant': 'i2v',
        'wan_version': '2.2',  # WAN 2.2
        'ffn_dim': 13824,
        'in_dim': 36,  # Larger in_dim for I2V
        'out_dim': 16,
        'text_len': 512,
        'freq_dim': 256,
        'text_dim': 4096,
        'patch_size': (1, 2, 2),
        'window_size': (-1, -1),
        'qk_norm': True,
        'cross_attn_norm': True,
        'eps': 1e-6,
    }
    
    # Import and create skeleton
    import comfy.ops
    from comfy.ldm.wan.model import WanModel, WanAttentionBlock
    
    # For WAN 2.2 I2V, we should use model_type='t2v'
    model_type = "t2v"  # WAN 2.2 I2V uses t2v architecture
    print(f"  model_type: {model_type}")
    print(f"  in_dim: {config['in_dim']}")
    
    # Create with meta device
    with torch.device('meta'):
        model = WanModel(
            model_type=model_type,
            dim=config['dim'],
            num_layers=2,  # Small for testing
            num_heads=config['num_heads'],
            in_dim=config['in_dim'],
            out_dim=config['out_dim'],
            ffn_dim=config['ffn_dim'],
            operations=comfy.ops.manual_cast,
            device='meta',
            dtype=torch.float16,
        )
    
    # Check model structure
    has_img_emb = hasattr(model, 'img_emb') and model.img_emb is not None
    print(f"  Has img_emb: {has_img_emb}")
    
    # Count params
    param_count = sum(1 for _ in model.parameters())
    print(f"  Parameter count: {param_count}")
    
    # List all state dict keys
    keys = list(model.state_dict().keys())
    print(f"  State dict keys: {len(keys)}")
    
    # Check for img_emb keys
    img_emb_keys = [k for k in keys if 'img_emb' in k]
    print(f"  img_emb keys: {len(img_emb_keys)}")
    if img_emb_keys:
        print(f"  WARNING: img_emb keys found but shouldn't be for WAN 2.2 I2V!")
        for k in img_emb_keys:
            print(f"    - {k}")
    
    # Check for k_img keys in cross_attn
    k_img_keys = [k for k in keys if 'k_img' in k or 'v_img' in k]
    print(f"  k_img/v_img keys: {len(k_img_keys)}")
    if k_img_keys:
        print(f"  WARNING: k_img/v_img keys found but shouldn't be for WAN 2.2 I2V!")
    
    if not img_emb_keys and not k_img_keys:
        print("\n✓ Model skeleton correct for WAN 2.2 I2V!")
    else:
        print("\n✗ Model skeleton has extra keys that won't be in state dict")

if __name__ == "__main__":
    main()
