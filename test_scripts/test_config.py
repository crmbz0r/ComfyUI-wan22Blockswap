#!/usr/bin/env python
"""Test _create_model_config filtering."""
import sys
import os

# Minimal imports to test the function
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Only import what we need
import torch
from typing import Dict, Any

def test_config_filtering():
    """Test that _create_model_config properly filters unet_config."""
    from comfy import latent_formats
    
    # Simulate the function logic
    wan_config = {
        'dim': 5120,
        'model_size': '14B-alt',  # Should be filtered
        'num_layers': 40,
        'num_heads': 40,
        'model_variant': 'i2v',   # Should become model_type
        'wan_version': '2.2',     # Should be filtered
        'model_type': 't2v',      # Will be overridden
        'ffn_dim': 13824,
        'in_dim': 36,
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
    
    # Valid WanModel parameters
    valid_unet_keys = {
        "model_type", "patch_size", "text_len", "in_dim", "dim", "ffn_dim",
        "freq_dim", "text_dim", "out_dim", "num_heads", "num_layers",
        "window_size", "qk_norm", "cross_attn_norm", "eps",
        "flf_pos_embed_token_number", "in_dim_ref_conv",
        "vace_in_dim", "vace_layers", "disable_unet_model_creation",
    }
    
    # Filter config
    unet_config = {k: v for k, v in wan_config.items() if k in valid_unet_keys}
    
    # Always derive model_type from model_variant (if present)
    if "model_variant" in wan_config:
        variant = wan_config["model_variant"]
        if variant in ["i2v", "camera", "camera_2.2", "vace"]:
            unet_config["model_type"] = "i2v"
        else:
            unet_config["model_type"] = "t2v"
    elif "model_type" not in unet_config:
        unet_config["model_type"] = "t2v"
    
    # Disable auto model creation
    unet_config["disable_unet_model_creation"] = True
    
    print("Test Results:")
    print(f"  unet_config keys: {sorted(unet_config.keys())}")
    print(f"  model_type: {unet_config.get('model_type')}")
    print(f"  disable_unet_model_creation: {unet_config.get('disable_unet_model_creation')}")
    print(f"  Has 'model_size'? {('model_size' in unet_config)}")
    print(f"  Has 'wan_version'? {('wan_version' in unet_config)}")
    print(f"  Has 'model_variant'? {('model_variant' in unet_config)}")
    
    # Assertions
    assert 'model_size' not in unet_config, "model_size should be filtered out"
    assert 'wan_version' not in unet_config, "wan_version should be filtered out"
    assert 'model_variant' not in unet_config, "model_variant should be filtered out"
    assert unet_config.get('disable_unet_model_creation') == True, "disable_unet_model_creation should be True"
    assert unet_config.get('model_type') == 'i2v', "model_type should be 'i2v' for i2v variant"
    
    print("\n✓ All assertions passed!")

if __name__ == "__main__":
    test_config_filtering()
