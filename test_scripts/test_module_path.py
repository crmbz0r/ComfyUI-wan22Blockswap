#!/usr/bin/env python
"""Test _module_path_exists function."""
import sys
import os

# Add ComfyUI to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from custom_nodes.ComfyUI_Wan22Blockswap.loader_helpers import (
    create_model_skeleton,
    _module_path_exists,
)

def test_module_path_exists():
    """Test that _module_path_exists correctly identifies module paths."""
    wan_config = {
        'wan_version': '2.2',
        'model_variant': 'i2v',
        'in_dim': 36,
        'dim': 5120,
        'ffn_dim': 5120 * 4,
        'num_heads': 40,
        'num_layers': 40,
        'text_dim': 4096,
        'freq_dim': 256,
        'text_len': 512,
        'out_dim': 16,
        'patch_size': (1, 2, 2),
    }

    print("Creating model skeleton with GGMLOps...")
    model = create_model_skeleton(wan_config, dtype=torch.float16, device='meta', is_gguf=True)
    
    # Test some paths
    test_cases = [
        ('blocks.0.self_attn.q.weight', True, "Linear weight (GGMLOps)"),
        ('blocks.0.self_attn.q.bias', True, "Linear bias (GGMLOps)"),
        ('blocks.0.modulation', True, "Modulation parameter"),
        ('blocks.0.norm1.weight', True, "LayerNorm weight"),
        ('time_embedding.0.weight', True, "Time embedding Linear"),
        ('head.head.weight', True, "Head Linear weight"),
        ('fake.path.to.nothing', False, "Non-existent path"),
        ('blocks.100.self_attn.q.weight', False, "Block index out of range"),
    ]
    
    print("\nTesting _module_path_exists:")
    all_passed = True
    for key, expected, description in test_cases:
        exists = _module_path_exists(model, key)
        status = "✓" if exists == expected else "✗"
        if exists != expected:
            all_passed = False
        print(f"  {status} {key}: {exists} (expected {expected}) - {description}")
    
    # Count total module attributes that could have weights
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            count += 1
            if count <= 10:
                has_weight = module.weight is not None
                print(f"  Module {name}.weight exists: {has_weight}")
    
    print(f"\nTotal modules with 'weight' attribute: {count}")
    print(f"\nAll tests passed: {all_passed}")
    return all_passed


if __name__ == "__main__":
    test_module_path_exists()
