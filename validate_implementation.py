"""Simple validation script for WAN22BlockSwap Looper implementation.

This script validates the basic structure and syntax of the implementation
without requiring the full ComfyUI environment. It can be run standalone
to verify the implementation is syntactically correct.
"""

import os
import sys
import ast
import traceback


def validate_file_syntax(file_path):
    """Validate that a Python file has correct syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the AST to check syntax
        ast.parse(content)
        return True, "Syntax OK"

    except SyntaxError as e:
        return False, f"Syntax Error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def validate_file_structure(file_path, required_elements=None):
    """Validate that a file contains required elements."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        results = []

        if required_elements:
            for element in required_elements:
                if element in content:
                    results.append(f"✓ Found: {element}")
                else:
                    results.append(f"✗ Missing: {element}")

        return True, results

    except Exception as e:
        return False, [f"Error reading file: {e}"]


def main():
    """Main validation function."""
    print("WAN22BlockSwap Looper Implementation Validation")
    print("=" * 50)

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Files to validate
    files_to_check = [
        {
            'path': 'blockswap_looper.py',
            'required': [
                'class WAN22BlockSwapLooper',
                'def prepare_looper_models',
                'NODE_CLASS_MAPPINGS',
                'wan22BlockSwapLooper'
            ]
        },
        {
            'path': 'looper_helpers.py',
            'required': [
                'def prepare_model_for_loop',
                'def create_fresh_blockswap_tracker',
                'def cleanup_loop_blockswap',
                'def validate_tensor_consistency',
                'def reset_model_blockswap_state'
            ]
        },
        {
            'path': '__init__.py',
            'required': [
                'WAN22BlockSwapLooper',
                'WAN22BlockSwapLooper'
            ]
        },
        {
            'path': 'test_blockswap_looper.py',
            'required': [
                'class TestBlockSwapTrackerCreation',
                'class TestModelPreparation',
                'class TestLoopCleanup',
                'class TestTensorConsistency',
                'class TestModelStateReset',
                'class TestWAN22BlockSwapLooperIntegration',
                'class TestMemoryLeakDetection',
                'class TestColorMatchingCompatibility'
            ]
        },
        {
            'path': 'test_looper_integration.py',
            'required': [
                'def test_imports',
                'def test_blockswap_tracker_creation',
                'def test_tensor_consistency_validation',
                'def test_memory_stability',
                'def test_node_registration',
                'def test_init_exports',
                'def test_compatibility_with_existing_nodes'
            ]
        },
        {
            'path': 'LOOPER_IMPLEMENTATION_GUIDE.md',
            'required': [
                '# WAN22BlockSwap Looper Implementation Guide',
                '## Problem Statement',
                '## Root Cause Analysis',
                '## Solution Architecture',
                '## Usage Guide'
            ]
        }
    ]

    all_valid = True

    for file_info in files_to_check:
        file_path = os.path.join(script_dir, file_info['path'])
        print(f"\nValidating: {file_info['path']}")

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            all_valid = False
            continue

        # Validate syntax (skip for markdown files)
        if file_path.endswith('.py'):
            syntax_ok, syntax_msg = validate_file_syntax(file_path)
            if syntax_ok:
                print(f"✓ Syntax: {syntax_msg}")
            else:
                print(f"✗ Syntax: {syntax_msg}")
                all_valid = False

        # Validate structure
        struct_ok, struct_results = validate_file_structure(file_path, file_info['required'])
        if struct_ok:
            for result in struct_results:
                print(f"  {result}")
                if result.startswith('✗'):
                    all_valid = False
        else:
            print(f"✗ Structure validation failed: {struct_results}")
            all_valid = False

    print("\n" + "=" * 50)
    if all_valid:
        print("🎉 All validations passed! Implementation looks good.")
        print("\nNext steps:")
        print("1. Restart ComfyUI to load the new node")
        print("2. Check that 'WAN 2.2 BlockSwap Looper (Loop-Aware)' appears in the node list")
        print("3. Test with your WanVideoLooper workflow")
        print("4. Enable debug mode to verify cleanup operations")
        return True
    else:
        print("❌ Some validations failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
