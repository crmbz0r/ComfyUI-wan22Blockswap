# ComfyUI_Wan22Blockswap

### **VRAM Optimization for WAN 2.1/2.2 with BlockSwap Forward Patching**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-brightgreen.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)

---
>[!CAUTION]
>
>THE SIMPLE UNET GGUF LOADER NODE USED IN THE EXAMPLE WORKFLOWS IS CREATING CUDA ISSUES AFTER ~3 RUNS, REPLACE IT WITH THE "Unet Loader (GGUF/Advanced)" NODE OR THE WAN MODEL LOADER IN THIS PACKAGE HERE! I WILL UPDATE THE WORKFLOWS SOON.

## 🚀 Overview

**ComfyUI_Wan22Blockswap** enables running WAN 2.1/2.2 14B GGUF models with lower VRAM usage by dynamically swapping transformer blocks between GPU and CPU during inference, allowing for generations with higher resolutions.

### Key Features

- ✅ **Forward Patching**: Patches model's forward method to swap blocks during inference
- ✅ **GGUF Lazy Loading**: Loads blocks directly to target device (prevents VRAM spikes)
- ✅ **Combo Patcher**: Automatic HIGH→LOW model switching for guidance distillation workflows
- ✅ **ON_CLEANUP Callbacks**: Automatic model switching when sampling completes
- ✅ **WanVideoLooper Compatible**: Works with multi-loop video generation
- ✅ **Full Cleanup Node**: Aggressive memory cleanup at end of workflow

---

## 🛠️ Installation

### Method 1: Manual Installation

1. Download the latest release from the [Releases page](https://github.com/crmbz0r/ComfyUI_Wan22Blockswap/releases)
2. Extract the contents to your ComfyUI custom nodes directory:
    ```
    ComfyUI/custom_nodes/ComfyUI_Wan22Blockswap/
    ```
3. Restart ComfyUI

### Method 2: Git Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/crmbz0r/ComfyUI_Wan22Blockswap.git
```

### Method 3: Manager Installation

If you're using ComfyUI Manager, search for "Wan22Blockswap" in the available nodes list and install directly. (not yet, will add soon)

---

# 🎯 Usage

### Basic Usage

1. Load your WAN model in ComfyUI
2. Add the "WAN 2.2 BlockSwap Patcher" node to your workflow
3. Connect your model to the BlockSwap Patcher node
4. Configure the parameters based on your VRAM requirements
5. Connect the output directly to the KSampler of your choice, I'd recommend to connect it directly to the KSampler without any nodes inbetween. (check workflow examples)

## 📦 Available Nodes

### Active Nodes (6 total)

| Node | Description |
|------|-------------|
| **WAN Model Loader** | Simple WAN model loader (no BlockSwap) - use with GGUF loaders |
| **WAN 2.2 BlockSwap Patcher** | Apply BlockSwap to any single loaded model |
| **WAN 2.2 BlockSwap Combo Patcher** | Apply BlockSwap to HIGH+LOW model pair with automatic switching |
| **WAN 2.2 BlockSwap Cleanup** | Clean up BlockSwap state after sampling |
| **WAN 2.2 BlockSwap Reposition** | Re-position blocks for next sampling run |
| **WAN 2.2 Full Cleanup (End)** | Aggressive cleanup at end of workflow (like "Free Model and Node Cache") |

---

## 🎯 Quick Start

### Recommended Workflow: Combo Patcher

For guidance distillation workflows (HIGH noise → LOW noise models):


<img width="3422" height="490" alt="grafik" src="https://github.com/user-attachments/assets/a1109fb3-cc80-4e75-a123-83b2357780cf" />

**Usage with the WanVideoLooper node is pretty similar, just replace the Integrated KSampler with the Looper Node and add the LoRA Sequencer if needed. I'll add an example workflow later too..**


### Or the basic High / Low KSampler workflow:


<img width="3212" height="946" alt="grafik" src="https://github.com/user-attachments/assets/231a6567-b93b-4b20-83f3-487dfcd76668" />


### How It Works

1. **Combo Patcher** receives both HIGH and LOW noise models
2. Positions HIGH noise blocks on GPU (28) and CPU (12)
3. Moves ALL LOW noise blocks to CPU (waiting)
4. Patches forward methods for dynamic block swapping
5. Registers ON_CLEANUP callback on HIGH noise model
6. When HIGH noise sampling completes → callback positions LOW noise blocks
7. LOW noise samples with its blocks properly positioned
8. **Full Cleanup** frees all memory at workflow end

---

## ⚙️ Node Parameters

### WAN 2.2 BlockSwap Combo Patcher

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_high` | MODEL | Required | High noise model (guidance distillation) |
| `model_low` | MODEL | Required | Low noise model (guidance distillation) |
| `blocks_to_swap` | INT | 12 | Number of blocks to offload to CPU (0-40) |

### WAN 2.2 BlockSwap Patcher

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | Required | Any WAN model to apply BlockSwap to |
| `blocks_to_swap` | INT | 20 | Number of blocks to offload to CPU (0-40) |

### WAN 2.2 Full Cleanup (End)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `any_input` | ANY | Optional | Connect any output to trigger execution order |
| `unload_models` | BOOL | True | Unload all models from GPU |
| `free_memory` | BOOL | True | Clear node cache after workflow |
| `clear_cuda_cache` | BOOL | True | Clear PyTorch CUDA cache |
| `run_gc` | BOOL | True | Run Python garbage collection |

### WAN 2.2 BlockSwap Cleanup

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | MODEL | Required | Model to clean up |
| `latent` | LATENT | Optional* | Latent pass-through (Integrated KSampler) |
| `images` | IMAGE | Optional* | Image pass-through (WanVideoLooper) |
| `move_to_cpu` | BOOL | True | Move all blocks to CPU |
| `unpatch` | BOOL | False | Remove BlockSwap patches entirely |
| `clear_cache` | BOOL | True | Clear CUDA cache and run GC |
* Use at least one of the latent or image input since they function as signal inputs to start cleaning after the KSampler is finished

---

## 📊 Performance

### VRAM Usage (14B GGUF Model)

| Configuration | VRAM Required | Notes |
|---------------|---------------|-------|
| No BlockSwap | ~24GB+ | OOM on 12GB cards |
| 12 blocks swapped | ~7.2GB | Fits on 12GB with margin |
| 20 blocks swapped | ~5.5GB | More headroom for batches |

### Timing (480x640, 2 steps per model)

| Phase | Time |
|-------|------|
| HIGH noise (2 steps) | ~26s |
| Model switch | ~1s |
| LOW noise (3 steps) | ~25s |
| **Total per segment** | ~52s |

### Block Transfer Times (typical)

```
Block 28: transfer_time=0.10s, compute_time=0.21s, to_cpu_transfer_time=0.10s
```

- **Transfer to GPU**: ~70-100ms per block
- **Compute**: ~200-300ms per block  
- **Transfer to CPU**: ~100-130ms per block

---

## 🔧 Technical Details

### Forward Patching Strategy

The BlockSwap patcher wraps the model's forward method:

```python
def patched_forward(*args, **kwargs):
    for block in swapped_blocks:
        # Move block to GPU
        block.to(gpu_device, non_blocking=True)
        torch.cuda.synchronize()
        
    # Execute original forward
    result = original_forward(*args, **kwargs)
    
    for block in swapped_blocks:
        # Move block back to CPU
        block.to(cpu_device, non_blocking=True)
    
    return result
```

### GGUF Lazy Loading

The lazy loader intercepts ComfyUI's model loading to prevent VRAM spikes:

1. **Hook Installation**: Patches GGUF loader's `load_torch_file`
2. **Block Detection**: Identifies transformer blocks during load
3. **Direct Routing**: Loads blocks directly to CPU/GPU based on swap config
4. **Zero Spike**: Never loads all blocks to GPU simultaneously

### ON_CLEANUP Callbacks

ComfyUI's `add_object_patch` with `"ON_CLEANUP"` key triggers after sampling:

```python
model_high.add_object_patch("ON_CLEANUP", switch_to_low_noise_callback)
```

This enables automatic HIGH→LOW model switching without manual intervention.

---

## 🏗️ Architecture

### File Structure

```
ComfyUI_Wan22Blockswap/
├── __init__.py           # Node registration (active nodes only)
├── blockswap_forward.py  # Main implementation (~1600 lines)
│   ├── BlockSwapForwardPatcher    # Core patching logic
│   ├── WAN22BlockSwapPatcher      # Single model patcher node
│   ├── WAN22BlockSwapComboPatcher # HIGH+LOW combo patcher node
│   ├── WAN22BlockSwapCleanup      # Cleanup node
│   ├── WAN22BlockSwapReposition   # Reposition node
│   └── WAN22FullCleanup           # End-of-workflow cleanup node
├── wan_loader.py         # Simple WAN model loader
├── block_manager.py      # Block management utilities
├── callbacks.py          # Lazy load and cleanup callbacks
├── utils.py              # Utility functions
├── config.py             # Configuration and constants
│
├── # DEPRECATED (code preserved, nodes disabled):
├── nodes.py              # Old callback-based nodes
├── blockswap_loader.py   # Old integrated loader
├── blockswap_looper.py   # Old looper integration
└── blockswap_meta_loader.py  # Old meta loader
```

### Key Classes

| Class | Purpose |
|-------|---------|
| `BlockSwapForwardPatcher` | Core logic for patching forward methods |
| `BlockManager` | Manages block state and device placement |
| `BlockSwapTracker` | Tracks which blocks are swapped for cleanup |

---

## ❓ FAQ

### Q: Can I use this with GGUF models?
**A:** Yes! The system includes a lazy loader specifically designed for GGUF models that prevents VRAM spikes during loading.

### Q: What's the difference between Patcher and Combo Patcher?
**A:** 
- **Patcher**: For single models (e.g., standard WAN 2.1 workflows)
- **Combo Patcher**: For guidance distillation with HIGH+LOW model pairs (automatic switching)

### Q: Why do I see "Tried to unpin tensor not pinned by ComfyUI"?
**A:** This is harmless. It occurs when ComfyUI tries to unpin tensors that BlockSwap already moved. Doesn't affect functionality.

### Q: How many blocks should I swap?
**A:** 
- **12GB VRAM**: Start with 12 blocks
- **16GB VRAM**: 8-10 blocks for faster inference
- **24GB+ VRAM**: May not need BlockSwap at all

### Q: Do I need the cleanup nodes?
**A:** 
- **Full Cleanup**: Recommended at end of workflow to free VRAM for next run
- **BlockSwap Cleanup**: Optional, useful between multiple sampling runs in same workflow

---

## 📝 Changelog

### v1.0.0 (Current)

- ✅ **Forward Patching**: New approach that patches model forward methods
- ✅ **Combo Patcher**: Automatic HIGH→LOW model switching
- ✅ **ON_CLEANUP Callbacks**: Automatic model switching after sampling
- ✅ **GGUF Lazy Loader**: Prevents VRAM spikes during model loading
- ✅ **Full Cleanup Node**: Aggressive end-of-workflow cleanup
- ✅ **WanVideoLooper Support**: Works with multi-loop workflows
- 🗑️ **Deprecated**: Old callback-based nodes (code preserved)

### v0.0.1 (Previous)

- Initial development with ON_LOAD callback approach
- Various experimental loaders and patchers that didn't really work as intended
- Realization that ComfyUI's core mechanics do not like external blockswap nodes
- Despair and on the verge of losing hope

---

## 🙏 Acknowledgments

- **[ComfyUI-wanBlockswap](https://github.com/orssorbit/ComfyUI-wanBlockswap)** - Original block swapping implementation
- **[ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)** - WAN 2.2 wrapper and techniques
- **[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)** - GGUF model support
- **Claude Opus** - AI pair programming assistance

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Support

If you encounter issues:

1. Check this README and FAQ
2. Search existing [Issues](https://github.com/crmbz0r/ComfyUI-Wan22Blockswap/issues)
3. Create a new issue with:
   - ComfyUI version
   - GPU model and VRAM
   - Full error traceback
   - Workflow description
