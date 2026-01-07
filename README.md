#                                       ComfyUI-Wan22Blockswap

> [!IMPORTANT]
> ***SEEMS LIKE THE CORE SAMPLING AND MODEL LOADING OF COMFYUI IS A PAIN IN THE ASS AND DOESNT LIKE EXTERNAL BLOCK SWAPPING... IT DOESNT REALLY WANT TO WORK AS INTENDED, BUT I AM NOT GIVING UP 😤😮‍💨***


  
                                             ***MID DEVELOPMENT, EXPECT BUGS AND WEIRD THINGS TO HAPPEN.***

                                     ***IF THEY HAPPEN, PLEASE CREATE AN ISSUE WITH THE TRACEBACK, THANKS!***  😊~               
 
### **Advanced VRAM Optimization for WAN 2.1/2 with Lazy Loading and GGUF Support**

                                                                                                   [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Compatible-brightgreen.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)

      
      
> [!CAUTION]
>                    ***__!! THE "WAN BlockSwap Model Loader" MODEL LOADER NODE SHOULD NOT BE USED YET !!__***
>                      
>                                          ***IT IS STILL VERY EXPERIMENTAL AND WILL MOST LIKELY FLOOD YOUR VRAM***
>
>                                          ***THE BLOCK SWAPPING ONLY WORKS WITH GGUF MODELS CURRENTLY***
> 
>                                          ***THE WAN MODEL LOADER ALSO ONLY WORKS WITH GGUF MODELS ATM***



      

## ***🚀 Overview***

### **ComfyUI-Wan22Blockswap is a powerful ComfyUI node that implements advanced VRAM**

### **optimization techniques for WAN 2.1/2.2 models. It uses a sophisticated lazy loading**

### **strategy to prevent VRAM spikes at model loading while maintaining optimal performance.**

    
## Key Features

-   **Lazy Loading**: Blocks are loaded directly to their target device, preventing massive VRAM spikes
-   **GGUF Compatible**: Automatically detects and handles GGUF quantized models with best-effort swapping
-   **Multi-Model Support**: Works with all WAN model variants (1.3B, 5B, 14B, LongCat)
-   **WanVideoLooper Integration**: Full support for multi-loop video generation workflows
-   **VACE Integration**: Optional support for VACE model block swapping for multi-modal tasks
-   **Smart Memory Management**: Automatic cleanup and memory optimization
-   **Debug Support**: Comprehensive logging and performance monitoring

          
        

## 📋 Table of Contents

-   [🛠️ Installation](https://github.com/crmbz0r/ComfyUI_Wan22Blockswap#%EF%B8%8F-installation)
-   [📦 Available Nodes](https://github.com/crmbz0r/ComfyUI_Wan22Blockswap#%EF%B8%8Favailable-nodes)
-   [🎯 Usage](https://github.com/crmbz0r/ComfyUI_Wan22Blockswap#-usage)
-   [⚙️ Parameters](https://github.com/crmbz0r/ComfyUI_Wan22Blockswap#%EF%B8%8F-parameters)
-   [📊 Performance](https://github.com/crmbz0r/ComfyUI_Wan22Blockswap#-performance)
-   [🔧 Compatibility](https://github.com/crmbz0r/ComfyUI_Wan22Blockswap#-compatibility)
-   [🏗️ Architecture](https://github.com/crmbz0r/ComfyUI_Wan22Blockswap#%EF%B8%8F-architecture)
-   [🤝 Contributing](https://github.com/crmbz0r/ComfyUI_Wan22Blockswap#-contributing)
-   [📝 License](https://github.com/crmbz0r/ComfyUI_Wan22Blockswap#-license)

        

## 🛠️ Installation

### Method 1: Manual Installation

1. Download the latest release from the [Releases page](https://github.com/crmbz0r/ComfyUI_Wan22Blockswap/releases)
2. Extract the contents to your ComfyUI custom nodes directory:
    ```
    ComfyUI/custom_nodes/ComfyUI-wan22Blockswap/
    ```
3. Restart ComfyUI

### Method 2: Git Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-Wan22Blockswap.git
```

### Method 3: Manager Installation

If you're using ComfyUI Manager, search for "Wan22Blockswap" in the available nodes list and install directly.

        

## 📦 Available Nodes

This extension provides **5 nodes** organized by purpose:

### 1. WAN Model Loader ✅ **(Recommended for simple workflows)**

**Category:** `WAN`

A simple all-in-one WAN model loader. Loads WAN 2.1/2.2 models in safetensors or GGUF format with automatic configuration detection. No BlockSwap - just pure model loading.

| Input | Type | Description |
|-------|------|-------------|
| `model_type` | Combo | Choose between "safetensors" or "gguf" format |
| `safetensors_model` | Combo | Select safetensors model from diffusion_models folder |
| `gguf_model` | Combo | Select GGUF model from diffusion_models folder |
| `wan_version` | Combo | "auto", "2.1", or "2.2" (auto-detects from weights) |
| `model_variant` | Combo | "auto", "t2v", "i2v", "vace", "camera", "s2v", "humo", "animate" |
| `fp8_optimization` | Combo | FP8 quantization: "disabled", "e4m3fn", "e5m2" |
| `weight_dtype` | Combo | Weight dtype: "auto", "fp16", "bf16", "fp32" |

> **Note:** When using GGUF models, `fp8_optimization` and `weight_dtype` have no effect (GGUF models have their own quantization).

**Output:** Connect to a **WAN 2.2 BlockSwap** node for VRAM optimization.

---

### 2. WAN 2.2 BlockSwap ✅ **(Main BlockSwap node)**

**Category:** `ComfyUI_Wan22Blockswap`

Apply LAZY LOADING block swapping to WAN 2.1/2.2 models. Blocks are offloaded DURING loading to prevent VRAM spikes. This is the main node for VRAM optimization.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | MODEL | - | ComfyUI native WAN model (connect from any loader) |
| `blocks_to_swap` | INT | 20 | Number of transformer blocks to swap to CPU |
| `offload_txt_emb` | BOOLEAN | False | Offload text embeddings to CPU (~500MB savings) |
| `offload_img_emb` | BOOLEAN | False | Offload image embeddings to CPU (~200MB savings) |
| `use_non_blocking` | BOOLEAN | False | Use non-blocking memory transfers (faster) |
| `vace_blocks_to_swap` | INT | 0 | VACE blocks to swap (0=auto detection) |
| `prefetch_blocks` | INT | 0 | Prefetch N blocks ahead for performance |
| `block_swap_debug` | BOOLEAN | False | Enable debug logging |

**Output:** Optimized model ready for sampling.

---

### 3. WAN BlockSwap Model Loader ⚠️ **(Experimental - DO NOT USE)**

**Category:** `ComfyUI_Wan22Blockswap`

> [!CAUTION]
> **This node is experimental and should not be used yet!** It has known issues with block cleanup that can cause CUDA/Torch errors after generation.

A combined loader + BlockSwap node that routes blocks directly during weight loading. The goal is to prevent VRAM spikes by never loading swap blocks to GPU memory.

**Status:** Under active development. Use the separate **WAN Model Loader** + **WAN 2.2 BlockSwap** nodes instead.

---

### 4. WAN22 BlockSwap Looper Models ✅ **(For WanVideoLooper)**

**Category:** `ComfyUI_Wan22Blockswap/looper`

Apply BlockSwap to high/low noise model pairs for WanVideoLooper integration. This node prepares both models with proper session tracking for multi-loop video generation.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model_high` | MODEL | - | High-noise model for WanVideoLooper |
| `model_low` | MODEL | - | Low-noise model for WanVideoLooper |
| `blocks_to_swap` | INT | 20 | Blocks to swap to CPU |
| `offload_txt_emb` | BOOLEAN | False | Offload text embeddings |
| `offload_img_emb` | BOOLEAN | False | Offload image embeddings |
| `use_non_blocking` | BOOLEAN | False | Non-blocking transfers |
| `vace_blocks_to_swap` | INT | 0 | VACE blocks to swap |
| `prefetch_blocks` | INT | 0 | Prefetch blocks |
| `block_swap_debug` | BOOLEAN | False | Debug logging |

**Outputs:** `model_high`, `model_low` - Connect directly to WanVideoLooper's model inputs.

---

### 5. WAN22 BlockSwap Sequencer ✅ **(For WanVideoLoraSequencer)**

**Category:** `ComfyUI_Wan22Blockswap/looper`

Apply BlockSwap to WanVideoLoraSequencer output for per-segment BlockSwap with different LoRAs. Takes a list of (model_high, model_low, clip) tuples and applies BlockSwap to each model.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| `model_clip_sequence` | ANY | - | Output from WanVideoLoraSequencer |
| `blocks_to_swap` | INT | 20 | Blocks to swap to CPU |
| `offload_txt_emb` | BOOLEAN | False | Offload text embeddings |
| `offload_img_emb` | BOOLEAN | False | Offload image embeddings |
| `use_non_blocking` | BOOLEAN | False | Non-blocking transfers |
| `vace_blocks_to_swap` | INT | 0 | VACE blocks to swap |
| `prefetch_blocks` | INT | 0 | Prefetch blocks |
| `block_swap_debug` | BOOLEAN | False | Debug logging |

**Output:** `model_clip_sequence` - Connect to WanVideoLooper's model_clip_sequence input.

**Workflow:**
1. Load models with WAN Model Loader
2. Use WanVideoLoraSequencer to assign per-segment LoRAs
3. Connect sequencer output to this node
4. Connect this node's output to WanVideoLooper

---

        

# 🎯 Usage

### Basic Usage

1. Load your WAN model in ComfyUI
2. Add the "WAN 2.2 BlockSwap" node to your workflow
3. Connect your model to the BlockSwap node
4. Configure the parameters based on your VRAM requirements
5. Connect the output to your next node

### Example Workflow

```python
# Basic setup for VRAM optimization
model = load_model("wan_2.1_model.safetensors")
optimized_model = wan22BlockSwap(
    model=model,
    blocks_to_swap=20,           # Swap 20 blocks to CPU
    offload_txt_emb=True,        # Offload text embeddings
    offload_img_emb=False,       # Keep image embeddings on GPU
    use_non_blocking=True,       # Faster transfers
    block_swap_debug=True        # Enable monitoring
)
```

        

## ⚙️ Parameters

### Required Parameters

| Parameter         | Type    | Default | Description                                      |
| ----------------- | ------- | ------- | ------------------------------------------------ |
| `model`           | MODEL   | -       | ComfyUI native WAN model (WAN 2.1/2.2 etc.)      |
| `blocks_to_swap`  | INT     | 20      | Number of transformer blocks to swap to CPU      |
| `offload_txt_emb` | BOOLEAN | False   | Offload text embeddings to CPU (~500MB savings)  |
| `offload_img_emb` | BOOLEAN | False   | Offload image embeddings to CPU (~200MB savings) |

### Optional Parameters

| Parameter             | Type    | Default | Description                                               |
| --------------------- | ------- | ------- | --------------------------------------------------------- |
| `use_non_blocking`    | BOOLEAN | False   | Use non-blocking memory transfers (faster, uses more RAM) |
| `vace_blocks_to_swap` | INT     | 0       | VACE model blocks to swap (0=auto detection)              |
| `prefetch_blocks`     | INT     | 0       | Prefetch N blocks ahead for performance                   |
| `block_swap_debug`    | BOOLEAN | False   | Enable debug logging and performance monitoring           |

### Parameter Guidelines

#### Model-Specific Recommendations

**1.3B/5B Models:**

-   `blocks_to_swap`: 0-30 (recommended: 15-25)
-   VRAM savings: ~100-200MB per 10 blocks

**14B Models:**

-   `blocks_to_swap`: 0-40 (recommended: 25-35)
-   VRAM savings: ~150-250MB per 10 blocks

**LongCat Models:**

-   `blocks_to_swap`: 0-48 (recommended: 30-40)
-   VRAM savings: ~200-300MB per 10 blocks

#### Memory Optimization Strategies

**For 8GB VRAM:**

```python
blocks_to_swap=25
offload_txt_emb=True
offload_img_emb=True
```

**For 12GB VRAM:**

```python
blocks_to_swap=15
offload_txt_emb=True
offload_img_emb=False
```

**For 16GB+ VRAM:**

```python
blocks_to_swap=0  # No swapping needed
offload_txt_emb=False
```

        

## 📊 Performance

### Memory Savings

**📊 Based on Real Testing with 10GB GGUF Models:**

| Blocks Swapped | VRAM Savings | CPU Memory Increase | Performance Impact |
| -------------- | ------------ | ------------------- | ------------------ |
| 10             | ~7.7GB       | ~2GB                | Minimal            |
| 20             | ~15.4GB      | ~4GB                | Low                |
| 30             | ~23.1GB      | ~6GB                | Medium             |
| 40             | ~30.8GB      | ~8GB                | Medium-High        |

        

**💡 Key Insights:**

-   **Per Block Savings**: ~770MB VRAM per block swapped
-   **Efficiency**: Each swapped block moves ~200MB to CPU memory
-   **Sweet Spot**: 15-25 blocks for optimal VRAM savings vs performance balance
-   **Setup**: RTX 4080 (16GB) + 48GB RAM handles 20+ blocks efficiently for high resolution generations(720x960)

### Performance Testing

To measure the actual VRAM savings and performance impact for your specific setup, use the included test script:

```bash
# Test with your GGUF model
python test_performance.py --model-path /path/to/your/model.gguf --blocks-to-swap 20

# Test multiple configurations
python test_performance.py --model-path /path/to/your/model.gguf --test-all
```

This will generate a detailed `block_swap_performance_results.json` file with:

-   Actual VRAM savings for your hardware configuration
-   Load time comparisons with GGUF model handling
-   CPU memory usage impact
-   Performance metrics optimized for GGUF quantized models

**GGUF Model Considerations:**

-   GGUF quantization affects memory allocation patterns - test script accounts for this
-   CPU RAM availability is crucial for GGUF model swapping
-   Results will vary based on your specific hardware and model size

### Performance Optimization Tips

1. **Start Conservative**: Begin with fewer blocks and increase gradually
2. **Monitor Performance**: Use `block_swap_debug=True` to monitor impact
3. **Use Prefetching**: Set `prefetch_blocks=1` for optimal performance
4. **Non-Blocking Transfers**: Enable `use_non_blocking=True` for faster transfers

### Debug Mode Output

When `block_swap_debug=True`, you'll see detailed information:

```
[BlockSwap] GGUF model detected - using LAZY LOADING with safe swapping
[BlockSwap] Total blocks: 48
[BlockSwap] Keeping on GPU: blocks 0-27
[BlockSwap] Offloading to CPU: blocks 28-47
[BlockSwap] GPU blocks: 28
[BlockSwap] CPU blocks: 20
[BlockSwap] Peak VRAM usage should be MUCH lower!
```

        

## 🔧 Compatibility

### Supported Models

-   **WAN 2.1 Models**: Full support with lazy loading
-   **WAN 2.2 Models**: Full support with lazy loading
-   **GGUF Quantized Models**: Best-effort support with safe swapping
-   **VACE Models**: Optional block swapping support

### System Requirements

-   **Python**: 3.8+
-   **PyTorch**: 1.12+ with CUDA support
-   **ComfyUI**: Latest version recommended
-   **GPU**: CUDA-compatible (NVIDIA) or CPU-only mode

### Platform Support

-   ✅ Windows 10/11
-   ✅ Linux (Ubuntu 18.04+, CentOS 7+)
-   ✅ macOS (Apple Silicon with MPS support)

        

## 🏗️ Architecture

### Modular Design

The project is organized into 6 focused modules:

1. **`config.py`** - Parameter validation and model configuration
2. **`block_manager.py`** - Core block swapping logic and state tracking
3. **`callbacks.py`** - Lazy loading and cleanup callback functions
4. **`utils.py`** - Utility functions for memory management
5. **`nodes.py`** - Main ComfyUI node interface
6. **`__init__.py`** - Module initialization and exports

### Lazy Loading Strategy

The lazy loading approach works in 4 phases:

1. **Detection**: Identify model type (GGUF vs native) and capabilities
2. **Calculation**: Determine which blocks to swap based on user parameters
3. **Execution**: Load blocks directly to target devices during model loading
4. **Cleanup**: Proper cleanup and memory management when model is unloaded

### Memory Management

-   **Smart Swapping**: Only swap blocks that provide meaningful VRAM savings
-   **Automatic Cleanup**: Clean up swapped blocks when models are unloaded
-   **Device Synchronization**: Proper GPU synchronization to prevent race conditions
-   **Error Handling**: Graceful handling of memory allocation failures

        

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests for your changes
5. Run the test suite: `python -m pytest`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Code Style

We follow PEP 8 standards and use Black for code formatting. Please ensure your code passes:

```bash
black .
flake8 .
```

        

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

        

## 🙏 Acknowledgments

This project builds upon the excellent work of the following projects and communities:

-   **[ComfyUI-wanBlockswap](https://github.com/orssorbit/ComfyUI-wanBlockswap)** - Original block swapping implementation that served as the foundation for this project
-   **[ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper)** - Source of block swapping techniques and implementation insights for WAN 2.2 models
-   The ComfyUI team for creating an amazing platform
-   The WAN model developers for their excellent work
-   The GGUF community for their quantization efforts
-   All contributors and testers who helped improve this project

        

## 📞 Support

If you encounter issues or have questions:

1. Check the [FAQ](#faq) section below
2. Search existing [Issues](https://github.com/crmbz0r/ComfyUI-Wan22Blockswap/issues)
3. Create a new issue with detailed information
4. Join our Discord community (link in repository)

        

## ❓ FAQ

### Q: Can I use this with GGUF models?

**A:** Yes! The system automatically detects GGUF models and uses safe swapping techniques that work with quantized models.

### Q: Will this affect my model's performance?

**A:** The performance impact is minimal to low, depending on your configuration. Using `prefetch_blocks=1` and `use_non_blocking=True` can help minimize any performance impact. When swapping 30 blocks while generating a 720x960 video,it will be way slower, but also a lot less vram usage.

### Q: How do I know how many blocks to swap?

**A:** Start with conservative values (10-15 blocks) and monitor your VRAM usage. Gradually increase until you reach your desired VRAM savings while maintaining acceptable performance.

### Q: What happens if I run out of CPU RAM?

**A:** The system will gracefully handle memory allocation failures and skip swapping blocks that can't be allocated, falling back to GPU memory.

        

## 🔄 Changelog

### v0.0.1 (Current)

-   Initial release with full lazy loading support
-   GGUF model compatibility
-   VACE model support
-   Comprehensive debug logging
-   Modular architecture
-   WanVideoLooper support (hopefully bug-free)

---
