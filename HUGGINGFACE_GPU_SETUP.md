# HuggingFace Spaces GPU Setup Guide 🚀

This guide will help you enable GPU acceleration for GRDN AI on HuggingFace Spaces with your Nvidia T4 grant.

## Prerequisites
- HuggingFace Space with GPU enabled (Nvidia T4 small: 4 vCPU, 15GB RAM, 16GB GPU)
- Model files uploaded to your Space

## Setup Steps

### 1. Enable GPU in Space Settings
1. Go to your Space settings on HuggingFace
2. Navigate to "Hardware" section
3. Select "T4 small" (or your granted GPU tier)
4. Save changes

### 2. Upload Model Files
Your Space needs the GGUF model files in the `src/models/` directory:
- `llama-2-7b-chat.Q4_K_M.gguf` (for Llama2)
- `decilm-7b-uniform-gqa-q8_0.gguf` (for DeciLM)

You can upload these via:
- HuggingFace web interface (Files tab)
- Git LFS (recommended for large files)
- HuggingFace Hub CLI

### 3. Install Dependencies
Make sure your Space has the updated `requirements.txt` which includes:
```
torch>=2.0.0
```

### 4. Verify GPU Detection
Once your Space restarts, check the sidebar in the app for:
- 🚀 **GPU Acceleration: ENABLED** - GPU is working!
- ⚠️ **GPU Acceleration: DISABLED** - Something's wrong

You should also see in the logs:
```
🤗 Running on HuggingFace Spaces
🚀 GPU detected: Tesla T4 with 15.xx GB memory
🚀 Will offload all layers to GPU (n_gpu_layers=-1)
✅ GPU acceleration ENABLED with -1 layers
```

## How It Works

The app now automatically:
1. **Detects HuggingFace Spaces environment** via `SPACE_ID` or `SPACE_AUTHOR_NAME` env variables
2. **Checks for GPU availability** using PyTorch's `torch.cuda.is_available()`
3. **Configures LlamaCPP** to use GPU with `n_gpu_layers=-1` (all layers on GPU)
4. **Shows status** in the sidebar UI

### GPU Configuration
- **CPU Mode**: `n_gpu_layers=0` - All computation on CPU (slow)
- **GPU Mode**: `n_gpu_layers=-1` - All model layers offloaded to GPU (fast)

## Performance Expectations

With GPU acceleration on Nvidia T4:
- **Response time**: ~2-5 seconds (vs 30-60+ seconds on CPU)
- **Token generation**: ~20-50 tokens/sec (vs 1-3 tokens/sec on CPU)
- **Memory**: Model fits comfortably in 16GB VRAM

## Troubleshooting

### GPU Not Detected
1. **Check Space hardware**: Ensure T4 is selected in settings
2. **Check logs**: Look for GPU detection messages
3. **Verify torch installation**: `torch.cuda.is_available()` should return `True`
4. **Try restarting**: Sometimes requires Space restart after hardware change

### Model File Not Found
If you see: `⚠️ Model not found at src/models/...`
- Upload the model files to the correct path
- Check file names match exactly
- Ensure files aren't corrupted during upload

### Out of Memory Errors
If GPU runs out of memory:
- The quantized models (Q4_K_M, q8_0) are designed to fit in 16GB
- Try restarting the Space
- Check if other processes are using GPU memory

### Still Slow After GPU Setup
1. Verify GPU is actually being used (check logs)
2. Ensure `n_gpu_layers=-1` is set (check initialization logs)
3. Check HuggingFace Space isn't in "Sleeping" mode
4. Verify model is fully loaded before making requests

## Code Changes Summary

The following changes enable automatic GPU detection:

1. **`src/backend/chatbot.py`**:
   - Added `detect_gpu_and_environment()` function
   - Modified `init_llm()` to use dynamic GPU configuration
   - Automatic path detection for HF Spaces vs local

2. **`app.py`**:
   - Added GPU status indicator in sidebar
   - Shows real-time GPU availability

3. **`src/requirements.txt`**:
   - Added `torch>=2.0.0` for GPU detection

## Testing Locally

To test GPU detection locally (if you have an Nvidia GPU):
```bash
# Install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Run the app
streamlit run app.py
```

Without GPU locally, you'll see:
```
⚠️ No GPU detected via torch.cuda
⚠️ Running on CPU (no GPU detected)
```

## Additional Resources

- [HuggingFace Spaces Hardware Documentation](https://huggingface.co/docs/hub/spaces-gpus)
- [LlamaCPP GPU Acceleration Guide](https://github.com/ggerganov/llama.cpp#cublas)
- [PyTorch CUDA Setup](https://pytorch.org/get-started/locally/)

---

**Note**: This GPU setup is backward compatible - the app will still work on CPU if no GPU is available!

