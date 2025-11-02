# ✅ vLLM Setup Complete - Apple Silicon (macOS)

## 🎯 Configuration Summary

You have vLLM installed via `uv` in `/Users/micyang/.venv`
- **Version:** 0.11.1rc6.dev45+gc2ed069b3
- **Location:** `/Users/micyang/vllm/` (development build)

## 📁 Key Files

### 1. `vllm_cpu.env` - Environment Configuration
Based on [vLLM CPU documentation](https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html#related-runtime-environment-variables):

```bash
# KV Cache Space (GB)
export VLLM_CPU_KVCACHE_SPACE=40

# CPU Thread Binding
export VLLM_CPU_OMP_THREADS_BIND=auto
```

**To customize:** Edit this file to adjust memory and threading settings.

### 2. `run_cpu.sh` - Updated Launch Script
Now includes:
- ✅ Activates `/Users/micyang/.venv`
- ✅ Loads `vllm_cpu.env` configuration
- ✅ Uses CPU-compatible flags

## 🚀 How to Run

```bash
# Navigate to project
cd /Users/micyang/vllm-webui

# Run vLLM server
./run_cpu.sh

# Or with custom model/port
./run_cpu.sh facebook/opt-350m 8001
```

## 🔍 What the Script Does

1. **Activates your venv:** `/Users/micyang/.venv`
2. **Loads CPU settings** from `vllm_cpu.env`
3. **Runs vLLM** with:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model facebook/opt-125m \
     --host 0.0.0.0 \
     --port 8000 \
     --dtype bfloat16
   ```

## ⚙️ Environment Variables in Use

From the [official documentation](https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html):

| Variable | Value | Purpose |
|----------|-------|---------|
| `VLLM_CPU_KVCACHE_SPACE` | 40 | Memory (GB) for KV cache |
| `VLLM_CPU_OMP_THREADS_BIND` | auto | Automatic thread binding to CPU cores |

## 🐛 If You Still Get Segmentation Faults

Edit `vllm_cpu.env` and uncomment these lines:

```bash
# Disable x86-specific optimizations
export VLLM_CPU_MOE_PREPACK=0
export VLLM_CPU_SGL_KERNEL=0
```

These optimizations are x86-specific and may cause issues on Apple Silicon.

## 🧪 Manual Testing

```bash
# Activate venv
source /Users/micyang/.venv/bin/activate

# Load environment
source vllm_cpu.env

# Run server
python -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-125m \
  --dtype bfloat16 \
  --port 8000
```

## 📊 Test the Server

Once running, test with:

```bash
# Check health
curl http://localhost:8000/health

# Test chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## 📝 Configuration Options

Edit `vllm_cpu.env` to customize (all from [official docs](https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html)):

```bash
# Reduce memory if needed
export VLLM_CPU_KVCACHE_SPACE=10

# Reserve CPUs for frontend
export VLLM_CPU_NUM_OF_RESERVED_CPU=1

# Disable problematic optimizations on Apple Silicon
export VLLM_CPU_MOE_PREPACK=0
export VLLM_CPU_SGL_KERNEL=0
```

## 🎯 Key Points

1. **Your vLLM is a dev build** (`0.11.1rc6.dev45`) from source
2. **It's in your venv** at `/Users/micyang/.venv`
3. **CPU mode is configured** via environment variables
4. **Script handles everything** - just run `./run_cpu.sh`

## 📚 References

- [vLLM CPU Installation](https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html)
- [CPU Environment Variables](https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html#related-runtime-environment-variables)

---

**Ready to test?** Run `./run_cpu.sh` and let me know if you encounter any errors!

