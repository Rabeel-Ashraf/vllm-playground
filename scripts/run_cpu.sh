#!/bin/bash
# Script to run vLLM on CPU (macOS compatible)
# Based on: https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html

echo "🚀 Starting vLLM in CPU mode (macOS compatible)"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
VENV_PATH="/Users/micyang/.venv"
if [ -d "$VENV_PATH" ]; then
    echo "🐍 Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "⚠️  Virtual environment not found at $VENV_PATH"
    echo "   Using system Python instead"
fi
echo ""

# Load environment variables from config file
if [ -f "$SCRIPT_DIR/vllm_cpu.env" ]; then
    echo "📄 Loading configuration from vllm_cpu.env"
    source "$SCRIPT_DIR/vllm_cpu.env"
else
    # Fallback to default values
    export VLLM_CPU_KVCACHE_SPACE=40
    export VLLM_CPU_OMP_THREADS_BIND=auto
fi

# Display settings
echo "📋 Environment Variables:"
echo "  VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE}"
echo "  VLLM_CPU_OMP_THREADS_BIND=${VLLM_CPU_OMP_THREADS_BIND}"
echo ""

# Default model
MODEL="${1:-facebook/opt-125m}"
PORT="${2:-8000}"

echo "🤖 Model: ${MODEL}"
echo "🌐 Port: ${PORT}"
echo ""
echo "Starting server..."
echo ""

# Run vLLM server with CPU-compatible settings
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype bfloat16

# Note: Removed GPU-specific flags:
# - --tensor-parallel-size
# - --gpu-memory-utilization
# - --load-format

