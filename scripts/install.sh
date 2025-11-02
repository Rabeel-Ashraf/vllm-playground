#!/bin/bash
# Installation script for vLLM WebUI

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           vLLM WebUI Installation Script                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
echo "🔍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
echo "✅ Found Python $PYTHON_VERSION"
echo ""

# Install dependencies
echo "📦 Installing WebUI dependencies..."
pip install -r requirements.txt
echo ""

# Check vLLM
echo "🔍 Checking vLLM installation..."
if python3 -c "import vllm" &> /dev/null; then
    echo "✅ vLLM is already installed"
else
    echo "⚠️  vLLM is not installed"
    echo ""
    read -p "Would you like to install vLLM now? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📦 Installing vLLM..."
        pip install vllm
    else
        echo "ℹ️  Skipping vLLM installation. You can install it later with:"
        echo "   pip install vllm"
    fi
fi
echo ""

# Run verification
echo "🔧 Running setup verification..."
python3 verify_setup.py
echo ""

# Done
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                  Installation Complete! 🎉                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 To start the WebUI, run:"
echo "   ./start.sh"
echo ""
echo "📚 Documentation:"
echo "   README.md       - Full documentation"
echo "   QUICKSTART.md   - Quick reference guide"
echo "   FEATURES.md     - Feature overview"
echo ""
echo "🌐 Access the WebUI at: http://localhost:7860"
echo ""

