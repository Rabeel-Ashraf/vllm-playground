# vLLM WebUI

A modern web interface for managing and interacting with vLLM (Very Large Language Model) servers. Supports both GPU and CPU modes, with special optimizations for macOS Apple Silicon.

## 📁 Project Structure

```
vllm-webui/
├── app.py                  # Main FastAPI backend application
├── run.py                  # Backend server launcher
├── index.html              # Main HTML interface
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
├── README.md              # This file
│
├── static/                # Frontend assets
│   ├── css/
│   │   └── style.css     # Main stylesheet
│   └── js/
│       └── app.js        # Frontend JavaScript
│
├── scripts/              # Utility scripts
│   ├── run_cpu.sh       # Start vLLM in CPU mode (macOS compatible)
│   ├── start.sh         # General start script
│   ├── install.sh       # Installation script
│   └── verify_setup.py  # Setup verification
│
├── config/               # Configuration files
│   ├── vllm_cpu.env     # CPU mode environment variables
│   └── example_configs.json  # Example configurations
│
└── docs/                 # Documentation
    ├── QUICKSTART.md    # Quick start guide
    ├── GETTING_STARTED.md
    ├── MACOS_CPU_GUIDE.md  # macOS CPU setup guide
    ├── QUICK_REFERENCE.md   # Command reference
    ├── FEATURES.md       # Feature documentation
    ├── PERFORMANCE_METRICS.md
    ├── COMMAND_PREVIEW.md
    ├── RESIZABLE_PANELS.md
    ├── PROJECT_SUMMARY.md
    └── README_SETUP.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install vLLM

```bash
# For macOS/CPU mode
pip install vllm
```

### 3. Start the WebUI

```bash
python run.py
```

Then open http://localhost:7860 in your browser.

### 4. Start vLLM Server

**Option A: Using the WebUI**
- Select CPU or GPU mode
- Click "Start Server"

**Option B: Using the script (macOS/CPU)**
```bash
./scripts/run_cpu.sh
```

## 💻 macOS Apple Silicon Support

For macOS users, vLLM runs in CPU mode. See [docs/MACOS_CPU_GUIDE.md](docs/MACOS_CPU_GUIDE.md) for detailed setup.

**Quick CPU Mode Setup:**
```bash
# Edit CPU configuration
nano config/vllm_cpu.env

# Run vLLM
./scripts/run_cpu.sh
```

## ✨ Features

- **Server Management**: Start/stop vLLM servers from the UI
- **Chat Interface**: Interactive chat with streaming responses
- **Performance Metrics**: Real-time token counts and generation speed
- **Model Support**: Pre-configured popular models + custom model support
- **CPU & GPU Modes**: Automatic detection and configuration
- **Benchmarking**: GuideLLM integration for performance testing
- **Resizable Panels**: Customizable layout
- **Command Preview**: See exact commands before execution

## 📖 Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [macOS CPU Setup](docs/MACOS_CPU_GUIDE.md)
- [Feature Overview](docs/FEATURES.md)
- [Performance Metrics](docs/PERFORMANCE_METRICS.md)
- [Command Reference](docs/QUICK_REFERENCE.md)

## 🔧 Configuration

### CPU Mode (macOS)

Edit `config/vllm_cpu.env`:
```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=auto
```

### Supported Models

- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (default)
- RedHatAI/Llama-3.2-1B-Instruct-FP8
- RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8
- Custom models via text input

## 🛠️ Development

### Project Structure

- **Backend**: FastAPI (`app.py`)
- **Frontend**: Vanilla JavaScript (`static/js/app.js`)
- **Styling**: Custom CSS (`static/css/style.css`)
- **Scripts**: Bash scripts in `scripts/`
- **Config**: Environment files in `config/`

### Running in Development

```bash
# Start backend with auto-reload
uvicorn app:app --reload --port 7860

# Or use the run script
python run.py
```

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details

## 🤝 Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

## 🔗 Links

- [vLLM Official Documentation](https://docs.vllm.ai/)
- [vLLM CPU Mode Guide](https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html)
- [vLLM GitHub](https://github.com/vllm-project/vllm)

## 🆘 Troubleshooting

### macOS Segmentation Fault

Use CPU mode with proper environment variables. See [docs/MACOS_CPU_GUIDE.md](docs/MACOS_CPU_GUIDE.md).

### Server Won't Start

1. Check if vLLM is installed: `python -c "import vllm; print(vllm.__version__)"`
2. Check port availability: `lsof -i :8000`
3. Review server logs in the WebUI

### Chat Not Streaming

Check browser console (F12) for errors and ensure the server is running.

---

Made with ❤️ for the vLLM community
