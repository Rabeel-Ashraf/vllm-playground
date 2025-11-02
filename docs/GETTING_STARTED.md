# 🚀 vLLM WebUI - Getting Started

Welcome to the vLLM WebUI! This guide will get you up and running in minutes.

## 🎯 What You'll Get

A beautiful web interface that lets you:
- 🎛️ **Configure** vLLM servers with a friendly UI
- 💬 **Chat** with your models interactively
- 📋 **Monitor** real-time server logs
- ⚡ **Control** server start/stop with one click

## ⚡ Quick Start (3 Steps)

### 1. Install Dependencies
```bash
cd webui
./install.sh
```

### 2. Start the WebUI
```bash
./start.sh
```

### 3. Open Your Browser
```
http://localhost:7860
```

That's it! 🎉

## 📖 First Time Usage

Once the WebUI is open:

1. **Select a Model**
   - Choose "facebook/opt-125m" for a quick test
   - Or pick any HuggingFace model you have access to

2. **Click "Start Server"**
   - Watch the logs panel on the right
   - Wait for "Application startup complete"

3. **Start Chatting!**
   - Type your message in the center panel
   - Adjust temperature/max tokens if desired
   - Press "Send" or Ctrl+Enter

## 🎓 Learn More

- **QUICKSTART.md** - Quick reference and tips
- **FEATURES.md** - Complete feature overview
- **README.md** - Full documentation
- **PROJECT_SUMMARY.md** - Technical details

## 🆘 Having Issues?

Run the verification script:
```bash
python3 verify_setup.py
```

Check the troubleshooting section in README.md

## 🎨 Screenshots

### Main Interface
```
┌────────────────────────────────────────────────────────┐
│  🚀 vLLM WebUI              ● Connected  00:05:23      │
├──────────────┬──────────────────┬──────────────────────┤
│              │                  │                       │
│ ⚙️ Config    │  💬 Chat         │  📋 Logs             │
│              │                  │                       │
│ Model: opt   │  User: Hello!    │  [INFO] Starting...  │
│ GPU: 90%     │  Bot: Hi there!  │  [INFO] Loading...   │
│ [▶️ Start]   │  [Send]          │  [✓] Ready!          │
│              │                  │                       │
└──────────────┴──────────────────┴──────────────────────┘
```

## 🔥 Popular Configurations

### Quick Test (< 1 minute to load)
```
Model: facebook/opt-125m
GPU Memory: 50%
Tensor Parallel: 1
```

### Production (Llama 2 7B)
```
Model: meta-llama/Llama-2-7b-chat-hf
GPU Memory: 90%
Tensor Parallel: 1
Enable Prefix Caching: ✓
```

### Large Model (Multi-GPU)
```
Model: meta-llama/Llama-2-13b-chat-hf
GPU Memory: 90%
Tensor Parallel: 2
Enable Prefix Caching: ✓
```

## 💡 Pro Tips

1. **Start Small**: Test with opt-125m first
2. **Watch Logs**: They show what's happening
3. **GPU Memory**: Start at 70-80% if unsure
4. **Temperature**: 0.7 is a good default
5. **Prefix Cache**: Enable for repeated prompts

## 🎮 Keyboard Shortcuts

- `Ctrl+Enter` / `Cmd+Enter` - Send message

## 📊 System Requirements

- Python 3.8+
- CUDA-compatible GPU
- 8GB+ GPU memory (depends on model)
- Modern web browser

## 🔧 Advanced Configuration

Edit `app.py` to change:
- Default WebUI port (currently 7860)
- vLLM server port (currently 8000)
- Other backend settings

Edit `static/css/style.css` to customize:
- Colors and theme
- Layout and spacing
- Fonts and sizes

## 📱 Access from Other Devices

To access from other devices on your network:

1. Start WebUI with:
```bash
WEBUI_PORT=7860 python3 run.py
```

2. Find your IP address:
```bash
# On Linux/Mac
hostname -I
# On Windows
ipconfig
```

3. Access from other devices:
```
http://YOUR_IP:7860
```

## 🐳 Docker Support (Future)

Docker support is planned for a future release.

## 🤝 Need Help?

1. Check the logs in the right panel
2. Run `python3 verify_setup.py`
3. Read the QUICKSTART.md guide
4. Open an issue on GitHub

## 🎉 Enjoy!

Have fun chatting with your models! If you create something cool, share it with the community!

---

**Made with ❤️ for the vLLM community**

Happy building! 🚀

