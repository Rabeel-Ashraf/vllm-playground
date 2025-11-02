# Project Folder Architecture

## 📁 Organized Structure

```
vllm-webui/
│
├── 📄 Core Files (Root Level)
│   ├── app.py                  # FastAPI backend server
│   ├── run.py                  # Server launcher script
│   ├── index.html              # Main HTML interface
│   ├── requirements.txt        # Python dependencies
│   ├── LICENSE                 # MIT License
│   └── README.md              # Main documentation
│
├── 📂 static/                 # Frontend Assets
│   ├── css/
│   │   └── style.css          # Main stylesheet (dark theme)
│   └── js/
│       └── app.js             # Frontend logic & API calls
│
├── 📂 scripts/                # Utility Scripts
│   ├── run_cpu.sh            # Launch vLLM in CPU mode (macOS)
│   ├── start.sh              # General start script
│   ├── install.sh            # Installation helper
│   └── verify_setup.py       # Setup verification tool
│
├── 📂 config/                 # Configuration Files
│   ├── vllm_cpu.env          # CPU environment variables
│   └── example_configs.json   # Example server configurations
│
└── 📂 docs/                   # Documentation
    ├── QUICKSTART.md          # Quick start guide
    ├── GETTING_STARTED.md     # Detailed getting started
    ├── MACOS_CPU_GUIDE.md     # macOS/Apple Silicon guide
    ├── QUICK_REFERENCE.md     # Command quick reference
    ├── FEATURES.md            # Feature documentation
    ├── PERFORMANCE_METRICS.md # Metrics documentation
    ├── COMMAND_PREVIEW.md     # Command preview feature
    ├── RESIZABLE_PANELS.md    # UI panel documentation
    ├── PROJECT_SUMMARY.md     # Project overview
    └── README_SETUP.md        # Setup instructions
```

## 📋 File Descriptions

### Core Application Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI backend server with WebSocket support |
| `run.py` | Simple launcher for the webserver |
| `index.html` | Single-page application interface |
| `requirements.txt` | Python package dependencies |

### Static Assets

| Directory | Contents |
|-----------|----------|
| `static/css/` | Stylesheets (dark theme, responsive design) |
| `static/js/` | Frontend JavaScript (streaming, metrics, etc.) |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_cpu.sh` | Launch vLLM with CPU optimizations for macOS |
| `scripts/start.sh` | General server start script |
| `scripts/install.sh` | Installation helper script |
| `scripts/verify_setup.py` | Verify installation and dependencies |

### Configuration

| File | Purpose |
|------|---------|
| `config/vllm_cpu.env` | CPU mode environment variables (KVCACHE, thread binding) |
| `config/example_configs.json` | Example server configurations |

### Documentation

| Document | Content |
|----------|---------|
| `docs/QUICKSTART.md` | Get started in < 5 minutes |
| `docs/MACOS_CPU_GUIDE.md` | Complete macOS/Apple Silicon guide |
| `docs/QUICK_REFERENCE.md` | Command cheat sheet |
| `docs/FEATURES.md` | Feature list and descriptions |
| `docs/PERFORMANCE_METRICS.md` | Performance metrics documentation |

## 🎯 Key Benefits of This Structure

### ✅ **Clear Separation of Concerns**
- Frontend assets in `static/`
- Scripts in `scripts/`
- Configuration in `config/`
- Documentation in `docs/`

### ✅ **Easy Navigation**
- Related files grouped together
- Clear naming conventions
- Logical hierarchy

### ✅ **Better Maintainability**
- Easy to find files
- Clear purpose for each directory
- Scalable structure

### ✅ **Professional Organization**
- Follows industry best practices
- Similar to popular open-source projects
- Easy for contributors to understand

## 🔄 Migration Notes

### Files Moved:

**To `scripts/`:**
- `run_cpu.sh` → `scripts/run_cpu.sh`
- `start.sh` → `scripts/start.sh`
- `install.sh` → `scripts/install.sh`
- `verify_setup.py` → `scripts/verify_setup.py`

**To `config/`:**
- `vllm_cpu.env` → `config/vllm_cpu.env`
- `example_configs.json` → `config/example_configs.json`

**To `docs/`:**
- All `*.md` files (except README.md) → `docs/`

### Updated References:

- Environment file path updated in app.py: `config/vllm_cpu.env`
- All documentation cross-references updated
- README.md now points to organized structure

## 🚀 Usage After Reorganization

### Run CPU Mode:
```bash
./scripts/run_cpu.sh
```

### Edit CPU Configuration:
```bash
nano config/vllm_cpu.env
```

### View Documentation:
```bash
cat docs/MACOS_CPU_GUIDE.md
```

### Verify Setup:
```bash
python scripts/verify_setup.py
```

---

This organized structure makes the project more professional, maintainable, and easier to navigate! 🎉

