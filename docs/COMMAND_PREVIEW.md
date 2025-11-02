# 🖥️ Command Preview Feature

## Overview

The Command Preview feature displays the exact vLLM command that will be executed based on your current configuration. This provides transparency and helps you understand what's happening under the hood.

## Location

The Command Preview section is located in the **left configuration panel**, between the configuration options and the Start/Stop buttons.

## Features

### 🔄 Real-time Updates
- Command updates instantly as you change any configuration option
- See exactly how each setting affects the command line arguments
- No need to guess what parameters are being used

### 📋 Copy to Clipboard
- One-click copy button to copy the entire command
- Visual confirmation when copied ("✓ Copied!")
- Run the command manually if needed

### 🎨 Formatted Display
- Multi-line formatting with backslashes for readability
- Monospace font for code clarity
- Scrollable if command is long
- Syntax highlighting-ready dark theme

## Example Commands

### Basic Configuration
```bash
python -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-125m \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --dtype auto \
  --load-format auto
```

### With Advanced Options
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-chat-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --dtype auto \
  --load-format auto \
  --max-model-len 4096 \
  --trust-remote-code \
  --enable-prefix-caching
```

### Multi-GPU Setup
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-13b-chat-hf \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --dtype auto \
  --load-format auto \
  --enable-prefix-caching \
  --disable-log-stats
```

## UI Layout

```
┌──────────────────────────────────────┐
│  ⚙️ Server Configuration             │
├──────────────────────────────────────┤
│                                       │
│  [Model Selection]                    │
│  [Host & Port]                        │
│  [GPU Settings]                       │
│  [Data Type]                          │
│  [Advanced Options]                   │
│                                       │
├──────────────────────────────────────┤
│  🖥️ Command Preview      [📋 Copy]   │
├──────────────────────────────────────┤
│  ┌────────────────────────────────┐  │
│  │ python -m vllm.entrypoints...  │  │
│  │   --model facebook/opt-125m    │  │
│  │   --host 0.0.0.0               │  │
│  │   --port 8000                  │  │
│  │   --tensor-parallel-size 1     │  │
│  │   ...                          │  │
│  └────────────────────────────────┘  │
│  This command will be executed...     │
├──────────────────────────────────────┤
│  [▶️ Start Server] [⏹️ Stop Server]  │
└──────────────────────────────────────┘
```

## Use Cases

### 1. **Learning & Understanding**
- See how UI options map to command-line arguments
- Learn vLLM command syntax
- Understand parameter relationships

### 2. **Manual Execution**
- Copy the command and run it manually
- Use in scripts or automation
- Modify for custom use cases

### 3. **Debugging**
- Verify configuration before starting
- Check if parameters are correct
- Share exact commands when asking for help

### 4. **Documentation**
- Copy commands for documentation
- Create setup guides
- Share configurations with team

## Interactive Behavior

### When Configuration Changes
1. User modifies any configuration field
2. JavaScript `updateCommandPreview()` is triggered
3. Command is regenerated with new values
4. Display updates immediately
5. Multi-line formatting applied

### Copy Button Behavior
1. User clicks "📋 Copy" button
2. Command text is copied to clipboard
3. Button changes to "✓ Copied!" with green background
4. Success notification appears
5. Button returns to normal after 2 seconds

## Technical Details

### Files Modified
1. **index.html** - Added command preview section
2. **style.css** - Added styling for command display
3. **app.js** - Added command generation logic

### Key Functions

#### `updateCommandPreview()`
- Reads all configuration values
- Builds command string with proper formatting
- Updates the display element
- Called on init and whenever config changes

#### `copyCommand()`
- Uses Clipboard API to copy text
- Provides visual feedback
- Shows success notification
- Handles errors gracefully

### Event Listeners
All configuration inputs trigger command update:
- Model selection (select & text input)
- Host, port, tensor parallel, GPU memory
- Data type selection
- Max model length
- All checkboxes (trust code, prefix caching, log stats)

## Benefits

✅ **Transparency** - See exactly what's being executed  
✅ **Learning** - Understand vLLM CLI options  
✅ **Debugging** - Verify configuration before start  
✅ **Portability** - Copy and run elsewhere  
✅ **Documentation** - Share configurations easily  

## Future Enhancements

Potential improvements:
- [ ] Syntax highlighting in command display
- [ ] Export command to file
- [ ] Command history / presets
- [ ] Explanation tooltips for each parameter
- [ ] One-click "Run in Terminal" button

---

This feature enhances the vLLM WebUI by providing full transparency into the server configuration process!

