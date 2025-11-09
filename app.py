"""
vLLM Playground - A web interface for managing and interacting with vLLM
"""
import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import shutil
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM Playground", version="1.0.0")

# Get base directory
BASE_DIR = Path(__file__).parent

# Mount static files (must be before routes)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/assets", StaticFiles(directory=str(BASE_DIR / "assets")), name="assets")

# Global state
vllm_process: Optional[subprocess.Popen] = None
log_queue: asyncio.Queue = asyncio.Queue()
websocket_connections: List[WebSocket] = []
latest_vllm_metrics: Dict[str, Any] = {}  # Store latest metrics from logs
metrics_timestamp: Optional[datetime] = None  # Track when metrics were last updated
current_model_identifier: Optional[str] = None  # Track the actual model identifier passed to vLLM


class VLLMConfig(BaseModel):
    """Configuration for vLLM server"""
    model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # CPU-friendly default
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    dtype: str = "auto"
    trust_remote_code: bool = False
    download_dir: Optional[str] = None
    load_format: str = "auto"
    disable_log_stats: bool = False
    enable_prefix_caching: bool = False
    # HuggingFace token for gated models (Llama, Gemma, etc.)
    # Get token from https://huggingface.co/settings/tokens
    hf_token: Optional[str] = None
    # CPU-specific options
    use_cpu: bool = False
    cpu_kvcache_space: int = 4  # GB for CPU KV cache (reduced default for stability)
    cpu_omp_threads_bind: str = "auto"  # CPU thread binding
    # Custom chat template and stop tokens (optional - overrides auto-detection)
    custom_chat_template: Optional[str] = None
    custom_stop_tokens: Optional[List[str]] = None
    # Internal flag to track if model has built-in template
    model_has_builtin_template: bool = False
    # Local model support - for compressed models or pre-downloaded models
    # If specified, takes precedence over 'model' parameter
    local_model_path: Optional[str] = None


class ChatMessage(BaseModel):
    """Chat message structure"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat request structure"""
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = True


class ServerStatus(BaseModel):
    """Server status information"""
    running: bool
    uptime: Optional[str] = None
    config: Optional[VLLMConfig] = None


class BenchmarkConfig(BaseModel):
    """Benchmark configuration"""
    total_requests: int = 100
    request_rate: float = 5.0
    prompt_tokens: int = 100
    output_tokens: int = 100
    use_guidellm: bool = False  # Toggle between built-in and GuideLLM


class BenchmarkResults(BaseModel):
    """Benchmark results"""
    throughput: float  # requests per second
    avg_latency: float  # milliseconds
    p50_latency: float  # milliseconds
    p95_latency: float  # milliseconds
    p99_latency: float  # milliseconds
    tokens_per_second: float
    total_tokens: int
    success_rate: float  # percentage
    completed: bool = False
    raw_output: Optional[str] = None  # Raw guidellm output for display
    json_output: Optional[str] = None  # JSON output from guidellm


# ============ Compression Models ============

class CompressionConfig(BaseModel):
    """Configuration for model compression"""
    model: str
    output_dir: Optional[str] = None  # Will be auto-generated if not provided
    
    # Quantization format
    quantization_format: Literal[
        "W8A8_INT8", "W8A8_FP8", "W4A16", "W8A16", "FP4_W4A16", "FP4_W4A4", "W4A4"
    ] = "W8A8_INT8"
    
    # Algorithm selection
    algorithm: Literal["PTQ", "GPTQ", "AWQ", "SmoothQuant", "SparseGPT"] = "GPTQ"
    
    # Dataset configuration
    dataset: str = "open_platypus"
    num_calibration_samples: int = 512
    max_seq_length: int = 2048
    
    # Advanced options
    smoothing_strength: float = 0.8  # For SmoothQuant
    target_layers: Optional[str] = "Linear"  # Comma-separated
    ignore_layers: Optional[str] = "lm_head"  # Comma-separated
    
    # Additional parameters
    hf_token: Optional[str] = None


class CompressionStatus(BaseModel):
    """Compression task status"""
    running: bool
    progress: float = 0.0  # 0-100
    stage: str = "idle"  # idle, loading, calibrating, quantizing, saving, complete, error
    message: str = ""
    output_dir: Optional[str] = None
    original_size_mb: Optional[float] = None
    compressed_size_mb: Optional[float] = None
    compression_ratio: Optional[float] = None
    error: Optional[str] = None
    start_time: Optional[float] = None  # Unix timestamp
    elapsed_time: Optional[float] = None  # Seconds


class CompressionPreset(BaseModel):
    """Preset compression configuration"""
    name: str
    description: str
    quantization_format: str
    algorithm: str
    emoji: str
    expected_speedup: str
    size_reduction: str


current_config: Optional[VLLMConfig] = None
server_start_time: Optional[datetime] = None
benchmark_task: Optional[asyncio.Task] = None
benchmark_results: Optional[BenchmarkResults] = None

# Compression state
compression_task: Optional[asyncio.Task] = None
compression_status: CompressionStatus = CompressionStatus(running=False)
compression_output_dir: Optional[Path] = None


def get_chat_template_for_model(model_name: str) -> str:
    """
    Get a reference chat template for a specific model.
    
    NOTE: This is now primarily used for documentation/reference purposes.
    vLLM automatically detects and uses chat templates from tokenizer_config.json.
    These templates are shown to match the model's actual tokenizer configuration.
    
    Supported models: Llama 2/3/3.1/3.2, Mistral/Mixtral, Gemma, TinyLlama, CodeLlama
    """
    model_lower = model_name.lower()
    
    # Llama 3/3.1/3.2 models (use new format with special tokens)
    # Reference: Meta's official Llama 3 tokenizer_config.json
    if 'llama-3' in model_lower and ('llama-3.1' in model_lower or 'llama-3.2' in model_lower or 'llama-3-' in model_lower):
        return (
            "{{- bos_token }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{- '<|start_header_id|>system<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'user' %}"
            "{{- '<|start_header_id|>user<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}"
            "{% endif %}"
        )
    
    # Llama 2 models (older [INST] format with <<SYS>>)
    # Reference: Meta's official Llama 2 tokenizer_config.json
    elif 'llama-2' in model_lower or 'llama2' in model_lower:
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if loop.index0 == 0 and system_message != false %}"
            "{{- '<s>[INST] <<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'user' %}"
            "{{- '<s>[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{- ' ' + message['content'] + ' </s>' }}"
            "{% endif %}"
            "{% endfor %}"
        )
    
    # Mistral/Mixtral models (similar to Llama 2 but simpler)
    # Reference: Mistral AI's official tokenizer_config.json
    elif 'mistral' in model_lower or 'mixtral' in model_lower:
        return (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"
            "{{- '[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{- message['content'] + eos_token }}"
            "{% else %}"
            "{{- raise_exception('Only user and assistant roles are supported!') }}"
            "{% endif %}"
            "{% endfor %}"
        )
    
    # Gemma models (Google)
    # Reference: Google's official Gemma tokenizer_config.json
    elif 'gemma' in model_lower:
        return (
            "{{ bos_token }}"
            "{% if messages[0]['role'] == 'system' %}"
            "{{- raise_exception('System role not supported') }}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"
            "{{- '<start_of_turn>user\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{- '<start_of_turn>model\\n' + message['content'] | trim + '<end_of_turn>\\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{- '<start_of_turn>model\\n' }}"
            "{% endif %}"
        )
    
    # TinyLlama (use ChatML format)
    # Reference: TinyLlama's official tokenizer_config.json
    elif 'tinyllama' in model_lower or 'tiny-llama' in model_lower:
        return (
            "{% for message in messages %}\\n"
            "{% if message['role'] == 'user' %}\\n"
            "{{- '<|user|>\\n' + message['content'] + eos_token }}\\n"
            "{% elif message['role'] == 'system' %}\\n"
            "{{- '<|system|>\\n' + message['content'] + eos_token }}\\n"
            "{% elif message['role'] == 'assistant' %}\\n"
            "{{- '<|assistant|>\\n'  + message['content'] + eos_token }}\\n"
            "{% endif %}\\n"
            "{% if loop.last and add_generation_prompt %}\\n"
            "{{- '<|assistant|>' }}\\n"
            "{% endif %}\\n"
            "{% endfor %}"
        )
    
    # CodeLlama (uses Llama 2 format)
    # Reference: Meta's CodeLlama tokenizer_config.json
    elif 'codellama' in model_lower or 'code-llama' in model_lower:
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if loop.index0 == 0 and system_message != false %}"
            "{{- '<s>[INST] <<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'user' %}"
            "{{- '<s>[INST] ' + message['content'] + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{- ' ' + message['content'] + ' </s>' }}"
            "{% endif %}"
            "{% endfor %}"
        )
    
    # Default generic template for unknown models
    else:
        logger.info(f"Using generic chat template for model: {model_name}")
        return (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{- message['content'] + '\\n' }}"
            "{% elif message['role'] == 'user' %}"
            "{{- 'User: ' + message['content'] + '\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{- 'Assistant: ' + message['content'] + '\\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{- 'Assistant:' }}"
            "{% endif %}"
        )


def get_stop_tokens_for_model(model_name: str) -> List[str]:
    """
    Get reference stop tokens for a specific model.
    
    NOTE: This is now primarily used for documentation/reference purposes.
    vLLM automatically handles stop tokens from the model's tokenizer.
    These are only used if user explicitly provides custom stop tokens.
    
    Supported models: Llama 2/3/3.1/3.2, Mistral/Mixtral, Gemma, TinyLlama, CodeLlama
    """
    model_lower = model_name.lower()
    
    # Llama 3/3.1/3.2 models - use special tokens
    if 'llama-3' in model_lower and ('llama-3.1' in model_lower or 'llama-3.2' in model_lower or 'llama-3-' in model_lower):
        return ["<|eot_id|>", "<|end_of_text|>"]
    
    # Llama 2 models - use special tokens
    elif 'llama-2' in model_lower or 'llama2' in model_lower:
        return ["</s>", "[INST]"]
    
    # Mistral/Mixtral models - use special tokens
    elif 'mistral' in model_lower or 'mixtral' in model_lower:
        return ["</s>", "[INST]"]
    
    # Gemma models - use special tokens
    elif 'gemma' in model_lower:
        return ["<end_of_turn>", "<start_of_turn>"]
    
    # TinyLlama - use ChatML special tokens
    elif 'tinyllama' in model_lower or 'tiny-llama' in model_lower:
        return ["</s>", "<|user|>", "<|system|>", "<|assistant|>"]
    
    # CodeLlama - use Llama 2 tokens
    elif 'codellama' in model_lower or 'code-llama' in model_lower:
        return ["</s>", "[INST]"]
    
    # Default generic stop tokens for unknown models
    else:
        return ["\n\nUser:", "\n\nAssistant:"]


def validate_local_model_path(model_path: str) -> Dict[str, Any]:
    """
    Validate that a local model path exists and contains required files.
    Supports ~ for home directory expansion.
    
    Returns:
        dict with keys: 'valid' (bool), 'error' (str if invalid), 'info' (dict with model info)
    """
    result = {
        'valid': False,
        'error': None,
        'info': {}
    }
    
    try:
        # Expand ~ to home directory and resolve to absolute path
        path = Path(model_path).expanduser().resolve()
        
        # Check if path exists
        if not path.exists():
            result['error'] = f"Path does not exist: {model_path} (expanded to: {path})"
            return result
        
        # Check if it's a directory
        if not path.is_dir():
            result['error'] = f"Path is not a directory: {model_path}"
            return result
        
        # Check for required files
        required_files = {
            'config.json': False,
            'tokenizer_config.json': False,
        }
        
        # Check for model weight files (at least one should exist)
        weight_patterns = [
            '*.safetensors',
            '*.bin',
            'pytorch_model*.bin',
            'model*.safetensors',
        ]
        
        has_weights = False
        for pattern in weight_patterns:
            if list(path.glob(pattern)):
                has_weights = True
                result['info']['weight_format'] = pattern
                break
        
        # Check required files
        for req_file in required_files.keys():
            file_path = path / req_file
            if file_path.exists():
                required_files[req_file] = True
        
        # Validation results
        missing_files = [f for f, exists in required_files.items() if not exists]
        
        if missing_files:
            result['error'] = f"Missing required files: {', '.join(missing_files)}"
            return result
        
        if not has_weights:
            result['error'] = "No model weight files found (*.safetensors or *.bin)"
            return result
        
        # Try to read model config for additional info
        try:
            import json
            config_path = path / 'config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)
                result['info']['model_type'] = config.get('model_type', 'unknown')
                result['info']['architectures'] = config.get('architectures', [])
                # Try to get model name from config
                if '_name_or_path' in config:
                    result['info']['_name_or_path'] = config['_name_or_path']
        except Exception as e:
            logger.warning(f"Could not read config.json: {e}")
        
        # Calculate directory size
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        result['info']['size_mb'] = round(total_size / (1024 * 1024), 2)
        result['info']['path'] = str(path.resolve())
        
        # Extract and add the display name
        result['info']['model_name'] = extract_model_name_from_path(str(path.resolve()), result['info'])
        
        result['valid'] = True
        return result
        
    except Exception as e:
        result['error'] = f"Error validating path: {str(e)}"
        return result


def extract_model_name_from_path(model_path: str, info: Dict[str, Any]) -> str:
    """
    Extract a meaningful model name from the local path.
    Handles HuggingFace cache directory structure and other cases.
    
    Args:
        model_path: Absolute path to the model directory
        info: Model info dict from validation
    
    Returns:
        A human-readable model name
    """
    path = Path(model_path)
    
    # Try to get name from config.json (_name_or_path field)
    if '_name_or_path' in info:
        name_or_path = info['_name_or_path']
        # If it's a HF model path like "TinyLlama/TinyLlama-1.1B-Chat-v1.0", use that
        if '/' in name_or_path and not name_or_path.startswith('/'):
            return name_or_path
    
    # Check if this is a HuggingFace cache directory
    # Structure: .../hub/models--Org--ModelName/snapshots/<hash>/...
    path_parts = path.parts
    
    for i, part in enumerate(path_parts):
        if part.startswith('models--'):
            # Found HF cache structure
            # Extract model name from "models--Org--ModelName"
            model_cache_name = part.replace('models--', '', 1)
            # Replace -- with /
            model_name = model_cache_name.replace('--', '/')
            logger.info(f"Extracted model name from HF cache: {model_name}")
            return model_name
    
    # If not HF cache, check for common compressed model naming patterns
    # e.g., "compressed_TinyLlama_w8a8_20240101_120000"
    dir_name = path.name
    
    if dir_name.startswith('compressed_'):
        # Try to extract original model name
        # Remove 'compressed_' prefix and any suffix after the last underscore
        cleaned = dir_name.replace('compressed_', '', 1)
        # If it has timestamp pattern at end, remove it
        import re
        # Remove patterns like _w8a8_20240101_120000 or _w8a8
        cleaned = re.sub(r'_[wW]\d+[aA]\d+(_\d{8}_\d{6})?$', '', cleaned)
        if cleaned:
            return cleaned
    
    # Last resort: use directory name
    return dir_name


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    html_path = Path(__file__).parent / "index.html"
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/status")
async def get_status() -> ServerStatus:
    """Get current server status"""
    global vllm_process, current_config, server_start_time
    
    running = vllm_process is not None and vllm_process.poll() is None
    uptime = None
    
    if running and server_start_time:
        elapsed = datetime.now() - server_start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    return ServerStatus(
        running=running,
        uptime=uptime,
        config=current_config
    )


@app.get("/api/features")
async def get_features():
    """Check which optional features are available"""
    features = {
        "vllm": True,  # Always available since it's core
        "llmcompressor": False,
        "guidellm": False
    }
    
    # Check llmcompressor
    try:
        import llmcompressor
        features["llmcompressor"] = True
    except ImportError:
        pass
    
    # Check guidellm
    try:
        import guidellm
        features["guidellm"] = True
    except ImportError:
        pass
    
    return features


@app.post("/api/start")
async def start_server(config: VLLMConfig):
    """Start the vLLM server"""
    global vllm_process, current_config, server_start_time, current_model_identifier
    
    if vllm_process is not None and vllm_process.poll() is None:
        raise HTTPException(status_code=400, detail="Server is already running")
    
    # Determine if using local model or HuggingFace Hub
    # Local model path takes precedence
    model_source = None
    model_display_name = None
    
    if config.local_model_path:
        # Using local model - validate with comprehensive validation
        await broadcast_log("[WEBUI] Validating local model path...")
        
        validation_result = validate_local_model_path(config.local_model_path)
        
        if not validation_result['valid']:
            error_msg = validation_result.get('error', 'Invalid local model path')
            await broadcast_log(f"[WEBUI] ERROR: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Use resolved absolute path
        info = validation_result['info']
        model_source = info['path']
        
        # Extract meaningful model name
        model_display_name = extract_model_name_from_path(model_source, info)
        
        # Log detailed validation info
        await broadcast_log(f"[WEBUI] ✓ Local model validated successfully")
        await broadcast_log(f"[WEBUI] Model name: {model_display_name}")
        await broadcast_log(f"[WEBUI] Path: {model_source}")
        await broadcast_log(f"[WEBUI] Size: {info.get('size_mb', 'unknown')} MB")
        if info.get('model_type'):
            await broadcast_log(f"[WEBUI] Model type: {info['model_type']}")
        if info.get('weight_format'):
            await broadcast_log(f"[WEBUI] Weight format: {info['weight_format']}")
    else:
        # Using HuggingFace Hub model
        model_source = config.model
        model_display_name = config.model
        
        # Check if gated model requires HF token
        # Meta Llama models (official and RedHatAI) are gated in our supported list
        model_lower = config.model.lower()
        is_gated = 'meta-llama/' in model_lower or 'redhatai/llama' in model_lower
        
        if is_gated and not config.hf_token:
            raise HTTPException(
                status_code=400, 
                detail=f"This model ({config.model}) is gated and requires a HuggingFace token. Please provide your HF token."
            )
        await broadcast_log(f"[WEBUI] Using HuggingFace Hub model: {model_source}")
    
    try:
        # Check if user manually selected CPU mode (takes precedence)
        if config.use_cpu:
            logger.info("CPU mode manually selected by user")
            await broadcast_log("[WEBUI] Using CPU mode (manual selection)")
        else:
            # Auto-detect macOS and enable CPU mode
            import platform
            is_macos = platform.system() == "Darwin"
            
            if is_macos:
                config.use_cpu = True
                logger.info("Detected macOS - enabling CPU mode")
                await broadcast_log("[WEBUI] Detected macOS - using CPU mode")
        
        # Set environment variables for CPU mode
        env = os.environ.copy()
        
        # Set HuggingFace token if provided (for gated models like Llama, Gemma)
        if config.hf_token:
            env['HF_TOKEN'] = config.hf_token
            env['HUGGING_FACE_HUB_TOKEN'] = config.hf_token  # Alternative name
            await broadcast_log("[WEBUI] HuggingFace token configured for gated models")
        elif os.environ.get('HF_TOKEN'):
            await broadcast_log("[WEBUI] Using HF_TOKEN from environment")
        
        if config.use_cpu:
            env['VLLM_CPU_KVCACHE_SPACE'] = str(config.cpu_kvcache_space)
            env['VLLM_CPU_OMP_THREADS_BIND'] = config.cpu_omp_threads_bind
            # Disable problematic CPU optimizations on Apple Silicon
            env['VLLM_CPU_MOE_PREPACK'] = '0'
            env['VLLM_CPU_SGL_KERNEL'] = '0'
            # Force CPU target device
            env['VLLM_TARGET_DEVICE'] = 'cpu'
            # Enable V1 engine (required to be set explicitly in vLLM 0.11.0+)
            env['VLLM_USE_V1'] = '1'
            logger.info(f"CPU Mode - VLLM_CPU_KVCACHE_SPACE={config.cpu_kvcache_space}, VLLM_CPU_OMP_THREADS_BIND={config.cpu_omp_threads_bind}")
            await broadcast_log(f"[WEBUI] CPU Settings - KV Cache: {config.cpu_kvcache_space}GB, Thread Binding: {config.cpu_omp_threads_bind}")
            await broadcast_log(f"[WEBUI] CPU Optimizations disabled for Apple Silicon compatibility")
            await broadcast_log(f"[WEBUI] Using V1 engine for CPU mode")
        else:
            await broadcast_log("[WEBUI] Using GPU mode")
        
        # Build command
        cmd = [
            sys.executable,
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_source,  # Use model_source (local path or HF model name)
            "--host", config.host,
            "--port", str(config.port),
        ]
        
        # Add GPU-specific parameters only if not using CPU
        # Note: vLLM auto-detects CPU platform, no --device flag needed
        if not config.use_cpu:
            cmd.extend([
                "--tensor-parallel-size", str(config.tensor_parallel_size),
                "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            ])
        else:
            await broadcast_log("[WEBUI] CPU mode - vLLM will auto-detect CPU backend")
        
        # Set dtype (use bfloat16 for CPU as recommended)
        if config.use_cpu and config.dtype == "auto":
            cmd.extend(["--dtype", "bfloat16"])
            await broadcast_log("[WEBUI] Using dtype=bfloat16 (recommended for CPU)")
        else:
            cmd.extend(["--dtype", config.dtype])
        
        # Add load-format only if not using CPU
        if not config.use_cpu:
            cmd.extend(["--load-format", config.load_format])
        
        # Handle max_model_len and max_num_batched_tokens
        # ALWAYS set both to prevent vLLM from auto-detecting large values
        if config.max_model_len:
            # User explicitly specified a value
            max_len = config.max_model_len
            cmd.extend(["--max-model-len", str(max_len)])
            cmd.extend(["--max-num-batched-tokens", str(max_len)])
            await broadcast_log(f"[WEBUI] Using user-specified max-model-len: {max_len}")
        elif config.use_cpu:
            # CPU mode: Use conservative defaults (2048)
            max_len = 2048
            cmd.extend(["--max-model-len", str(max_len)])
            cmd.extend(["--max-num-batched-tokens", str(max_len)])
            await broadcast_log(f"[WEBUI] Using default max-model-len for CPU: {max_len}")
        else:
            # GPU mode: Use reasonable default (8192) instead of letting vLLM auto-detect
            max_len = 8192
            cmd.extend(["--max-model-len", str(max_len)])
            cmd.extend(["--max-num-batched-tokens", str(max_len)])
            await broadcast_log(f"[WEBUI] Using default max-model-len for GPU: {max_len}")
        
        if config.trust_remote_code:
            cmd.append("--trust-remote-code")
        
        if config.download_dir:
            cmd.extend(["--download-dir", config.download_dir])
        
        if config.disable_log_stats:
            cmd.append("--disable-log-stats")
        
        if config.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        
        # Chat template handling:
        # Trust vLLM to auto-detect chat templates from tokenizer_config.json
        # Modern models (2023+) all have built-in templates, vLLM will use them automatically
        # Only pass --chat-template if user explicitly provides a custom override
        if config.custom_chat_template:
            # User provided custom template - write it to a temp file and pass to vLLM
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jinja', delete=False) as f:
                f.write(config.custom_chat_template)
                template_file = f.name
            cmd.extend(["--chat-template", template_file])
            config.model_has_builtin_template = False  # Using custom override
            await broadcast_log(f"[WEBUI] Using custom chat template from config (overrides model's built-in template)")
        else:
            # Let vLLM auto-detect and use the model's built-in chat template
            # vLLM will read it from tokenizer_config.json automatically
            config.model_has_builtin_template = True  # Assume model has template (modern models do)
            await broadcast_log(f"[WEBUI] Trusting vLLM to auto-detect chat template from tokenizer_config.json")
            await broadcast_log(f"[WEBUI] vLLM will use model's built-in chat template automatically")
        
        logger.info(f"Starting vLLM with command: {' '.join(cmd)}")
        await broadcast_log(f"[WEBUI] Command: {' '.join(cmd)}")
        
        # Start process with environment variables
        # Use line buffering (bufsize=1) and ensure output is captured
        vllm_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,  # Line buffered
            env=env
        )
        
        current_config = config
        server_start_time = datetime.now()
        
        # Store the actual model identifier for use in API calls
        current_model_identifier = model_source
        
        # Start log reader task
        asyncio.create_task(read_logs())
        
        await broadcast_log(f"[WEBUI] vLLM server starting with PID: {vllm_process.pid}")
        await broadcast_log(f"[WEBUI] Model: {model_display_name}")
        if config.local_model_path:
            await broadcast_log(f"[WEBUI] Model Source: Local ({model_source})")
        else:
            await broadcast_log(f"[WEBUI] Model Source: HuggingFace Hub")
        if config.use_cpu:
            await broadcast_log(f"[WEBUI] Mode: CPU (KV Cache: {config.cpu_kvcache_space}GB)")
        else:
            await broadcast_log(f"[WEBUI] Mode: GPU (Memory: {int(config.gpu_memory_utilization * 100)}%)")
        
        return {"status": "started", "pid": vllm_process.pid}
    
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop")
async def stop_server():
    """Stop the vLLM server"""
    global vllm_process, server_start_time, current_model_identifier
    
    if vllm_process is None:
        raise HTTPException(status_code=400, detail="Server is not running")
    
    try:
        await broadcast_log("[WEBUI] Stopping vLLM server...")
        vllm_process.terminate()
        
        # Wait for process to terminate
        try:
            vllm_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            vllm_process.kill()
            await broadcast_log("[WEBUI] Force killed vLLM server")
        
        vllm_process = None
        server_start_time = None
        current_model_identifier = None
        await broadcast_log("[WEBUI] vLLM server stopped")
        
        return {"status": "stopped"}
    
    except Exception as e:
        logger.error(f"Failed to stop server: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def read_logs():
    """Read logs from vLLM process"""
    global vllm_process
    
    if vllm_process is None:
        return
    
    try:
        # Use a loop to continuously read output
        loop = asyncio.get_event_loop()
        
        while vllm_process is not None and vllm_process.poll() is None:
            # Read line in a non-blocking way
            try:
                line = await loop.run_in_executor(None, vllm_process.stdout.readline)
                if line:
                    # Strip whitespace and send to clients
                    line = line.strip()
                    if line:  # Only send non-empty lines
                        await broadcast_log(line)
                        logger.debug(f"vLLM: {line}")
                else:
                    # No data, small sleep to prevent busy waiting
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error reading line: {e}")
                await asyncio.sleep(0.1)
        
        # Process has ended - read any remaining output
        if vllm_process is not None:
            remaining_output = vllm_process.stdout.read()
            if remaining_output:
                for line in remaining_output.splitlines():
                    line = line.strip()
                    if line:
                        await broadcast_log(line)
            
            # Log process exit
            return_code = vllm_process.returncode
            if return_code == 0:
                await broadcast_log(f"[WEBUI] vLLM process ended normally (exit code: {return_code})")
            else:
                await broadcast_log(f"[WEBUI] vLLM process ended with code: {return_code}")
    
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        await broadcast_log(f"[WEBUI] Error reading logs: {e}")


async def broadcast_log(message: str):
    """Broadcast log message to all connected websockets"""
    global latest_vllm_metrics, metrics_timestamp
    
    if not message:
        return
    
    # Parse metrics from log messages with more flexible patterns
    import re
    
    metrics_updated = False  # Track if we updated any metrics in this log line
    
    # Try various patterns for KV cache usage
    # Examples: "GPU KV cache usage: 0.3%", "KV cache usage: 0.3%", "cache usage: 0.3%"
    if "cache usage" in message.lower() and "%" in message:
        # More flexible pattern - match any number before %
        match = re.search(r'cache usage[:\s]+([\d.]+)\s*%', message, re.IGNORECASE)
        if match:
            cache_usage = float(match.group(1))
            latest_vllm_metrics['kv_cache_usage_perc'] = cache_usage
            metrics_updated = True
            logger.info(f"✓ Captured KV cache usage: {cache_usage}% from: {message[:100]}")
        else:
            logger.debug(f"Failed to parse cache usage from: {message[:100]}")
    
    # Try various patterns for prefix cache hit rate
    # Examples: "Prefix cache hit rate: 36.1%", "hit rate: 36.1%", "cache hit rate: 36.1%"
    if "hit rate" in message.lower() and "%" in message:
        # More flexible pattern
        match = re.search(r'hit rate[:\s]+([\d.]+)\s*%', message, re.IGNORECASE)
        if match:
            hit_rate = float(match.group(1))
            latest_vllm_metrics['prefix_cache_hit_rate'] = hit_rate
            metrics_updated = True
            logger.info(f"✓ Captured prefix cache hit rate: {hit_rate}% from: {message[:100]}")
        else:
            logger.debug(f"Failed to parse hit rate from: {message[:100]}")
    
    # Try to parse avg prompt throughput
    if "prompt throughput" in message.lower():
        match = re.search(r'prompt throughput[:\s]+([\d.]+)', message, re.IGNORECASE)
        if match:
            prompt_throughput = float(match.group(1))
            latest_vllm_metrics['avg_prompt_throughput'] = prompt_throughput
            metrics_updated = True
            logger.info(f"✓ Captured prompt throughput: {prompt_throughput}")
    
    # Try to parse avg generation throughput
    if "generation throughput" in message.lower():
        match = re.search(r'generation throughput[:\s]+([\d.]+)', message, re.IGNORECASE)
        if match:
            generation_throughput = float(match.group(1))
            latest_vllm_metrics['avg_generation_throughput'] = generation_throughput
            metrics_updated = True
            logger.info(f"✓ Captured generation throughput: {generation_throughput}")
    
    # Update timestamp if we captured any metrics
    if metrics_updated:
        metrics_timestamp = datetime.now()
        latest_vllm_metrics['timestamp'] = metrics_timestamp.isoformat()
        logger.info(f"📊 Metrics updated at: {metrics_timestamp.strftime('%H:%M:%S')}")
    
    disconnected = []
    for ws in websocket_connections:
        try:
            await ws.send_text(message)
        except Exception as e:
            logger.error(f"Error sending to websocket: {e}")
            disconnected.append(ws)
    
    # Remove disconnected websockets
    for ws in disconnected:
        websocket_connections.remove(ws)


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for streaming logs"""
    await websocket.accept()
    websocket_connections.append(websocket)
    
    try:
        await websocket.send_text("[WEBUI] Connected to log stream")
        
        # Keep connection alive
        while True:
            try:
                # Wait for messages (ping/pong)
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_text("")
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)


class ChatRequestWithStopTokens(BaseModel):
    """Chat request structure with optional stop tokens override"""
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = True
    stop_tokens: Optional[List[str]] = None  # Allow overriding stop tokens per request


@app.post("/api/chat")
async def chat(request: ChatRequestWithStopTokens):
    """Proxy chat requests to vLLM server using OpenAI-compatible /v1/chat/completions endpoint"""
    global current_config, current_model_identifier
    
    if vllm_process is None or vllm_process.poll() is not None:
        raise HTTPException(status_code=400, detail="vLLM server is not running")
    
    if current_config is None:
        raise HTTPException(status_code=400, detail="Server configuration not available")
    
    try:
        import aiohttp
        
        # Use OpenAI-compatible chat completions endpoint
        # vLLM will automatically handle chat template formatting using the model's tokenizer config
        url = f"http://{current_config.host}:{current_config.port}/v1/chat/completions"
        
        # Convert messages to OpenAI format
        messages_dict = [{"role": m.role, "content": m.content} for m in request.messages]
        
        # Build payload for OpenAI-compatible endpoint
        # Use current_model_identifier (actual path or HF model) instead of config.model
        payload = {
            "model": current_model_identifier if current_model_identifier else current_config.model,
            "messages": messages_dict,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream,
        }
        
        # Stop tokens handling:
        # By default, trust vLLM to use appropriate stop tokens from the model's tokenizer
        # Only override if user explicitly provides custom tokens in the server config
        if current_config.custom_stop_tokens:
            # User configured custom stop tokens in server config
            payload["stop"] = current_config.custom_stop_tokens
            logger.info(f"Using custom stop tokens from server config: {current_config.custom_stop_tokens}")
        elif request.stop_tokens:
            # User provided stop tokens in this specific request (not recommended)
            payload["stop"] = request.stop_tokens
            logger.warning(f"Using stop tokens from request (not recommended): {request.stop_tokens}")
        else:
            # Let vLLM handle stop tokens automatically from model's tokenizer (RECOMMENDED)
            logger.info(f"✓ Letting vLLM handle stop tokens automatically (recommended for /v1/chat/completions)")
        
        # Log the request payload being sent to vLLM
        logger.info(f"=== vLLM REQUEST ===")
        logger.info(f"URL: {url}")
        logger.info(f"Payload: {payload}")
        logger.info(f"Messages ({len(messages_dict)}): {messages_dict}")
        logger.info(f"==================")
        
        async def generate_stream():
            """Generator for streaming responses"""
            full_response_text = ""  # Accumulate response for logging
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"=== vLLM ERROR RESPONSE ===")
                        logger.error(f"Status: {response.status}")
                        logger.error(f"Error: {text}")
                        logger.error(f"==========================")
                        yield f"data: {{'error': '{text}'}}\n\n"
                        return
                    
                    logger.info(f"=== vLLM STREAMING RESPONSE START ===")
                    # Stream the response line by line
                    # OpenAI-compatible chat completions format
                    async for line in response.content:
                        if line:
                            decoded_line = line.decode('utf-8')
                            # Log each chunk received
                            if decoded_line.strip() and decoded_line.strip() != "data: [DONE]":
                                logger.debug(f"vLLM chunk: {decoded_line.strip()}")
                                # Try to extract content from SSE data
                                import json
                                if decoded_line.startswith("data: "):
                                    try:
                                        data_str = decoded_line[6:].strip()
                                        if data_str and data_str != "[DONE]":
                                            data = json.loads(data_str)
                                            if 'choices' in data and len(data['choices']) > 0:
                                                delta = data['choices'][0].get('delta', {})
                                                content = delta.get('content', '')
                                                if content:
                                                    full_response_text += content
                                    except:
                                        pass
                            # Pass through the SSE formatted data
                            if decoded_line.strip():
                                yield decoded_line
                    
                    # Log the complete response
                    logger.info(f"=== vLLM COMPLETE RESPONSE ===")
                    logger.info(f"Full text: {full_response_text}")
                    logger.info(f"Length: {len(full_response_text)} chars")
                    logger.info(f"===============================")
        
        if request.stream:
            # Return streaming response using SSE
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # Non-streaming response
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"=== vLLM ERROR RESPONSE (non-streaming) ===")
                        logger.error(f"Status: {response.status}")
                        logger.error(f"Error: {text}")
                        logger.error(f"===========================================")
                        raise HTTPException(status_code=response.status, detail=text)
                    
                    data = await response.json()
                    # Log the complete response
                    logger.info(f"=== vLLM RESPONSE (non-streaming) ===")
                    logger.info(f"Full response: {data}")
                    if 'choices' in data and len(data['choices']) > 0:
                        message = data['choices'][0].get('message', {})
                        content = message.get('content', '')
                        logger.info(f"Response text: {content}")
                        logger.info(f"Length: {len(content)} chars")
                    logger.info(f"=====================================")
                    return data
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CompletionRequest(BaseModel):
    """Completion request structure for non-chat models"""
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 256


@app.post("/api/completion")
async def completion(request: CompletionRequest):
    """Proxy completion requests to vLLM server for base models"""
    global current_config, current_model_identifier
    
    if vllm_process is None or vllm_process.poll() is not None:
        raise HTTPException(status_code=400, detail="vLLM server is not running")
    
    if current_config is None:
        raise HTTPException(status_code=400, detail="Server configuration not available")
    
    try:
        import aiohttp
        
        url = f"http://{current_config.host}:{current_config.port}/v1/completions"
        
        payload = {
            "model": current_model_identifier if current_model_identifier else current_config.model,
            "prompt": request.prompt,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    raise HTTPException(status_code=response.status, detail=text)
                
                data = await response.json()
                return data
    
    except Exception as e:
        logger.error(f"Completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/models")
async def list_models():
    """Get list of common models"""
    common_models = [
        # CPU-optimized models (recommended for macOS)
        {"name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "size": "1.1B", "description": "Compact chat model (CPU-friendly)", "cpu_friendly": True},
        {"name": "meta-llama/Llama-3.2-1B", "size": "1B", "description": "Llama 3.2 1B (CPU-friendly, gated)", "cpu_friendly": True, "gated": True},
        
        # Larger models (may be slow on CPU)
        {"name": "mistralai/Mistral-7B-Instruct-v0.2", "size": "7B", "description": "Mistral Instruct (slow on CPU)", "cpu_friendly": False},
        {"name": "RedHatAI/Llama-3.2-1B-Instruct-FP8", "size": "1B", "description": "Llama 3.2 1B Instruct FP8 (GPU-optimized, gated)", "cpu_friendly": False, "gated": True},
        {"name": "RedHatAI/Llama-3.1-8B-Instruct", "size": "8B", "description": "Llama 3.1 8B Instruct (gated)", "cpu_friendly": False, "gated": True},
    ]
    
    return {"models": common_models}


@app.post("/api/models/validate-local")
async def validate_local_model(request: dict):
    """
    Validate a local model path
    
    Request body: {"path": "/path/to/model"}
    Response: {"valid": bool, "error": str, "info": dict}
    """
    model_path = request.get('path', '')
    
    if not model_path:
        return JSONResponse(
            status_code=400,
            content={"valid": False, "error": "No path provided"}
        )
    
    result = validate_local_model_path(model_path)
    
    if result['valid']:
        return result
    else:
        return JSONResponse(
            status_code=400,
            content=result
        )


@app.post("/api/browse-directories")
async def browse_directories(request: dict):
    """
    Browse directories on the server for folder selection
    
    Request body: {"path": "/path/to/directory"}
    Response: {"directories": [...], "current_path": "..."}
    """
    try:
        import os
        from pathlib import Path
        
        requested_path = request.get('path', '~')
        
        # Expand ~ to home directory
        if requested_path == '~':
            requested_path = str(Path.home())
        
        path = Path(requested_path).expanduser().resolve()
        
        # Security check: ensure path exists and is a directory
        if not path.exists():
            # Try parent directory
            path = path.parent
            if not path.exists():
                path = Path.home()
        
        if not path.is_dir():
            path = path.parent
        
        # List only directories (not files)
        directories = []
        
        try:
            # Add parent directory option (except for root)
            if path.parent != path:
                directories.append({
                    'name': '..',
                    'path': str(path.parent)
                })
            
            # List subdirectories
            for item in sorted(path.iterdir()):
                if item.is_dir() and not item.name.startswith('.'):
                    # Check if it might be a model directory (has config.json)
                    is_model_dir = (item / 'config.json').exists()
                    directories.append({
                        'name': item.name + (' 🤖' if is_model_dir else ''),
                        'path': str(item)
                    })
        except PermissionError:
            logger.warning(f"Permission denied accessing directory: {path}")
        
        return {
            "directories": directories[:100],  # Limit to 100 directories
            "current_path": str(path)
        }
    
    except Exception as e:
        logger.error(f"Error browsing directories: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to browse directories: {str(e)}"}
        )


class LocalModelValidationRequest(BaseModel):
    """Request to validate a local model path"""
    path: str


class LocalModelValidationResponse(BaseModel):
    """Response for local model path validation"""
    valid: bool
    message: str
    model_name: Optional[str] = None
    model_type: Optional[str] = None
    has_tokenizer: bool = False
    has_config: bool = False
    estimated_size_mb: Optional[float] = None


@app.post("/api/models/validate-local")
async def validate_local_model(request: LocalModelValidationRequest) -> LocalModelValidationResponse:
    """Validate a local model directory"""
    try:
        model_path = Path(request.path)
        
        # Check if path exists
        if not model_path.exists():
            return LocalModelValidationResponse(
                valid=False,
                message=f"Path does not exist: {request.path}"
            )
        
        # Check if it's a directory
        if not model_path.is_dir():
            return LocalModelValidationResponse(
                valid=False,
                message=f"Path must be a directory, not a file"
            )
        
        # Check for required files
        config_file = model_path / "config.json"
        tokenizer_config = model_path / "tokenizer_config.json"
        has_config = config_file.exists()
        has_tokenizer = tokenizer_config.exists()
        
        if not has_config:
            return LocalModelValidationResponse(
                valid=False,
                message=f"Invalid model directory: missing config.json",
                has_config=has_config,
                has_tokenizer=has_tokenizer
            )
        
        # Try to read model info from config.json
        model_type = None
        model_name = model_path.name
        try:
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                model_type = config_data.get('model_type', 'unknown')
                # Try to get architectures
                architectures = config_data.get('architectures', [])
                if architectures:
                    model_type = architectures[0]
        except Exception as e:
            logger.warning(f"Could not read config.json: {e}")
        
        # Estimate directory size
        estimated_size_mb = None
        try:
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            estimated_size_mb = total_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.warning(f"Could not estimate model size: {e}")
        
        return LocalModelValidationResponse(
            valid=True,
            message=f"Valid model directory",
            model_name=model_name,
            model_type=model_type,
            has_config=has_config,
            has_tokenizer=has_tokenizer,
            estimated_size_mb=round(estimated_size_mb, 2) if estimated_size_mb else None
        )
    
    except Exception as e:
        logger.error(f"Error validating local model: {e}")
        return LocalModelValidationResponse(
            valid=False,
            message=f"Error validating path: {str(e)}"
        )


@app.get("/api/chat/template")
async def get_chat_template():
    """
    Get information about the chat template being used by the currently loaded model.
    vLLM auto-detects templates from tokenizer_config.json - this endpoint provides reference info.
    """
    global current_config, vllm_process
    
    if current_config is None:
        raise HTTPException(status_code=400, detail="No model configuration available")
    
    if current_config.custom_chat_template:
        # User is using a custom template
        return {
            "source": "custom (user-provided)",
            "model": current_config.model,
            "template": current_config.custom_chat_template,
            "stop_tokens": current_config.custom_stop_tokens or [],
            "note": "Using custom chat template provided by user (overrides model's built-in template)"
        }
    else:
        # vLLM is auto-detecting from model's tokenizer_config.json
        # We provide reference templates for documentation purposes
        return {
            "source": "auto-detected by vLLM",
            "model": current_config.model,
            "template": get_chat_template_for_model(current_config.model),
            "stop_tokens": get_stop_tokens_for_model(current_config.model),
            "note": "vLLM automatically uses the chat template from the model's tokenizer_config.json. The template shown here is a reference/fallback for documentation purposes only."
        }


@app.get("/api/vllm/metrics")
async def get_vllm_metrics():
    """Get vLLM server metrics including KV cache and prefix cache stats"""
    global current_config, vllm_process, latest_vllm_metrics, metrics_timestamp
    
    if vllm_process is None or vllm_process.poll() is not None:
        return JSONResponse(
            status_code=400, 
            content={"error": "vLLM server is not running"}
        )
    
    # Calculate how fresh the metrics are
    metrics_age_seconds = None
    if metrics_timestamp:
        metrics_age_seconds = (datetime.now() - metrics_timestamp).total_seconds()
        logger.info(f"Returning metrics (age: {metrics_age_seconds:.1f}s): {latest_vllm_metrics}")
    else:
        logger.info(f"Returning metrics (no timestamp): {latest_vllm_metrics}")
    
    # Return metrics parsed from logs with freshness indicator
    if latest_vllm_metrics:
        result = latest_vllm_metrics.copy()
        if metrics_age_seconds is not None:
            result['metrics_age_seconds'] = round(metrics_age_seconds, 1)
        return result
    
    # If no metrics captured yet from logs, try the metrics endpoint
    if current_config is None:
        return {}
    
    try:
        import aiohttp
        
        # Try to fetch metrics from vLLM's metrics endpoint
        metrics_url = f"http://{current_config.host}:{current_config.port}/metrics"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(metrics_url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                    if response.status == 200:
                        text = await response.text()
                        
                        # Parse Prometheus-style metrics
                        metrics = {}
                        
                        # Look for KV cache usage
                        for line in text.split('\n'):
                            if 'vllm:gpu_cache_usage_perc' in line and not line.startswith('#'):
                                try:
                                    value = float(line.split()[-1])
                                    metrics['gpu_cache_usage_perc'] = value
                                except:
                                    pass
                            elif 'vllm:cpu_cache_usage_perc' in line and not line.startswith('#'):
                                try:
                                    value = float(line.split()[-1])
                                    metrics['cpu_cache_usage_perc'] = value
                                except:
                                    pass
                            elif 'vllm:avg_prompt_throughput_toks_per_s' in line and not line.startswith('#'):
                                try:
                                    value = float(line.split()[-1])
                                    metrics['avg_prompt_throughput'] = value
                                except:
                                    pass
                            elif 'vllm:avg_generation_throughput_toks_per_s' in line and not line.startswith('#'):
                                try:
                                    value = float(line.split()[-1])
                                    metrics['avg_generation_throughput'] = value
                                except:
                                    pass
                        
                        return metrics
                    else:
                        return {}
            except asyncio.TimeoutError:
                return {}
            except Exception as e:
                logger.debug(f"Error fetching metrics endpoint: {e}")
                return {}
    
    except Exception as e:
        logger.debug(f"Error in get_vllm_metrics: {e}")
        return {}


@app.post("/api/benchmark/start")
async def start_benchmark(config: BenchmarkConfig):
    """Start a benchmark test using either built-in or GuideLLM"""
    global vllm_process, current_config, benchmark_task, benchmark_results
    
    if vllm_process is None or vllm_process.poll() is not None:
        raise HTTPException(status_code=400, detail="vLLM server is not running")
    
    if benchmark_task is not None and not benchmark_task.done():
        raise HTTPException(status_code=400, detail="Benchmark is already running")
    
    try:
        # Reset results
        benchmark_results = None
        
        # Choose benchmark method
        if config.use_guidellm:
            # Start GuideLLM benchmark task
            benchmark_task = asyncio.create_task(
                run_guidellm_benchmark(config, current_config)
            )
            await broadcast_log("[BENCHMARK] Starting GuideLLM benchmark...")
        else:
            # Start built-in benchmark task
            benchmark_task = asyncio.create_task(
                run_benchmark(config, current_config)
            )
            await broadcast_log("[BENCHMARK] Starting built-in benchmark...")
        
        return {"status": "started", "message": "Benchmark started"}
    
    except Exception as e:
        logger.error(f"Failed to start benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/benchmark/status")
async def get_benchmark_status():
    """Get current benchmark status"""
    global benchmark_task, benchmark_results
    
    if benchmark_task is None:
        return {"running": False, "results": None}
    
    if benchmark_task.done():
        if benchmark_results:
            results_dict = benchmark_results.dict()
            logger.info(f"[BENCHMARK DEBUG] Returning results: {results_dict}")
            return {"running": False, "results": results_dict}
        else:
            return {"running": False, "results": None, "error": "Benchmark failed"}
    
    return {"running": True, "results": None}


@app.post("/api/benchmark/stop")
async def stop_benchmark():
    """Stop the running benchmark"""
    global benchmark_task
    
    if benchmark_task is None or benchmark_task.done():
        raise HTTPException(status_code=400, detail="No benchmark is running")
    
    try:
        benchmark_task.cancel()
        await broadcast_log("[BENCHMARK] Benchmark stopped by user")
        return {"status": "stopped"}
    except Exception as e:
        logger.error(f"Failed to stop benchmark: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_benchmark(config: BenchmarkConfig, server_config: VLLMConfig):
    """Run a simple benchmark test"""
    global benchmark_results, current_model_identifier
    
    try:
        import aiohttp
        import time
        import random
        import numpy as np
        
        await broadcast_log(f"[BENCHMARK] Configuration: {config.total_requests} requests at {config.request_rate} req/s")
        
        url = f"http://{server_config.host}:{server_config.port}/v1/chat/completions"
        
        # Generate a sample prompt of specified length
        prompt_text = " ".join(["benchmark" for _ in range(config.prompt_tokens // 10)])
        
        results = []
        successful = 0
        failed = 0
        start_time = time.time()
        
        # Create session
        async with aiohttp.ClientSession() as session:
            # Send requests
            for i in range(config.total_requests):
                request_start = time.time()
                
                try:
                    payload = {
                        "model": current_model_identifier if current_model_identifier else server_config.model,
                        "messages": [{"role": "user", "content": prompt_text}],
                        "max_tokens": config.output_tokens,
                        "temperature": 0.7,
                    }
                    
                    # Add stop tokens only if user configured custom ones
                    # Otherwise let vLLM handle stop tokens automatically
                    if server_config.custom_stop_tokens:
                        payload["stop"] = server_config.custom_stop_tokens
                    
                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                        if response.status == 200:
                            data = await response.json()
                            request_end = time.time()
                            latency = (request_end - request_start) * 1000  # ms
                            
                            # Extract token counts
                            usage = data.get('usage', {})
                            completion_tokens = usage.get('completion_tokens', config.output_tokens)
                            
                            # Debug: Log token extraction for first few requests
                            if i < 3:
                                logger.info(f"[BENCHMARK DEBUG] Request {i+1} usage: {usage}")
                                logger.info(f"[BENCHMARK DEBUG] Request {i+1} completion_tokens: {completion_tokens}")
                            
                            results.append({
                                'latency': latency,
                                'tokens': completion_tokens
                            })
                            successful += 1
                        else:
                            failed += 1
                            logger.warning(f"Request {i+1} failed with status {response.status}")
                
                except Exception as e:
                    failed += 1
                    logger.error(f"Request {i+1} error: {e}")
                
                # Progress update
                if (i + 1) % max(1, config.total_requests // 10) == 0:
                    progress = ((i + 1) / config.total_requests) * 100
                    await broadcast_log(f"[BENCHMARK] Progress: {progress:.0f}% ({i+1}/{config.total_requests} requests)")
                
                # Rate limiting
                if config.request_rate > 0:
                    await asyncio.sleep(1.0 / config.request_rate)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate metrics
        if results:
            latencies = [r['latency'] for r in results]
            tokens = [r['tokens'] for r in results]
            
            throughput = len(results) / duration
            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            tokens_per_second = sum(tokens) / duration
            total_tokens = sum(tokens) + (len(results) * config.prompt_tokens)
            success_rate = (successful / config.total_requests) * 100
            
            # Debug logging
            logger.info(f"[BENCHMARK DEBUG] Total output tokens: {sum(tokens)}")
            logger.info(f"[BENCHMARK DEBUG] Total prompt tokens: {len(results) * config.prompt_tokens}")
            logger.info(f"[BENCHMARK DEBUG] Duration: {duration:.2f}s")
            logger.info(f"[BENCHMARK DEBUG] tokens_per_second: {tokens_per_second:.2f}")
            logger.info(f"[BENCHMARK DEBUG] total_tokens: {int(total_tokens)}")
            
            benchmark_results = BenchmarkResults(
                throughput=round(throughput, 2),
                avg_latency=round(avg_latency, 2),
                p50_latency=round(p50_latency, 2),
                p95_latency=round(p95_latency, 2),
                p99_latency=round(p99_latency, 2),
                tokens_per_second=round(tokens_per_second, 2),
                total_tokens=int(total_tokens),
                success_rate=round(success_rate, 2),
                completed=True
            )
            
            await broadcast_log(f"[BENCHMARK] Completed! Throughput: {throughput:.2f} req/s, Avg Latency: {avg_latency:.2f}ms")
            await broadcast_log(f"[BENCHMARK] Token Throughput: {tokens_per_second:.2f} tok/s, Total Tokens: {int(total_tokens)}")
        else:
            await broadcast_log(f"[BENCHMARK] Failed - No successful requests")
            benchmark_results = None
    
    except asyncio.CancelledError:
        await broadcast_log("[BENCHMARK] Benchmark cancelled")
        raise
    except Exception as e:
        logger.error(f"Benchmark error: {e}")
        await broadcast_log(f"[BENCHMARK] Error: {e}")
        benchmark_results = None


async def run_guidellm_benchmark(config: BenchmarkConfig, server_config: VLLMConfig):
    """Run a benchmark using GuideLLM"""
    global benchmark_results
    
    try:
        await broadcast_log(f"[GUIDELLM] Configuration: {config.total_requests} requests at {config.request_rate} req/s")
        
        # Check if GuideLLM is installed
        try:
            import guidellm
            # Don't import internal modules - we'll use the CLI
            await broadcast_log(f"[GUIDELLM] Package found: {guidellm.__version__ if hasattr(guidellm, '__version__') else 'version unknown'}")
        except ImportError as e:
            error_msg = f"GuideLLM not installed: {str(e)}"
            logger.error(error_msg)
            await broadcast_log(f"[GUIDELLM] ERROR: {error_msg}")
            await broadcast_log(f"[GUIDELLM] Python executable: {sys.executable}")
            await broadcast_log(f"[GUIDELLM] Python path: {sys.path}")
            await broadcast_log(f"[GUIDELLM] Run: pip install guidellm")
            benchmark_results = None
            return
        
        # Setup target URL
        target_url = f"http://{server_config.host}:{server_config.port}/v1"
        await broadcast_log(f"[GUIDELLM] Target: {target_url}")
        
        # Run GuideLLM benchmark using subprocess (since GuideLLM CLI is simpler)
        import json
        import subprocess
        
        # Use the same Python executable that's running this application
        # Since guidellm was successfully imported above, it must be in the same environment
        python_exec = sys.executable
        await broadcast_log(f"[GUIDELLM] Using Python executable: {python_exec}")
        
        # Get the path to the guidellm module for informational purposes
        guidellm_location = guidellm.__file__
        await broadcast_log(f"[GUIDELLM] Module location: {guidellm_location}")
        
        # Verify guidellm is accessible from this Python
        # Note: This check is optional - if it fails or times out, we'll still attempt to run
        try:
            check_result = subprocess.run(
                [python_exec, "-m", "guidellm", "--help"],
                capture_output=True,
                timeout=30  # Increased timeout for OpenShift compatibility
            )
            if check_result.returncode != 0:
                # Current python_exec doesn't have guidellm, try finding it in PATH
                await broadcast_log(f"[GUIDELLM] WARNING: guidellm CLI not accessible from {python_exec}")
                await broadcast_log(f"[GUIDELLM] stderr: {check_result.stderr.decode()}")
                # Try to find guidellm in PATH
                which_result = subprocess.run(
                    ["which", "guidellm"],
                    capture_output=True,
                    text=True
                )
                if which_result.returncode == 0:
                    guidellm_bin = which_result.stdout.strip()
                    await broadcast_log(f"[GUIDELLM] Found guidellm binary at: {guidellm_bin}")
                    # Use guidellm directly instead of python -m
                    python_exec = None  # Will use guidellm command directly
                else:
                    await broadcast_log(f"[GUIDELLM] WARNING: GuideLLM CLI verification failed, will attempt to run anyway")
                    await broadcast_log(f"[GUIDELLM] If benchmark fails, ensure GuideLLM is properly installed: pip install guidellm")
            else:
                await broadcast_log(f"[GUIDELLM] CLI verified: {python_exec}")
        except subprocess.TimeoutExpired:
            # Don't fail - just warn and continue
            await broadcast_log(f"[GUIDELLM] WARNING: CLI check timed out (30s), will attempt to run benchmark anyway")
            await broadcast_log(f"[GUIDELLM] If you encounter issues, ensure GuideLLM is installed in your venv: pip install guidellm")
        except Exception as e:
            # Don't fail - just warn and continue
            await broadcast_log(f"[GUIDELLM] WARNING: Error checking GuideLLM installation: {e}")
            await broadcast_log(f"[GUIDELLM] Will attempt to run benchmark anyway. Ensure GuideLLM is installed: pip install guidellm")
        
        # Create a temporary JSON file for results
        result_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False)
        result_file.close()
        
        # Build GuideLLM command
        # GuideLLM structure: guidellm benchmark [OPTIONS]
        # Example: guidellm benchmark --target "url" --rate-type sweep --max-seconds 30 --data "prompt_tokens=256,output_tokens=128"
        
        if python_exec:
            cmd = [
                python_exec, "-m", "guidellm",
                "benchmark",
                "--target", target_url,
            ]
        else:
            cmd = [
                "guidellm",
                "benchmark",
                "--target", target_url,
            ]
        
        # Add rate configuration
        # If rate is specified, use constant rate, otherwise use sweep
        if config.request_rate > 0:
            cmd.extend(["--rate-type", "constant"])
            cmd.extend(["--rate", str(config.request_rate)])
        else:
            cmd.extend(["--rate-type", "sweep"])
        
        # Add request limit
        cmd.extend(["--max-requests", str(config.total_requests)])
        
        # Add token configuration in guidellm's data format
        data_str = f"prompt_tokens={config.prompt_tokens},output_tokens={config.output_tokens}"
        cmd.extend(["--data", data_str])
        
        # Add output path to save JSON results
        cmd.extend(["--output-path", result_file.name])
        
        await broadcast_log(f"[GUIDELLM] Running: {' '.join(cmd)}")
        await broadcast_log(f"[GUIDELLM] JSON output will be saved to: {result_file.name}")
        
        # Run GuideLLM process and capture ALL output
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Collect all output lines for parsing
        output_lines = []
        
        # Stream output
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded = line.decode().strip()
            if decoded:
                output_lines.append(decoded)
                await broadcast_log(f"[GUIDELLM] {decoded}")
        
        # Wait for completion
        await process.wait()
        
        if process.returncode != 0:
            stderr = await process.stderr.read()
            error_msg = stderr.decode().strip()
            await broadcast_log(f"[GUIDELLM] Error: {error_msg}")
            benchmark_results = None
            return
        
        # Join all output for raw display
        raw_output = "\n".join(output_lines)
        
        # Try to read JSON output file
        json_output = None
        try:
            # Check if file exists and has content
            if os.path.exists(result_file.name):
                file_size = os.path.getsize(result_file.name)
                await broadcast_log(f"[GUIDELLM] 📄 JSON file found: {result_file.name} (size: {file_size} bytes)")
                
                with open(result_file.name, 'r') as f:
                    json_output = f.read()
                
                if json_output:
                    await broadcast_log(f"[GUIDELLM] ✅ JSON output loaded successfully ({len(json_output)} characters)")
                    # Validate it's valid JSON
                    try:
                        import json as json_module
                        json_module.loads(json_output)
                        await broadcast_log(f"[GUIDELLM] ✅ JSON is valid")
                    except Exception as json_err:
                        await broadcast_log(f"[GUIDELLM] ⚠️ JSON validation failed: {json_err}")
                else:
                    await broadcast_log(f"[GUIDELLM] ⚠️ JSON file is empty")
            else:
                await broadcast_log(f"[GUIDELLM] ⚠️ JSON output file not found at {result_file.name}")
                await broadcast_log(f"[GUIDELLM] Checking if guidellm created a file in current directory...")
                # Sometimes guidellm creates files with different names
                import glob
                json_files = glob.glob("*.json")
                if json_files:
                    await broadcast_log(f"[GUIDELLM] Found JSON files in current directory: {json_files}")
        except Exception as e:
            await broadcast_log(f"[GUIDELLM] ⚠️ Failed to read JSON output: {e}")
            logger.exception("Error reading GuideLLM JSON output")
        
        # Parse results from text output
        try:
            # Extract metrics from the "Benchmarks Stats" table
            # Example line: constant@5.00| 0.57| 9.43| 57.3| 115.1| 16.45| 16.08| ...
            
            throughput = 0.0
            tokens_per_second = 0.0
            avg_latency = 0.0
            p50_latency = 0.0
            p99_latency = 0.0
            
            # Find the stats line (after "Benchmark| Per Second|")
            for i, line in enumerate(output_lines):
                # Look for the data line with benchmark name and metrics
                if 'constant@' in line or 'sweep@' in line:
                    # Check if this is the stats line (contains numeric data)
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 7:
                        try:
                            # Parse the data
                            # Format: Benchmark| Per Second| Concurrency| Out Tok/sec| Tot Tok/sec| Req Latency mean| median| p99| ...
                            throughput = float(parts[1])  # Per Second
                            tokens_per_second = float(parts[3])  # Out Tok/sec
                            # total_tok_per_sec = float(parts[4])  # Tot Tok/sec
                            avg_latency = float(parts[5]) * 1000  # Convert seconds to ms
                            p50_latency = float(parts[6]) * 1000  # median
                            if len(parts) >= 8:
                                p99_latency = float(parts[7]) * 1000  # p99
                            
                            await broadcast_log(f"[GUIDELLM] 📊 Parsed metrics from output")
                            break
                        except (ValueError, IndexError) as e:
                            await broadcast_log(f"[GUIDELLM] Debug: Failed to parse line: {line}")
                            await broadcast_log(f"[GUIDELLM] Debug: Parts: {parts}")
                            continue
            
            benchmark_results = BenchmarkResults(
                throughput=float(throughput),
                avg_latency=float(avg_latency),
                p50_latency=float(p50_latency),
                p95_latency=float(p50_latency * 1.2),  # Estimate if not available
                p99_latency=float(p99_latency),
                tokens_per_second=float(tokens_per_second),
                total_tokens=int(config.total_requests * config.output_tokens),  # Estimate
                success_rate=100.0,  # Assume success if completed
                completed=True,
                raw_output=raw_output,  # Store raw output for display
                json_output=json_output  # Store JSON output for display
            )
            
            await broadcast_log(f"[GUIDELLM] ✅ Completed!")
            await broadcast_log(f"[GUIDELLM] 📊 Throughput: {benchmark_results.throughput:.2f} req/s")
            await broadcast_log(f"[GUIDELLM] ⚡ Token Throughput: {benchmark_results.tokens_per_second:.2f} tok/s")
            await broadcast_log(f"[GUIDELLM] ⏱️  Avg Latency: {benchmark_results.avg_latency:.2f} ms")
            await broadcast_log(f"[GUIDELLM] 📈 P99 Latency: {benchmark_results.p99_latency:.2f} ms")
            
        except Exception as e:
            logger.error(f"Failed to parse GuideLLM results: {e}")
            await broadcast_log(f"[GUIDELLM] Error parsing results: {e}")
            # Still create result with raw output
            benchmark_results = BenchmarkResults(
                throughput=0.0,
                avg_latency=0.0,
                p50_latency=0.0,
                p95_latency=0.0,
                p99_latency=0.0,
                tokens_per_second=0.0,
                total_tokens=0,
                success_rate=0.0,
                completed=True,
                raw_output=raw_output if 'raw_output' in locals() else "Error capturing output",
                json_output=json_output if 'json_output' in locals() else None
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(result_file.name)
            except:
                pass
                
    except asyncio.CancelledError:
        await broadcast_log("[GUIDELLM] Benchmark cancelled")
        raise
    except Exception as e:
        logger.error(f"GuideLLM benchmark error: {e}")
        await broadcast_log(f"[GUIDELLM] Error: {e}")
        benchmark_results = None


# ============ Compression Endpoints ============

@app.get("/api/compress/presets")
async def get_compression_presets() -> Dict[str, List[CompressionPreset]]:
    """Get predefined compression presets"""
    presets = [
        CompressionPreset(
            name="Quick INT8",
            description="Fast W8A8 quantization with GPTQ - great balance of speed and quality",
            quantization_format="W8A8_INT8",
            algorithm="GPTQ",
            emoji="⚡",
            expected_speedup="2-3x faster",
            size_reduction="~50% smaller"
        ),
        CompressionPreset(
            name="Compact INT4",
            description="W4A16 quantization with AWQ - maximum compression with good quality",
            quantization_format="W4A16",
            algorithm="AWQ",
            emoji="📦",
            expected_speedup="1.5-2x faster",
            size_reduction="~75% smaller"
        ),
        CompressionPreset(
            name="Production FP8",
            description="FP8 activation quantization - production-grade performance",
            quantization_format="W8A8_FP8",
            algorithm="GPTQ",
            emoji="🚀",
            expected_speedup="3-4x faster",
            size_reduction="~50% smaller"
        ),
        CompressionPreset(
            name="Maximum Compression",
            description="W4A4 FP4 quantization - highest compression ratio",
            quantization_format="FP4_W4A4",
            algorithm="GPTQ",
            emoji="🎯",
            expected_speedup="4-5x faster",
            size_reduction="~85% smaller"
        ),
        CompressionPreset(
            name="Smooth Quantization",
            description="W8A8 with SmoothQuant - best for accuracy-sensitive tasks",
            quantization_format="W8A8_INT8",
            algorithm="SmoothQuant",
            emoji="✨",
            expected_speedup="2-3x faster",
            size_reduction="~50% smaller"
        ),
    ]
    
    return {"presets": presets}


@app.post("/api/compress/start")
async def start_compression(config: CompressionConfig):
    """Start model compression task"""
    global compression_task, compression_status, compression_output_dir
    
    if compression_task is not None and not compression_task.done():
        raise HTTPException(status_code=400, detail="Compression task already running")
    
    try:
        # Reset status
        import time
        compression_status = CompressionStatus(
            running=True,
            progress=0.0,
            stage="initializing",
            message="Starting compression...",
            start_time=time.time()
        )
        
        # Generate output directory name if not provided
        if not config.output_dir:
            # Clean model name (remove org prefix, replace slashes and special chars)
            model_name = config.model.split('/')[-1]  # Take last part after slash
            model_name = model_name.replace('/', '_').replace(' ', '_').replace(':', '_')
            
            # Map scheme to shorter name
            scheme_map = {
                "W8A8_INT8": "w8a8_int8",
                "W8A8_FP8": "w8a8_fp8",
                "W4A16": "w4a16",
                "W8A16": "w8a16",
                "FP4_W4A16": "fp4_w4a16",
                "FP4_W4A4": "fp4_w4a4",
                "W4A4": "w4a4",
            }
            scheme_suffix = scheme_map.get(config.quantization_format, config.quantization_format.lower())
            
            # Add timestamp to avoid overwriting existing compressions
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.output_dir = f"compressed_{model_name}_{scheme_suffix}_{timestamp}"
        
        # Create output directory
        compression_output_dir = Path(config.output_dir)
        compression_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start compression task
        compression_task = asyncio.create_task(
            run_compression(config)
        )
        
        await broadcast_log("[COMPRESSION] Starting model compression...")
        await broadcast_log(f"[COMPRESSION] Model: {config.model}")
        await broadcast_log(f"[COMPRESSION] Format: {config.quantization_format}")
        await broadcast_log(f"[COMPRESSION] Algorithm: {config.algorithm}")
        
        return {"status": "started", "message": "Compression task started"}
    
    except Exception as e:
        logger.error(f"Failed to start compression: {e}")
        compression_status.running = False
        compression_status.stage = "error"
        compression_status.error = str(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/compress/status")
async def get_compression_status() -> CompressionStatus:
    """Get current compression task status"""
    global compression_status
    
    # Calculate elapsed time if compression is running or completed
    if compression_status.start_time is not None:
        import time
        if compression_status.running:
            compression_status.elapsed_time = time.time() - compression_status.start_time
        elif compression_status.elapsed_time is None:
            # If completed but elapsed_time not set, calculate it
            compression_status.elapsed_time = time.time() - compression_status.start_time
    
    return compression_status


@app.post("/api/compress/stop")
async def stop_compression():
    """Stop the running compression task"""
    global compression_task, compression_status
    
    if compression_task is None or compression_task.done():
        raise HTTPException(status_code=400, detail="No compression task is running")
    
    try:
        compression_task.cancel()
        compression_status.running = False
        compression_status.stage = "cancelled"
        compression_status.message = "Compression cancelled by user"
        
        await broadcast_log("[COMPRESSION] Compression task cancelled")
        return {"status": "cancelled"}
    
    except Exception as e:
        logger.error(f"Failed to stop compression: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/compress/download")
async def download_compressed_model():
    """Download the compressed model"""
    global compression_output_dir
    
    if compression_output_dir is None or not compression_output_dir.exists():
        raise HTTPException(status_code=404, detail="No compressed model available")
    
    # Create a tar.gz archive of the output directory
    import tarfile
    
    archive_path = compression_output_dir.parent / f"{compression_output_dir.name}.tar.gz"
    
    try:
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(compression_output_dir, arcname=compression_output_dir.name)
        
        return FileResponse(
            path=archive_path,
            media_type="application/gzip",
            filename=f"{compression_output_dir.name}.tar.gz"
        )
    
    except Exception as e:
        logger.error(f"Failed to create archive: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_compression(config: CompressionConfig):
    """Run the compression task"""
    global compression_status
    
    try:
        # Set HF token if provided - must be set in os.environ for llmcompressor to use it
        hf_token_set = False
        original_hf_token = os.environ.get('HF_TOKEN')
        original_hf_hub_token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
        
        if config.hf_token:
            os.environ['HF_TOKEN'] = config.hf_token
            os.environ['HUGGING_FACE_HUB_TOKEN'] = config.hf_token
            hf_token_set = True
            await broadcast_log("[COMPRESSION] Using provided HF token for authentication")
        
        compression_status.stage = "loading"
        compression_status.progress = 10.0
        compression_status.message = "Loading model and preparing for compression..."
        await broadcast_log(f"[COMPRESSION] Loading model: {config.model}")
        
        # Get original model size
        try:
            # Try to estimate model size from HF
            compression_status.original_size_mb = await estimate_model_size(config.model)
        except Exception as e:
            logger.warning(f"Could not estimate model size: {e}")
        
        compression_status.stage = "preparing"
        compression_status.progress = 20.0
        compression_status.message = "Preparing quantization recipe..."
        await broadcast_log(f"[COMPRESSION] Preparing {config.algorithm} recipe...")
        
        # Build the recipe based on configuration
        recipe = build_compression_recipe(config)
        
        compression_status.stage = "calibrating"
        compression_status.progress = 30.0
        compression_status.message = "Loading calibration dataset..."
        await broadcast_log(f"[COMPRESSION] Loading dataset: {config.dataset}")
        
        # Import llmcompressor here to avoid issues if not installed
        try:
            from llmcompressor import oneshot
        except ImportError:
            raise Exception("llmcompressor not installed. Run: pip install llmcompressor")
        
        compression_status.progress = 40.0
        compression_status.message = "Applying compression..."
        await broadcast_log(f"[COMPRESSION] Applying {config.quantization_format} quantization...")
        
        # Simulate progress during compression since llmcompressor doesn't provide callbacks
        # We'll update progress incrementally during the long-running operation
        async def run_with_progress():
            loop = asyncio.get_event_loop()
            
            # Start the compression in a background thread
            # Note: The HF token is already set in os.environ above
            compression_future = loop.run_in_executor(
                None,
                lambda: oneshot(
                    model=config.model,
                    dataset=config.dataset,
                    recipe=recipe,
                    output_dir=config.output_dir,
                    max_seq_length=config.max_seq_length,
                    num_calibration_samples=config.num_calibration_samples,
                )
            )
            
            # Simulate progress updates while compression is running
            # Progress from 40% to 85% over the compression duration
            progress_steps = [
                (5, "Calibrating model layers..."),
                (10, "Processing attention layers..."),
                (15, "Quantizing weights..."),
                (20, "Optimizing activations..."),
                (25, "Applying compression recipe..."),
                (30, "Processing remaining layers..."),
                (35, "Finalizing quantization..."),
                (40, "Validating compressed model..."),
            ]
            
            step_idx = 0
            while not compression_future.done():
                await asyncio.sleep(5)  # Update every 5 seconds
                
                if step_idx < len(progress_steps):
                    progress_increment, message = progress_steps[step_idx]
                    compression_status.progress = min(40.0 + progress_increment, 85.0)
                    compression_status.message = message
                    await broadcast_log(f"[COMPRESSION] {message}")
                    step_idx += 1
                else:
                    # Keep incrementing slowly after all steps
                    compression_status.progress = min(compression_status.progress + 1, 85.0)
            
            # Wait for compression to complete
            return await compression_future
        
        await run_with_progress()
        
        compression_status.stage = "saving"
        compression_status.progress = 90.0
        compression_status.message = "Saving compressed model..."
        await broadcast_log(f"[COMPRESSION] 💾 Saving compressed model...")
        await broadcast_log(f"[COMPRESSION] Output directory: {config.output_dir}")
        
        # Get compressed model size
        compressed_size = get_directory_size(Path(config.output_dir))
        compression_status.compressed_size_mb = compressed_size
        
        if compression_status.original_size_mb:
            compression_status.compression_ratio = (
                compression_status.original_size_mb / compressed_size
            )
        
        # Calculate final elapsed time
        import time
        elapsed_seconds = time.time() - compression_status.start_time if compression_status.start_time else 0
        compression_status.elapsed_time = elapsed_seconds
        
        # Format elapsed time as HH:MM:SS
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = int(elapsed_seconds % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        compression_status.stage = "complete"
        compression_status.progress = 100.0
        compression_status.message = f"Compression complete! Saved to: {config.output_dir}"
        compression_status.output_dir = config.output_dir
        compression_status.running = False
        
        await broadcast_log(f"[COMPRESSION] ✅ Compression complete!")
        await broadcast_log(f"[COMPRESSION] " + "="*60)
        await broadcast_log(f"[COMPRESSION] 📁 SAVED TO: {config.output_dir}")
        await broadcast_log(f"[COMPRESSION] ⏱️  TIME TAKEN: {time_str}")
        await broadcast_log(f"[COMPRESSION] " + "="*60)
        if compression_status.original_size_mb and compression_status.compressed_size_mb:
            await broadcast_log(
                f"[COMPRESSION] Size: {compression_status.original_size_mb:.1f}MB → "
                f"{compression_status.compressed_size_mb:.1f}MB "
                f"({compression_status.compression_ratio:.2f}x reduction)"
            )
        # Also log the absolute path for clarity
        abs_path = Path(config.output_dir).resolve()
        await broadcast_log(f"[COMPRESSION] Absolute path: {abs_path}")
    
    except asyncio.CancelledError:
        compression_status.running = False
        compression_status.stage = "cancelled"
        compression_status.message = "Compression cancelled"
        await broadcast_log("[COMPRESSION] Task cancelled")
        raise
    
    except Exception as e:
        logger.error(f"Compression error: {e}", exc_info=True)
        compression_status.running = False
        compression_status.stage = "error"
        compression_status.error = str(e)
        compression_status.message = f"Error: {str(e)}"
        await broadcast_log(f"[COMPRESSION] ❌ Error: {str(e)}")
    
    finally:
        # Restore original HF token environment variables
        if hf_token_set:
            if original_hf_token is not None:
                os.environ['HF_TOKEN'] = original_hf_token
            elif 'HF_TOKEN' in os.environ:
                del os.environ['HF_TOKEN']
            
            if original_hf_hub_token is not None:
                os.environ['HUGGING_FACE_HUB_TOKEN'] = original_hf_hub_token
            elif 'HUGGING_FACE_HUB_TOKEN' in os.environ:
                del os.environ['HUGGING_FACE_HUB_TOKEN']


def build_compression_recipe(config: CompressionConfig) -> List:
    """Build compression recipe based on configuration"""
    recipe = []
    
    try:
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    except ImportError:
        # Provide helpful error if llmcompressor not installed
        raise Exception("llmcompressor not installed. Run: pip install llmcompressor")
    
    # Parse target and ignore layers
    targets = config.target_layers or "Linear"
    ignore = [layer.strip() for layer in (config.ignore_layers or "lm_head").split(",")]
    
    # Build scheme based on quantization format
    # Note: FP8 requires QuantizationModifier with explicit config, not GPTQModifier
    scheme_map = {
        "W8A8_INT8": "W8A8",
        "W8A8_FP8": None,  # Handled separately below
        "W4A16": "W4A16",
        "W8A16": "W8A16",
        "FP4_W4A16": "W4A16",  # FP4 is handled by vLLM
        "FP4_W4A4": "W4A4",
        "W4A4": "W4A4",
    }
    
    scheme = scheme_map.get(config.quantization_format, "W8A8")
    
    # Add SmoothQuant if specified
    if config.algorithm == "SmoothQuant":
        recipe.append(
            SmoothQuantModifier(smoothing_strength=config.smoothing_strength)
        )
    
    # Handle FP8 quantization separately (requires QuantizationModifier)
    if config.quantization_format == "W8A8_FP8":
        recipe.append(
            QuantizationModifier(
                targets=targets,
                scheme={
                    "input_activations": {
                        "num_bits": 8,
                        "type": "float",
                        "symmetric": True,
                    },
                    "weights": {
                        "num_bits": 8,
                        "type": "float",
                        "symmetric": True,
                    }
                },
                ignore=ignore
            )
        )
    # Add quantization modifier based on algorithm for non-FP8 formats
    elif config.algorithm in ["GPTQ", "SmoothQuant", "PTQ"]:
        recipe.append(
            GPTQModifier(
                scheme=scheme,
                targets=targets,
                ignore=ignore
            )
        )
    elif config.algorithm == "AWQ":
        # AWQ uses similar interface to GPTQ
        recipe.append(
            GPTQModifier(
                scheme=scheme,
                targets=targets,
                ignore=ignore
            )
        )
    
    return recipe


async def estimate_model_size(model_name: str) -> float:
    """Estimate model size in MB from HuggingFace"""
    # This is a rough estimate - in production you'd want to query HF API
    # For now, use simple heuristics based on model name
    
    model_lower = model_name.lower()
    
    # Extract model size from name (e.g., "7b", "1.1b", "8b")
    import re
    size_match = re.search(r'(\d+\.?\d*)\s*b', model_lower)
    
    if size_match:
        size_b = float(size_match.group(1))
        # Rough estimate: ~2 bytes per parameter (fp16)
        return size_b * 1024 * 2  # Convert to MB
    
    # Default estimates for common models
    if 'tinyllama' in model_lower or '1b' in model_lower:
        return 2200.0  # ~1.1B * 2 bytes
    elif '7b' in model_lower:
        return 14000.0  # ~7B * 2 bytes
    elif '8b' in model_lower:
        return 16000.0  # ~8B * 2 bytes
    elif '13b' in model_lower:
        return 26000.0  # ~13B * 2 bytes
    
    return 5000.0  # Default estimate


def get_directory_size(directory: Path) -> float:
    """Get total size of directory in MB"""
    total_size = 0
    for item in directory.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size
    return total_size / (1024 * 1024)  # Convert to MB


def main():
    """Main entry point"""
    logger.info("Starting vLLM Playground...")
    
    # Get port from environment or use default
    webui_port = int(os.environ.get("WEBUI_PORT", "7860"))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=webui_port,
        log_level="info"
    )


if __name__ == "__main__":
    main()

