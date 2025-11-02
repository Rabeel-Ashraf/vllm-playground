"""
vLLM WebUI - A web interface for managing and interacting with vLLM
"""
import asyncio
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM WebUI", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global state
vllm_process: Optional[subprocess.Popen] = None
log_queue: asyncio.Queue = asyncio.Queue()
websocket_connections: List[WebSocket] = []


class VLLMConfig(BaseModel):
    """Configuration for vLLM server"""
    model: str = "facebook/opt-125m"
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
    # CPU-specific options
    use_cpu: bool = False
    cpu_kvcache_space: int = 40  # GB for CPU KV cache
    cpu_omp_threads_bind: str = "auto"  # CPU thread binding


class ChatMessage(BaseModel):
    """Chat message structure"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat request structure"""
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 512
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


current_config: Optional[VLLMConfig] = None
server_start_time: Optional[datetime] = None
benchmark_task: Optional[asyncio.Task] = None
benchmark_results: Optional[BenchmarkResults] = None


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


@app.post("/api/start")
async def start_server(config: VLLMConfig):
    """Start the vLLM server"""
    global vllm_process, current_config, server_start_time
    
    if vllm_process is not None and vllm_process.poll() is None:
        raise HTTPException(status_code=400, detail="Server is already running")
    
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
        if config.use_cpu:
            env['VLLM_CPU_KVCACHE_SPACE'] = str(config.cpu_kvcache_space)
            env['VLLM_CPU_OMP_THREADS_BIND'] = config.cpu_omp_threads_bind
            logger.info(f"CPU Mode - VLLM_CPU_KVCACHE_SPACE={config.cpu_kvcache_space}, VLLM_CPU_OMP_THREADS_BIND={config.cpu_omp_threads_bind}")
            await broadcast_log(f"[WEBUI] CPU Settings - KV Cache: {config.cpu_kvcache_space}GB, Thread Binding: {config.cpu_omp_threads_bind}")
        else:
            await broadcast_log("[WEBUI] Using GPU mode")
        
        # Build command
        cmd = [
            sys.executable,
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.model,
            "--host", config.host,
            "--port", str(config.port),
        ]
        
        # Add GPU-specific parameters only if not using CPU
        if not config.use_cpu:
            cmd.extend([
                "--tensor-parallel-size", str(config.tensor_parallel_size),
                "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            ])
        
        # Set dtype (use bfloat16 for CPU as recommended)
        if config.use_cpu and config.dtype == "auto":
            cmd.extend(["--dtype", "bfloat16"])
            await broadcast_log("[WEBUI] Using dtype=bfloat16 (recommended for CPU)")
        else:
            cmd.extend(["--dtype", config.dtype])
        
        # Add load-format only if not using CPU
        if not config.use_cpu:
            cmd.extend(["--load-format", config.load_format])
        
        if config.max_model_len:
            cmd.extend(["--max-model-len", str(config.max_model_len)])
            # Set max-num-batched-tokens to match max-model-len to avoid validation errors
            # This is especially important for CPU mode
            cmd.extend(["--max-num-batched-tokens", str(config.max_model_len)])
            await broadcast_log(f"[WEBUI] Using user-specified max-model-len: {config.max_model_len}")
        
        if config.trust_remote_code:
            cmd.append("--trust-remote-code")
        
        if config.download_dir:
            cmd.extend(["--download-dir", config.download_dir])
        
        if config.disable_log_stats:
            cmd.append("--disable-log-stats")
        
        if config.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        
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
        
        # Start log reader task
        asyncio.create_task(read_logs())
        
        await broadcast_log(f"[WEBUI] vLLM server starting with PID: {vllm_process.pid}")
        await broadcast_log(f"[WEBUI] Model: {config.model}")
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
    global vllm_process, server_start_time
    
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
        
        while vllm_process.poll() is None:
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
    if not message:
        return
    
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


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Proxy chat requests to vLLM server"""
    global current_config
    
    if vllm_process is None or vllm_process.poll() is not None:
        raise HTTPException(status_code=400, detail="vLLM server is not running")
    
    if current_config is None:
        raise HTTPException(status_code=400, detail="Server configuration not available")
    
    try:
        import aiohttp
        
        url = f"http://{current_config.host}:{current_config.port}/v1/chat/completions"
        
        payload = {
            "model": current_config.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream
        }
        
        async def generate_stream():
            """Generator for streaming responses"""
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        yield f"data: {{'error': '{text}'}}\n\n"
                        return
                    
                    # Stream the response line by line
                    async for line in response.content:
                        if line:
                            decoded_line = line.decode('utf-8')
                            # Pass through the SSE formatted data
                            if decoded_line.strip():
                                yield decoded_line
        
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
                        raise HTTPException(status_code=response.status, detail=text)
                    
                    data = await response.json()
                    return data
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CompletionRequest(BaseModel):
    """Completion request structure for non-chat models"""
    prompt: str
    temperature: float = 0.7
    max_tokens: int = 512


@app.post("/api/completion")
async def completion(request: CompletionRequest):
    """Proxy completion requests to vLLM server for base models"""
    global current_config
    
    if vllm_process is None or vllm_process.poll() is not None:
        raise HTTPException(status_code=400, detail="vLLM server is not running")
    
    if current_config is None:
        raise HTTPException(status_code=400, detail="Server configuration not available")
    
    try:
        import aiohttp
        
        url = f"http://{current_config.host}:{current_config.port}/v1/completions"
        
        payload = {
            "model": current_config.model,
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
        {"name": "facebook/opt-125m", "size": "125M", "description": "Small test model"},
        {"name": "facebook/opt-350m", "size": "350M", "description": "Small test model"},
        {"name": "facebook/opt-1.3b", "size": "1.3B", "description": "Medium model"},
        {"name": "facebook/opt-2.7b", "size": "2.7B", "description": "Medium model"},
        {"name": "meta-llama/Llama-2-7b-chat-hf", "size": "7B", "description": "Llama 2 Chat"},
        {"name": "meta-llama/Llama-2-13b-chat-hf", "size": "13B", "description": "Llama 2 Chat"},
        {"name": "mistralai/Mistral-7B-Instruct-v0.2", "size": "7B", "description": "Mistral Instruct"},
        {"name": "codellama/CodeLlama-7b-Instruct-hf", "size": "7B", "description": "Code Llama"},
    ]
    
    return {"models": common_models}


@app.post("/api/benchmark/start")
async def start_benchmark(config: BenchmarkConfig):
    """Start a benchmark test using simple load testing"""
    global vllm_process, current_config, benchmark_task, benchmark_results
    
    if vllm_process is None or vllm_process.poll() is not None:
        raise HTTPException(status_code=400, detail="vLLM server is not running")
    
    if benchmark_task is not None and not benchmark_task.done():
        raise HTTPException(status_code=400, detail="Benchmark is already running")
    
    try:
        # Reset results
        benchmark_results = None
        
        # Start benchmark task
        benchmark_task = asyncio.create_task(
            run_benchmark(config, current_config)
        )
        
        await broadcast_log("[BENCHMARK] Starting performance benchmark...")
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
            return {"running": False, "results": benchmark_results.dict()}
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
    global benchmark_results
    
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
                        "model": server_config.model,
                        "messages": [{"role": "user", "content": prompt_text}],
                        "max_tokens": config.output_tokens,
                        "temperature": 0.7
                    }
                    
                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                        if response.status == 200:
                            data = await response.json()
                            request_end = time.time()
                            latency = (request_end - request_start) * 1000  # ms
                            
                            # Extract token counts
                            usage = data.get('usage', {})
                            completion_tokens = usage.get('completion_tokens', config.output_tokens)
                            
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


def main():
    """Main entry point"""
    logger.info("Starting vLLM WebUI...")
    
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

