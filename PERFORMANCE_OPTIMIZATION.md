# Performance Optimization Guide for Streaming API

## Overview

This guide covers performance optimization strategies for the real-time hazard detection streaming API, focusing on latency reduction, throughput improvement, and resource efficiency.

## OpenVINO Model Optimizations

### 1. Model Selection & Configuration
```python
# Optimal settings in .env
OPENVINO_DEVICE=CPU  # or GPU if available
OPENVINO_PERFORMANCE_MODE=LATENCY  # For real-time streaming
OPENVINO_ASYNC_INFERENCE=true
OPENVINO_CACHE_ENABLED=true

# Model path (FP16 preferred for speed)
MODEL_PATH=/app/server/openvino_fp16/best.xml
```

### 2. Device-Specific Optimizations

#### CPU Optimization
- Use multi-threading: `CPU_THREADS_NUM=4`
- Enable thread binding: `CPU_BIND_THREAD=YES`
- Batch size: Keep at 1 for streaming
- Precision: FP16 for best speed/accuracy balance

#### GPU Optimization (if available)
- Enable GPU device: `OPENVINO_DEVICE=GPU`
- Use GPU-optimized models
- Consider batch processing for higher throughput

### 3. Model Preprocessing Optimization
```python
# In streaming_service.py - optimized preprocessing
def _preprocess_image_optimized(self, image: Image.Image):
    """Optimized image preprocessing for streaming"""
    # Skip unnecessary conversions
    if image.mode == "RGB" and image.size == (640, 640):
        # Direct processing for optimal case
        img_array = np.array(image, dtype=np.float32) / 255.0
        return np.transpose(img_array, (2, 0, 1))[np.newaxis, ...]
    
    # Standard preprocessing path
    return self._preprocess_image_standard(image)
```

## Network & Protocol Optimizations

### 1. WebSocket Optimizations
```javascript
// Client-side optimizations
const ws = new WebSocket('ws://api/stream/detection', [], {
    perMessageDeflate: true,  // Enable compression
    maxPayload: 1024 * 1024   // 1MB max payload
});

// Frame rate limiting on client
const targetFPS = 15;
const frameInterval = 1000 / targetFPS;
let lastFrameTime = 0;

function sendFrameIfReady(imageData) {
    const now = Date.now();
    if (now - lastFrameTime >= frameInterval) {
        sendFrame(imageData);
        lastFrameTime = now;
    }
}
```

### 2. SSE Optimizations
```python
# Server-side SSE optimization
@router.get("/stream/events/{client_id}")
async def streaming_events_endpoint(client_id: str, request: Request):
    return StreamingResponse(
        streaming_service.get_event_stream(client_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx optimization
            "Content-Encoding": "none"   # Disable compression for real-time
        }
    )
```

### 3. Image Data Optimization
```python
# Optimized image encoding
def optimize_image_for_streaming(image_path: str, quality: int = 85) -> str:
    """Optimize image for streaming with quality/size balance"""
    with Image.open(image_path) as img:
        # Resize if too large
        if img.width > 1280 or img.height > 1280:
            img.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Compress with optimal settings
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        
        return base64.b64encode(buffer.getvalue()).decode()
```

## Memory & Resource Management

### 1. Frame Queue Optimization
```python
# In streaming_service.py
class OptimizedFrameProcessor:
    def __init__(self, max_queue_size: int = 25):  # Reduced from 50
        self._processing_queue = asyncio.Queue(maxsize=max_queue_size)
        self._worker_pool = ThreadPoolExecutor(max_workers=4)
        
    async def process_with_backpressure(self, frame_data):
        """Process with intelligent backpressure"""
        if self._processing_queue.full():
            # Drop oldest frame to make space
            try:
                self._processing_queue.get_nowait()
                logger.debug("Dropped frame due to backpressure")
            except asyncio.QueueEmpty:
                pass
        
        await self._processing_queue.put(frame_data)
```

### 2. Memory Pool for Image Processing
```python
import numpy as np
from typing import Dict

class ImageBufferPool:
    """Reusable buffer pool for image processing"""
    
    def __init__(self, buffer_size: int = 10):
        self._buffers = {}
        self._available = {}
        self._buffer_size = buffer_size
    
    def get_buffer(self, shape: tuple) -> np.ndarray:
        """Get reusable buffer for image processing"""
        key = str(shape)
        
        if key not in self._buffers:
            self._buffers[key] = []
            self._available[key] = []
        
        if self._available[key]:
            return self._available[key].pop()
        
        if len(self._buffers[key]) < self._buffer_size:
            buffer = np.empty(shape, dtype=np.float32)
            self._buffers[key].append(buffer)
            return buffer
        
        # Reuse oldest buffer
        return self._buffers[key][0]
    
    def return_buffer(self, buffer: np.ndarray):
        """Return buffer to pool"""
        key = str(buffer.shape)
        if key in self._available:
            self._available[key].append(buffer)
```

### 3. Async Processing Pipeline
```python
class StreamingPipeline:
    """Optimized async processing pipeline"""
    
    def __init__(self):
        self.decode_pool = ThreadPoolExecutor(max_workers=2)
        self.inference_pool = ThreadPoolExecutor(max_workers=1)  # Model is thread-safe
        self.encode_pool = ThreadPoolExecutor(max_workers=2)
    
    async def process_frame_pipeline(self, frame_data: dict):
        """Optimized pipeline with parallel stages"""
        # Stage 1: Decode (CPU-bound)
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(
            self.decode_pool, 
            self._decode_frame, 
            frame_data['image_data']
        )
        
        # Stage 2: Inference (GPU/CPU-bound)
        detections = await loop.run_in_executor(
            self.inference_pool,
            model_service.predict_sync,
            image
        )
        
        # Stage 3: Post-process and encode (CPU-bound)
        result = await loop.run_in_executor(
            self.encode_pool,
            self._encode_result,
            detections,
            frame_data
        )
        
        return result
```

## Configuration Tuning

### 1. Environment Variables for Production
```bash
# Core performance settings
STREAMING_MAX_FPS=30
STREAMING_DEFAULT_FPS=15
STREAMING_QUEUE_SIZE=25
STREAMING_SESSION_TIMEOUT=300

# OpenVINO optimization
OPENVINO_DEVICE=CPU
OPENVINO_PERFORMANCE_MODE=LATENCY
OPENVINO_ASYNC_INFERENCE=true
OPENVINO_CACHE_ENABLED=true

# Detection thresholds (balance speed vs accuracy)
CONFIDENCE_THRESHOLD=0.5
IOU_THRESHOLD=0.45

# Thread pool settings
UVICORN_WORKERS=1  # Single worker for WebSocket state
```

### 2. Adaptive Quality Settings
```python
class AdaptiveQualityController:
    """Automatically adjust quality based on performance"""
    
    def __init__(self):
        self.target_latency = 100  # ms
        self.performance_history = deque(maxlen=50)
        
    def adjust_settings(self, session: StreamSession, processing_time_ms: float):
        """Adjust session settings based on performance"""
        self.performance_history.append(processing_time_ms)
        avg_latency = sum(self.performance_history) / len(self.performance_history)
        
        if avg_latency > self.target_latency * 1.5:
            # Reduce quality for better performance
            if session.config.fps_limit > 5:
                session.config.fps_limit = max(5, session.config.fps_limit - 2)
                logger.info(f"Reduced FPS to {session.config.fps_limit} for session {session.session_id}")
                
        elif avg_latency < self.target_latency * 0.7:
            # Increase quality if performance allows
            if session.config.fps_limit < 25:
                session.config.fps_limit = min(25, session.config.fps_limit + 1)
                logger.info(f"Increased FPS to {session.config.fps_limit} for session {session.session_id}")
```

## Monitoring & Profiling

### 1. Performance Metrics Collection
```python
import time
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class PerformanceMetrics:
    """Comprehensive performance tracking"""
    frame_decode_time: float = 0.0
    preprocessing_time: float = 0.0
    inference_time: float = 0.0
    postprocessing_time: float = 0.0
    total_time: float = 0.0
    queue_wait_time: float = 0.0

class PerformanceProfiler:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.bottlenecks = defaultdict(int)
    
    def profile_frame_processing(self, session_id: str):
        """Context manager for profiling frame processing"""
        return FrameProcessingProfiler(session_id, self)

class FrameProcessingProfiler:
    def __init__(self, session_id: str, profiler: PerformanceProfiler):
        self.session_id = session_id
        self.profiler = profiler
        self.metrics = PerformanceMetrics()
        self.stage_start = None
    
    def start_stage(self, stage_name: str):
        """Start timing a processing stage"""
        if self.stage_start:
            # End previous stage
            stage_time = time.time() - self.stage_start
            setattr(self.metrics, f"{self.current_stage}_time", stage_time)
        
        self.current_stage = stage_name
        self.stage_start = time.time()
    
    def __enter__(self):
        self.total_start = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metrics.total_time = time.time() - self.total_start
        self.profiler.metrics[self.session_id].append(self.metrics)
        
        # Identify bottlenecks
        max_stage_time = max([
            self.metrics.frame_decode_time,
            self.metrics.preprocessing_time,
            self.metrics.inference_time,
            self.metrics.postprocessing_time
        ])
        
        if max_stage_time == self.metrics.inference_time:
            self.profiler.bottlenecks['inference'] += 1
        elif max_stage_time == self.metrics.preprocessing_time:
            self.profiler.bottlenecks['preprocessing'] += 1
        # ... etc
```

### 2. Real-time Performance Dashboard
```python
@router.get("/stream/performance")
async def get_performance_metrics():
    """Get real-time performance metrics"""
    streaming_service = get_streaming_service()
    
    metrics = {
        "active_sessions": len(streaming_service.sessions),
        "total_throughput_fps": sum(s.metrics.fps for s in streaming_service.sessions.values()),
        "avg_processing_time_ms": 0,
        "bottlenecks": {},
        "system_resources": {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "gpu_usage": get_gpu_usage() if has_gpu() else None
        }
    }
    
    # Calculate average processing time
    all_times = []
    for session in streaming_service.sessions.values():
        all_times.extend(session.metrics.processing_times)
    
    if all_times:
        metrics["avg_processing_time_ms"] = sum(all_times) / len(all_times) * 1000
    
    return metrics
```

## Deployment Optimizations

### 1. Docker Configuration
```dockerfile
# Optimized Dockerfile for streaming API
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set optimal CPU flags
ENV OPENVINO_NUM_THREADS=4
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Optimize Python runtime
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run with optimized settings
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--loop", "uvloop"]
```

### 2. Nginx Configuration for Production
```nginx
upstream streaming_api {
    server app:8080;
    keepalive 32;
}

server {
    listen 80;
    
    # WebSocket proxy
    location /stream/detection {
        proxy_pass http://streaming_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }
    
    # SSE proxy
    location /stream/events/ {
        proxy_pass http://streaming_api;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 24h;
    }
    
    # Regular API endpoints
    location / {
        proxy_pass http://streaming_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## Expected Performance Benchmarks

### Optimal Configuration Results
- **Latency**: < 50ms end-to-end (local network)
- **Throughput**: 15-30 FPS per session
- **Concurrent Sessions**: 5-10 on moderate hardware
- **Memory Usage**: ~150MB per active session
- **CPU Usage**: ~25-40% per session (quad-core CPU)

### Hardware Recommendations
- **CPU**: Intel i5/i7 or AMD Ryzen 5/7 (4+ cores)
- **RAM**: 8GB+ (16GB recommended for multiple sessions)
- **Storage**: SSD for model loading
- **Network**: Gigabit Ethernet or high-quality WiFi
- **GPU**: Optional Intel GPU or dedicated GPU for acceleration

This optimization guide provides a comprehensive approach to maximizing the performance of your streaming detection API while maintaining accuracy and reliability.