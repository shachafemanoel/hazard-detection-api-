# Live Detection Streaming API Architecture

## Overview

This document outlines the comprehensive API architecture for hosting an OpenVINO model in a Python backend with live detection streaming capabilities for road hazard detection.

## Architecture Components

### 1. API Endpoint Design

#### WebSocket Endpoints
- **`/stream/detection`** - Real-time bidirectional streaming
  - Accepts video frames via WebSocket
  - Returns detection results in real-time
  - Supports configuration updates mid-stream
  - Optimized for lowest latency

#### REST + SSE Endpoints
- **`POST /stream/sessions`** - Create streaming session
- **`GET /stream/events/{client_id}`** - SSE endpoint for results
- **`POST /stream/process/{client_id}`** - Submit frames for processing
- **`GET /stream/sessions/{client_id}/stats`** - Performance metrics
- **`PATCH /stream/sessions/{client_id}/config`** - Update configuration
- **`DELETE /stream/sessions/{client_id}`** - Close session

#### Health & Monitoring
- **`GET /stream/health`** - Service health status
- **`GET /stream/stats`** - Global streaming statistics

### 2. WebSocket Protocol Design

#### Message Types

**Client to Server:**
```json
{
  "type": "frame",
  "image_data": "base64_encoded_image",
  "frame_id": "unique_frame_id",
  "timestamp": 1641234567.123,
  "metadata": {}
}
```

**Server to Client:**
```json
{
  "type": "detection_result",
  "data": {
    "frame_id": "unique_frame_id",
    "detections": [...],
    "processing_time_ms": 45.2,
    "timestamp": 1641234567.123
  }
}
```

#### Configuration Updates
```json
{
  "type": "config_update",
  "config": {
    "fps_limit": 15,
    "confidence_threshold": 0.6,
    "enable_tracking": true
  }
}
```

### 3. Frame Processing Pipeline

#### High-Level Flow
```
Frame Input → Rate Limiting → Queue → OpenVINO Inference → Post-processing → Result Output
```

#### Detailed Pipeline
1. **Frame Reception**
   - WebSocket or HTTP POST
   - Base64 decode validation
   - Rate limiting check

2. **Pre-processing**
   - Image format conversion
   - Resize to 640x640 (YOLOv12n input)
   - Normalization and tensor preparation

3. **Model Inference**
   - OpenVINO async inference
   - Thread pool execution for non-blocking
   - GPU/CPU optimization based on device

4. **Post-processing**
   - NMS (Non-Maximum Suppression)
   - Coordinate transformation
   - Confidence filtering
   - Object tracking (if enabled)

5. **Result Streaming**
   - WebSocket JSON response
   - SSE event broadcasting
   - Performance metrics collection

### 4. Client Connection Management

#### Session Lifecycle
```
Create Session → Configure → Process Frames → Monitor → Cleanup
```

#### Connection Types
- **WebSocket**: Bidirectional, lowest latency
- **SSE**: Unidirectional, HTTP-friendly

#### Session Management
- Unique client IDs
- Configurable timeouts (300s default)
- Automatic cleanup for expired sessions
- Connection health monitoring

### 5. Performance Optimization Strategies

#### Rate Limiting
- Configurable FPS limits (1-30 FPS)
- Frame dropping for overload scenarios
- Queue size management (50 frame limit)

#### Memory Optimization
- Image processing in thread pools
- Async/await for I/O operations
- Limited detection count per frame
- Circular buffers for metrics

#### OpenVINO Optimizations
- Async inference requests
- Device-specific configurations
- Model caching enabled
- FP16 precision for speed

#### Network Optimizations
- WebSocket compression
- SSE keepalive pings
- Efficient JSON serialization
- Base64 image streaming

### 6. Error Handling & Fallback Mechanisms

#### Error Categories
1. **Connection Errors**
   - WebSocket disconnection
   - Network timeouts
   - Client-side failures

2. **Processing Errors**
   - Invalid image data
   - Model inference failures
   - Resource exhaustion

3. **Configuration Errors**
   - Invalid parameters
   - Unsupported settings
   - Resource limits exceeded

#### Fallback Strategies
- Graceful degradation of FPS
- Quality mode switching
- Error event broadcasting
- Automatic session recovery

#### Error Response Format
```json
{
  "type": "error",
  "error": "Processing failed",
  "error_code": "INFERENCE_ERROR",
  "frame_id": "frame_123",
  "timestamp": 1641234567.123
}
```

### 7. Integration with Existing Codebase

#### Model Service Integration
- Uses existing `model_service` instance
- OpenVINO model: `/app/server/openvino_fp16/best.xml`
- YOLOv12n 2-class detection: "crack", "pothole"
- 640x640 input size maintained

#### Session Service Integration
- Compatible with existing session management
- Redis-based session storage
- Performance monitoring integration

#### Configuration Integration
- Uses existing `settings` configuration
- Environment variable support
- Backward compatibility maintained

### 8. Deployment Considerations

#### Environment Variables
```bash
STREAMING_SESSION_TIMEOUT=300
STREAMING_MAX_FPS=30
STREAMING_DEFAULT_FPS=10
STREAMING_QUEUE_SIZE=50
STREAMING_ENABLE_TRACKING=true
```

#### Resource Requirements
- CPU: Multi-core recommended for async processing
- Memory: 2GB+ for model and frame buffers
- Network: Low-latency connection for real-time streaming

#### Scaling Strategies
- Horizontal scaling with load balancer
- WebSocket sticky sessions
- Redis for distributed session storage

## Usage Examples

### WebSocket Client (JavaScript)
```javascript
const ws = new WebSocket('ws://localhost:8080/stream/detection?fps_limit=15');

ws.onopen = () => {
    console.log('Connected to streaming');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'detection_result') {
        processDetections(data.data.detections);
    }
};

// Send frame
const sendFrame = (imageData) => {
    ws.send(JSON.stringify({
        type: 'frame',
        image_data: imageData,
        frame_id: Date.now().toString(),
        timestamp: Date.now() / 1000
    }));
};
```

### REST + SSE Client (JavaScript)
```javascript
// Create session
const response = await fetch('/stream/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        fps_limit: 15,
        confidence_threshold: 0.6
    })
});
const session = await response.json();

// Connect to SSE
const eventSource = new EventSource(session.sse_endpoint);
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'detection_result') {
        processDetections(data.data.detections);
    }
};

// Send frames
const sendFrame = async (imageData) => {
    await fetch(session.upload_endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image_data: imageData,
            frame_id: Date.now().toString(),
            timestamp: Date.now() / 1000
        })
    });
};
```

### Python Client
```python
import asyncio
import websockets
import json
import base64

async def stream_detection():
    uri = "ws://localhost:8080/stream/detection?fps_limit=10"
    
    async with websockets.connect(uri) as websocket:
        # Send frame
        with open("test_frame.jpg", "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        frame_data = {
            "type": "frame",
            "image_data": image_data,
            "frame_id": "test_frame_1",
            "timestamp": time.time()
        }
        
        await websocket.send(json.dumps(frame_data))
        
        # Receive result
        result = await websocket.recv()
        detection_result = json.loads(result)
        print("Detections:", detection_result['data']['detections'])

asyncio.run(stream_detection())
```

## Performance Metrics

### Expected Performance
- **Latency**: < 100ms end-to-end
- **Throughput**: 10-30 FPS depending on configuration
- **Concurrent Sessions**: 10+ with proper resource allocation
- **Memory Usage**: ~100MB per active session
- **CPU Usage**: ~20-50% per session (depends on hardware)

### Monitoring Endpoints
- Session-level metrics: processing times, FPS, success rates
- Global metrics: active sessions, total throughput
- Health checks: model status, service availability
- Error tracking: failure rates, error types

This architecture provides a robust, scalable foundation for real-time hazard detection streaming while maintaining integration with your existing FastAPI service.