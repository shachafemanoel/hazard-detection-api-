# Hazard Detection API - Questions & Clarifications  

## Critical Questions Before Implementation

### 1. OpenVINO Model Loading Strategy 🧠
**Question**: Should model loading be synchronous (blocking startup) or asynchronous (lazy loading)?

**Current State**: Model loads synchronously on startup, causing ~30s initialization time.

**Options**:
- A) **Sync loading**: Ensures model ready before accepting requests (current approach)
- B) **Async loading**: Non-blocking startup, load on first request  
- C) **Hybrid**: Start loading async, block requests until ready
- D) **Multi-model**: Load lightweight model first, swap to optimized model later

**Trade-offs**:
- Startup time vs first-request latency  
- Memory usage vs availability
- Deployment complexity vs user experience

**Recommendation**: Option C - Async loading with request blocking until ready.

---

### 2. Session TTL and Cleanup Policy ⏰
**Question**: What should be the session lifecycle and cleanup strategy?

**Current State**: Sessions created but no explicit expiry or cleanup visible.

**Session Scenarios**:
- **Active detection**: Continuous API calls every 100-200ms
- **Idle session**: User paused/minimized, no activity for minutes
- **Abandoned session**: User closed browser, session never ended properly
- **Long session**: User detecting for hours continuously

**TTL Options**:
- **Fixed TTL**: 1 hour from creation (simple)
- **Sliding window**: Extend TTL on each activity (complex)  
- **Hybrid**: Fixed max age (4h) + sliding activity (1h)

**Recommendation**: Hybrid approach - 1 hour sliding window, 4 hour maximum age.

---

### 3. Concurrent Request Handling 🔄  
**Question**: How many concurrent detection requests should one model instance handle?

**Current Constraints**:
- OpenVINO model inference is CPU-bound
- Each request requires ~50-200ms processing time
- Memory usage per request: ~100-500MB peak
- Target: P95 latency ≤150ms

**Concurrency Models**:
- A) **Single-threaded**: Process requests sequentially (safe, slow)
- B) **Thread pool**: Limited concurrent threads (2-4 workers)  
- C) **Async queue**: Queue requests, process with semaphore
- D) **Model replicas**: Multiple model instances per container

**Current Server Resources**: Unknown - need to verify Railway container limits.

**Recommendation**: Option C - Async queue with semaphore limiting to 4 concurrent requests.

---

### 4. Redis Session State Schema 🗄️
**Question**: What data structure should we use for session persistence in Redis?

**Current Implementation**: Basic session storage, format unclear from code review.

**Session Data Requirements**:
```json
{
  "session_id": "uuid",
  "created_at": "2025-08-08T10:00:00Z",
  "last_activity": "2025-08-08T10:30:00Z", 
  "status": "active|ended",
  "detection_count": 42,
  "unique_hazards": ["pothole", "crack"],
  "confidence_threshold": 0.5,
  "metadata": {
    "source": "web_app",
    "user_id": "optional",
    "client_version": "1.0.0"
  },
  "detections": [
    {
      "timestamp": "2025-08-08T10:15:30Z",
      "hazards": [...],
      "confidence": 0.85,
      "image_url": "cloudinary_url_optional"
    }
  ]
}
```

**Storage Considerations**:  
- **Size**: Full detection history could be large (>1MB per long session)
- **Performance**: Frequent updates during active detection
- **Cleanup**: TTL and compression for old detections

**Recommendation**: Store session metadata in Redis, detection details in time-series structure.

---

### 5. Error Response Standardization 📋
**Question**: What error response format should we use for client consumption?

**Current State**: Mix of HTTP status codes and error message formats.

**Error Categories**:
- **4xx Client Errors**: Invalid image, missing session, bad parameters
- **5xx Server Errors**: Model failure, Redis unavailable, processing timeout
- **Custom Errors**: Session expired, rate limited, model not ready

**Response Format Options**:
```python
# Option A: Simple
{"error": "Session not found"}

# Option B: Structured  
{
  "error": {
    "code": "SESSION_NOT_FOUND",
    "message": "Session abc123 does not exist or has expired",
    "details": {"session_id": "abc123", "expired_at": "..."}
  }
}

# Option C: JSON:API compliant
{
  "errors": [{
    "status": "404",
    "code": "SESSION_NOT_FOUND", 
    "title": "Session Not Found",
    "detail": "Session abc123 does not exist or has expired"
  }]
}
```

**Recommendation**: Option B - Structured but not overly complex.

---

### 6. Circuit Breaker Thresholds ⚡
**Question**: What failure thresholds should trigger circuit breaker activation?

**Dependencies to Monitor**:
- **Redis connection**: Session operations
- **Cloudinary uploads**: Image storage  
- **Model inference**: Core detection functionality

**Failure Scenarios**:
- Redis: Connection timeout, memory full, authentication failure
- Cloudinary: Rate limit, network timeout, quota exceeded  
- Model: Inference crash, memory exhaustion, invalid input

**Circuit Breaker Parameters**:
```python
CIRCUIT_BREAKER_CONFIG = {
    "redis": {
        "failure_threshold": 5,      # failures before opening
        "recovery_timeout": 30,      # seconds before retry
        "success_threshold": 3       # successes to close circuit
    },
    "cloudinary": {
        "failure_threshold": 10,     # more tolerant for uploads
        "recovery_timeout": 60, 
        "success_threshold": 5
    }
}
```

**Recommendation**: Conservative thresholds initially, tune based on production data.

---

### 7. Performance Monitoring Strategy 📊
**Question**: What metrics should we collect and expose for monitoring?

**Performance Metrics**:
- Request latency (P50, P95, P99) per endpoint
- Request rate (requests/second)  
- Error rate (errors/total requests)
- Model inference time breakdown
- Memory usage and garbage collection

**Infrastructure Metrics**:
- Redis connection pool size and usage
- OpenVINO model memory consumption
- CPU usage during inference
- Network I/O for image uploads

**Export Formats**:
- A) **Custom JSON endpoint**: `/metrics` with custom format
- B) **Prometheus format**: Standard /metrics endpoint
- C) **Railway integration**: Use Railway's built-in monitoring  
- D) **Hybrid**: Both custom and Prometheus

**Recommendation**: Option D - Prometheus for standardization + custom endpoint for debugging.

---

### 8. Image Processing Pipeline 🖼️
**Question**: What's the optimal image preprocessing pipeline for performance?

**Current Pipeline** (inferred):
```
Upload → PIL Image → OpenCV → NumPy → OpenVINO Tensor
```

**Alternative Pipelines**:
```
# Option A: PIL direct
Upload → PIL → resize/normalize → NumPy → OpenVINO

# Option B: OpenCV optimized  
Upload → OpenCV → preprocessing → NumPy → OpenVINO

# Option C: NumPy only
Upload → bytes → NumPy → preprocessing → OpenVINO
```

**Performance Considerations**:
- Memory allocations during conversion
- Image decode/encode overhead
- Preprocessing operation efficiency  

**Recommendation**: Profile all three options, likely Option A for simplicity.

---

### 9. Model Validation and Warmup 🏃‍♂️
**Question**: How should we validate model functionality and ensure consistent performance?

**Validation Needs**:  
- Model loads correctly after deployment
- Inference produces expected output format
- Performance is within acceptable ranges  
- Memory usage is stable

**Warmup Strategy**:
```python  
# Option A: Fixed test images
WARMUP_IMAGES = ["test_pothole.jpg", "test_crack.jpg", "test_clear.jpg"]

# Option B: Generated synthetic images  
synthetic_images = generate_test_patterns(count=5)

# Option C: Real sample from training data
warmup_samples = load_validation_samples(count=3)
```

**Performance Validation**:
- First request after warmup should be <150ms
- Memory usage should stabilize after warmup
- All warmup inferences should succeed

**Recommendation**: Option C - Use real validation samples for accurate warmup.

---

### 10. Deployment and Rollback Strategy 🚀  
**Question**: How should we handle API deployments and potential rollbacks?

**Current Deployment**: Direct Railway deployment from main branch.

**Deployment Risks**:  
- Model loading changes could prevent startup
- Performance degradation could impact all users
- Database/Redis schema changes could cause data loss

**Deployment Strategy Options**:
- A) **Blue/Green**: Deploy new version alongside old, switch traffic
- B) **Canary**: Roll out to small percentage of users first  
- C) **Feature flags**: Deploy code but control feature activation
- D) **Rolling update**: Gradual replacement of instances

**Health Check Requirements**:
- New deployment must pass health checks before serving traffic
- Rollback triggers: >5% error rate, >300ms P95 latency
- Manual rollback capability within 5 minutes

**Recommendation**: Option B - Canary deployment with 10% traffic initially.

---

## Implementation Priorities

### Must Answer Before Starting:
1. **Model loading strategy** (#1) - affects all subsequent work
2. **Session TTL policy** (#2) - foundational for session management  
3. **Concurrent request handling** (#3) - performance critical
4. **Error response format** (#5) - client integration dependency

### Can Decide During Implementation:  
5. **Redis schema details** (#4) - can evolve incrementally
6. **Circuit breaker thresholds** (#6) - tune based on testing
7. **Monitoring metrics** (#7) - add progressively
8. **Image pipeline optimization** (#8) - performance optimization phase

### Address After Core Implementation:
9. **Model validation details** (#9) - operational improvement
10. **Deployment strategy refinement** (#10) - production readiness

---
*Questions compiled on 2025-08-08 - Please review and prioritize before API improvements begin.*