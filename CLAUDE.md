# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Core Development
```bash
# Install dependencies
make dev-install

# Start development server
make run
# Or directly: python main.py

# Run all quality checks
make check-all
```

### Code Quality
```bash
# Format code
make format
# Runs: black app/ main.py && ruff format app/ main.py

# Lint code  
make lint
# Runs: ruff check app/ main.py

# Type checking
make typecheck
# Runs: mypy app/ main.py
```

### Testing
```bash
# Run all tests
make test
# Or: pytest -v

# Run tests with coverage
make test-cov
# Or: pytest --cov=app --cov-report=html --cov-report=term

# Run specific test files
pytest app/tests/test_detection_api.py -v
pytest app/tests/test_model_service.py -v
```

### Docker
```bash
# Build Docker image
make docker-build

# Run Docker container
make docker-run
```

## Architecture Overview

This is a FastAPI-based road hazard detection service using computer vision models (YOLOv12n) with OpenVINO optimization.

### Core Architecture

**Service-Oriented Design**: The application follows a modular service architecture with clear separation of concerns:

- **API Layer** (`app/api/`): Route definitions and request handling
- **Services Layer** (`app/services/`): Business logic and external integrations  
- **Core Layer** (`app/core/`): Shared utilities, configuration, exceptions
- **Models Layer** (`app/models/`): Pydantic models for validation

### Key Services

**Model Service** (`app/services/model_service.py`): 
- Handles OpenVINO and PyTorch model loading with intelligent backend selection
- Manages YOLOv12n model for 2-class detection: "crack" and "pothole"
- Supports FP16/FP32 OpenVINO models in `/app/server/openvino_fp16/` and `/app/server/openvino_fp32/`
- Falls back gracefully between backends

**Session Service** (`app/services/session_service.py`):
- Manages stateful detection sessions with tracking
- Handles session lifecycle and cleanup

**Report Service** (`app/services/report_service.py`):
- Manages detection reports with Redis persistence
- Integrates with Cloudinary for image storage
- Supports geocoding via Google Maps API

**Redis Service** (`app/services/redis_service.py`):
- Provides synchronous Redis client for report storage
- Handles connection management and error handling

**Streaming Service** (`app/services/streaming_service.py`):
- Manages real-time WebSocket and SSE connections for live detection
- Handles frame processing queues and connection lifecycle
- Supports configurable FPS limits and quality settings

**SSE Service** (`app/services/sse_service.py`):
- Server-Sent Events for real-time streaming without WebSocket complexity
- HTTP-based streaming for broad client compatibility

### Configuration System

**Environment-Based Config** (`app/core/config.py`):
- Uses Pydantic Settings for type-safe configuration
- Supports `.env` files and environment variables
- Key settings: `MODEL_BACKEND`, `MODEL_DIR`, `OPENVINO_DEVICE`

**Model Configuration**:
- Primary model location: `/app/server/openvino_fp16/best.xml`
- Fallback locations include FP32 models and legacy paths
- Input size: 640x640 for YOLOv12n model
- Supports 2-class detection: "crack" and "pothole"

### Testing Architecture

**Comprehensive Test Suite** (`app/tests/`):
- Unit tests for services with extensive mocking
- Integration tests for API endpoints  
- Fixtures in `conftest.py` for reusable test data
- 90%+ coverage target with pytest-cov

### Deployment

**Railway Deployment**: Configured via `railway.toml`
- Uses Nixpacks builder
- Health checks on `/health` endpoint  
- Environment variables for production configuration

**Docker Support**: Multi-stage Dockerfile with OpenVINO optimization
- Efficient model loading and caching
- Production-ready container setup

## Important Implementation Notes

### Model Loading
- OpenVINO is the primary backend for server inference
- Models are loaded asynchronously during startup
- Graceful fallback if model loading fails initially

### Error Handling
- Custom exceptions in `app/core/exceptions.py`
- Structured error responses with correlation IDs
- Performance monitoring with request/response logging

### Session Management
- Sessions track detection history and generate reports
- Automatic cleanup of old sessions (configurable)
- Redis-backed persistence for scalability

### Performance Monitoring
- Built-in performance metrics collection
- Request timing and success rate tracking
- Resource utilization monitoring

## File Structure Patterns

When adding new functionality:
- **API endpoints**: Add to `app/api/` with route definitions only
- **Business logic**: Implement in `app/services/` 
- **Data models**: Define in `app/models/` using Pydantic
- **Tests**: Mirror structure in `app/tests/test_*.py`
- **Configuration**: Extend `app/core/config.py` for new settings

## External Integrations

- **Google Maps API**: For geocoding services
- **Cloudinary**: For image storage and management  
- **Redis**: For session and report persistence
- **Railway**: Primary deployment platform

## Streaming and Real-Time Features

**WebSocket Support** (`app/api/streaming.py`):
- Real-time bidirectional streaming for live video feeds
- Configurable FPS limits and processing queues
- Connection health monitoring and automatic reconnection

**Server-Sent Events (SSE)**:
- HTTP-based streaming for broader client compatibility
- Real-time detection results without WebSocket complexity

**Authentication** (`app/api/auth.py`):
- JWT token-based authentication system
- Session-based access control for streaming endpoints