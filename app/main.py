"""
Main FastAPI application for the Hazard Detection API
Refactored modular structure with proper separation of concerns
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.logging_config import get_logger
from .core.exceptions import (
    HazardDetectionException,
    http_exception_handler,
    hazard_detection_exception_handler,
    general_exception_handler,
)
from .services.model_service import model_service
from .services.session_service import session_service
from .services.performance_monitor import performance_monitor
from .api import health, sessions, detection, external_apis, reports

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown"""
    # Startup
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"🌍 Environment: {settings.environment}")
    logger.info(
        f"🔧 Configuration: {settings.ml_model_backend} backend, {settings.openvino_device} device"
    )

    # Initialize model service (async loading)
    try:
        await model_service.load_model()
        logger.info("✅ Model service initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ Model loading failed during startup: {e}")
        logger.info("🔄 Model will be loaded on first request")

    # Initialize session service
    logger.info("✅ Session service initialized")

    yield

    # Shutdown
    logger.info("🛑 Shutting down application...")

    # Cleanup old sessions
    cleaned_sessions = session_service.cleanup_old_sessions(max_age_hours=1)
    if cleaned_sessions > 0:
        logger.info(f"🧹 Cleaned up {cleaned_sessions} old sessions")

    logger.info("✅ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Advanced computer vision API for road hazard detection using OpenVINO and PyTorch",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

# Add exception handlers
app.add_exception_handler(Exception, general_exception_handler)
app.add_exception_handler(HazardDetectionException, hazard_detection_exception_handler)

# Include API routers
app.include_router(health.router)
app.include_router(sessions.router)
app.include_router(detection.router)
app.include_router(reports.router)
app.include_router(external_apis.router)


# Add structured request logging and performance monitoring middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests with structured logging and track performance metrics"""
    import uuid
    start_time = __import__("time").time()
    
    # Generate request ID for correlation
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    response = await call_next(request)

    process_time = __import__("time").time() - start_time
    success = 200 <= response.status_code < 400

    # Record performance metrics
    performance_monitor.record_request(
        endpoint=request.url.path, duration=process_time, success=success
    )

    # Structured logging with request ID and timing (Agent G requirement)
    logger.info(
        f"REQUEST_COMPLETED",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(process_time * 1000, 2),
            "success": success,
            "user_agent": request.headers.get("user-agent", "unknown"),
            "client_ip": request.client.host if request.client else "unknown"
        }
    )

    # Add request ID to response headers for debugging
    response.headers["X-Request-ID"] = request_id

    return response


if __name__ == "__main__":
    import uvicorn

    logger.info(f"🚀 Starting server on {settings.host}:{settings.port}")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=True,
        reload=settings.debug,
    )
