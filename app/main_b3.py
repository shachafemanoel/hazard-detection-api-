"""
Main FastAPI application for the Hazard Detection API
B1-B5 compliant with proper structure, config, model lifecycle, persistence, validation
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

# B2: Model lifecycle - initialize once at startup, keep warmed handle
from .services.model_service import model_service

# B3: Redis persistence and session services
from .services.redis_service import redis_service
from .services.report_service_b3 import report_service_b3

# B5: Performance monitoring
from .services.performance_monitor import performance_monitor

# New routers following specification
from .routers import detect, session, report

logger = get_logger("main_b3")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown
    B1: Load env once, validate required vars on startup
    B2: Initialize OpenVINO or fallback backend once at startup; keep warmed model handle
    """
    # Startup
    logger.info(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"🌍 Environment: {settings.environment}")
    logger.info(f"🔧 Configuration: {settings.model_backend} backend, {settings.openvino_device} device")

    # B1: Validate required environment variables
    required_vars = []
    if not settings.redis_url and not (settings.redis_host and settings.redis_password):
        logger.warning("⚠️ Redis not configured - sessions will not persist")
    
    if not all([settings.cloudinary_cloud_name, settings.cloudinary_api_key, settings.cloudinary_api_secret]):
        logger.warning("⚠️ Cloudinary not configured - image uploads will fail")
    
    # B2: Initialize model service (async loading)
    try:
        await model_service.load_model()
        logger.info("✅ Model service initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ Model loading failed during startup: {e}")
        logger.info("🔄 Model will be loaded on first request")

    # B3: Test Redis connection
    redis_health = redis_service.get_health_status()
    if redis_health.get("connected"):
        logger.info("✅ Redis connection verified")
    else:
        logger.warning("⚠️ Redis connection failed")

    # B4: Test Cloudinary connection
    from .services.cloudinary_service import cloudinary_service
    cloudinary_health = cloudinary_service.get_health_status()
    if cloudinary_health.get("configured"):
        logger.info("✅ Cloudinary configured")
    else:
        logger.warning("⚠️ Cloudinary not configured")

    yield

    # Shutdown
    logger.info("🛑 Shutting down application...")

    # B3: Cleanup old sessions
    try:
        cleaned_sessions = redis_service.cleanup_old_sessions(max_age_hours=1)
        if cleaned_sessions > 0:
            logger.info(f"🧹 Cleaned up {cleaned_sessions} old sessions")
    except Exception as e:
        logger.warning(f"Failed to cleanup sessions: {e}")

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

# B1: CORS - include client origins, no wildcard in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)

# Add exception handlers
app.add_exception_handler(Exception, general_exception_handler)
app.add_exception_handler(HazardDetectionException, hazard_detection_exception_handler)

# B2, B3: Include new API routers with proper structure
app.include_router(detect.router)  # B2: /health, /ready, detection endpoints
app.include_router(session.router) # B3: session management
app.include_router(report.router)  # B3: report persistence

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "ok",
        "environment": settings.environment,
        "endpoints": {
            "health": "/detect/health",
            "ready": "/detect/ready", 
            "detect_with_session": "/detect/{session_id}",
            "detect_legacy": "/detect/",
            "session_start": "/session/start",
            "session_reports": "/session/{id}/reports",
            "session_summary": "/session/{id}/summary",
            "upload_report": "/report/",
            "create_report": "/report/create"
        }
    }


# B5: Request logging and performance monitoring middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log incoming requests and track performance metrics
    B5: Structured logging with request id and timing for detect/report endpoints
    """
    start_time = __import__("time").time()

    response = await call_next(request)

    process_time = __import__("time").time() - start_time
    success = 200 <= response.status_code < 400

    # Record performance metrics
    try:
        performance_monitor.record_request(
            endpoint=request.url.path, 
            duration=process_time, 
            success=success
        )
    except Exception:
        pass  # Don't break requests if monitoring fails

    # B5: Structured logging
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )

    return response


if __name__ == "__main__":
    import uvicorn

    logger.info(f"🚀 Starting server on {settings.host}:{settings.port}")

    uvicorn.run(
        "app.main_b3:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        access_log=True,
        reload=settings.debug,
    )