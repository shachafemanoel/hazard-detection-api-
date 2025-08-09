"""
Health check and status API endpoints
"""

import platform
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Response, Request

from ..core.config import settings
from ..core.logging_config import get_logger
from ..services.model_service import model_service
from ..services.session_service import session_service
from ..services.performance_monitor import performance_monitor
from ..services.redis_service import redis_service
from ..models.api_models import HealthResponse, RootResponse, StatusResponse

logger = get_logger("health_api")

router = APIRouter(prefix="", tags=["health"])


@router.get("/health")
async def health_check():
    """Enhanced health check endpoint with model status and Redis check"""
    try:
        # Get model status (don't fail if model service has issues)
        try:
            model_status = model_service.get_model_status()
        except Exception as e:
            logger.warning(f"Model status check failed: {e}")
            model_status = "unknown"
        
        # Check Redis status (non-fatal)
        try:
            redis_ok = bool(redis_service.get_redis().ping()) if redis_service.get_redis() else False
        except Exception:
            redis_ok = False
        
        # Get version info
        version = settings.app_version
        
        return {
            "status": "healthy",
            "model_status": model_status,  # "not_loaded", "warming", "ready", "error" 
            "version": version,
            "redis": "up" if redis_ok else "down"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded", 
            "model_status": "error",
            "version": settings.app_version,
            "redis": "down",
            "error": str(e)
        }


@router.options("/health", include_in_schema=False)
async def health_options() -> Response:
    """Handle OPTIONS requests for health checks"""
    return Response(status_code=200)


@router.get("/")
async def root():
    """Root endpoint with basic service information"""
    return {
        "status": "ok",
        "service": settings.app_name.lower().replace(" ", "-"),
        "version": settings.app_version,
        "message": "FastAPI service is running",
        "endpoints": ["/health", "/status", "/session/start", "/detect"],
    }


@router.get("/debug")
async def debug_info():
    """Debug endpoint for troubleshooting deployment issues"""
    import os
    import sys
    from pathlib import Path
    
    try:
        # Check file system
        model_files = []
        try:
            for path in ["/app/best*", "/app/best*/*"]:
                files = list(Path("/app").glob(path.replace("/app/", "")))
                model_files.extend([str(f) for f in files])
        except Exception as e:
            model_files = [f"Error scanning: {e}"]
        
        return {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "environment": dict(os.environ),
            "model_files_found": model_files,
            "app_directory_contents": [str(p) for p in Path("/app").glob("*")],
            "settings": {
                "ml_model_dir": settings.ml_model_dir,
                "ml_model_backend": settings.ml_model_backend,
                "environment": settings.environment,
            }
        }
    except Exception as e:
        return {"error": str(e), "message": "Debug info failed"}


@router.get("/status")
async def get_detailed_status():
    """Comprehensive status information about the service"""
    try:
        # Model status information
        model_info = model_service.get_model_info()

        if model_info["status"] == "loaded":
            model_status = f"loaded_{model_info['backend']}"
            backend_inference = True
            backend_type = model_info["backend"]
        else:
            model_status = "loading"
            backend_inference = False
            backend_type = "auto"

        # Environment information
        env_info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "deployment_env": settings.environment,
            "port": str(settings.port),
            "cors_enabled": True,
            "mobile_friendly": True,
        }

        # Model file information
        model_files_info = {
            "openvino_model": str(settings.ml_model_dir)
            + "/best0408_openvino_model/best0408.xml",
            "pytorch_model": str(settings.ml_model_dir) + "/best.pt",
            "current_backend": backend_type,
            "model_classes": len(model_info.get("classes", [])),
            "input_size": settings.ml_model_input_size,
        }

        # API endpoints
        endpoints_info = {
            "health": "/health",
            "status": "/status",
            "session_start": "/session/start",
            "session_detect": "/detect/{session_id}",
            "legacy_detect": "/detect",
            "batch_detect": "/detect-batch",
        }

        return {
            "status": "healthy",
            "model_status": model_status,
            "backend_inference": backend_inference,
            "backend_type": backend_type,
            "active_sessions": session_service.get_active_session_count(),
            "device_info": model_info,
            "environment": env_info,
            "model_files": model_files_info,
            "endpoints": endpoints_info,
            "configuration": {
                "confidence_threshold": settings.confidence_threshold,
                "iou_threshold": settings.iou_threshold,
                "tracking_enabled": True,
                "openvino_device": settings.openvino_device,
                "performance_mode": settings.openvino_performance_mode,
            },
        }

    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.get("/metrics")
async def get_performance_metrics():
    """Get detailed performance metrics and system health"""
    try:
        report = performance_monitor.get_performance_report()
        return {
            "success": True,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "performance": report,
        }
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
