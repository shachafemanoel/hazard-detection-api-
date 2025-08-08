"""
Detection router - detection endpoints with warm model on startup
B2 requirement: warm model on startup, detection with minimal latency
"""

import time
from typing import Dict, List, Any, Optional
from io import BytesIO
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Depends
from fastapi.responses import JSONResponse

from ..core.config import settings
from ..core.logging_config import get_logger
from ..models.schemas import (
    DetectionRequest, DetectionResponse, DetectionMeta, 
    HealthResponse, ReadyResponse
)
from ..services.model_service import model_service
from ..services.redis_service import redis_service
from ..services.cloudinary_service import cloudinary_service

logger = get_logger("detect_router")

router = APIRouter(prefix="/detect", tags=["detection"])


@router.post("/{session_id}", response_model=DetectionResponse)
async def detect_with_session(
    session_id: str,
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Form(None),
    save_detections: bool = Form(False)
) -> DetectionResponse:
    """
    Run detection on uploaded image for a specific session
    Optionally save detections as reports
    """
    start_time = time.time()
    
    try:
        # Validate model is loaded (B2 requirement)
        if not model_service.is_loaded:
            raise HTTPException(
                status_code=503, 
                detail="Model not loaded. Please wait for startup to complete."
            )
        
        # Validate image file
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=422,
                detail="Invalid file type. Only images are supported."
            )
        
        # Read and validate image
        image_data = await file.read()
        try:
            image = Image.open(BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid image data: {str(e)}"
            )
        
        # Override confidence threshold if provided
        if confidence_threshold is not None:
            if not (0.0 <= confidence_threshold <= 1.0):
                raise HTTPException(
                    status_code=422,
                    detail="Confidence threshold must be between 0.0 and 1.0"
                )
        else:
            confidence_threshold = settings.confidence_threshold
        
        # Run inference
        detections = await model_service.predict(image, confidence_threshold)
        
        # Convert to response format
        boxes = []
        scores = []
        classes = []
        
        for detection in detections:
            boxes.append(detection.bbox)
            scores.append(detection.confidence)
            classes.append(detection.class_id)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log performance
        logger.info(
            f"Detection completed: {len(detections)} hazards found in {processing_time:.1f}ms "
            f"(session: {session_id})"
        )
        
        return DetectionResponse(
            success=True,
            sessionId=session_id,
            boxes=boxes,
            scores=scores, 
            classes=classes,
            raw={"detections": [d.to_dict() for d in detections]},
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )


@router.post("/", response_model=DetectionResponse)
async def detect_legacy(
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Form(None)
) -> DetectionResponse:
    """
    Legacy detection endpoint without session
    Maintains compatibility with existing clients
    """
    start_time = time.time()
    
    try:
        # Validate model is loaded
        if not model_service.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please wait for startup to complete."
            )
        
        # Validate image file
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=422,
                detail="Invalid file type. Only images are supported."
            )
        
        # Read and validate image
        image_data = await file.read()
        try:
            image = Image.open(BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid image data: {str(e)}"
            )
        
        # Override confidence threshold if provided
        if confidence_threshold is not None:
            if not (0.0 <= confidence_threshold <= 1.0):
                raise HTTPException(
                    status_code=422,
                    detail="Confidence threshold must be between 0.0 and 1.0"
                )
        else:
            confidence_threshold = settings.confidence_threshold
        
        # Run inference
        detections = await model_service.predict(image, confidence_threshold)
        
        # Convert to response format
        boxes = []
        scores = []
        classes = []
        
        for detection in detections:
            boxes.append(detection.bbox)
            scores.append(detection.confidence)
            classes.append(detection.class_id)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log performance
        logger.info(
            f"Legacy detection completed: {len(detections)} hazards found in {processing_time:.1f}ms"
        )
        
        return DetectionResponse(
            success=True,
            sessionId=None,
            boxes=boxes,
            scores=scores,
            classes=classes,
            raw={"detections": [d.to_dict() for d in detections]},
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Legacy detection failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Simple health check - returns 200 if service is live (B2 requirement)
    """
    return HealthResponse(status="healthy")


@router.get("/ready", response_model=ReadyResponse) 
async def readiness_check() -> ReadyResponse:
    """
    Readiness check - returns 200 after model loaded & Redis/Cloudinary reachable (B2 requirement)
    """
    try:
        # Check model status
        model_status = model_service.get_health_status()
        model_loaded = model_status.get("model_loaded", False)
        
        # Check Redis connection
        redis_status = redis_service.get_health_status()
        redis_connected = redis_status.get("connected", False)
        
        # Check Cloudinary configuration
        cloudinary_status = cloudinary_service.get_health_status()
        cloudinary_configured = cloudinary_status.get("configured", False)
        
        # Overall ready status
        is_ready = model_loaded and redis_connected and cloudinary_configured
        
        if is_ready:
            logger.debug("Readiness check: all systems ready")
        else:
            logger.warning(
                f"Readiness check failed: model={model_loaded}, "
                f"redis={redis_connected}, cloudinary={cloudinary_configured}"
            )
        
        return ReadyResponse(
            status="ready" if is_ready else "not_ready",
            model_loaded=model_loaded,
            redis_connected=redis_connected, 
            cloudinary_configured=cloudinary_configured
        )
        
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return ReadyResponse(
            status="not_ready",
            model_loaded=False,
            redis_connected=False,
            cloudinary_configured=False
        )


@router.get("/model/info")
async def get_model_info() -> Dict[str, Any]:
    """
    Get detailed model information for debugging
    """
    try:
        model_info = model_service.get_model_info()
        health_info = model_service.get_health_status()
        
        return {
            "model": model_info,
            "health": health_info,
            "settings": {
                "input_size": settings.model_input_size,
                "confidence_threshold": settings.confidence_threshold,
                "iou_threshold": settings.iou_threshold,
                "backend_preference": settings.model_backend,
                "openvino_device": settings.openvino_device
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )