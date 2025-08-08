"""
Report router - POST /report to accept blobs, upload to Cloudinary, persist in Redis
B3 requirement: accept client-sent blob, upload to Cloudinary inside server, persist
"""

from typing import Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from ..core.logging_config import get_logger
from ..models.schemas import (
    DetectionMeta, GeoLocation, CreateReportRequest, CreateReportResponse,
    UploadDetectionResponse, ErrorResponse
)
from ..services.report_service_b3 import report_service_b3

logger = get_logger("report_router")

router = APIRouter(prefix="/report", tags=["report"])


@router.post("/", response_model=UploadDetectionResponse)
async def upload_detection_report(
    file: UploadFile = File(..., description="Image file containing the detection"),
    sessionId: str = Form(..., description="Session identifier"),
    className: str = Form(..., description="Detection class name"),
    confidence: float = Form(..., ge=0.0, le=1.0, description="Detection confidence"),
    ts: int = Form(..., description="Timestamp in epoch milliseconds"),
    lat: float = Form(None, description="Latitude (optional)"),
    lon: float = Form(None, description="Longitude (optional)")
) -> UploadDetectionResponse:
    """
    POST /report → create from client-sent blob (Cloudinary upload inside the server) and persist
    B3 requirement: accept blobs, upload to Cloudinary, persist in Redis
    
    This endpoint:
    1. Accepts an image file from the client
    2. Uploads it to Cloudinary (with retries and validation)  
    3. Stores the report metadata in Redis
    4. Returns the report ID and Cloudinary URL
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=422,
                detail="No file provided"
            )
        
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=422,
                detail="File must be an image"
            )
        
        # Build geo location if coordinates provided
        geo = None
        if lat is not None and lon is not None:
            geo = GeoLocation(lat=lat, lon=lon)
        
        # Create detection metadata
        detection_meta = DetectionMeta(
            sessionId=sessionId,
            className=className,
            confidence=confidence,
            ts=ts,
            geo=geo
        )
        
        # Upload detection and persist in Redis
        result = await report_service_b3.upload_detection(file, detection_meta)
        
        logger.info(
            f"✅ Detection report uploaded: {result.id} for session {sessionId} "
            f"(class: {className}, confidence: {confidence:.2f})"
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to upload detection report: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload detection report: {str(e)}"
        )


@router.post("/create", response_model=CreateReportResponse)
async def create_report_from_url(
    sessionId: str = Form(..., description="Session identifier"), 
    cloudinaryUrl: str = Form(..., description="Already uploaded Cloudinary URL"),
    className: str = Form(..., description="Detection class name"),
    confidence: float = Form(..., ge=0.0, le=1.0, description="Detection confidence"),
    ts: int = Form(..., description="Timestamp in epoch milliseconds"),
    lat: float = Form(None, description="Latitude (optional)"),
    lon: float = Form(None, description="Longitude (optional)")
) -> CreateReportResponse:
    """
    Create a report record referencing an already-uploaded Cloudinary URL
    Used when the client has already uploaded the image separately
    """
    try:
        # Build geo location if coordinates provided
        geo = None
        if lat is not None and lon is not None:
            geo = GeoLocation(lat=lat, lon=lon)
        
        # Create detection metadata (without sessionId since it's separate)
        detection_meta = DetectionMeta(
            sessionId=sessionId,  # Will be ignored in create_report
            className=className,
            confidence=confidence,
            ts=ts,
            geo=geo
        )
        
        # Create report with existing Cloudinary URL
        result = await report_service_b3.create_report(
            sessionId, 
            cloudinaryUrl, 
            detection_meta
        )
        
        logger.info(
            f"✅ Report created from URL: {result.id} for session {sessionId} "
            f"(class: {className}, URL: {cloudinaryUrl})"
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to create report from URL: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create report: {str(e)}"
        )


@router.get("/health")
async def get_report_health() -> Dict[str, Any]:
    """
    Get report service health status
    """
    try:
        health_status = report_service_b3.get_health_status()
        
        return {
            "status": "healthy" if health_status["status"] == "healthy" else "degraded",
            "details": health_status
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get report health: {str(e)}")
        return {
            "status": "error",
            "details": {"error": str(e)}
        }