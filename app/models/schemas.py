"""
Pydantic models for the Hazard Detection API following B3 persistence contract
These models define the exact data structures for Detection, Report, and SessionSummary
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class Detection(BaseModel):
    """Detection result with bounding box and confidence"""
    
    bbox: List[float] = Field(
        ..., 
        description="Bounding box [x1, y1, x2, y2] in model coordinates",
        example=[100.0, 100.0, 200.0, 200.0]
    )
    confidence: float = Field(
        ..., 
        description="Detection confidence score", 
        example=0.85, 
        ge=0.0, 
        le=1.0
    )
    class_id: int = Field(
        ..., 
        description="Class ID (0: crack, 1: knocked, 2: pothole, 3: surface damage)", 
        example=2
    )
    class_name: str = Field(
        ..., 
        description="Human-readable class name", 
        example="pothole"
    )
    
    # Additional computed fields
    center_x: Optional[float] = Field(None, description="Bounding box center X")
    center_y: Optional[float] = Field(None, description="Bounding box center Y")
    width: Optional[float] = Field(None, description="Bounding box width")
    height: Optional[float] = Field(None, description="Bounding box height")
    area: Optional[float] = Field(None, description="Bounding box area")


class GeoLocation(BaseModel):
    """Geographical coordinates"""
    
    lat: Optional[float] = Field(None, description="Latitude", example=40.7128)
    lon: Optional[float] = Field(None, description="Longitude", example=-74.0060)


class DetectionMeta(BaseModel):
    """Metadata for a detection report"""
    
    sessionId: str = Field(..., description="Session identifier")
    className: str = Field(..., description="Detection class name")
    confidence: float = Field(..., ge=0.0, le=1.0)
    geo: Optional[GeoLocation] = Field(None, description="GPS coordinates if available")
    ts: int = Field(..., description="Timestamp in epoch milliseconds")


class Report(BaseModel):
    """Report persistence model following Redis contract:
    Stored under report:{uuid} with fields: sessionId, class, confidence, ts, geo, cloudinaryUrl, status
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Report UUID")
    sessionId: str = Field(..., description="Session ID this report belongs to")
    className: str = Field(
        ..., 
        description="Detection class name", 
        example="pothole"
    )
    confidence: float = Field(
        ..., 
        description="Detection confidence score", 
        ge=0.0, 
        le=1.0
    )
    ts: int = Field(
        ..., 
        description="Timestamp in epoch milliseconds",
        example=1691328600000
    )
    geo: Optional[GeoLocation] = Field(
        None, 
        description="GPS coordinates if available"
    )
    cloudinaryUrl: str = Field(
        ..., 
        description="Cloudinary secure URL for the detection image"
    )
    status: Literal["pending", "confirmed", "dismissed"] = Field(
        default="pending",
        description="Report status"
    )
    
    # Additional metadata
    bbox: Optional[List[float]] = Field(
        None, 
        description="Original detection bounding box"
    )
    created_at: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Report creation timestamp"
    )


class SessionSummary(BaseModel):
    """Session summary model for Summary modal display
    Aggregates counts, types, avg confidence, start/end times, plus list of {uuid, cloudinaryUrl}
    """
    
    sessionId: str = Field(..., description="Session identifier")
    count: int = Field(
        ..., 
        description="Total number of reports in session",
        example=15
    )
    types: Dict[str, int] = Field(
        ..., 
        description="Count per hazard type",
        example={"pothole": 5, "crack": 8, "surface damage": 2}
    )
    avgConfidence: float = Field(
        ..., 
        description="Average confidence across all reports",
        example=0.78,
        ge=0.0,
        le=1.0
    )
    durationMs: int = Field(
        ..., 
        description="Session duration in milliseconds",
        example=1800000  # 30 minutes
    )
    reports: List[Dict[str, Any]] = Field(
        ...,
        description="List of reports with {id, cloudinaryUrl, className, confidence, ts, geo}",
        example=[
            {
                "id": "123e4567-e89b-12d3-a456-426614174001",
                "cloudinaryUrl": "https://res.cloudinary.com/demo/image/upload/v1691328600/hazard_123.jpg",
                "className": "pothole",
                "confidence": 0.85,
                "ts": 1691328600000,
                "geo": {"lat": 40.7128, "lon": -74.0060}
            }
        ]
    )
    
    # Session timing
    startTime: Optional[int] = Field(
        None, 
        description="Session start timestamp (epoch ms)"
    )
    endTime: Optional[int] = Field(
        None, 
        description="Session end timestamp (epoch ms)"
    )


# API Request/Response models
class DetectionRequest(BaseModel):
    """Request model for detection endpoint"""
    
    sessionId: Optional[str] = Field(None, description="Optional session ID")
    confidence_threshold: Optional[float] = Field(
        None, 
        description="Override confidence threshold",
        ge=0.0,
        le=1.0
    )
    save_detections: bool = Field(
        default=False,
        description="Whether to save detections as reports"
    )


class DetectionResponse(BaseModel):
    """Response from detection endpoints"""
    
    success: bool = Field(..., example=True)
    sessionId: Optional[str] = Field(None, description="Session ID if provided")
    boxes: List[List[float]] = Field(
        ..., 
        description="Detection bounding boxes",
        example=[[100.0, 100.0, 200.0, 200.0]]
    )
    scores: List[float] = Field(
        ..., 
        description="Confidence scores",
        example=[0.85]
    )
    classes: List[int] = Field(
        ..., 
        description="Class IDs",
        example=[2]
    )
    raw: Optional[Dict[str, Any]] = Field(
        None, 
        description="Raw detection data for debugging"
    )
    processing_time_ms: float = Field(
        ..., 
        description="Processing time in milliseconds",
        example=150.5
    )


class CreateReportRequest(BaseModel):
    """Request to create a report from uploaded image"""
    
    sessionId: str = Field(..., description="Session ID")
    cloudinaryUrl: str = Field(..., description="Cloudinary URL of uploaded image")
    meta: DetectionMeta = Field(..., description="Detection metadata")


class CreateReportResponse(BaseModel):
    """Response from report creation"""
    
    id: str = Field(..., description="Generated report ID")
    message: str = Field(default="Report created successfully")


class UploadDetectionRequest(BaseModel):
    """Request to upload detection image to Cloudinary"""
    pass  # File upload handled via multipart form data


class UploadDetectionResponse(BaseModel):
    """Response from detection image upload"""
    
    id: str = Field(..., description="Generated report ID")
    cloudinaryUrl: str = Field(..., description="Cloudinary secure URL")


class HealthResponse(BaseModel):
    """Health check response"""
    
    status: Literal["healthy", "unhealthy"] = Field(..., example="healthy")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    
class ReadyResponse(BaseModel):
    """Readiness check response"""
    
    status: Literal["ready", "not_ready"] = Field(..., example="ready")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    redis_connected: bool = Field(..., description="Whether Redis is reachable")
    cloudinary_configured: bool = Field(..., description="Whether Cloudinary is configured")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standard error response"""
    
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request tracking ID")


# Session management models
class SessionStartResponse(BaseModel):
    """Response from session start"""
    
    session_id: str = Field(..., description="Generated session ID")
    message: str = Field(default="Session started successfully")


class ReportActionResponse(BaseModel):
    """Response from report confirm/dismiss actions"""
    
    message: str = Field(..., example="Report confirmed")
    report_id: str = Field(..., description="Report ID that was acted upon")
    new_status: Literal["confirmed", "dismissed"] = Field(..., description="New status")