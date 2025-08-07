"""
Pydantic models for API request/response validation and documentation
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


# Global model configuration to suppress warnings
model_config = ConfigDict(protected_namespaces=())

# Base response models
class HealthResponse(BaseModel):
    """Health check response"""
    model_config = model_config

    status: str = Field(..., example="healthy")


class RootResponse(BaseModel):
    """Root endpoint response"""
    model_config = model_config

    status: str = Field(..., example="ok")
    service: str = Field(..., example="hazard-detection-api")
    version: str = Field(..., example="1.0.0")
    message: str = Field(..., example="FastAPI service is running")
    endpoints: List[str] = Field(
        ..., example=["/health", "/status", "/session/start", "/detect"]
    )


# Detection models
class BoundingBox(BaseModel):
    """Bounding box coordinates"""

    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")


class Detection(BaseModel):
    """Single detection result"""
    model_config = model_config

    bbox: List[float] = Field(
        ...,
        description="Bounding box [x1, y1, x2, y2]",
        example=[100.0, 100.0, 200.0, 200.0],
    )
    confidence: float = Field(
        ..., description="Detection confidence score", example=0.85, ge=0.0, le=1.0
    )
    class_id: int = Field(..., description="Class ID", example=0)
    class_name: str = Field(..., description="Class name", example="pothole")
    center_x: float = Field(
        ..., description="Bounding box center X coordinate", example=150.0
    )
    center_y: float = Field(
        ..., description="Bounding box center Y coordinate", example=150.0
    )
    width: float = Field(..., description="Bounding box width", example=100.0)
    height: float = Field(..., description="Bounding box height", example=100.0)
    area: float = Field(..., description="Bounding box area", example=10000.0)
    is_new: Optional[bool] = Field(
        None, description="Whether this is a new detection (session-based)"
    )
    report_id: Optional[str] = Field(
        None, description="Associated report ID if generated"
    )


class ImageSize(BaseModel):
    """Image dimensions"""

    width: int = Field(..., example=480)
    height: int = Field(..., example=480)


class ModelInfo(BaseModel):
    """Model information"""

    status: str = Field(..., example="loaded")
    backend: str = Field(..., example="openvino")
    classes: List[str] = Field(..., example=["crack", "knocked", "pothole", "surface damage"])
    class_count: int = Field(..., example=4)
    confidence_threshold: Optional[float] = Field(None, example=0.6)
    tracking_enabled: Optional[bool] = Field(None, example=True)
    input_shape: Optional[List[int]] = Field(None, example=[1, 3, 480, 480])
    output_shape: Optional[List[int]] = Field(None, example=[1, 25200, 15])
    device: Optional[str] = Field(None, example="AUTO")
    performance_mode: Optional[str] = Field(None, example="LATENCY")
    async_inference: Optional[bool] = Field(None, example=True)


# Session models
class SessionStartResponse(BaseModel):
    """Session start response"""

    session_id: str = Field(..., example="123e4567-e89b-12d3-a456-426614174000")


class SessionEndResponse(BaseModel):
    """Session end response"""

    message: str = Field(..., example="Session ended")
    session_id: str = Field(..., example="123e4567-e89b-12d3-a456-426614174000")


class DetectionReport(BaseModel):
    """Detection report model"""

    report_id: str = Field(..., example="123e4567-e89b-12d3-a456-426614174001")
    session_id: str = Field(..., example="123e4567-e89b-12d3-a456-426614174000")
    detection: Detection = Field(
        ..., description="Detection that generated this report"
    )
    timestamp: str = Field(..., example="2024-08-06T10:30:00")
    status: str = Field(
        ...,
        example="pending",
        description="Report status: pending, confirmed, dismissed",
    )
    location: Dict[str, Any] = Field(
        ..., example={"bbox": [100, 100, 200, 200], "center": [150, 150]}
    )
    frame_info: Dict[str, Any] = Field(
        ..., example={"has_image": True, "image_size": 1024}
    )


class SessionStats(BaseModel):
    """Session statistics"""

    total_detections: int = Field(..., example=25)
    unique_hazards: int = Field(..., example=5)
    pending_reports: int = Field(..., example=3)
    confirmed_reports: Optional[int] = Field(None, example=1)
    dismissed_reports: Optional[int] = Field(None, example=1)


class SessionSummary(BaseModel):
    """Complete session summary"""

    id: str = Field(..., example="123e4567-e89b-12d3-a456-426614174000")
    start_time: str = Field(..., example="2024-08-06T10:00:00")
    reports: List[DetectionReport] = Field(
        ..., description="All reports in this session"
    )
    detection_count: int = Field(..., example=25)
    unique_hazards: int = Field(..., example=5)
    pending_reports: int = Field(..., example=3)
    confirmed_reports: int = Field(..., example=1)
    dismissed_reports: int = Field(..., example=1)


# Detection response models
class DetectionResponse(BaseModel):
    """Detection endpoint response"""

    success: bool = Field(..., example=True)
    detections: List[Detection] = Field(..., description="Detected hazards")
    new_reports: Optional[List[DetectionReport]] = Field(
        None, description="New reports generated"
    )
    session_stats: Optional[SessionStats] = Field(
        None, description="Session statistics"
    )
    processing_time_ms: float = Field(
        ..., example=150.5, description="Processing time in milliseconds"
    )
    image_size: ImageSize = Field(..., description="Input image dimensions")
    model_info: ModelInfo = Field(..., description="Model information")


class LegacyDetectionResponse(BaseModel):
    """Legacy detection endpoint response"""

    success: bool = Field(..., example=True)
    detections: List[Detection] = Field(..., description="Detected hazards")
    processing_time_ms: float = Field(
        ..., example=150.5, description="Processing time in milliseconds"
    )
    image_size: ImageSize = Field(..., description="Input image dimensions")
    model_info: ModelInfo = Field(..., description="Model information")


class BatchDetectionResult(BaseModel):
    """Single file result in batch detection"""

    file_index: int = Field(..., example=0)
    filename: str = Field(..., example="image1.jpg")
    success: Optional[bool] = Field(None, example=True)
    detections: Optional[List[Detection]] = Field(None, description="Detected hazards")
    image_size: Optional[ImageSize] = Field(None, description="Input image dimensions")
    error: Optional[str] = Field(None, example="File must be an image")


class BatchDetectionResponse(BaseModel):
    """Batch detection endpoint response"""

    success: bool = Field(..., example=True)
    results: List[BatchDetectionResult] = Field(
        ..., description="Results for each file"
    )
    total_processing_time_ms: float = Field(
        ..., example=450.2, description="Total processing time"
    )
    processed_count: int = Field(
        ..., example=3, description="Number of files processed"
    )
    successful_count: int = Field(
        ..., example=2, description="Number of successfully processed files"
    )
    model_info: ModelInfo = Field(..., description="Model information")


# Report action models
class ReportActionResponse(BaseModel):
    """Report confirmation/dismissal response"""

    message: str = Field(..., example="Report confirmed")
    report_id: str = Field(..., example="123e4567-e89b-12d3-a456-426614174001")


# Status models
class DeviceInfo(BaseModel):
    """Device information"""
    model_config = model_config

    device: Optional[str] = Field(None, example="AUTO")
    input_shape: Optional[List[int]] = Field(None, example=[1, 3, 480, 480])
    output_shape: Optional[List[int]] = Field(None, example=[1, 25200, 15])
    model_path: Optional[str] = Field(
        None, example="best0408_openvino_model/best0408.xml"
    )
    cache_enabled: Optional[bool] = Field(None, example=False)
    backend: Optional[str] = Field(None, example="openvino")
    performance_mode: Optional[str] = Field(None, example="LATENCY")
    async_inference: Optional[bool] = Field(None, example=True)
    openvino_version: Optional[str] = Field(None, example="2024_optimized")


class EnvironmentInfo(BaseModel):
    """Environment information"""

    platform: str = Field(..., example="Linux")
    python_version: str = Field(..., example="3.11.0")
    hostname: str = Field(..., example="api-server-001")
    deployment_env: str = Field(..., example="production")
    port: str = Field(..., example="8080")
    cors_enabled: bool = Field(..., example=True)
    mobile_friendly: bool = Field(..., example=True)


class ModelFiles(BaseModel):
    """Model file information"""
    model_config = model_config

    openvino_model: str = Field(
        ..., example="/app/best0408_openvino_model/best0408.xml"
    )
    pytorch_model: str = Field(..., example="/app/best.pt")
    current_backend: str = Field(..., example="openvino")
    model_classes: int = Field(..., example=4)  # Updated for best0408 model (4 classes)
    input_size: int = Field(..., example=480)


class ConfigurationInfo(BaseModel):
    """Configuration information"""

    confidence_threshold: float = Field(..., example=0.5)
    iou_threshold: float = Field(..., example=0.45)
    tracking_enabled: bool = Field(..., example=True)
    openvino_device: str = Field(..., example="AUTO")
    performance_mode: str = Field(..., example="LATENCY")


class EndpointsInfo(BaseModel):
    """Available endpoints information"""

    health: str = Field(..., example="/health")
    status: str = Field(..., example="/status")
    session_start: str = Field(..., example="/session/start")
    session_detect: str = Field(..., example="/detect/{session_id}")
    legacy_detect: str = Field(..., example="/detect")
    batch_detect: str = Field(..., example="/detect-batch")


class StatusResponse(BaseModel):
    """Detailed status response"""
    model_config = model_config

    status: str = Field(..., example="healthy")
    model_status: str = Field(..., example="loaded_openvino")
    backend_inference: bool = Field(..., example=True)
    backend_type: str = Field(..., example="openvino")
    active_sessions: int = Field(..., example=3)
    device_info: DeviceInfo = Field(..., description="Device and model information")
    environment: EnvironmentInfo = Field(..., description="Environment information")
    model_files: ModelFiles = Field(..., description="Model file information")
    endpoints: EndpointsInfo = Field(..., description="Available endpoints")
    configuration: ConfigurationInfo = Field(..., description="Current configuration")


# Error models
class ErrorDetail(BaseModel):
    """Error response detail"""

    error: str = Field(..., example="ValidationError")
    detail: str = Field(..., example="Invalid input data")
    timestamp: str = Field(..., example="2024-08-06T10:30:00")
    path: Optional[str] = Field(None, example="/detect/invalid-session-id")
    request_id: Optional[str] = Field(None, example="req-123456")
    additional_info: Optional[Dict[str, Any]] = Field(
        None, description="Additional error context"
    )
