"""
Report data models for the Hazard Detection API
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ReportStatus(str, Enum):
    """Report status enumeration"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"
    ARCHIVED = "archived"


class HazardSeverity(str, Enum):
    """Hazard severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LocationInfo(BaseModel):
    """Location information for a report"""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    address: Optional[str] = None
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    center: List[float] = Field(..., description="Center point [x, y]")


class ImageInfo(BaseModel):
    """Image information for a report"""
    url: str = Field(..., description="Cloudinary URL")
    public_id: str = Field(..., description="Cloudinary public ID")
    width: int
    height: int
    format: Optional[str] = None
    size_bytes: Optional[int] = None
    thumbnail_url: Optional[str] = None


class DetectionInfo(BaseModel):
    """Detection information from the model"""
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    area: float
    center_x: float
    center_y: float


class ReportMetadata(BaseModel):
    """Additional metadata for reports"""
    session_id: str
    source: Optional[str] = "api"  # api, manual, batch, etc.
    device_info: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[float] = None
    model_version: Optional[str] = None
    api_version: Optional[str] = None


class ReportCreateRequest(BaseModel):
    """Request model for creating a new report"""
    detection: DetectionInfo
    image_data: Optional[str] = Field(None, description="Base64 encoded image")
    location: Optional[LocationInfo] = None
    metadata: Optional[ReportMetadata] = None
    description: Optional[str] = None
    severity: HazardSeverity = HazardSeverity.MEDIUM
    tags: Optional[List[str]] = None


class ReportUpdateRequest(BaseModel):
    """Request model for updating a report"""
    status: Optional[ReportStatus] = None
    description: Optional[str] = None
    severity: Optional[HazardSeverity] = None
    location: Optional[LocationInfo] = None
    tags: Optional[List[str]] = None


class ReportResponse(BaseModel):
    """Response model for report operations"""
    id: str
    status: ReportStatus
    detection: DetectionInfo
    location: Optional[LocationInfo] = None
    image: Optional[ImageInfo] = None
    description: Optional[str] = None
    severity: HazardSeverity
    tags: List[str] = []
    metadata: Optional[ReportMetadata] = None
    created_at: datetime
    updated_at: datetime
    confirmed_at: Optional[datetime] = None


class ReportListResponse(BaseModel):
    """Response model for listing reports"""
    reports: List[ReportResponse]
    pagination: Dict[str, Any]
    total_count: int
    filters_applied: Dict[str, Any]


class ReportStatsResponse(BaseModel):
    """Response model for report statistics"""
    total_reports: int
    pending_reports: int
    confirmed_reports: int
    dismissed_reports: int
    reports_by_severity: Dict[str, int]
    reports_by_class: Dict[str, int]
    recent_reports_count: int
    avg_confidence: float


class ReportFilterRequest(BaseModel):
    """Request model for filtering reports"""
    status: Optional[List[ReportStatus]] = None
    severity: Optional[List[HazardSeverity]] = None
    class_ids: Optional[List[int]] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    session_id: Optional[str] = None
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    location_bounds: Optional[List[float]] = Field(
        None, description="Bounding box [min_lat, min_lon, max_lat, max_lon]"
    )
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    page: int = Field(1, ge=1)
    limit: int = Field(20, ge=1, le=100)
    sort_by: str = "created_at"
    sort_order: str = Field("desc", pattern="^(asc|desc)$")