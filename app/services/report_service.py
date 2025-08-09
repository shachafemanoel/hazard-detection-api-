"""
Report service for managing hazard detection reports
Handles CRUD operations, Redis storage, and integration with Cloudinary
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from ..core.config import settings
from ..core.logging_config import get_logger
from ..core.exceptions import SessionNotFoundException
from ..models.report_models import (
    ReportResponse,
    ReportCreateRequest,
    ReportUpdateRequest,
    ReportFilterRequest,
    ReportListResponse,
    ReportStatsResponse,
    ReportStatus,
    ImageInfo,
    LocationInfo,
    DetectionInfo,
    ReportMetadata,
    HazardSeverity
)
from .cloudinary_service import cloudinary_service
from .model_service import DetectionResult
from .redis_service import redis_service

logger = get_logger("report_service")


class ReportService:
    """Service for managing hazard detection reports"""

    def __init__(self):
        self.redis_client = None  # Will be set from main.py lifespan
        self.geocoder: Optional[Nominatim] = None
        self._setup_geocoder()

    def _setup_geocoder(self):
        """Setup geocoding service (lightweight, non-blocking)"""
        if settings.geocoding_enabled:
            try:
                # Just create the geocoder object, don't test it
                self.geocoder = Nominatim(user_agent="hazard-detection-api")
                logger.info("🔄 Geocoding service initialized (will test on first use)")
            except Exception as e:
                logger.warning(f"⚠️ Geocoder setup issue (will retry on use): {e}")
                self.geocoder = None

    async def create_report_from_detection(
        self,
        detection: DetectionResult,
        session_id: str,
        image_data: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> ReportResponse:
        """
        Create a report from a detection result
        
        Args:
            detection: DetectionResult from model inference
            session_id: Session ID for tracking
            image_data: Base64 encoded image data
            additional_metadata: Additional metadata to include
            
        Returns:
            ReportResponse: Created report
        """
        try:
            report_id = str(uuid.uuid4())
            current_time = datetime.now()
            
            # Convert DetectionResult to DetectionInfo
            detection_info = DetectionInfo(
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
                area=detection.area,
                center_x=detection.center_x,
                center_y=detection.center_y
            )
            
            # Upload image if provided
            image_info = None
            if image_data:
                try:
                    image_info = await cloudinary_service.upload_base64_image(
                        image_data,
                        f"report_{report_id}_{current_time.strftime('%Y%m%d_%H%M%S')}",
                        folder="hazard-reports",
                        create_thumbnail=True
                    )
                except Exception as e:
                    logger.error(f"Failed to upload image for report {report_id}: {e}")

            # Create location info from detection bbox
            location_info = LocationInfo(
                bbox=detection.bbox,
                center=[detection.center_x, detection.center_y]
            )

            # Create metadata
            metadata = ReportMetadata(
                session_id=session_id,
                source="detection_pipeline"
            )
            if additional_metadata:
                metadata.device_info = additional_metadata.get("device_info")
                metadata.processing_time_ms = additional_metadata.get("processing_time_ms")
                metadata.model_version = additional_metadata.get("model_version")

            # Determine severity based on confidence and class
            severity = self._determine_severity(detection)

            # Create report
            report = ReportResponse(
                id=report_id,
                status=ReportStatus.PENDING,
                detection=detection_info,
                location=location_info,
                image=image_info,
                severity=severity,
                tags=[],
                metadata=metadata,
                created_at=current_time,
                updated_at=current_time
            )

            # Store in Redis
            await self._store_report(report)
            
            logger.info(f"✅ Created report {report_id} from detection (confidence: {detection.confidence:.2f})")
            return report

        except Exception as e:
            logger.error(f"❌ Failed to create report from detection: {e}")
            raise

    async def create_report(self, request: ReportCreateRequest) -> ReportResponse:
        """Create a new report from request"""
        try:
            report_id = str(uuid.uuid4())
            current_time = datetime.now()

            # Upload image if provided
            image_info = None
            if request.image_data:
                try:
                    image_info = await cloudinary_service.upload_base64_image(
                        request.image_data,
                        f"report_{report_id}_{current_time.strftime('%Y%m%d_%H%M%S')}",
                        folder="hazard-reports",
                        create_thumbnail=True
                    )
                except Exception as e:
                    logger.error(f"Failed to upload image for report {report_id}: {e}")

            # Geocode location if address is provided
            location_info = request.location
            if location_info and location_info.address and not location_info.latitude:
                try:
                    coords = await self._geocode_address(location_info.address)
                    if coords:
                        location_info.latitude = coords[0]
                        location_info.longitude = coords[1]
                except Exception as e:
                    logger.warning(f"Geocoding failed for {location_info.address}: {e}")

            # Create report
            report = ReportResponse(
                id=report_id,
                status=ReportStatus.PENDING,
                detection=request.detection,
                location=location_info,
                image=image_info,
                description=request.description,
                severity=request.severity,
                tags=request.tags or [],
                metadata=request.metadata,
                created_at=current_time,
                updated_at=current_time
            )

            # Store in Redis
            await self._store_report(report)
            
            logger.info(f"✅ Created report {report_id}")
            return report

        except Exception as e:
            logger.error(f"❌ Failed to create report: {e}")
            raise

    async def get_report(self, report_id: str) -> Optional[ReportResponse]:
        """Get a report by ID"""
        try:
            if not self.redis_client:
                logger.error("Redis not available")
                return None

            report_key = f"report:{report_id}"
            report_data = self.redis_client.get(report_key)
            
            if not report_data:
                return None

            report_dict = json.loads(report_data)
            return ReportResponse(**report_dict)

        except Exception as e:
            logger.error(f"❌ Failed to get report {report_id}: {e}")
            return None

    async def update_report(self, report_id: str, request: ReportUpdateRequest) -> Optional[ReportResponse]:
        """Update an existing report"""
        try:
            # Get existing report
            report = await self.get_report(report_id)
            if not report:
                return None

            # Update fields
            update_data = request.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(report, field):
                    setattr(report, field, value)

            # Update timestamp
            report.updated_at = datetime.now()

            # Geocode location if address changed
            if request.location and request.location.address:
                if not request.location.latitude:
                    try:
                        coords = await self._geocode_address(request.location.address)
                        if coords:
                            request.location.latitude = coords[0]
                            request.location.longitude = coords[1]
                            report.location = request.location
                    except Exception as e:
                        logger.warning(f"Geocoding failed: {e}")

            # Store updated report
            await self._store_report(report)
            
            logger.info(f"✅ Updated report {report_id}")
            return report

        except Exception as e:
            logger.error(f"❌ Failed to update report {report_id}: {e}")
            raise

    async def delete_report(self, report_id: str) -> bool:
        """Delete a report"""
        try:
            if not self.redis_client:
                logger.error("Redis not available")
                return False

            # Get report to find image for deletion
            report = await self.get_report(report_id)
            if report and report.image:
                try:
                    await cloudinary_service.delete_image(report.image.public_id)
                except Exception as e:
                    logger.warning(f"Failed to delete image for report {report_id}: {e}")

            # Delete from Redis
            report_key = f"report:{report_id}"
            deleted = self.redis_client.delete(report_key)
            
            if deleted:
                logger.info(f"✅ Deleted report {report_id}")
                return True
            else:
                logger.warning(f"Report {report_id} not found for deletion")
                return False

        except Exception as e:
            logger.error(f"❌ Failed to delete report {report_id}: {e}")
            return False

    async def list_reports(self, filters: ReportFilterRequest) -> ReportListResponse:
        """List reports with filtering and pagination"""
        try:
            if not self.redis_client:
                return ReportListResponse(
                    reports=[],
                    pagination={"page": 1, "limit": 0, "total": 0, "pages": 0},
                    total_count=0,
                    filters_applied=filters.dict()
                )

            # Get all report keys
            report_keys = self.redis_client.keys("report:*")
            all_reports = []

            # Load all reports (for filtering)
            for key in report_keys:
                try:
                    report_data = self.redis_client.get(key)
                    if report_data:
                        report_dict = json.loads(report_data)
                        report = ReportResponse(**report_dict)
                        all_reports.append(report)
                except Exception as e:
                    logger.warning(f"Failed to load report from key {key}: {e}")

            # Apply filters
            filtered_reports = self._apply_filters(all_reports, filters)

            # Apply sorting
            filtered_reports = self._apply_sorting(filtered_reports, filters.sort_by, filters.sort_order)

            # Apply pagination
            total_count = len(filtered_reports)
            start_idx = (filters.page - 1) * filters.limit
            end_idx = start_idx + filters.limit
            paginated_reports = filtered_reports[start_idx:end_idx]

            pagination = {
                "page": filters.page,
                "limit": filters.limit,
                "total": total_count,
                "pages": (total_count + filters.limit - 1) // filters.limit
            }

            return ReportListResponse(
                reports=paginated_reports,
                pagination=pagination,
                total_count=total_count,
                filters_applied=filters.dict()
            )

        except Exception as e:
            logger.error(f"❌ Failed to list reports: {e}")
            raise

    async def get_report_stats(self) -> ReportStatsResponse:
        """Get report statistics"""
        try:
            if not self.redis_client:
                return ReportStatsResponse(
                    total_reports=0,
                    pending_reports=0,
                    confirmed_reports=0,
                    dismissed_reports=0,
                    reports_by_severity={},
                    reports_by_class={},
                    recent_reports_count=0,
                    avg_confidence=0.0
                )

            # Get all reports
            filters = ReportFilterRequest(page=1, limit=10000)  # Get all reports
            result = await self.list_reports(filters)
            reports = result.reports

            # Calculate statistics
            total_reports = len(reports)
            pending_reports = len([r for r in reports if r.status == ReportStatus.PENDING])
            confirmed_reports = len([r for r in reports if r.status == ReportStatus.CONFIRMED])
            dismissed_reports = len([r for r in reports if r.status == ReportStatus.DISMISSED])

            # Reports by severity
            reports_by_severity = {}
            for severity in HazardSeverity:
                reports_by_severity[severity.value] = len([r for r in reports if r.severity == severity])

            # Reports by class
            reports_by_class = {}
            for report in reports:
                class_name = report.detection.class_name
                reports_by_class[class_name] = reports_by_class.get(class_name, 0) + 1

            # Recent reports (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_reports_count = len([r for r in reports if r.created_at >= recent_cutoff])

            # Average confidence
            if reports:
                avg_confidence = sum(r.detection.confidence for r in reports) / len(reports)
            else:
                avg_confidence = 0.0

            return ReportStatsResponse(
                total_reports=total_reports,
                pending_reports=pending_reports,
                confirmed_reports=confirmed_reports,
                dismissed_reports=dismissed_reports,
                reports_by_severity=reports_by_severity,
                reports_by_class=reports_by_class,
                recent_reports_count=recent_reports_count,
                avg_confidence=avg_confidence
            )

        except Exception as e:
            logger.error(f"❌ Failed to get report stats: {e}")
            raise

    async def confirm_report(self, report_id: str) -> Optional[ReportResponse]:
        """Confirm a pending report"""
        request = ReportUpdateRequest(status=ReportStatus.CONFIRMED)
        report = await self.update_report(report_id, request)
        if report:
            report.confirmed_at = datetime.now()
            await self._store_report(report)
        return report

    async def dismiss_report(self, report_id: str) -> Optional[ReportResponse]:
        """Dismiss a pending report"""
        request = ReportUpdateRequest(status=ReportStatus.DISMISSED)
        return await self.update_report(report_id, request)

    async def _store_report(self, report: ReportResponse):
        """Store report in Redis"""
        if not self.redis_client:
            logger.error("Redis not available for storing report")
            return

        try:
            report_key = f"report:{report.id}"
            report_data = report.json()
            self.redis_client.set(report_key, report_data)
            
            # Set expiration if configured
            if settings.report_retention_days > 0:
                self.redis_client.expire(report_key, settings.report_retention_days * 24 * 3600)

        except Exception as e:
            logger.error(f"❌ Failed to store report in Redis: {e}")
            raise

    async def _geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """Geocode an address to coordinates"""
        if not self.geocoder:
            return None

        try:
            location = self.geocoder.geocode(address, timeout=10)
            if location:
                return (location.latitude, location.longitude)
            return None
        except GeocoderTimedOut:
            logger.warning(f"Geocoding timed out for address: {address}")
            return None
        except Exception as e:
            logger.error(f"Geocoding error for address {address}: {e}")
            return None

    def _determine_severity(self, detection: DetectionResult) -> HazardSeverity:
        """Determine hazard severity based on detection confidence and class (YOLOv12n: crack, pothole)"""
        class_name = detection.class_name.lower()
        confidence = detection.confidence
        
        # YOLOv12n classification: pothole is more critical than crack
        if class_name == "pothole":
            if confidence >= 0.85:
                return HazardSeverity.CRITICAL
            elif confidence >= 0.7:
                return HazardSeverity.HIGH
            else:
                return HazardSeverity.MEDIUM
        elif class_name == "crack":
            if confidence >= 0.9:
                return HazardSeverity.HIGH
            elif confidence >= 0.7:
                return HazardSeverity.MEDIUM
            else:
                return HazardSeverity.LOW
        else:
            # Fallback for unknown classes
            return HazardSeverity.LOW

    def _apply_filters(self, reports: List[ReportResponse], filters: ReportFilterRequest) -> List[ReportResponse]:
        """Apply filtering to reports list"""
        filtered_reports = reports

        if filters.status:
            filtered_reports = [r for r in filtered_reports if r.status in filters.status]

        if filters.severity:
            filtered_reports = [r for r in filtered_reports if r.severity in filters.severity]

        if filters.class_ids:
            filtered_reports = [r for r in filtered_reports if r.detection.class_id in filters.class_ids]

        if filters.min_confidence is not None:
            filtered_reports = [r for r in filtered_reports if r.detection.confidence >= filters.min_confidence]

        if filters.max_confidence is not None:
            filtered_reports = [r for r in filtered_reports if r.detection.confidence <= filters.max_confidence]

        if filters.session_id:
            filtered_reports = [r for r in filtered_reports 
                              if r.metadata and r.metadata.session_id == filters.session_id]

        if filters.source:
            filtered_reports = [r for r in filtered_reports 
                              if r.metadata and r.metadata.source == filters.source]

        if filters.date_from:
            filtered_reports = [r for r in filtered_reports if r.created_at >= filters.date_from]

        if filters.date_to:
            filtered_reports = [r for r in filtered_reports if r.created_at <= filters.date_to]

        return filtered_reports

    def _apply_sorting(self, reports: List[ReportResponse], sort_by: str, sort_order: str) -> List[ReportResponse]:
        """Apply sorting to reports list"""
        reverse = sort_order == "desc"
        
        if sort_by == "created_at":
            return sorted(reports, key=lambda r: r.created_at, reverse=reverse)
        elif sort_by == "confidence":
            return sorted(reports, key=lambda r: r.detection.confidence, reverse=reverse)
        elif sort_by == "severity":
            severity_order = [HazardSeverity.LOW, HazardSeverity.MEDIUM, HazardSeverity.HIGH, HazardSeverity.CRITICAL]
            return sorted(reports, key=lambda r: severity_order.index(r.severity), reverse=reverse)
        else:
            return reports


# Global report service instance
report_service = ReportService()