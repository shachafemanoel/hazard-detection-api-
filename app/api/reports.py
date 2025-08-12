"""
Report API endpoints for hazard detection reports
"""

import asyncio
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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
    ReportStatus
)
from ..services.report_service import report_service
from ..services.sse_service import sse_service

logger = get_logger("reports_api")

router = APIRouter(tags=["reports"])


class ReportConfirmRequest(BaseModel):
    """Request to confirm a report"""
    notes: Optional[str] = None


class ReportDismissRequest(BaseModel):
    """Request to dismiss a report"""
    reason: Optional[str] = None


@router.post("/reports", response_model=ReportResponse)
async def create_report(request: ReportCreateRequest) -> ReportResponse:
    """
    Create a new hazard report
    
    Creates a new report from detection data, uploads images to Cloudinary,
    and stores metadata in Redis for efficient retrieval.
    """
    try:
        start_time = time.time()
        
        # Validate image size if provided
        if request.image_data:
            import base64
            try:
                image_bytes = base64.b64decode(request.image_data.split(',')[-1])
                image_size_mb = len(image_bytes) / (1024 * 1024)
                if image_size_mb > settings.report_image_max_size_mb:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Image too large: {image_size_mb:.2f}MB. Max allowed: {settings.report_image_max_size_mb}MB"
                    )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

        report = await report_service.create_report(request)
        
        # Broadcast SSE event for new report (non-blocking)
        try:
            report_dict = report.model_dump() if hasattr(report, 'model_dump') else report.dict()
            await sse_service.broadcast_report_created(report_dict)
        except Exception as e:
            # Don't fail the request if SSE fails
            logger.warning(f"Failed to broadcast report creation event: {e}")
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Created report {report.id} in {processing_time}ms")
        
        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create report: {str(e)}")


@router.get("/reports", response_model=ReportListResponse)
async def list_reports(
    # Status filters
    status: Optional[str] = None,
    
    # Detection filters
    class_ids: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    
    # Metadata filters
    session_id: Optional[str] = None,
    source: Optional[str] = None,
    
    # Time filters
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    
    # Location filters
    location_bounds: Optional[str] = None,
    
    # Pagination
    page: int = 1,
    limit: int = 20,
    
    # Sorting
    sort_by: str = "created_at",
    sort_order: str = "desc"
) -> ReportListResponse:
    """
    List hazard reports with filtering and pagination
    
    Supports comprehensive filtering by status, detection properties,
    metadata, time ranges, and geographic bounds.
    """
    try:
        # Validate and parse parameters
        if limit > settings.max_reports_per_request:
            limit = settings.max_reports_per_request

        # Parse comma-separated values
        status_list = None
        if status:
            try:
                status_list = [ReportStatus(s.strip()) for s in status.split(',')]
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid status value: {e}")

        class_ids_list = None
        if class_ids:
            try:
                class_ids_list = [int(id.strip()) for id in class_ids.split(',')]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid class_ids format")

        location_bounds_list = None
        if location_bounds:
            try:
                location_bounds_list = [float(x.strip()) for x in location_bounds.split(',')]
                if len(location_bounds_list) != 4:
                    raise ValueError("Location bounds must have 4 values")
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid location_bounds format. Expected: min_lat,min_lon,max_lat,max_lon"
                )

        # Parse dates
        from datetime import datetime
        date_from_dt = None
        date_to_dt = None
        if date_from:
            try:
                date_from_dt = datetime.fromisoformat(date_from)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_from format. Use ISO format.")
        
        if date_to:
            try:
                date_to_dt = datetime.fromisoformat(date_to)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid date_to format. Use ISO format.")

        # Create filter request
        filters = ReportFilterRequest(
            status=status_list,
            class_ids=class_ids_list,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            session_id=session_id,
            source=source,
            location_bounds=location_bounds_list,
            date_from=date_from_dt,
            date_to=date_to_dt,
            page=page,
            limit=limit,
            sort_by=sort_by,
            sort_order=sort_order
        )

        start_time = time.time()
        result = await report_service.list_reports(filters)
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        logger.info(
            f"Listed {len(result.reports)} reports (page {page}, total {result.total_count}) in {processing_time}ms"
        )
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list reports: {str(e)}")


@router.get("/reports/stats", response_model=ReportStatsResponse)
async def get_report_stats() -> ReportStatsResponse:
    """
    Get comprehensive report statistics
    
    Returns aggregated statistics including counts by status,
    severity, class, and recent activity metrics.
    """
    try:
        start_time = time.time()
        stats = await report_service.get_report_stats()
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        logger.info(f"Generated report stats in {processing_time}ms")
        return stats

    except Exception as e:
        logger.error(f"Report stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get report stats: {str(e)}")


@router.get("/reports/{report_id}", response_model=ReportResponse)
async def get_report(report_id: str) -> ReportResponse:
    """
    Get a specific report by ID
    
    Returns detailed report information including detection data,
    images, location, and metadata.
    """
    try:
        report = await report_service.get_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
        
        logger.info(f"Retrieved report {report_id}")
        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report retrieval failed for {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")


@router.patch("/reports/{report_id}", response_model=ReportResponse)
async def update_report(report_id: str, request: ReportUpdateRequest) -> ReportResponse:
    """
    Update an existing report
    
    Allows updating status, description, severity, location,
    and tags for existing reports.
    """
    try:
        start_time = time.time()
        
        report = await report_service.update_report(report_id, request)
        if not report:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
        
        # Broadcast SSE event for report update (non-blocking)
        try:
            report_dict = report.model_dump() if hasattr(report, 'model_dump') else report.dict()
            await sse_service.broadcast_report_updated(report_dict)
        except Exception as e:
            # Don't fail the request if SSE fails
            logger.warning(f"Failed to broadcast report update event: {e}")
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Updated report {report_id} in {processing_time}ms")
        
        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report update failed for {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update report: {str(e)}")


@router.delete("/reports/{report_id}")
async def delete_report(report_id: str) -> Dict[str, Any]:
    """
    Delete a report
    
    Permanently removes the report from storage and deletes
    associated images from Cloudinary.
    """
    try:
        start_time = time.time()
        
        success = await report_service.delete_report(report_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
        
        # Broadcast SSE event for report deletion (non-blocking)
        try:
            await sse_service.broadcast_report_deleted(report_id)
        except Exception as e:
            # Don't fail the request if SSE fails
            logger.warning(f"Failed to broadcast report deletion event: {e}")
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Deleted report {report_id} in {processing_time}ms")
        
        return {
            "success": True,
            "message": f"Report {report_id} deleted successfully",
            "processing_time_ms": processing_time
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report deletion failed for {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete report: {str(e)}")


@router.post("/reports/{report_id}/confirm", response_model=ReportResponse)
async def confirm_report(report_id: str, request: ReportConfirmRequest) -> ReportResponse:
    """
    Confirm a pending report
    
    Changes report status to confirmed and optionally adds notes.
    This typically happens after manual review.
    """
    try:
        start_time = time.time()
        
        report = await report_service.confirm_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
        
        # Add notes if provided
        if request.notes:
            update_request = ReportUpdateRequest(description=request.notes)
            report = await report_service.update_report(report_id, update_request)
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Confirmed report {report_id} in {processing_time}ms")
        
        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report confirmation failed for {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to confirm report: {str(e)}")


@router.post("/reports/{report_id}/dismiss", response_model=ReportResponse)
async def dismiss_report(report_id: str, request: ReportDismissRequest) -> ReportResponse:
    """
    Dismiss a pending report
    
    Changes report status to dismissed and optionally adds dismissal reason.
    This typically happens when a report is determined to be invalid.
    """
    try:
        start_time = time.time()
        
        report = await report_service.dismiss_report(report_id)
        if not report:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
        
        # Add dismissal reason if provided
        if request.reason:
            update_request = ReportUpdateRequest(description=f"Dismissed: {request.reason}")
            report = await report_service.update_report(report_id, update_request)
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        logger.info(f"Dismissed report {report_id} in {processing_time}ms")
        
        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report dismissal failed for {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to dismiss report: {str(e)}")


@router.post("/reports/upload", response_model=ReportResponse)
async def upload_report_with_file(
    file: UploadFile = File(...),
    detection_data: str = Form(...),
    metadata: Optional[str] = Form(None)
) -> ReportResponse:
    """
    Create a report with file upload
    
    Alternative endpoint for creating reports by uploading image files
    directly instead of using base64 encoding.
    """
    try:
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read file content
        file_content = await file.read()
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if file_size_mb > settings.report_image_max_size_mb:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: {file_size_mb:.2f}MB. Max allowed: {settings.report_image_max_size_mb}MB"
            )

        # Parse detection data
        import json
        try:
            detection_dict = json.loads(detection_data)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid detection_data JSON")

        # Parse metadata if provided
        metadata_dict = None
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON")

        # Convert file to base64
        import base64
        image_base64 = base64.b64encode(file_content).decode('utf-8')
        
        # Create detection request
        from ..models.report_models import DetectionInfo, ReportMetadata
        
        detection_info = DetectionInfo(**detection_dict)
        report_metadata = ReportMetadata(**metadata_dict) if metadata_dict else None
        
        request = ReportCreateRequest(
            detection=detection_info,
            image_data=image_base64,
            metadata=report_metadata
        )

        # Create report
        start_time = time.time()
        report = await report_service.create_report(request)
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        logger.info(f"Created report {report.id} from file upload in {processing_time}ms")
        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create report from upload: {str(e)}")


@router.get("/events")
async def reports_events_stream(request: Request):
    """
    Server-Sent Events (SSE) endpoint for real-time report updates
    
    Provides a persistent connection for dashboard clients to receive
    real-time notifications about new reports, updates, and deletions.
    """
    try:
        # Connect client to SSE service
        client_id = await sse_service.connect_client(request)
        
        logger.info(f"📡 New SSE client connected for reports: {client_id}")
        
        # Return streaming response with proper headers
        return StreamingResponse(
            sse_service.get_client_stream(client_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            }
        )
        
    except Exception as e:
        logger.error(f"SSE connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to establish SSE connection: {str(e)}")


@router.get("/events/stats")
async def sse_stats():
    """
    Get SSE service statistics
    Returns information about active connections and service status
    """
    try:
        stats = sse_service.get_stats()
        return {"sse_stats": stats}
    except Exception as e:
        logger.error(f"Failed to get SSE stats: {e}")
        return {"error": str(e), "sse_stats": {}}