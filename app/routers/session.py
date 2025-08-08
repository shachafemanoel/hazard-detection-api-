"""
Session router - session management endpoints
B3 requirements: GET /session/{id}/reports, GET /session/{id}/summary, confirm/dismiss endpoints
"""

import uuid
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Path
from fastapi.responses import JSONResponse

from ..core.logging_config import get_logger
from ..models.schemas import (
    SessionStartResponse, SessionSummary, ReportActionResponse,
    ErrorResponse
)
from ..services.report_service_b3 import report_service_b3

logger = get_logger("session_router")

router = APIRouter(prefix="/session", tags=["session"])


@router.post("/start", response_model=SessionStartResponse)
async def start_session() -> SessionStartResponse:
    """
    Start a new detection session
    Returns a unique session ID
    """
    try:
        session_id = str(uuid.uuid4())
        logger.info(f"✅ Started new session: {session_id}")
        
        return SessionStartResponse(
            session_id=session_id,
            message="Session started successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start session: {str(e)}"
        )


@router.get("/{session_id}/reports")
async def get_session_reports(
    session_id: str = Path(..., description="Session ID")
) -> List[Dict[str, Any]]:
    """
    GET /session/{id}/reports → returns full objects, including cloudinaryUrl
    B3 requirement: return full report objects with cloudinaryUrl
    """
    try:
        # Validate session_id format
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="Invalid session ID format"
            )
        
        reports = await report_service_b3.get_session_reports(session_id)
        
        logger.info(f"Retrieved {len(reports)} reports for session {session_id}")
        
        return reports
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get reports for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session reports: {str(e)}"
        )


@router.get("/{session_id}/summary", response_model=SessionSummary)
async def get_session_summary(
    session_id: str = Path(..., description="Session ID")
) -> SessionSummary:
    """
    GET /session/{id}/summary → aggregates counts, types, avg conf, start/end times,
    plus the list of {uuid, cloudinaryUrl} for the Summary modal
    B3 requirement: return aggregated session data for client Summary modal
    """
    try:
        # Validate session_id format
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="Invalid session ID format"
            )
        
        summary = await report_service_b3.get_session_summary(session_id)
        
        if summary is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        logger.info(
            f"Retrieved summary for session {session_id}: "
            f"{summary.count} reports, {len(summary.types)} types"
        )
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get summary for session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session summary: {str(e)}"
        )


@router.post("/{session_id}/report/{report_id}/confirm", response_model=ReportActionResponse)
async def confirm_report(
    session_id: str = Path(..., description="Session ID"),
    report_id: str = Path(..., description="Report ID")
) -> ReportActionResponse:
    """
    POST /session/{id}/report/{uuid}/confirm → update status to 'confirmed'
    B3 requirement: confirm/dismiss report status updates
    """
    try:
        # Validate UUIDs
        try:
            uuid.UUID(session_id)
            uuid.UUID(report_id)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="Invalid session ID or report ID format"
            )
        
        success = await report_service_b3.confirm_report(session_id, report_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Report {report_id} not found in session {session_id}"
            )
        
        logger.info(f"✅ Report {report_id} confirmed for session {session_id}")
        
        return ReportActionResponse(
            message="Report confirmed",
            report_id=report_id,
            new_status="confirmed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to confirm report {report_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to confirm report: {str(e)}"
        )


@router.post("/{session_id}/report/{report_id}/dismiss", response_model=ReportActionResponse)
async def dismiss_report(
    session_id: str = Path(..., description="Session ID"),
    report_id: str = Path(..., description="Report ID")
) -> ReportActionResponse:
    """
    POST /session/{id}/report/{uuid}/dismiss → update status to 'dismissed'
    B3 requirement: confirm/dismiss report status updates
    """
    try:
        # Validate UUIDs
        try:
            uuid.UUID(session_id)
            uuid.UUID(report_id)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="Invalid session ID or report ID format"
            )
        
        success = await report_service_b3.dismiss_report(session_id, report_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Report {report_id} not found in session {session_id}"
            )
        
        logger.info(f"✅ Report {report_id} dismissed for session {session_id}")
        
        return ReportActionResponse(
            message="Report dismissed",
            report_id=report_id,
            new_status="dismissed"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to dismiss report {report_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to dismiss report: {str(e)}"
        )


@router.delete("/{session_id}")
async def end_session(
    session_id: str = Path(..., description="Session ID")
) -> Dict[str, str]:
    """
    End a session (optional - for cleanup)
    This doesn't delete reports, just marks the session as ended
    """
    try:
        # Validate session_id format
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="Invalid session ID format"
            )
        
        # For now, just log the session end
        # In the future, this could set an end timestamp or trigger cleanup
        logger.info(f"Session {session_id} ended")
        
        return {
            "message": f"Session {session_id} ended successfully",
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to end session {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to end session: {str(e)}"
        )


@router.get("/{session_id}/health")
async def get_session_health(
    session_id: str = Path(..., description="Session ID")
) -> Dict[str, Any]:
    """
    Get health status for a specific session
    Useful for debugging session-specific issues
    """
    try:
        # Validate session_id format
        try:
            uuid.UUID(session_id)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail="Invalid session ID format"
            )
        
        # Get basic session info
        reports = await report_service_b3.get_session_reports(session_id)
        summary = await report_service_b3.get_session_summary(session_id)
        
        # Get service health
        service_health = report_service_b3.get_health_status()
        
        return {
            "session_id": session_id,
            "report_count": len(reports),
            "has_summary": summary is not None,
            "summary_count": summary.count if summary else 0,
            "service_health": service_health
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get session health {session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session health: {str(e)}"
        )