"""
Session management API endpoints
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.logging_config import get_logger
from ..core.exceptions import SessionNotFoundException
from ..services.session_service import session_service

logger = get_logger("sessions_api")

router = APIRouter(prefix="/session", tags=["sessions"])


class SessionStartRequest(BaseModel):
    confidence_threshold: Optional[float] = 0.5
    source: Optional[str] = "web_app"
    user_id: Optional[str] = None


@router.post("/start")
async def start_session(session_config: SessionStartRequest = None) -> Dict[str, Any]:
    """Start a new detection session with optional configuration"""
    try:
        # Create session with optional configuration
        session_metadata = {}
        if session_config:
            session_metadata = {
                "confidence_threshold": session_config.confidence_threshold,
                "source": session_config.source,
                "user_id": session_config.user_id
            }
        
        session_id = session_service.create_session(metadata=session_metadata)
        logger.info(f"Started new session: {session_id} with config: {session_metadata}")
        return {
            "session_id": session_id,
            "confidence_threshold": session_config.confidence_threshold if session_config else 0.5,
            "status": "created",
        }
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start session: {str(e)}"
        )


@router.post("/{session_id}/end")
async def end_session(session_id: str) -> Dict[str, str]:
    """End a detection session and return summary"""
    try:
        result = session_service.end_session(session_id)
        logger.info(f"Ended session: {session_id}")
        return result
    except SessionNotFoundException as e:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to end session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to end session: {str(e)}")


@router.get("/{session_id}/summary")
async def get_session_summary(session_id: str) -> Dict[str, Any]:
    """Get session summary with all reports"""
    try:
        summary = session_service.get_session_summary(session_id)
        return summary
    except SessionNotFoundException as e:
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get session summary {session_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get session summary: {str(e)}"
        )


@router.post("/{session_id}/report/{report_id}/confirm")
async def confirm_report(session_id: str, report_id: str) -> Dict[str, str]:
    """Confirm a report for submission"""
    try:
        result = session_service.confirm_report(session_id, report_id)
        return result
    except SessionNotFoundException as e:
        logger.warning(f"Session or report not found: {session_id}/{report_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to confirm report {report_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to confirm report: {str(e)}"
        )


@router.post("/{session_id}/report/{report_id}/dismiss")
async def dismiss_report(session_id: str, report_id: str) -> Dict[str, str]:
    """Dismiss a report"""
    try:
        result = session_service.dismiss_report(session_id, report_id)
        return result
    except SessionNotFoundException as e:
        logger.warning(f"Session or report not found: {session_id}/{report_id}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to dismiss report {report_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to dismiss report: {str(e)}"
        )
