"""
Report service implementing B3 persistence contract exactly:
- Store detection record under report:{uuid} 
- Fields: sessionId, class, confidence, ts, geo, cloudinaryUrl, status
- RPUSH session:{sessionId}:reports {uuid} to index per session
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import UploadFile

from ..core.config import settings
from ..core.logging_config import get_logger
from ..models.schemas import (
    Report, SessionSummary, DetectionMeta, GeoLocation,
    CreateReportRequest, CreateReportResponse, UploadDetectionResponse
)
from .redis_service import redis_service
from .cloudinary_service import cloudinary_service

logger = get_logger("report_service_b3")


class ReportServiceB3:
    """
    Report service following B3 persistence contract exactly
    """
    
    def __init__(self):
        self.redis_service = redis_service
        self.cloudinary_service = cloudinary_service
    
    async def upload_detection(self, file: UploadFile, meta: DetectionMeta) -> UploadDetectionResponse:
        """
        Upload detection image to Cloudinary and persist metadata in Redis
        Used by POST /report endpoint
        
        Args:
            file: Uploaded image file
            meta: Detection metadata (sessionId, className, confidence, ts, geo)
            
        Returns:
            UploadDetectionResponse with id and cloudinaryUrl
        """
        try:
            # Read image data
            image_data = await file.read()
            
            # Upload to Cloudinary via server
            upload_result = await self.cloudinary_service.upload_detection_blob(
                image_data, 
                meta.dict()
            )
            
            # Create report record for Redis storage
            report_id = str(uuid.uuid4())
            report_data = {
                "sessionId": meta.sessionId,
                "className": meta.className,  # Changed from 'class' to match B3 spec
                "confidence": meta.confidence,
                "ts": meta.ts,
                "geo": meta.geo.dict() if meta.geo else None,
                "cloudinaryUrl": upload_result["cloudinaryUrl"],
                "status": "pending",
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Store in Redis following B3 contract
            success = self.redis_service.store_report(report_id, report_data)
            if not success:
                logger.error(f"Failed to store report {report_id} in Redis")
                # Continue anyway - Cloudinary upload succeeded
            
            # Add to session index
            self.redis_service.add_report_to_session(meta.sessionId, report_id)
            
            logger.info(f"✅ Detection uploaded: {report_id} for session {meta.sessionId}")
            
            return UploadDetectionResponse(
                id=report_id,
                cloudinaryUrl=upload_result["cloudinaryUrl"]
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to upload detection: {e}")
            raise
    
    async def create_report(self, session_id: str, cloudinary_url: str, meta: DetectionMeta) -> CreateReportResponse:
        """
        Create a report record referencing an already-uploaded Cloudinary URL
        Used by createReport from client
        
        Args:
            session_id: Session identifier
            cloudinary_url: Already uploaded Cloudinary URL
            meta: Detection metadata (without sessionId)
            
        Returns:
            CreateReportResponse with generated report ID
        """
        try:
            report_id = str(uuid.uuid4())
            report_data = {
                "sessionId": session_id,
                "className": meta.className,
                "confidence": meta.confidence, 
                "ts": meta.ts,
                "geo": meta.geo.dict() if meta.geo else None,
                "cloudinaryUrl": cloudinary_url,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Store in Redis following B3 contract
            success = self.redis_service.store_report(report_id, report_data)
            if not success:
                raise Exception("Failed to store report in Redis")
            
            # Add to session index
            self.redis_service.add_report_to_session(session_id, report_id)
            
            logger.info(f"✅ Report created: {report_id} for session {session_id}")
            
            return CreateReportResponse(id=report_id)
            
        except Exception as e:
            logger.error(f"❌ Failed to create report: {e}")
            raise
    
    async def get_session_reports(self, session_id: str) -> List[Dict[str, Any]]:
        """
        GET /session/{id}/reports → returns full objects, including cloudinaryUrl
        """
        try:
            report_ids = self.redis_service.get_session_reports(session_id)
            reports = []
            
            for report_id in report_ids:
                report_data = self.redis_service.get_report(report_id)
                if report_data:
                    # Ensure cloudinaryUrl is included
                    report_data["id"] = report_id
                    reports.append(report_data)
            
            logger.debug(f"Retrieved {len(reports)} reports for session {session_id}")
            return reports
            
        except Exception as e:
            logger.error(f"❌ Failed to get session reports {session_id}: {e}")
            return []
    
    async def get_session_summary(self, session_id: str) -> Optional[SessionSummary]:
        """
        GET /session/{id}/summary → aggregates counts, types, avg confidence, 
        start/end times, plus list of {uuid, cloudinaryUrl} for Summary modal
        """
        try:
            # Get aggregated summary from Redis
            summary_data = self.redis_service.get_session_summary(session_id)
            
            if not summary_data:
                # Return empty summary if no session found
                return SessionSummary(
                    sessionId=session_id,
                    count=0,
                    types={},
                    avgConfidence=0.0,
                    durationMs=0,
                    reports=[]
                )
            
            # Convert to SessionSummary model
            return SessionSummary(**summary_data)
            
        except Exception as e:
            logger.error(f"❌ Failed to get session summary {session_id}: {e}")
            return None
    
    async def confirm_report(self, session_id: str, report_id: str) -> bool:
        """
        POST /session/{id}/report/{uuid}/confirm → update status to 'confirmed'
        """
        try:
            success = self.redis_service.update_report_status(report_id, "confirmed")
            if success:
                logger.info(f"✅ Report {report_id} confirmed for session {session_id}")
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to confirm report {report_id}: {e}")
            return False
    
    async def dismiss_report(self, session_id: str, report_id: str) -> bool:
        """
        POST /session/{id}/report/{uuid}/dismiss → update status to 'dismissed'
        """
        try:
            success = self.redis_service.update_report_status(report_id, "dismissed")
            if success:
                logger.info(f"✅ Report {report_id} dismissed for session {session_id}")
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to dismiss report {report_id}: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get report service health status"""
        redis_status = self.redis_service.get_health_status()
        cloudinary_status = self.cloudinary_service.get_health_status()
        
        overall_healthy = (
            redis_status.get("connected", False) and 
            cloudinary_status.get("configured", False)
        )
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "redis": redis_status,
            "cloudinary": cloudinary_status
        }


# Global report service instance following B3 contract
report_service_b3 = ReportServiceB3()