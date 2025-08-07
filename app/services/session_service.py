"""
Session service for managing detection sessions and report tracking
"""

import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import math

from ..core.config import settings
from ..core.logging_config import get_logger
from ..core.exceptions import SessionNotFoundException
from .model_service import DetectionResult

logger = get_logger("session_service")


class DetectionReport:
    """Represents a detection report in a session"""

    def __init__(
        self,
        detection: DetectionResult,
        session_id: str,
        image_data: Optional[str] = None,
    ):
        self.report_id = str(uuid.uuid4())
        self.session_id = session_id
        self.detection = detection
        self.timestamp = datetime.now().isoformat()
        self.status = "pending"  # pending, confirmed, dismissed
        self.image_data = image_data
        self.thumbnail = None
        self.persistent_report_id = None  # ID of persistent report if created
        self.create_persistent = False  # Flag for persistent report creation
        self.image_data_for_persistent = None  # Image data for persistent report

        # Location information
        self.location = {
            "bbox": detection.bbox,
            "center": [detection.center_x, detection.center_y],
        }

        # Frame information
        self.frame_info = {
            "has_image": image_data is not None,
            "image_size": len(image_data) if image_data else 0,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            "report_id": self.report_id,
            "session_id": self.session_id,
            "detection": self.detection.to_dict(),
            "timestamp": self.timestamp,
            "status": self.status,
            "location": self.location,
            "frame_info": self.frame_info,
            "persistent_report_id": self.persistent_report_id,
            "has_persistent_report": self.persistent_report_id is not None,
        }


class DetectionSession:
    """Represents a detection session"""

    def __init__(self, session_id: str, metadata: Optional[Dict[str, Any]] = None):
        self.id = session_id
        self.start_time = datetime.now().isoformat()
        self.metadata = metadata or {}
        self.reports: List[DetectionReport] = []
        self.detection_count = 0
        self.unique_hazards = 0
        self.active_detections: List[Dict[str, Any]] = []

    def add_report(self, report: DetectionReport):
        """Add a report to the session"""
        self.reports.append(report)
        self.unique_hazards += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get session summary"""
        # Calculate session duration
        from datetime import datetime
        start_dt = datetime.fromisoformat(self.start_time)
        current_dt = datetime.now()
        duration_seconds = (current_dt - start_dt).total_seconds()
        
        # Calculate average confidence from reports
        total_confidence = 0.0
        confidence_count = 0
        for report in self.reports:
            if report.detection.confidence > 0:
                total_confidence += report.detection.confidence
                confidence_count += 1
        
        avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0.0
        
        # Get class distribution
        class_counts = {}
        for report in self.reports:
            class_name = report.detection.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            "session_id": self.id,
            "start_time": self.start_time,
            "duration_seconds": duration_seconds,
            "metadata": self.metadata,
            "stats": {
                "total_detections": self.detection_count,
                "unique_hazards": self.unique_hazards,
                "pending_reports": len([r for r in self.reports if r.status == "pending"]),
                "confirmed_reports": len([r for r in self.reports if r.status == "confirmed"]),
                "dismissed_reports": len([r for r in self.reports if r.status == "dismissed"]),
                "average_confidence": round(avg_confidence, 3),
                "class_distribution": class_counts,
                "active_detections_count": len(self.active_detections)
            },
            "reports": [report.to_dict() for report in self.reports]
        }


class SessionService:
    """Service for managing detection sessions"""

    def __init__(self):
        self.sessions: Dict[str, DetectionSession] = {}
        self.active_detections: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def create_session(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new detection session"""
        session_id = str(uuid.uuid4())
        session = DetectionSession(session_id, metadata)
        self.sessions[session_id] = session
        self.active_detections[session_id] = []

        logger.info(f"Created new session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> DetectionSession:
        """Get a session by ID"""
        if session_id not in self.sessions:
            raise SessionNotFoundException(f"Session {session_id} not found")
        return self.sessions[session_id]

    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a session and clean up resources"""
        if session_id not in self.sessions:
            raise SessionNotFoundException(f"Session {session_id} not found")

        session = self.sessions.pop(session_id)
        self.active_detections.pop(session_id, None)

        logger.info(f"Ended session: {session_id}")
        return {"message": "Session ended", "session_id": session_id}

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get session summary"""
        session = self.get_session(session_id)
        return session.get_summary()

    async def process_detections(
        self,
        session_id: str,
        detections: List[DetectionResult],
        image_data: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process detections for a session with tracking and report generation"""
        session = self.get_session(session_id)

        new_reports = []
        processed_detections = []
        current_time = time.time()

        for detection in detections:
            # Increment detection count
            session.detection_count += 1

            detection_dict = detection.to_dict()
            detection_dict["timestamp"] = current_time

            # Check for high-confidence detections that should generate reports
            if detection.confidence >= settings.min_confidence_for_report:
                is_duplicate, existing_report_id = self._is_duplicate_detection(
                    detection, self.active_detections[session_id]
                )

                if not is_duplicate:
                    # Create new report
                    report = DetectionReport(detection, session_id, image_data)
                    session.add_report(report)
                    new_reports.append(report)

                    # Mark for persistent report creation (will be handled after main processing)
                    if settings.auto_create_reports:
                        report.create_persistent = True
                        report.image_data_for_persistent = image_data

                    # Add to active detections for tracking
                    tracking_detection = detection.to_dict()
                    tracking_detection.update(
                        {"report_id": report.report_id, "timestamp": current_time}
                    )
                    self.active_detections[session_id].append(tracking_detection)

                    # Mark as new detection
                    detection_dict["is_new"] = True
                    detection_dict["report_id"] = report.report_id
                else:
                    # Mark as existing detection
                    detection_dict["is_new"] = False
                    detection_dict["report_id"] = existing_report_id
            else:
                # Low confidence detections are not tracked
                detection_dict["is_new"] = False
                detection_dict["report_id"] = None

            processed_detections.append(detection_dict)

        # Return processing results
        result = {
            "detections": processed_detections,
            "new_reports": [report.to_dict() for report in new_reports],
            "session_stats": {
                "total_detections": session.detection_count,
                "unique_hazards": session.unique_hazards,
                "pending_reports": len(
                    [r for r in session.reports if r.status == "pending"]
                ),
            },
        }
        
        # Create persistent reports asynchronously (don't block the response)
        reports_to_persist = [r for r in new_reports if getattr(r, 'create_persistent', False)]
        if reports_to_persist:
            try:
                import asyncio
                current_loop = asyncio.current_task()
                if current_loop:
                    # Create background task for persistent report creation
                    asyncio.create_task(self._create_persistent_reports(reports_to_persist, session_id))
                    logger.info(f"Scheduled {len(reports_to_persist)} reports for persistent creation")
            except Exception as e:
                logger.error(f"Failed to schedule persistent reports: {e}")
        
        return result

    def confirm_report(self, session_id: str, report_id: str) -> Dict[str, Any]:
        """Confirm a report for submission"""
        session = self.get_session(session_id)

        for report in session.reports:
            if report.report_id == report_id:
                report.status = "confirmed"
                logger.info(f"Confirmed report {report_id} in session {session_id}")
                return {"message": "Report confirmed", "report_id": report_id}

        raise SessionNotFoundException(
            f"Report {report_id} not found in session {session_id}"
        )

    def dismiss_report(self, session_id: str, report_id: str) -> Dict[str, Any]:
        """Dismiss a report"""
        session = self.get_session(session_id)

        for report in session.reports:
            if report.report_id == report_id:
                report.status = "dismissed"
                logger.info(f"Dismissed report {report_id} in session {session_id}")
                return {"message": "Report dismissed", "report_id": report_id}

        raise SessionNotFoundException(
            f"Report {report_id} not found in session {session_id}"
        )

    def _is_duplicate_detection(
        self, new_detection: DetectionResult, existing_detections: List[Dict[str, Any]]
    ) -> tuple[bool, Optional[str]]:
        """Check if detection is duplicate based on location and time"""
        current_time = time.time()
        new_bbox = new_detection.bbox

        for existing in existing_detections:
            # Check if same class and within time threshold
            if (
                existing["class_id"] == new_detection.class_id
                and current_time - existing["timestamp"]
                < settings.tracking_time_threshold
            ):

                # Check spatial proximity
                distance = self._calculate_distance(new_bbox, existing["bbox"])
                if distance < settings.tracking_distance_threshold:
                    return True, existing.get("report_id")

        return False, None

    def _calculate_distance(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Euclidean distance between box centers"""
        x1, y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
        x2, y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_active_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)

    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old inactive sessions"""
        current_time = datetime.now()
        sessions_to_remove = []

        for session_id, session in self.sessions.items():
            session_start = datetime.fromisoformat(session.start_time)
            age_hours = (current_time - session_start).total_seconds() / 3600

            if age_hours > max_age_hours:
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            self.end_session(session_id)
            logger.info(f"Cleaned up old session: {session_id}")

        return len(sessions_to_remove)

    async def _create_persistent_reports(self, reports: List[DetectionReport], session_id: str):
        """Create persistent reports in the background"""
        try:
            from .report_service import report_service
            
            for report in reports:
                try:
                    additional_metadata = {
                        "processing_time_ms": time.time() * 1000,
                        "session_report_id": report.report_id
                    }
                    
                    persistent_report = await report_service.create_report_from_detection(
                        report.detection, session_id, 
                        report.image_data_for_persistent, additional_metadata
                    )
                    
                    # Update session report with persistent ID
                    report.persistent_report_id = persistent_report.id
                    logger.info(f"Created persistent report {persistent_report.id} for session report {report.report_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to create persistent report for {report.report_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to create persistent reports: {e}")


# Global session service instance
session_service = SessionService()
