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
    
    def __init__(self, detection: DetectionResult, session_id: str, image_data: Optional[str] = None):
        self.report_id = str(uuid.uuid4())
        self.session_id = session_id
        self.detection = detection
        self.timestamp = datetime.now().isoformat()
        self.status = "pending"  # pending, confirmed, dismissed
        self.image_data = image_data
        self.thumbnail = None
        
        # Location information
        self.location = {
            "bbox": detection.bbox,
            "center": [detection.center_x, detection.center_y]
        }
        
        # Frame information
        self.frame_info = {
            "has_image": image_data is not None,
            "image_size": len(image_data) if image_data else 0
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
            "frame_info": self.frame_info
        }


class DetectionSession:
    """Represents a detection session"""
    
    def __init__(self, session_id: str):
        self.id = session_id
        self.start_time = datetime.now().isoformat()
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
        return {
            "id": self.id,
            "start_time": self.start_time,
            "reports": [report.to_dict() for report in self.reports],
            "detection_count": self.detection_count,
            "unique_hazards": self.unique_hazards,
            "pending_reports": len([r for r in self.reports if r.status == "pending"]),
            "confirmed_reports": len([r for r in self.reports if r.status == "confirmed"]),
            "dismissed_reports": len([r for r in self.reports if r.status == "dismissed"])
        }


class SessionService:
    """Service for managing detection sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, DetectionSession] = {}
        self.active_detections: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def create_session(self) -> str:
        """Create a new detection session"""
        session_id = str(uuid.uuid4())
        session = DetectionSession(session_id)
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
    
    def process_detections(self, session_id: str, detections: List[DetectionResult], 
                          image_data: Optional[str] = None) -> Dict[str, Any]:
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
                    
                    # Add to active detections for tracking
                    tracking_detection = detection.to_dict()
                    tracking_detection.update({
                        "report_id": report.report_id,
                        "timestamp": current_time
                    })
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
        
        return {
            "detections": processed_detections,
            "new_reports": [report.to_dict() for report in new_reports],
            "session_stats": {
                "total_detections": session.detection_count,
                "unique_hazards": session.unique_hazards,
                "pending_reports": len([r for r in session.reports if r.status == "pending"])
            }
        }
    
    def confirm_report(self, session_id: str, report_id: str) -> Dict[str, Any]:
        """Confirm a report for submission"""
        session = self.get_session(session_id)
        
        for report in session.reports:
            if report.report_id == report_id:
                report.status = "confirmed"
                logger.info(f"Confirmed report {report_id} in session {session_id}")
                return {"message": "Report confirmed", "report_id": report_id}
        
        raise SessionNotFoundException(f"Report {report_id} not found in session {session_id}")
    
    def dismiss_report(self, session_id: str, report_id: str) -> Dict[str, Any]:
        """Dismiss a report"""
        session = self.get_session(session_id)
        
        for report in session.reports:
            if report.report_id == report_id:
                report.status = "dismissed"
                logger.info(f"Dismissed report {report_id} in session {session_id}")
                return {"message": "Report dismissed", "report_id": report_id}
        
        raise SessionNotFoundException(f"Report {report_id} not found in session {session_id}")
    
    def _is_duplicate_detection(self, new_detection: DetectionResult, 
                               existing_detections: List[Dict[str, Any]]) -> tuple[bool, Optional[str]]:
        """Check if detection is duplicate based on location and time"""
        current_time = time.time()
        new_bbox = new_detection.bbox
        
        for existing in existing_detections:
            # Check if same class and within time threshold
            if (existing["class_id"] == new_detection.class_id and 
                current_time - existing["timestamp"] < settings.tracking_time_threshold):
                
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


# Global session service instance
session_service = SessionService()