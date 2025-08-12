"""
Streaming service for real-time hazard detection
Manages WebSocket and SSE connections for live video processing
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass, field
from collections import deque
from fastapi import WebSocket
import base64
from io import BytesIO
from PIL import Image

from ..core.config import settings
from ..core.logging_config import get_logger
from ..services.model_service import model_service, DetectionResult
from ..models.streaming_models import (
    StreamingConfiguration, 
    FrameProcessingResult,
    DetectionStreamEvent,
    StreamingStatsResponse
)

logger = get_logger("streaming_service")


@dataclass
class FrameMetrics:
    """Performance metrics for frame processing"""
    total_frames: int = 0
    processed_frames: int = 0
    failed_frames: int = 0
    avg_processing_time: float = 0.0
    fps: float = 0.0
    last_frame_time: Optional[float] = None
    processing_times: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class StreamSession:
    """Represents an active streaming session"""
    session_id: str
    client_id: str
    connection_type: str  # "websocket" or "sse"
    config: StreamingConfiguration
    created_at: float
    last_activity: float
    websocket: Optional[WebSocket] = None
    event_queue: Optional[asyncio.Queue] = None
    metrics: FrameMetrics = field(default_factory=FrameMetrics)
    status: str = "active"
    
    def is_expired(self, timeout_seconds: int = 300) -> bool:
        """Check if session has expired"""
        return (time.time() - self.last_activity) > timeout_seconds


class FrameProcessor:
    """Handles frame processing with rate limiting and quality optimization"""
    
    def __init__(self, streaming_service=None):
        self._processing_queue = asyncio.Queue(maxsize=50)  # Limit queue size
        self._worker_task: Optional[asyncio.Task] = None
        self._is_processing = False
        self._streaming_service = streaming_service
        
    async def start_processing(self):
        """Start the frame processing worker"""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_frames_worker())
            logger.info("🎬 Frame processing worker started")
    
    async def stop_processing(self):
        """Stop the frame processing worker"""
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            logger.info("🎬 Frame processing worker stopped")
    
    async def _process_frames_worker(self):
        """Background worker for processing frames"""
        while True:
            try:
                session_id, frame_data = await self._processing_queue.get()
                await self._process_single_frame(session_id, frame_data)
                self._processing_queue.task_done()
            except asyncio.CancelledError:
                logger.info("Frame processing worker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in frame processing worker: {e}")
    
    async def queue_frame(self, session_id: str, frame_data: Dict[str, Any]) -> bool:
        """Queue a frame for processing"""
        try:
            self._processing_queue.put_nowait((session_id, frame_data))
            return True
        except asyncio.QueueFull:
            logger.warning(f"Frame processing queue full for session {session_id}")
            return False
    
    async def _process_single_frame(self, session_id: str, frame_data: Dict[str, Any]):
        """Process a single frame"""
        try:
            if self._streaming_service:
                await self._streaming_service._handle_frame_processing(session_id, frame_data)
            else:
                logger.error(f"No streaming service reference available for session {session_id}")
        except Exception as e:
            logger.error(f"Frame processing failed for session {session_id}: {e}")


class StreamingService:
    """Main streaming service for managing real-time detection sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, StreamSession] = {}
        self.active_connections: Set[str] = set()
        self.frame_processor = FrameProcessor(streaming_service=self)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stats_update_task: Optional[asyncio.Task] = None
        self._global_metrics = {
            "total_sessions": 0,
            "active_sessions": 0,
            "frames_processed": 0,
            "avg_fps": 0.0,
            "uptime": time.time()
        }
        
    async def initialize(self):
        """Initialize the streaming service"""
        await self.frame_processor.start_processing()
        self._start_background_tasks()
        logger.info("🎬 Streaming service initialized")
    
    async def cleanup(self):
        """Cleanup streaming service resources"""
        await self.frame_processor.stop_processing()
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        if self._stats_update_task and not self._stats_update_task.done():
            self._stats_update_task.cancel()
        logger.info("🎬 Streaming service cleanup complete")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        if self._stats_update_task is None or self._stats_update_task.done():
            self._stats_update_task = asyncio.create_task(self._update_global_stats())
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(60)  # Clean up every minute
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                logger.info("Streaming cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in streaming cleanup: {e}")
    
    async def _update_global_stats(self):
        """Update global statistics"""
        while True:
            try:
                await asyncio.sleep(5)  # Update every 5 seconds
                self._calculate_global_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating global stats: {e}")
    
    def _calculate_global_metrics(self):
        """Calculate global streaming metrics"""
        active_sessions = len([s for s in self.sessions.values() if s.status == "active"])
        total_fps = sum(s.metrics.fps for s in self.sessions.values() if s.status == "active")
        avg_fps = total_fps / max(active_sessions, 1)
        
        self._global_metrics.update({
            "active_sessions": active_sessions,
            "avg_fps": round(avg_fps, 2),
            "frames_processed": sum(s.metrics.processed_frames for s in self.sessions.values())
        })
    
    async def create_session(
        self,
        client_id: str,
        session_id: Optional[str] = None,
        connection_type: str = "sse",
        websocket: Optional[WebSocket] = None,
        config: Optional[StreamingConfiguration] = None
    ) -> StreamSession:
        """Create a new streaming session"""
        session_id = session_id or str(uuid.uuid4())
        config = config or StreamingConfiguration()
        
        session = StreamSession(
            session_id=session_id,
            client_id=client_id,
            connection_type=connection_type,
            config=config,
            created_at=time.time(),
            last_activity=time.time(),
            websocket=websocket
        )
        
        if connection_type == "sse":
            session.event_queue = asyncio.Queue()
        
        self.sessions[client_id] = session
        self.active_connections.add(client_id)
        self._global_metrics["total_sessions"] += 1
        
        logger.info(f"🎬 Created streaming session: {session_id} ({connection_type})")
        return session
    
    async def close_session(self, client_id: str) -> bool:
        """Close a streaming session"""
        if client_id not in self.sessions:
            return False
        
        session = self.sessions[client_id]
        session.status = "closed"
        
        # Close WebSocket if active
        if session.websocket:
            try:
                await session.websocket.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket for {client_id}: {e}")
        
        # Clean up
        self.sessions.pop(client_id, None)
        self.active_connections.discard(client_id)
        
        logger.info(f"🎬 Closed streaming session: {client_id}")
        return True
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_sessions = []
        timeout = getattr(settings, 'streaming_session_timeout', 300)
        
        for client_id, session in self.sessions.items():
            if session.is_expired(timeout):
                expired_sessions.append(client_id)
        
        for client_id in expired_sessions:
            await self.close_session(client_id)
            logger.info(f"🧹 Removed expired streaming session: {client_id}")
    
    def get_session(self, client_id: str) -> Optional[StreamSession]:
        """Get a streaming session by client ID"""
        return self.sessions.get(client_id)
    
    async def process_frame(self, session_id: str, frame_data: Dict[str, Any]):
        """Process a frame for streaming detection (WebSocket)"""
        # Find session by session_id
        session = None
        for s in self.sessions.values():
            if s.session_id == session_id:
                session = s
                break
        
        if not session:
            logger.warning(f"Session not found for processing: {session_id}")
            return
        
        # Update activity timestamp
        session.last_activity = time.time()
        session.metrics.total_frames += 1
        
        # Rate limiting check
        if not self._should_process_frame(session):
            return
        
        # Queue frame for processing
        queued = await self.frame_processor.queue_frame(session.client_id, frame_data)
        if not queued:
            session.metrics.failed_frames += 1
            logger.warning(f"Failed to queue frame for session {session_id}")
    
    async def process_frame_background(self, client_id: str, frame_data: Dict[str, Any]):
        """Process frame in background for SSE clients"""
        session = self.get_session(client_id)
        if not session:
            return
        
        session.last_activity = time.time()
        session.metrics.total_frames += 1
        
        # Rate limiting check
        if not self._should_process_frame(session):
            return
        
        # Queue frame for processing
        await self.frame_processor.queue_frame(client_id, frame_data)
    
    def _should_process_frame(self, session: StreamSession) -> bool:
        """Check if frame should be processed based on rate limiting"""
        now = time.time()
        
        # FPS limiting
        if session.metrics.last_frame_time:
            time_since_last = now - session.metrics.last_frame_time
            min_interval = 1.0 / session.config.fps_limit
            if time_since_last < min_interval:
                return False
        
        session.metrics.last_frame_time = now
        return True
    
    async def _handle_frame_processing(self, client_id: str, frame_data: Dict[str, Any]):
        """Handle actual frame processing"""
        session = self.get_session(client_id)
        if not session:
            return
        
        start_time = time.time()
        
        try:
            # Decode image
            image = self._decode_frame_data(frame_data.get("image_data", ""))
            if not image:
                session.metrics.failed_frames += 1
                return
            
            # Run inference
            detections = await model_service.predict(image)
            
            # Create detection event
            processing_time = time.time() - start_time
            event = DetectionStreamEvent(
                frame_id=frame_data.get("frame_id", str(uuid.uuid4())),
                timestamp=frame_data.get("timestamp", time.time()),
                detections=[det.to_dict() for det in detections],
                processing_time_ms=round(processing_time * 1000, 2),
                session_id=session.session_id
            )
            
            # Update metrics
            session.metrics.processed_frames += 1
            session.metrics.processing_times.append(processing_time)
            self._update_session_metrics(session)
            
            # Send result based on connection type
            if session.connection_type == "websocket" and session.websocket:
                await self._send_websocket_result(session, event)
            elif session.connection_type == "sse" and session.event_queue:
                await self._send_sse_result(session, event)
            
        except Exception as e:
            session.metrics.failed_frames += 1
            logger.error(f"Frame processing error for {client_id}: {e}")
            
            # Send error event
            error_event = {
                "type": "error",
                "error": str(e),
                "frame_id": frame_data.get("frame_id"),
                "timestamp": time.time()
            }
            
            if session.connection_type == "websocket" and session.websocket:
                try:
                    await session.websocket.send_json(error_event)
                except:
                    pass
            elif session.event_queue:
                await session.event_queue.put(error_event)
    
    def _decode_frame_data(self, image_data: str) -> Optional[Image.Image]:
        """Decode base64 image data"""
        try:
            # Remove data URL prefix if present
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_data)
            
            # Open image
            image = Image.open(BytesIO(image_bytes))
            return image
            
        except Exception as e:
            logger.error(f"Failed to decode image data: {e}")
            return None
    
    def _update_session_metrics(self, session: StreamSession):
        """Update session performance metrics"""
        if session.metrics.processing_times:
            session.metrics.avg_processing_time = sum(session.metrics.processing_times) / len(session.metrics.processing_times)
        
        # Calculate FPS
        now = time.time()
        time_window = 10.0  # Calculate FPS over 10 seconds
        if session.metrics.last_frame_time and (now - session.metrics.last_frame_time) < time_window:
            # Estimate FPS based on recent processing
            if len(session.metrics.processing_times) > 1:
                session.metrics.fps = min(len(session.metrics.processing_times) / time_window, session.config.fps_limit)
        
    async def _send_websocket_result(self, session: StreamSession, event: DetectionStreamEvent):
        """Send detection result via WebSocket"""
        if session.websocket:
            try:
                await session.websocket.send_json({
                    "type": "detection_result",
                    "data": event.model_dump(),
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.warning(f"Failed to send WebSocket result: {e}")
                session.status = "disconnected"
    
    async def _send_sse_result(self, session: StreamSession, event: DetectionStreamEvent):
        """Send detection result via SSE"""
        if session.event_queue:
            try:
                await session.event_queue.put({
                    "type": "detection_result",
                    "data": event.model_dump(),
                    "timestamp": time.time()
                })
            except Exception as e:
                logger.warning(f"Failed to queue SSE result: {e}")
    
    async def get_event_stream(self, client_id: str):
        """Get SSE event stream for a client"""
        session = self.get_session(client_id)
        if not session or not session.event_queue:
            return
        
        try:
            # Send welcome message
            welcome_event = {
                "type": "stream_started",
                "session_id": session.session_id,
                "client_id": client_id,
                "timestamp": time.time()
            }
            yield f"data: {json.dumps(welcome_event)}\n\n"
            
            while client_id in self.active_connections and session.status == "active":
                try:
                    # Wait for event with timeout
                    event_data = await asyncio.wait_for(session.event_queue.get(), timeout=30.0)
                    
                    # Format as SSE
                    event_json = json.dumps(event_data)
                    yield f"data: {event_json}\n\n"
                    
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    yield f"data: {{\"type\":\"ping\",\"timestamp\":{time.time()}}}\n\n"
                    
        except Exception as e:
            logger.error(f"Error in SSE stream for {client_id}: {e}")
        finally:
            await self.close_session(client_id)
    
    async def update_session_config(self, client_id: str, config_update: Dict[str, Any]) -> bool:
        """Update session configuration"""
        session = self.get_session(client_id)
        if not session:
            return False
        
        try:
            # Update configuration
            for key, value in config_update.items():
                if hasattr(session.config, key):
                    setattr(session.config, key, value)
            
            session.last_activity = time.time()
            
            logger.info(f"Updated config for session {client_id}: {config_update}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update config for {client_id}: {e}")
            return False
    
    def get_session_stats(self, client_id: str) -> StreamingStatsResponse:
        """Get performance statistics for a session"""
        session = self.get_session(client_id)
        if not session:
            raise ValueError(f"Session {client_id} not found")
        
        return StreamingStatsResponse(
            session_id=session.session_id,
            client_id=client_id,
            uptime_seconds=time.time() - session.created_at,
            total_frames=session.metrics.total_frames,
            processed_frames=session.metrics.processed_frames,
            failed_frames=session.metrics.failed_frames,
            success_rate=session.metrics.processed_frames / max(session.metrics.total_frames, 1),
            avg_processing_time_ms=round(session.metrics.avg_processing_time * 1000, 2),
            current_fps=session.metrics.fps,
            status=session.status,
            config=session.config
        )
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global streaming service statistics"""
        return {
            **self._global_metrics,
            "uptime_seconds": time.time() - self._global_metrics["uptime"],
            "sessions_by_type": {
                "websocket": len([s for s in self.sessions.values() if s.connection_type == "websocket"]),
                "sse": len([s for s in self.sessions.values() if s.connection_type == "sse"])
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get streaming service health status"""
        try:
            active_sessions = len([s for s in self.sessions.values() if s.status == "active"])
            
            return {
                "status": "healthy",
                "active_sessions": active_sessions,
                "frame_processor_running": self._worker_task is not None and not self._worker_task.done(),
                "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done(),
                "model_ready": model_service.is_loaded,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def schedule_session_cleanup(self, client_id: str, delay_seconds: int = 300):
        """Schedule session cleanup after delay"""
        await asyncio.sleep(delay_seconds)
        await self.close_session(client_id)


# Global streaming service instance - will be initialized in main.py
streaming_service_instance: Optional[StreamingService] = None