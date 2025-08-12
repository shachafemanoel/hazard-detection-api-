"""
Streaming API endpoints for real-time hazard detection
Provides WebSocket and SSE endpoints for live video stream processing
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set
from fastapi import (
    APIRouter, 
    WebSocket, 
    WebSocketDisconnect, 
    HTTPException, 
    Request,
    Depends,
    Query,
    BackgroundTasks
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import base64
from io import BytesIO
from PIL import Image

from ..core.config import settings
from ..core.logging_config import get_logger
from ..services.model_service import model_service, DetectionResult
from ..services.session_service import session_service
from ..services.streaming_service import StreamSession
from ..models.streaming_models import (
    StreamingSessionResponse,
    StreamingStatsResponse,
    FrameProcessingResult,
    StreamingConfiguration,
    DetectionStreamEvent
)

logger = get_logger("streaming_api")

router = APIRouter(tags=["streaming"])

def get_streaming_service():
    """Get the streaming service instance from main module"""
    import sys
    main_module = sys.modules.get('app.main')
    if main_module and hasattr(main_module, 'streaming_service_instance'):
        return getattr(main_module, 'streaming_service_instance')
    raise RuntimeError("Streaming service not initialized")


class FrameData(BaseModel):
    """Frame data for processing"""
    image_data: str = Field(..., description="Base64 encoded image data")
    frame_id: Optional[str] = Field(default=None, description="Client-provided frame identifier")
    timestamp: Optional[float] = Field(default=None, description="Client timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional frame metadata")


class StreamingConfigRequest(BaseModel):
    """Streaming configuration request"""
    fps_limit: Optional[int] = Field(default=10, ge=1, le=30, description="Max FPS for processing")
    confidence_threshold: Optional[float] = Field(default=None, ge=0.1, le=1.0)
    iou_threshold: Optional[float] = Field(default=None, ge=0.1, le=1.0)
    enable_tracking: bool = Field(default=True, description="Enable object tracking")
    batch_processing: bool = Field(default=False, description="Enable batch frame processing")
    quality_mode: str = Field(default="balanced", pattern="^(speed|balanced|quality)$")


@router.websocket("/stream/detection")
async def websocket_detection_stream(
    websocket: WebSocket,
    session_id: Optional[str] = Query(None),
    fps_limit: int = Query(10, ge=1, le=30),
    confidence_threshold: Optional[float] = Query(None, ge=0.1, le=1.0),
    enable_tracking: bool = Query(True)
):
    """
    WebSocket endpoint for real-time detection streaming
    
    Accepts video frames via WebSocket and returns detection results in real-time.
    Optimized for low-latency processing with configurable quality settings.
    """
    client_id = str(uuid.uuid4())
    await websocket.accept()
    
    logger.info(f"🔗 WebSocket client connected: {client_id}")
    
    # Create streaming session
    config = StreamingConfiguration(
        fps_limit=fps_limit,
        confidence_threshold=confidence_threshold or settings.confidence_threshold,
        iou_threshold=settings.iou_threshold,
        enable_tracking=enable_tracking
    )
    
    try:
        streaming_service = get_streaming_service()
        session = await streaming_service.create_session(
            client_id=client_id,
            session_id=session_id,
            connection_type="websocket",
            websocket=websocket,
            config=config
        )
        
        # Send welcome message with session info
        await websocket.send_json({
            "type": "session_created",
            "session_id": session.session_id,
            "client_id": client_id,
            "config": config.model_dump(),
            "timestamp": time.time()
        })
        
        # Process frames in real-time
        while True:
            try:
                # Receive frame data with timeout
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                
                if data.get("type") == "frame":
                    # Process frame asynchronously
                    streaming_service = get_streaming_service()
                    await streaming_service.process_frame(session.session_id, data)
                    
                elif data.get("type") == "config_update":
                    # Update streaming configuration
                    streaming_service = get_streaming_service()
                    await streaming_service.update_session_config(session.session_id, data.get("config", {}))
                    
                elif data.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": time.time()
                    })
                    
            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": time.time()
                    })
                except:
                    break
                    
    except WebSocketDisconnect:
        logger.info(f"🔗 WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": str(e),
                "timestamp": time.time()
            })
        except:
            pass
    finally:
        streaming_service = get_streaming_service()
        await streaming_service.close_session(client_id)


@router.post("/stream/sessions", response_model=StreamingSessionResponse)
async def create_streaming_session(
    config: StreamingConfigRequest,
    session_id: Optional[str] = Query(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Create a new streaming session for SSE-based detection
    
    Returns session configuration and SSE endpoint URL for receiving results.
    """
    try:
        client_id = str(uuid.uuid4())
        
        streaming_config = StreamingConfiguration(
            fps_limit=config.fps_limit,
            confidence_threshold=config.confidence_threshold or settings.confidence_threshold,
            iou_threshold=config.iou_threshold or settings.iou_threshold,
            enable_tracking=config.enable_tracking,
            batch_processing=config.batch_processing,
            quality_mode=config.quality_mode
        )
        
        streaming_service = get_streaming_service()
        session = await streaming_service.create_session(
            client_id=client_id,
            session_id=session_id,
            connection_type="sse",
            config=streaming_config
        )
        
        # Schedule session cleanup
        background_tasks.add_task(
            get_streaming_service().schedule_session_cleanup,
            client_id,
            delay_seconds=getattr(settings, 'streaming_session_timeout', 300)
        )
        
        logger.info(f"📡 Created streaming session: {session.session_id} for client: {client_id}")
        
        return StreamingSessionResponse(
            session_id=session.session_id,
            client_id=client_id,
            sse_endpoint=f"/stream/events/{client_id}",
            upload_endpoint=f"/stream/process/{client_id}",
            config=streaming_config,
            created_at=session.created_at,
            status="active"
        )
        
    except Exception as e:
        logger.error(f"Failed to create streaming session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("/stream/events/{client_id}")
async def streaming_events_endpoint(client_id: str, request: Request):
    """
    Server-Sent Events endpoint for streaming detection results
    
    Provides real-time detection results for frames submitted via POST endpoint.
    """
    try:
        streaming_service = get_streaming_service()
        session = streaming_service.get_session(client_id)
        if not session:
            raise HTTPException(status_code=404, detail="Streaming session not found")
        
        logger.info(f"📡 SSE client connected for streaming: {client_id}")
        
        streaming_service = get_streaming_service()
        return StreamingResponse(
            streaming_service.get_event_stream(client_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            }
        )
        
    except Exception as e:
        logger.error(f"SSE streaming connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to establish SSE connection: {str(e)}")


@router.post("/stream/process/{client_id}")
async def process_streaming_frame(
    client_id: str,
    frame_data: FrameData,
    background_tasks: BackgroundTasks
):
    """
    Process a single frame for streaming detection
    
    Accepts frame data and queues it for processing. Results are sent via SSE endpoint.
    Optimized for high-throughput frame processing.
    """
    try:
        streaming_service = get_streaming_service()
        session = streaming_service.get_session(client_id)
        if not session:
            raise HTTPException(status_code=404, detail="Streaming session not found")
        
        # Process frame in background for better responsiveness
        background_tasks.add_task(
            get_streaming_service().process_frame_background,
            client_id,
            frame_data.model_dump()
        )
        
        return {
            "status": "queued",
            "frame_id": frame_data.frame_id,
            "client_id": client_id,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Frame processing failed for {client_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process frame: {str(e)}")


@router.get("/stream/sessions/{client_id}/stats", response_model=StreamingStatsResponse)
async def get_streaming_session_stats(client_id: str):
    """
    Get performance statistics for a streaming session
    
    Returns throughput metrics, processing times, and session health info.
    """
    try:
        streaming_service = get_streaming_service()
        session = streaming_service.get_session(client_id)
        if not session:
            raise HTTPException(status_code=404, detail="Streaming session not found")
        
        streaming_service = get_streaming_service()
        stats = streaming_service.get_session_stats(client_id)
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session stats for {client_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session stats: {str(e)}")


@router.patch("/stream/sessions/{client_id}/config")
async def update_streaming_config(
    client_id: str,
    config_update: StreamingConfigRequest
):
    """
    Update streaming configuration for an active session
    
    Allows real-time adjustment of processing parameters without reconnecting.
    """
    try:
        streaming_service = get_streaming_service()
        session = streaming_service.get_session(client_id)
        if not session:
            raise HTTPException(status_code=404, detail="Streaming session not found")
        
        streaming_service = get_streaming_service()
        success = await streaming_service.update_session_config(
            client_id, 
            config_update.model_dump(exclude_none=True)
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update configuration")
        
        return {
            "status": "updated",
            "client_id": client_id,
            "config": get_streaming_service().get_session(client_id).config.model_dump(),
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Config update failed for {client_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")


@router.delete("/stream/sessions/{client_id}")
async def close_streaming_session(client_id: str):
    """
    Close a streaming session and clean up resources
    
    Terminates the session and removes all associated data.
    """
    try:
        streaming_service = get_streaming_service()
        success = await streaming_service.close_session(client_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Streaming session not found")
        
        logger.info(f"📡 Closed streaming session for client: {client_id}")
        
        return {
            "status": "closed",
            "client_id": client_id,
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to close session {client_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to close session: {str(e)}")


@router.get("/stream/health")
async def streaming_service_health():
    """
    Health check endpoint for streaming service
    
    Returns service status and performance metrics.
    """
    try:
        streaming_service = get_streaming_service()
        health_status = streaming_service.get_health_status()
        model_status = model_service.get_health_status()
        
        return {
            "status": "healthy" if health_status["status"] == "healthy" else "degraded",
            "streaming_service": health_status,
            "model_service": model_status,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


@router.get("/stream/stats", response_model=Dict[str, Any])
async def get_global_streaming_stats():
    """
    Get global streaming service statistics
    
    Returns aggregate metrics across all active sessions.
    """
    try:
        streaming_service = get_streaming_service()
        stats = streaming_service.get_global_stats()
        return {
            "streaming_stats": stats,
            "model_info": model_service.get_model_info(),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get global stats: {e}")
        return {
            "error": str(e),
            "timestamp": time.time()
        }