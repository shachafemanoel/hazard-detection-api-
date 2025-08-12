"""
Server-Sent Events (SSE) service for real-time notifications
Handles broadcasting report updates and other events to connected dashboard clients
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from fastapi import Request

from ..core.logging_config import get_logger
from ..core.config import settings

logger = get_logger("sse_service")


@dataclass
class SSEClient:
    """Represents a connected SSE client"""
    id: str
    queue: asyncio.Queue
    request: Request
    connected_at: float
    last_ping: float
    
    def is_expired(self, timeout_seconds: int = 300) -> bool:
        """Check if client connection has expired"""
        return (time.time() - self.last_ping) > timeout_seconds


class SSEService:
    """
    Server-Sent Events service for broadcasting real-time updates
    Manages client connections and event broadcasting
    """
    
    def __init__(self):
        self.clients: Dict[str, SSEClient] = {}
        self.active_connections: Set[str] = set()
        self._cleanup_task: Optional[asyncio.Task] = None
        
    def start_cleanup_task(self):
        """Start the periodic cleanup task for expired connections"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("🧹 SSE cleanup task started")
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired client connections"""
        while True:
            try:
                await asyncio.sleep(60)  # Clean up every minute
                await self.cleanup_expired_clients()
            except asyncio.CancelledError:
                logger.info("SSE cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in SSE cleanup task: {e}")
    
    async def cleanup_expired_clients(self, timeout_seconds: int = 300):
        """Remove expired client connections"""
        expired_clients = []
        
        for client_id, client in self.clients.items():
            if client.is_expired(timeout_seconds):
                expired_clients.append(client_id)
        
        for client_id in expired_clients:
            await self.disconnect_client(client_id)
            logger.info(f"🧹 Removed expired SSE client: {client_id}")
        
        if expired_clients:
            logger.info(f"🧹 Cleaned up {len(expired_clients)} expired SSE clients")
    
    async def connect_client(self, request: Request) -> str:
        """Connect a new SSE client and return client ID"""
        client_id = str(uuid.uuid4())
        
        client = SSEClient(
            id=client_id,
            queue=asyncio.Queue(),
            request=request,
            connected_at=time.time(),
            last_ping=time.time()
        )
        
        self.clients[client_id] = client
        self.active_connections.add(client_id)
        
        logger.info(f"📡 SSE client connected: {client_id} (total: {len(self.clients)})")
        
        # Start cleanup task if not already running
        self.start_cleanup_task()
        
        # Send welcome message
        await self.send_to_client(client_id, {
            "type": "connection",
            "message": "Connected to hazard detection updates",
            "clientId": client_id,
            "timestamp": time.time()
        })
        
        return client_id
    
    async def disconnect_client(self, client_id: str):
        """Disconnect an SSE client"""
        if client_id in self.clients:
            try:
                # Send final message
                await self.send_to_client(client_id, {
                    "type": "disconnect",
                    "message": "Connection closed",
                    "timestamp": time.time()
                })
            except Exception:
                pass  # Client may already be disconnected
            
            # Clean up
            self.clients.pop(client_id, None)
            self.active_connections.discard(client_id)
            
            logger.info(f"📡 SSE client disconnected: {client_id} (remaining: {len(self.clients)})")
    
    async def send_to_client(self, client_id: str, data: Dict[str, Any]):
        """Send data to a specific client"""
        if client_id not in self.clients:
            return False
        
        try:
            client = self.clients[client_id]
            await client.queue.put(data)
            client.last_ping = time.time()
            return True
        except Exception as e:
            logger.error(f"Failed to send to client {client_id}: {e}")
            # Remove problematic client
            await self.disconnect_client(client_id)
            return False
    
    async def broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast an event to all connected clients"""
        if not self.clients:
            logger.debug(f"No SSE clients to broadcast {event_type} event to")
            return
        
        event_data = {
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        }
        
        logger.info(f"📡 Broadcasting SSE event '{event_type}' to {len(self.clients)} clients")
        
        # Send to all clients concurrently
        tasks = []
        for client_id in list(self.clients.keys()):  # Copy keys to avoid modification during iteration
            tasks.append(self.send_to_client(client_id, event_data))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failed_count = sum(1 for result in results if not result or isinstance(result, Exception))
            if failed_count > 0:
                logger.warning(f"Failed to send event to {failed_count} clients")
    
    async def broadcast_report_created(self, report_data: Dict[str, Any]):
        """Broadcast new report creation event"""
        await self.broadcast_event("new_report", {
            "report": report_data,
            "message": f"New {report_data.get('hazard_type', 'hazard')} report created"
        })
    
    async def broadcast_report_updated(self, report_data: Dict[str, Any]):
        """Broadcast report update event"""
        await self.broadcast_event("report_updated", {
            "report": report_data,
            "message": f"Report {report_data.get('id', 'unknown')} updated"
        })
    
    async def broadcast_report_deleted(self, report_id: str):
        """Broadcast report deletion event"""
        await self.broadcast_event("report_deleted", {
            "reportId": report_id,
            "message": f"Report {report_id} deleted"
        })
    
    async def get_client_stream(self, client_id: str):
        """Get the event stream for a specific client"""
        if client_id not in self.clients:
            return
        
        client = self.clients[client_id]
        
        try:
            # Send periodic ping to keep connection alive
            ping_task = asyncio.create_task(self._send_periodic_ping(client_id))
            
            while client_id in self.active_connections:
                try:
                    # Wait for event with timeout
                    event_data = await asyncio.wait_for(client.queue.get(), timeout=30.0)
                    
                    # Format as SSE
                    event_json = json.dumps(event_data)
                    sse_data = f"data: {event_json}\n\n"
                    
                    yield sse_data
                    
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    yield "data: {\"type\":\"ping\",\"timestamp\":" + str(time.time()) + "}\n\n"
                except Exception as e:
                    logger.error(f"Error in client stream {client_id}: {e}")
                    break
        
        finally:
            # Clean up
            ping_task.cancel()
            await self.disconnect_client(client_id)
    
    async def _send_periodic_ping(self, client_id: str):
        """Send periodic ping to keep client connection alive"""
        try:
            while client_id in self.active_connections:
                await asyncio.sleep(25)  # Ping every 25 seconds
                if client_id in self.clients:
                    await self.send_to_client(client_id, {
                        "type": "ping",
                        "timestamp": time.time()
                    })
        except asyncio.CancelledError:
            pass  # Task was cancelled
        except Exception as e:
            logger.error(f"Error in periodic ping for {client_id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get SSE service statistics"""
        return {
            "active_clients": len(self.clients),
            "total_connections": len(self.active_connections),
            "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done()
        }


# Global SSE service instance
sse_service = SSEService()