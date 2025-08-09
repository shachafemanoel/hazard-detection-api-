"""
Redis service for the Hazard Detection API
Provides connection pool and Redis operations following B3 persistence contract
"""

import json
import redis
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

from ..core.config import settings
from ..core.logging_config import get_logger

logger = get_logger("redis_service")


class RedisService:
    """Redis service with connection pool and helper methods"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.connection_pool: Optional[redis.ConnectionPool] = None
        self._setup_connection()
    
    def get_redis(self) -> Optional[redis.Redis]:
        """Get the singleton Redis client"""
        return self.redis_client
    
    def _setup_connection(self):
        """Setup Redis connection with connection pool"""
        try:
            # Create connection pool for better performance
            if settings.redis_url:
                self.connection_pool = redis.ConnectionPool.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    max_connections=20,
                    socket_connect_timeout=10,
                    socket_timeout=10,
                    retry_on_timeout=True
                )
                self.redis_client = redis.Redis(connection_pool=self.connection_pool)
            elif settings.redis_host and settings.redis_password:
                self.connection_pool = redis.ConnectionPool(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    username=settings.redis_username or "default",
                    password=settings.redis_password,
                    db=settings.redis_db,
                    decode_responses=True,
                    max_connections=20,
                    socket_connect_timeout=10,
                    socket_timeout=10,
                    retry_on_timeout=True
                )
                self.redis_client = redis.Redis(connection_pool=self.connection_pool)
            else:
                logger.warning("⚠️ Redis not configured - reports will not persist")
                return
            
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connection pool established")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Redis connection: {e}")
            self.redis_client = None
            self.connection_pool = None
    
    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        try:
            if not self.redis_client:
                return False
            self.redis_client.ping()
            return True
        except Exception:
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get Redis health status for /ready endpoint"""
        try:
            if not self.redis_client:
                return {
                    "status": "not_configured",
                    "connected": False,
                    "error": "Redis not configured"
                }
            
            # Test connection and get info
            info = self.redis_client.info()
            
            return {
                "status": "healthy",
                "connected": True,
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_connections_received": info.get("total_connections_received"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }
    
    # B3 Report persistence contract methods
    
    def store_report(self, report_id: str, report_data: Dict[str, Any], ttl_days: Optional[int] = None) -> bool:
        """
        Store detection record under report:{uuid}
        Fields: sessionId, class, confidence, ts, geo, cloudinaryUrl, status
        """
        if not self.redis_client:
            logger.error("Redis not available for storing report")
            return False
        
        try:
            report_key = f"report:{report_id}"
            report_json = json.dumps(report_data, default=str)
            
            # Store report
            self.redis_client.set(report_key, report_json)
            
            # Set TTL if specified
            if ttl_days:
                self.redis_client.expire(report_key, ttl_days * 24 * 3600)
            elif settings.report_retention_days > 0:
                self.redis_client.expire(report_key, settings.report_retention_days * 24 * 3600)
            
            logger.debug(f"Stored report {report_id} in Redis")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store report {report_id}: {e}")
            return False
    
    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Get report data by ID"""
        if not self.redis_client:
            return None
        
        try:
            report_key = f"report:{report_id}"
            report_data = self.redis_client.get(report_key)
            
            if report_data:
                return json.loads(report_data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get report {report_id}: {e}")
            return None
    
    def add_report_to_session(self, session_id: str, report_id: str) -> bool:
        """
        RPUSH session:{sessionId}:reports {uuid} to index per session
        """
        if not self.redis_client:
            return False
        
        try:
            session_key = f"session:{session_id}:reports"
            self.redis_client.rpush(session_key, report_id)
            
            # Set TTL on session list
            if settings.report_retention_days > 0:
                self.redis_client.expire(session_key, settings.report_retention_days * 24 * 3600)
            
            logger.debug(f"Added report {report_id} to session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add report {report_id} to session {session_id}: {e}")
            return False
    
    def get_session_reports(self, session_id: str) -> List[str]:
        """Get list of report IDs for a session"""
        if not self.redis_client:
            return []
        
        try:
            session_key = f"session:{session_id}:reports"
            report_ids = self.redis_client.lrange(session_key, 0, -1)
            return report_ids or []
            
        except Exception as e:
            logger.error(f"Failed to get session reports for {session_id}: {e}")
            return []
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session summary with aggregated data:
        - count: number of reports
        - types: count per hazard type  
        - avgConfidence: average confidence
        - durationMs: session duration
        - reports: list with {id, cloudinaryUrl, className, confidence, ts, geo}
        """
        if not self.redis_client:
            return None
        
        try:
            # Get all report IDs for this session
            report_ids = self.get_session_reports(session_id)
            
            if not report_ids:
                return {
                    "sessionId": session_id,
                    "count": 0,
                    "types": {},
                    "avgConfidence": 0.0,
                    "durationMs": 0,
                    "reports": []
                }
            
            # Get all report data
            reports = []
            confidences = []
            types_count = {}
            timestamps = []
            
            for report_id in report_ids:
                report_data = self.get_report(report_id)
                if report_data:
                    reports.append({
                        "id": report_id,
                        "cloudinaryUrl": report_data.get("cloudinaryUrl", ""),
                        "className": report_data.get("className", "unknown"),
                        "confidence": report_data.get("confidence", 0.0),
                        "ts": report_data.get("ts", 0),
                        "geo": report_data.get("geo")
                    })
                    
                    # Aggregate data
                    confidences.append(report_data.get("confidence", 0.0))
                    class_name = report_data.get("className", "unknown")
                    types_count[class_name] = types_count.get(class_name, 0) + 1
                    timestamps.append(report_data.get("ts", 0))
            
            # Calculate aggregates
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            duration_ms = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
            
            return {
                "sessionId": session_id,
                "count": len(reports),
                "types": types_count,
                "avgConfidence": avg_confidence,
                "durationMs": duration_ms,
                "reports": reports,
                "startTime": min(timestamps) if timestamps else None,
                "endTime": max(timestamps) if timestamps else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get session summary for {session_id}: {e}")
            return None
    
    def update_report_status(self, report_id: str, status: str) -> bool:
        """Update report status (confirm/dismiss)"""
        if not self.redis_client:
            return False
        
        try:
            report_data = self.get_report(report_id)
            if not report_data:
                return False
            
            report_data["status"] = status
            report_data["updated_at"] = datetime.utcnow().isoformat()
            
            return self.store_report(report_id, report_data)
            
        except Exception as e:
            logger.error(f"Failed to update report {report_id} status to {status}: {e}")
            return False
    
    def delete_report(self, report_id: str) -> bool:
        """Delete a report and remove from session lists"""
        if not self.redis_client:
            return False
        
        try:
            # Get report to find session ID
            report_data = self.get_report(report_id)
            if report_data:
                session_id = report_data.get("sessionId")
                if session_id:
                    # Remove from session list
                    session_key = f"session:{session_id}:reports"
                    self.redis_client.lrem(session_key, 0, report_id)
            
            # Delete report
            report_key = f"report:{report_id}"
            deleted = self.redis_client.delete(report_key)
            
            return deleted > 0
            
        except Exception as e:
            logger.error(f"Failed to delete report {report_id}: {e}")
            return False
    
    # General Redis operations
    
    def set_with_ttl(self, key: str, value: Union[str, Dict, List], ttl_seconds: int) -> bool:
        """Set key with TTL"""
        if not self.redis_client:
            return False
        
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str)
            
            self.redis_client.setex(key, ttl_seconds, value)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set key {key} with TTL: {e}")
            return False
    
    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON value by key"""
        if not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get JSON key {key}: {e}")
            return None
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern"""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Failed to delete pattern {pattern}: {e}")
            return 0
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Cleanup old session data"""
        if not self.redis_client:
            return 0
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
            
            # Find session keys
            session_keys = self.redis_client.keys("session:*:reports")
            cleaned_count = 0
            
            for session_key in session_keys:
                try:
                    # Get session ID from key
                    session_id = session_key.split(":")[1]
                    
                    # Get report IDs
                    report_ids = self.redis_client.lrange(session_key, 0, -1)
                    
                    # Check if any reports are newer than cutoff
                    has_recent = False
                    for report_id in report_ids:
                        report_data = self.get_report(report_id)
                        if report_data and report_data.get("ts", 0) > cutoff_timestamp:
                            has_recent = True
                            break
                    
                    # If no recent reports, clean up the session
                    if not has_recent and report_ids:
                        # Delete all reports in session
                        for report_id in report_ids:
                            self.delete_report(report_id)
                        
                        # Delete session list
                        self.redis_client.delete(session_key)
                        cleaned_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to cleanup session key {session_key}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old sessions")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0


# Global Redis service instance
redis_service = RedisService()