"""
Performance monitoring service for tracking API and model performance
"""

import time
import psutil
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..core.logging_config import get_logger

logger = get_logger("performance_monitor")


class PerformanceMetrics:
    """Container for performance metrics"""
    
    def __init__(self):
        self.request_times = deque(maxlen=1000)  # Last 1000 requests
        self.inference_times = deque(maxlen=1000)  # Last 1000 inferences
        self.error_count = 0
        self.total_requests = 0
        self.total_inferences = 0
        self.start_time = time.time()
        
        # Endpoint-specific metrics
        self.endpoint_metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0,
            'error_count': 0,
            'avg_time': 0
        })
        
        # Model performance metrics
        self.model_metrics = {
            'load_time': None,
            'average_inference_time': 0,
            'fastest_inference': float('inf'),
            'slowest_inference': 0,
            'backend': None
        }
    
    def record_request(self, endpoint: str, duration: float, success: bool = True):
        """Record a request metric"""
        self.total_requests += 1
        self.request_times.append(duration)
        
        # Update endpoint metrics
        endpoint_data = self.endpoint_metrics[endpoint]
        endpoint_data['count'] += 1
        endpoint_data['total_time'] += duration
        if not success:
            endpoint_data['error_count'] += 1
            self.error_count += 1
        
        # Calculate average
        endpoint_data['avg_time'] = endpoint_data['total_time'] / endpoint_data['count']
    
    def record_inference(self, duration: float, backend: str):
        """Record an inference metric"""
        self.total_inferences += 1
        self.inference_times.append(duration)
        
        # Update model metrics
        self.model_metrics['backend'] = backend
        self.model_metrics['average_inference_time'] = sum(self.inference_times) / len(self.inference_times)
        self.model_metrics['fastest_inference'] = min(self.model_metrics['fastest_inference'], duration)
        self.model_metrics['slowest_inference'] = max(self.model_metrics['slowest_inference'], duration)
    
    def record_model_load(self, duration: float, backend: str):
        """Record model loading time"""
        self.model_metrics['load_time'] = duration
        self.model_metrics['backend'] = backend
        logger.info(f"Model loaded in {duration:.2f}s using {backend} backend")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_percent': cpu_percent,
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'percent': memory.percent
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                }
            }
        except Exception as e:
            logger.warning(f"Failed to get system metrics: {e}")
            return {}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        uptime = time.time() - self.start_time
        
        # Calculate request statistics
        avg_request_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
        requests_per_second = self.total_requests / uptime if uptime > 0 else 0
        error_rate = (self.error_count / self.total_requests) if self.total_requests > 0 else 0
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.total_requests,
            'total_inferences': self.total_inferences,
            'requests_per_second': requests_per_second,
            'average_request_time_ms': avg_request_time * 1000,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'endpoint_metrics': dict(self.endpoint_metrics),
            'model_metrics': self.model_metrics,
            'system_metrics': self.get_system_metrics()
        }


class PerformanceMonitor:
    """Service for monitoring and tracking performance metrics"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.alerts = []
        
        # Performance thresholds
        self.thresholds = {
            'request_time_ms': 5000,  # 5 seconds
            'inference_time_ms': 2000,  # 2 seconds
            'error_rate': 0.1,  # 10%
            'cpu_percent': 90,
            'memory_percent': 90
        }
    
    def record_request(self, endpoint: str, duration: float, success: bool = True):
        """Record request performance"""
        self.metrics.record_request(endpoint, duration, success)
        
        # Check for performance issues
        self._check_request_performance(endpoint, duration, success)
    
    def record_inference(self, duration: float, backend: str):
        """Record inference performance"""
        self.metrics.record_inference(duration, backend)
        
        # Check inference performance
        self._check_inference_performance(duration)
    
    def record_model_load(self, duration: float, backend: str):
        """Record model loading performance"""
        self.metrics.record_model_load(duration, backend)
    
    def _check_request_performance(self, endpoint: str, duration: float, success: bool):
        """Check request performance against thresholds"""
        duration_ms = duration * 1000
        
        if duration_ms > self.thresholds['request_time_ms']:
            alert = {
                'type': 'slow_request',
                'endpoint': endpoint,
                'duration_ms': duration_ms,
                'threshold_ms': self.thresholds['request_time_ms'],
                'timestamp': datetime.now().isoformat()
            }
            self.alerts.append(alert)
            logger.warning(f"Slow request detected: {endpoint} took {duration_ms:.1f}ms")
        
        if not success:
            logger.warning(f"Request failed: {endpoint}")
    
    def _check_inference_performance(self, duration: float):
        """Check inference performance against thresholds"""
        duration_ms = duration * 1000
        
        if duration_ms > self.thresholds['inference_time_ms']:
            alert = {
                'type': 'slow_inference',
                'duration_ms': duration_ms,
                'threshold_ms': self.thresholds['inference_time_ms'],
                'timestamp': datetime.now().isoformat()
            }
            self.alerts.append(alert)
            logger.warning(f"Slow inference detected: {duration_ms:.1f}ms")
    
    def check_system_health(self):
        """Check system health metrics"""
        try:
            system_metrics = self.metrics.get_system_metrics()
            
            if system_metrics:
                cpu_percent = system_metrics.get('cpu_percent', 0)
                memory_percent = system_metrics.get('memory', {}).get('percent', 0)
                
                if cpu_percent > self.thresholds['cpu_percent']:
                    alert = {
                        'type': 'high_cpu',
                        'cpu_percent': cpu_percent,
                        'threshold': self.thresholds['cpu_percent'],
                        'timestamp': datetime.now().isoformat()
                    }
                    self.alerts.append(alert)
                    logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                
                if memory_percent > self.thresholds['memory_percent']:
                    alert = {
                        'type': 'high_memory',
                        'memory_percent': memory_percent,
                        'threshold': self.thresholds['memory_percent'],
                        'timestamp': datetime.now().isoformat()
                    }
                    self.alerts.append(alert)
                    logger.warning(f"High memory usage: {memory_percent:.1f}%")
        
        except Exception as e:
            logger.error(f"Failed to check system health: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        # Check system health before generating report
        self.check_system_health()
        
        summary = self.metrics.get_summary()
        
        # Add alert information
        summary['alerts'] = {
            'total_alerts': len(self.alerts),
            'recent_alerts': [alert for alert in self.alerts 
                             if datetime.fromisoformat(alert['timestamp']) > 
                             datetime.now() - timedelta(minutes=30)],
            'alert_types': {}
        }
        
        # Count alert types
        for alert in self.alerts:
            alert_type = alert['type']
            if alert_type not in summary['alerts']['alert_types']:
                summary['alerts']['alert_types'][alert_type] = 0
            summary['alerts']['alert_types'][alert_type] += 1
        
        # Performance recommendations
        summary['recommendations'] = self._generate_recommendations(summary)
        
        return summary
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations based on metrics"""
        recommendations = []
        
        # Check average request time
        avg_request_time_ms = summary.get('average_request_time_ms', 0)
        if avg_request_time_ms > 1000:
            recommendations.append(
                "Consider optimizing request processing - average response time is high"
            )
        
        # Check error rate
        error_rate = summary.get('error_rate', 0)
        if error_rate > 0.05:  # 5%
            recommendations.append(
                "High error rate detected - investigate failing requests"
            )
        
        # Check inference performance
        model_metrics = summary.get('model_metrics', {})
        avg_inference_time = model_metrics.get('average_inference_time', 0)
        if avg_inference_time > 1.0:  # 1 second
            recommendations.append(
                "Consider optimizing model inference - processing time is high"
            )
        
        # Check system resources
        system_metrics = summary.get('system_metrics', {})
        cpu_percent = system_metrics.get('cpu_percent', 0)
        memory_percent = system_metrics.get('memory', {}).get('percent', 0)
        
        if cpu_percent > 70:
            recommendations.append(
                "High CPU usage detected - consider scaling up or optimizing processing"
            )
        
        if memory_percent > 70:
            recommendations.append(
                "High memory usage detected - monitor for memory leaks"
            )
        
        return recommendations
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        self.metrics = PerformanceMetrics()
        self.alerts = []
        logger.info("Performance metrics reset")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()