"""Performance monitoring service.

This module provides a lightweight monitoring utility that tracks request,
model loading and inference metrics. The implementation intentionally keeps
simple data structures (lists of dictionaries) to make it easy to analyse the
metrics in tests and health endpoints.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List

import psutil

from ..core.logging_config import get_logger

logger = get_logger("performance_monitor")


class PerformanceMonitor:
    """Collects basic performance metrics for the API and model."""

    def __init__(self) -> None:
        self.request_metrics: List[Dict[str, Any]] = []
        self.inference_metrics: List[Dict[str, Any]] = []
        self.model_load_metrics: List[Dict[str, Any]] = []
        self.start_time = time.time()

        # Thresholds used for alert generation
        self._slow_request_threshold = 1.0  # seconds
        self._cpu_threshold = 90.0
        self._memory_threshold = 90.0
        self._disk_threshold = 85.0

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------
    def record_request(self, endpoint: str, duration: float, success: bool = True) -> None:
        self.request_metrics.append(
            {
                "endpoint": endpoint,
                "duration": duration,
                "success": success,
                "timestamp": time.time(),
            }
        )

    def record_model_load(self, duration: float, backend: str) -> None:
        self.model_load_metrics.append(
            {
                "duration": duration,
                "backend": backend,
                "timestamp": time.time(),
            }
        )
        logger.info(f"Model loaded in {duration:.2f}s using {backend} backend")

    def record_inference(self, duration: float, backend: str) -> None:
        self.inference_metrics.append(
            {
                "duration": duration,
                "backend": backend,
                "timestamp": time.time(),
            }
        )

    # ------------------------------------------------------------------
    # Metric summaries
    # ------------------------------------------------------------------
    def get_system_metrics(self) -> Dict[str, Any]:
        """Return basic system utilisation statistics."""
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024 ** 3), 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024 ** 3), 2),
            "timestamp": datetime.now().isoformat(),
        }

    def get_request_stats(self) -> Dict[str, Any]:
        total = len(self.request_metrics)
        success_count = len([m for m in self.request_metrics if m["success"]])
        fail_count = total - success_count
        avg = sum(m["duration"] for m in self.request_metrics) / total if total else 0

        by_endpoint: Dict[str, Any] = {}
        for m in self.request_metrics:
            ep = m["endpoint"]
            data = by_endpoint.setdefault(
                ep, {"count": 0, "success_count": 0, "failure_count": 0, "total_time": 0}
            )
            data["count"] += 1
            data["total_time"] += m["duration"]
            if m["success"]:
                data["success_count"] += 1
            else:
                data["failure_count"] += 1

        for data in by_endpoint.values():
            data["avg_time"] = data["total_time"] / data["count"]
            data["success_rate"] = (
                data["success_count"] / data["count"] if data["count"] else 0
            )

        return {
            "total_requests": total,
            "successful_requests": success_count,
            "failed_requests": fail_count,
            "success_rate": success_count / total if total else 0,
            "avg_response_time": avg,
            "by_endpoint": by_endpoint,
        }

    def get_inference_stats(self) -> Dict[str, Any]:
        total = len(self.inference_metrics)
        avg = sum(m["duration"] for m in self.inference_metrics) / total if total else 0
        by_backend: Dict[str, Any] = {}
        for m in self.inference_metrics:
            backend = m["backend"]
            data = by_backend.setdefault(backend, {"count": 0, "total_time": 0})
            data["count"] += 1
            data["total_time"] += m["duration"]

        for data in by_backend.values():
            data["avg_time"] = data["total_time"] / data["count"] if data["count"] else 0

        return {
            "total_inferences": total,
            "avg_inference_time": avg,
            "by_backend": by_backend,
        }

    def get_model_load_stats(self) -> Dict[str, Any]:
        total = len(self.model_load_metrics)
        avg = sum(m["duration"] for m in self.model_load_metrics) / total if total else 0
        by_backend: Dict[str, Any] = {}
        for m in self.model_load_metrics:
            backend = m["backend"]
            data = by_backend.setdefault(backend, {"count": 0, "total_time": 0})
            data["count"] += 1
            data["total_time"] += m["duration"]

        for data in by_backend.values():
            data["avg_time"] = data["total_time"] / data["count"] if data["count"] else 0

        return {
            "total_loads": total,
            "avg_load_time": avg,
            "by_backend": by_backend,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        return {
            "uptime_seconds": time.time() - self.start_time,
            "system_metrics": self.get_system_metrics(),
            "request_stats": self.get_request_stats(),
            "inference_stats": self.get_inference_stats(),
            "model_load_stats": self.get_model_load_stats(),
        }

    # Alias used by health endpoint
    def get_performance_report(self) -> Dict[str, Any]:
        return self.get_performance_summary()

    # ------------------------------------------------------------------
    # Alerts and maintenance
    # ------------------------------------------------------------------
    def get_alerts(self) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []
        system = self.get_system_metrics()
        if system["cpu_percent"] > self._cpu_threshold:
            alerts.append({"type": "high_cpu_usage", "value": system["cpu_percent"]})
        if system["memory_percent"] > self._memory_threshold:
            alerts.append({"type": "high_memory_usage", "value": system["memory_percent"]})
        if system["disk_percent"] > self._disk_threshold:
            alerts.append({"type": "high_disk_usage", "value": system["disk_percent"]})
        if any(m["duration"] > self._slow_request_threshold for m in self.request_metrics):
            alerts.append({"type": "slow_requests"})
        if any(not m["success"] for m in self.request_metrics):
            alerts.append({"type": "failed_requests"})
        return alerts

    def cleanup_old_metrics(self, max_age_seconds: float = 3600) -> int:
        cutoff = time.time() - max_age_seconds
        before = len(self.request_metrics)
        self.request_metrics = [m for m in self.request_metrics if m["timestamp"] >= cutoff]
        return before - len(self.request_metrics)

    def reset_metrics(self) -> None:
        self.request_metrics.clear()
        self.inference_metrics.clear()
        self.model_load_metrics.clear()
        self.start_time = time.time()
        logger.info("Performance metrics reset")


# Global performance monitor instance
performance_monitor = PerformanceMonitor()

