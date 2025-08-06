"""
Tests for performance monitoring service
"""

import pytest
import time
from unittest.mock import patch

from app.services.performance_monitor import PerformanceMonitor


class TestPerformanceMonitor:
    """Test performance monitoring functionality"""

    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initialization"""
        monitor = PerformanceMonitor()

        assert monitor.request_metrics == []
        assert monitor.model_load_metrics == []
        assert monitor.inference_metrics == []
        assert monitor.start_time is not None

    def test_record_request(self):
        """Test recording request metrics"""
        monitor = PerformanceMonitor()

        monitor.record_request("/detect", 0.150, True)
        monitor.record_request("/health", 0.005, True)
        monitor.record_request("/detect", 0.200, False)

        assert len(monitor.request_metrics) == 3

        # Check first metric
        metric = monitor.request_metrics[0]
        assert metric["endpoint"] == "/detect"
        assert metric["duration"] == 0.150
        assert metric["success"] is True
        assert "timestamp" in metric

    def test_record_model_load(self):
        """Test recording model load metrics"""
        monitor = PerformanceMonitor()

        monitor.record_model_load(2.5, "openvino")
        monitor.record_model_load(1.8, "pytorch")

        assert len(monitor.model_load_metrics) == 2

        # Check first metric
        metric = monitor.model_load_metrics[0]
        assert metric["duration"] == 2.5
        assert metric["backend"] == "openvino"
        assert "timestamp" in metric

    def test_record_inference(self):
        """Test recording inference metrics"""
        monitor = PerformanceMonitor()

        monitor.record_inference(0.050, "openvino")
        monitor.record_inference(0.080, "pytorch")
        monitor.record_inference(0.045, "openvino")

        assert len(monitor.inference_metrics) == 3

        # Check metrics by backend
        openvino_metrics = [
            m for m in monitor.inference_metrics if m["backend"] == "openvino"
        ]
        pytorch_metrics = [
            m for m in monitor.inference_metrics if m["backend"] == "pytorch"
        ]

        assert len(openvino_metrics) == 2
        assert len(pytorch_metrics) == 1

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_get_system_metrics(self, mock_disk, mock_memory, mock_cpu):
        """Test getting system metrics"""
        monitor = PerformanceMonitor()

        # Mock system metrics
        mock_cpu.return_value = 45.5
        mock_memory.return_value.percent = 68.2
        mock_memory.return_value.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_disk.return_value.percent = 75.0
        mock_disk.return_value.free = 100 * 1024 * 1024 * 1024  # 100GB

        metrics = monitor.get_system_metrics()

        assert metrics["cpu_percent"] == 45.5
        assert metrics["memory_percent"] == 68.2
        assert metrics["memory_available_gb"] == 4.0
        assert metrics["disk_percent"] == 75.0
        assert metrics["disk_free_gb"] == 100.0
        assert "timestamp" in metrics

    def test_get_request_stats(self):
        """Test getting request statistics"""
        monitor = PerformanceMonitor()

        # Add some test metrics
        monitor.record_request("/detect", 0.150, True)
        monitor.record_request("/detect", 0.120, True)
        monitor.record_request("/detect", 0.300, False)
        monitor.record_request("/health", 0.005, True)

        stats = monitor.get_request_stats()

        assert stats["total_requests"] == 4
        assert stats["successful_requests"] == 3
        assert stats["failed_requests"] == 1
        assert stats["success_rate"] == 0.75

        # Check average response times
        assert stats["avg_response_time"] == 0.14375  # (0.15 + 0.12 + 0.3 + 0.005) / 4

        # Check endpoint breakdown
        assert "/detect" in stats["by_endpoint"]
        assert "/health" in stats["by_endpoint"]

        detect_stats = stats["by_endpoint"]["/detect"]
        assert detect_stats["count"] == 3
        assert detect_stats["success_rate"] == 2 / 3

    def test_get_inference_stats(self):
        """Test getting inference statistics"""
        monitor = PerformanceMonitor()

        # Add inference metrics
        monitor.record_inference(0.050, "openvino")
        monitor.record_inference(0.080, "pytorch")
        monitor.record_inference(0.045, "openvino")
        monitor.record_inference(0.060, "openvino")

        stats = monitor.get_inference_stats()

        assert stats["total_inferences"] == 4
        assert (
            stats["avg_inference_time"] == 0.05875
        )  # (0.05 + 0.08 + 0.045 + 0.06) / 4

        # Check backend breakdown
        assert "openvino" in stats["by_backend"]
        assert "pytorch" in stats["by_backend"]

        openvino_stats = stats["by_backend"]["openvino"]
        assert openvino_stats["count"] == 3
        assert openvino_stats["avg_time"] == (0.050 + 0.045 + 0.060) / 3

        pytorch_stats = stats["by_backend"]["pytorch"]
        assert pytorch_stats["count"] == 1
        assert pytorch_stats["avg_time"] == 0.080

    def test_get_model_load_stats(self):
        """Test getting model load statistics"""
        monitor = PerformanceMonitor()

        # Add model load metrics
        monitor.record_model_load(2.5, "openvino")
        monitor.record_model_load(1.8, "pytorch")
        monitor.record_model_load(2.2, "openvino")

        stats = monitor.get_model_load_stats()

        assert stats["total_loads"] == 3
        assert stats["avg_load_time"] == (2.5 + 1.8 + 2.2) / 3

        # Check backend breakdown
        openvino_stats = stats["by_backend"]["openvino"]
        assert openvino_stats["count"] == 2
        assert openvino_stats["avg_time"] == (2.5 + 2.2) / 2

        pytorch_stats = stats["by_backend"]["pytorch"]
        assert pytorch_stats["count"] == 1
        assert pytorch_stats["avg_time"] == 1.8

    def test_get_performance_summary(self):
        """Test getting comprehensive performance summary"""
        monitor = PerformanceMonitor()

        # Add various metrics
        monitor.record_request("/detect", 0.150, True)
        monitor.record_request("/health", 0.005, True)
        monitor.record_model_load(2.5, "openvino")
        monitor.record_inference(0.050, "openvino")

        with patch.object(monitor, "get_system_metrics") as mock_system:
            mock_system.return_value = {
                "cpu_percent": 45.0,
                "memory_percent": 68.0,
                "disk_percent": 75.0,
            }

            summary = monitor.get_performance_summary()

        assert "uptime_seconds" in summary
        assert summary["system_metrics"]["cpu_percent"] == 45.0
        assert summary["request_stats"]["total_requests"] == 2
        assert summary["inference_stats"]["total_inferences"] == 1
        assert summary["model_load_stats"]["total_loads"] == 1

    def test_get_alerts(self):
        """Test getting performance alerts"""
        monitor = PerformanceMonitor()

        # Add metrics that should trigger alerts
        monitor.record_request("/detect", 2.5, True)  # Slow request
        monitor.record_request("/detect", 0.1, False)  # Failed request

        with patch.object(monitor, "get_system_metrics") as mock_system:
            mock_system.return_value = {
                "cpu_percent": 95.0,  # High CPU
                "memory_percent": 92.0,  # High memory
                "disk_percent": 88.0,  # High disk usage
            }

            alerts = monitor.get_alerts()

        # Should have alerts for high resource usage and slow requests
        alert_types = [alert["type"] for alert in alerts]

        assert "high_cpu_usage" in alert_types
        assert "high_memory_usage" in alert_types
        assert "high_disk_usage" in alert_types
        assert "slow_requests" in alert_types
        assert "failed_requests" in alert_types

    def test_cleanup_old_metrics(self):
        """Test cleanup of old metrics"""
        monitor = PerformanceMonitor()

        # Add metrics with different timestamps
        current_time = time.time()

        # Add old metric (manually set timestamp)
        old_metric = {
            "endpoint": "/detect",
            "duration": 0.1,
            "success": True,
            "timestamp": current_time - 7200,  # 2 hours ago
        }
        monitor.request_metrics.append(old_metric)

        # Add recent metric
        monitor.record_request("/health", 0.005, True)

        assert len(monitor.request_metrics) == 2

        # Cleanup metrics older than 1 hour
        cleaned_count = monitor.cleanup_old_metrics(max_age_seconds=3600)

        assert cleaned_count > 0
        assert len(monitor.request_metrics) == 1
        assert monitor.request_metrics[0]["endpoint"] == "/health"
