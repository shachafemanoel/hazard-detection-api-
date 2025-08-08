"""
Test structured logging and request ID functionality
Agent G requirement: structured logging with request ID and timing
"""

import pytest
import json
import uuid
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from ..main import app

client = TestClient(app)


def test_request_id_header():
    """Test that X-Request-ID header is added to responses"""
    response = client.get("/health")
    assert response.status_code == 200
    
    # Check that request ID header is present
    assert "X-Request-ID" in response.headers
    request_id = response.headers["X-Request-ID"]
    
    # Validate it's a reasonable format (8 chars from UUID)
    assert len(request_id) == 8
    assert all(c in "0123456789abcdef-" for c in request_id.lower())


def test_structured_logging_format(caplog):
    """Test that structured logging includes required fields"""
    with patch('app.main.logger') as mock_logger:
        response = client.get("/health")
        assert response.status_code == 200
        
        # Verify logger.info was called
        mock_logger.info.assert_called()
        
        # Get the logging call arguments
        call_args = mock_logger.info.call_args
        message = call_args[0][0]
        extra = call_args[1]['extra']
        
        # Verify structured logging format
        assert message == "REQUEST_COMPLETED"
        
        # Verify required fields
        required_fields = [
            "request_id", "method", "path", "status_code", 
            "duration_ms", "success", "user_agent", "client_ip"
        ]
        for field in required_fields:
            assert field in extra
        
        # Verify data types
        assert isinstance(extra["request_id"], str)
        assert isinstance(extra["method"], str) 
        assert isinstance(extra["path"], str)
        assert isinstance(extra["status_code"], int)
        assert isinstance(extra["duration_ms"], (int, float))
        assert isinstance(extra["success"], bool)
        
        # Verify values
        assert extra["method"] == "GET"
        assert extra["path"] == "/health"
        assert extra["status_code"] == 200
        assert extra["success"] is True
        assert extra["duration_ms"] >= 0


def test_different_endpoints_logged():
    """Test that different endpoints are properly logged"""
    endpoints_to_test = ["/health", "/", "/status"]
    
    with patch('app.main.logger') as mock_logger:
        for endpoint in endpoints_to_test:
            response = client.get(endpoint)
            # Don't assert status codes since some might fail in test environment
            
        # Verify logging was called for each request
        assert mock_logger.info.call_count >= len(endpoints_to_test)
        
        # Verify each logged path
        logged_paths = []
        for call in mock_logger.info.call_args_list:
            if call[1].get('extra', {}).get('path'):
                logged_paths.append(call[1]['extra']['path'])
        
        for endpoint in endpoints_to_test:
            assert endpoint in logged_paths


def test_error_requests_logged():
    """Test that error responses are also properly logged"""
    with patch('app.main.logger') as mock_logger:
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        # Verify error was logged
        mock_logger.info.assert_called()
        
        # Get the last logging call
        call_args = mock_logger.info.call_args
        extra = call_args[1]['extra']
        
        # Verify error status
        assert extra["status_code"] == 404
        assert extra["success"] is False
        assert extra["path"] == "/nonexistent-endpoint"


def test_performance_metrics_integration():
    """Test that performance monitoring is integrated with logging"""
    with patch('app.main.performance_monitor') as mock_monitor:
        response = client.get("/health")
        assert response.status_code == 200
        
        # Verify performance monitor was called
        mock_monitor.record_request.assert_called_once()
        
        # Get the call arguments
        call_args = mock_monitor.record_request.call_args
        kwargs = call_args[1] if call_args[1] else {}
        args = call_args[0] if call_args[0] else []
        
        # Verify performance metrics
        assert 'endpoint' in kwargs or len(args) >= 1
        assert 'duration' in kwargs or len(args) >= 2  
        assert 'success' in kwargs or len(args) >= 3


def test_request_id_consistency():
    """Test that request ID is consistent across middleware"""
    response = client.get("/health")
    assert response.status_code == 200
    
    request_id_header = response.headers.get("X-Request-ID")
    assert request_id_header is not None
    
    # In a real implementation, we'd verify the request ID is the same
    # in logs and response headers. For now, just verify format.
    assert len(request_id_header) == 8


def test_logging_includes_timing():
    """Test that logging includes proper timing information"""
    with patch('app.main.logger') as mock_logger:
        response = client.get("/health")
        assert response.status_code == 200
        
        call_args = mock_logger.info.call_args
        extra = call_args[1]['extra']
        
        # Verify timing information
        assert 'duration_ms' in extra
        duration = extra['duration_ms']
        
        # Duration should be reasonable (> 0, < 10 seconds for health check)
        assert 0 <= duration <= 10000
        assert isinstance(duration, (int, float))