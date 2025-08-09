"""
Tests for health and status API endpoints
"""

import pytest
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """Test enhanced health check endpoint with model status"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_status" in data  # Should be "not_loaded", "warming", "ready", or "error"
    assert "version" in data
    
    # Verify model_status is one of the expected values
    assert data["model_status"] in ["not_loaded", "warming", "ready", "error"]


def test_root_endpoint(client: TestClient):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data
    assert "version" in data
    assert "endpoints" in data
    assert isinstance(data["endpoints"], list)


def test_status_endpoint(client: TestClient):
    """Test detailed status endpoint"""
    response = client.get("/status")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "model_status" in data
    assert "backend_inference" in data
    assert "backend_type" in data
    assert "active_sessions" in data
    assert "device_info" in data
    assert "environment" in data
    assert "model_files" in data
    assert "endpoints" in data
    assert "configuration" in data

    # Check environment info
    env_info = data["environment"]
    assert "platform" in env_info
    assert "python_version" in env_info
    assert "deployment_env" in env_info

    # Check configuration info
    config_info = data["configuration"]
    assert "confidence_threshold" in config_info
    assert "iou_threshold" in config_info
    assert "tracking_enabled" in config_info


def test_status_endpoint_model_info(client: TestClient):
    """Test status endpoint returns proper model information"""
    response = client.get("/status")
    assert response.status_code == 200

    data = response.json()
    model_files = data["model_files"]

    assert "openvino_model" in model_files
    assert "pytorch_model" in model_files
    assert "current_backend" in model_files
    assert "model_classes" in model_files
    assert "input_size" in model_files

    # Verify model paths contain expected files
    assert "best0408.xml" in model_files["openvino_model"]
    assert "best.pt" in model_files["pytorch_model"]
