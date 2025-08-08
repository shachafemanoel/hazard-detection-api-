"""
Health endpoint tests for B2 requirements
"""

import pytest
from fastapi.testclient import TestClient
from app.main_b3 import app

client = TestClient(app)


def test_health_endpoint():
    """
    Test /detect/health returns 200 (B2 requirement)
    """
    response = client.get("/detect/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_ready_endpoint():
    """
    Test /detect/ready returns 200 after model + Redis + Cloudinary ok (B2 requirement)
    """
    response = client.get("/detect/ready")
    assert response.status_code == 200
    data = response.json()
    
    # Should have all the required fields
    assert "status" in data
    assert "model_loaded" in data
    assert "redis_connected" in data
    assert "cloudinary_configured" in data
    
    # Status should be either ready or not_ready
    assert data["status"] in ["ready", "not_ready"]


def test_model_info_endpoint():
    """
    Test model info endpoint returns correct information
    """
    response = client.get("/detect/model/info")
    assert response.status_code == 200
    data = response.json()
    
    # Should have model and health info
    assert "model" in data
    assert "health" in data
    assert "settings" in data
    
    # Settings should have the required B2 values
    settings = data["settings"]
    assert settings["input_size"] == 480  # B2 requirement
    assert "confidence_threshold" in settings
    assert "backend_preference" in settings