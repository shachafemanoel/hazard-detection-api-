"""
Detection endpoint tests
"""

import pytest
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
from app.main_b3 import app

client = TestClient(app)


def create_test_image() -> BytesIO:
    """Create a simple test image"""
    image = Image.new('RGB', (480, 480), color='blue')
    img_bytes = BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_detect_with_session():
    """Test detection with session ID"""
    # Start a session first
    session_response = client.post("/session/start")
    assert session_response.status_code == 200
    session_id = session_response.json()["session_id"]
    
    # Create test image
    test_image = create_test_image()
    
    # Run detection
    response = client.post(
        f"/detect/{session_id}",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={
            "confidence_threshold": 0.5,
            "save_detections": False
        }
    )
    
    # Should succeed if model is loaded, or return 503 if not loaded yet
    if response.status_code == 200:
        data = response.json()
        assert data["success"] is True
        assert data["sessionId"] == session_id
        assert "boxes" in data
        assert "scores" in data
        assert "classes" in data
        assert "processing_time_ms" in data
        
        # Verify response format matches B3 contract
        assert isinstance(data["boxes"], list)
        assert isinstance(data["scores"], list)
        assert isinstance(data["classes"], list)
        
    elif response.status_code == 503:
        # Model not loaded yet - acceptable
        assert "Model not loaded" in response.json()["detail"]
    else:
        pytest.fail(f"Unexpected status code: {response.status_code}")


def test_detect_legacy():
    """Test legacy detection endpoint without session"""
    test_image = create_test_image()
    
    response = client.post(
        "/detect/",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"confidence_threshold": 0.3}
    )
    
    # Should succeed if model is loaded, or return 503 if not loaded yet
    if response.status_code == 200:
        data = response.json()
        assert data["success"] is True
        assert data["sessionId"] is None  # Legacy endpoint has no session
        assert "boxes" in data
        assert "scores" in data
        assert "classes" in data
        assert "processing_time_ms" in data
        
    elif response.status_code == 503:
        # Model not loaded yet - acceptable
        assert "Model not loaded" in response.json()["detail"]


def test_detect_invalid_file():
    """Test detection with invalid file types"""
    session_response = client.post("/session/start")
    session_id = session_response.json()["session_id"]
    
    # Test with text file
    response = client.post(
        f"/detect/{session_id}",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        data={"confidence_threshold": 0.5}
    )
    
    assert response.status_code == 422
    assert "Invalid file type" in response.json()["detail"]


def test_detect_invalid_confidence():
    """Test detection with invalid confidence threshold"""
    session_response = client.post("/session/start")
    session_id = session_response.json()["session_id"]
    
    test_image = create_test_image()
    
    # Test with confidence > 1.0
    response = client.post(
        f"/detect/{session_id}",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={"confidence_threshold": 1.5}
    )
    
    assert response.status_code == 422
    assert "must be between 0.0 and 1.0" in response.json()["detail"]


def test_detect_no_file():
    """Test detection without providing a file"""
    session_response = client.post("/session/start")
    session_id = session_response.json()["session_id"]
    
    response = client.post(
        f"/detect/{session_id}",
        data={"confidence_threshold": 0.5}
    )
    
    assert response.status_code == 422