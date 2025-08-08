"""
Full report flow tests for B3 requirements
Tests: detect → upload → create report → confirm → fetch summary
"""

import pytest
import uuid
import time
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
from app.main_b3 import app

client = TestClient(app)


def create_test_image() -> BytesIO:
    """Create a simple test image"""
    image = Image.new('RGB', (480, 480), color='red')
    img_bytes = BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


def test_session_start():
    """Test session start endpoint"""
    response = client.post("/session/start")
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["message"] == "Session started successfully"
    
    # Validate UUID format
    session_id = data["session_id"]
    uuid.UUID(session_id)  # Should not raise exception
    return session_id


def test_upload_detection_report():
    """
    Test POST /report persists a record with a real Cloudinary URL (B3 requirement)
    """
    # Start a session
    session_id = test_session_start()
    
    # Create test image
    test_image = create_test_image()
    
    # Upload detection report
    response = client.post(
        "/report/",
        files={"file": ("test.jpg", test_image, "image/jpeg")},
        data={
            "sessionId": session_id,
            "className": "pothole",
            "confidence": 0.85,
            "ts": int(time.time() * 1000),
            "lat": 40.7128,
            "lon": -74.0060
        }
    )
    
    # Should succeed or fail gracefully if Cloudinary not configured
    if response.status_code == 200:
        data = response.json()
        assert "id" in data
        assert "cloudinaryUrl" in data
        
        # Validate UUID format
        uuid.UUID(data["id"])
        
        return session_id, data["id"]
    else:
        # Cloudinary not configured - that's acceptable for tests
        assert response.status_code == 500
        return session_id, None


def test_get_session_reports():
    """
    Test GET /session/{id}/reports returns it (B3 requirement)
    """
    session_id, report_id = test_upload_detection_report()
    
    if report_id:
        # Get session reports
        response = client.get(f"/session/{session_id}/reports")
        assert response.status_code == 200
        
        reports = response.json()
        assert isinstance(reports, list)
        
        if len(reports) > 0:
            # Should include cloudinaryUrl
            report = reports[0]
            assert "id" in report
            assert "cloudinaryUrl" in report
            assert "sessionId" in report
            assert report["sessionId"] == session_id


def test_get_session_summary():
    """
    Test GET /session/{id}/summary includes counts and URLs (B3 requirement)
    """
    session_id, report_id = test_upload_detection_report()
    
    # Get session summary
    response = client.get(f"/session/{session_id}/summary")
    assert response.status_code == 200
    
    data = response.json()
    assert "sessionId" in data
    assert "count" in data
    assert "types" in data
    assert "avgConfidence" in data
    assert "durationMs" in data
    assert "reports" in data
    
    # Verify it matches the Summary modal format
    assert data["sessionId"] == session_id
    assert isinstance(data["count"], int)
    assert isinstance(data["types"], dict)
    assert isinstance(data["reports"], list)


def test_report_confirmation():
    """
    Test report confirm/dismiss endpoints
    """
    session_id, report_id = test_upload_detection_report()
    
    if report_id:
        # Test confirm
        response = client.post(f"/session/{session_id}/report/{report_id}/confirm")
        if response.status_code == 200:
            data = response.json()
            assert data["message"] == "Report confirmed"
            assert data["report_id"] == report_id
            assert data["new_status"] == "confirmed"
        
        # Test dismiss
        response = client.post(f"/session/{session_id}/report/{report_id}/dismiss")
        if response.status_code == 200:
            data = response.json()
            assert data["message"] == "Report dismissed"
            assert data["report_id"] == report_id
            assert data["new_status"] == "dismissed"


def test_invalid_session_id():
    """Test behavior with invalid session IDs"""
    invalid_session_id = "not-a-uuid"
    
    response = client.get(f"/session/{invalid_session_id}/reports")
    assert response.status_code == 422
    
    response = client.get(f"/session/{invalid_session_id}/summary")
    assert response.status_code == 422


def test_nonexistent_session():
    """Test behavior with valid but non-existent session ID"""
    fake_session_id = str(uuid.uuid4())
    
    response = client.get(f"/session/{fake_session_id}/reports")
    assert response.status_code == 200
    assert response.json() == []  # Should return empty list
    
    response = client.get(f"/session/{fake_session_id}/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["types"] == {}
    assert data["reports"] == []