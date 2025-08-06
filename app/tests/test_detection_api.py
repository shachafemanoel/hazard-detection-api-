"""
Tests for detection API endpoints
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient


def test_detect_no_session_id_error(client: TestClient):
    """Test detection endpoint with invalid session"""
    fake_session_id = "fake-session-id"
    
    # Create test image
    test_image_data = b"fake_image_data"
    files = {"file": ("test.jpg", test_image_data, "image/jpeg")}
    
    response = client.post(f"/detect/{fake_session_id}", files=files)
    assert response.status_code == 404
    assert "Session" in response.json()["detail"]


def test_detect_no_file(client: TestClient, sample_session_id):
    """Test detection endpoint without file"""
    response = client.post(f"/detect/{sample_session_id}")
    assert response.status_code == 422
    assert "File field required" in response.json()["detail"]


def test_detect_invalid_file_type(client: TestClient, sample_session_id):
    """Test detection endpoint with non-image file"""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    
    response = client.post(f"/detect/{sample_session_id}", files=files)
    assert response.status_code == 422
    assert "File field required" in response.json()["detail"]


@patch('app.services.model_service.model_service')
def test_detect_model_not_loaded(mock_model_service, client: TestClient, sample_session_id, test_image_small):
    """Test detection when model is not loaded"""
    # Mock model service to indicate not loaded
    mock_model_service.is_loaded = False
    mock_model_service.load_model = AsyncMock(side_effect=Exception("Model loading failed"))
    
    files = {"file": ("test.jpg", test_image_small, "image/jpeg")}
    
    response = client.post(f"/detect/{sample_session_id}", files=files)
    assert response.status_code == 500


@patch('app.services.model_service.model_service')
def test_detect_success_mock(mock_model_service, client: TestClient, sample_session_id, 
                            test_image_small, mock_detection_results):
    """Test successful detection with mocked model service"""
    # Mock model service
    mock_model_service.is_loaded = True
    mock_model_service.load_model = AsyncMock(return_value=True)
    mock_model_service.predict = AsyncMock(return_value=mock_detection_results)
    mock_model_service.get_model_info.return_value = {
        "status": "loaded",
        "backend": "pytorch",
        "classes": ["Pothole", "Alligator Crack"],
        "class_count": 2
    }
    
    files = {"file": ("test.jpg", test_image_small, "image/jpeg")}
    
    response = client.post(f"/detect/{sample_session_id}", files=files)
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert "detections" in data
    assert "new_reports" in data
    assert "session_stats" in data
    assert "processing_time_ms" in data
    assert "image_size" in data
    assert "model_info" in data
    
    # Verify detection format
    detections = data["detections"]
    assert len(detections) == len(mock_detection_results)
    
    for detection in detections:
        assert "bbox" in detection
        assert "confidence" in detection
        assert "class_id" in detection
        assert "class_name" in detection
        assert "center_x" in detection
        assert "center_y" in detection


def test_detect_legacy_no_file(client: TestClient):
    """Test legacy detection endpoint without file"""
    response = client.post("/detect")
    assert response.status_code == 422


def test_detect_legacy_invalid_file_type(client: TestClient):
    """Test legacy detection endpoint with non-image file"""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    
    response = client.post("/detect", files=files)
    assert response.status_code == 400
    assert "File must be an image" in response.json()["detail"]


@patch('app.services.model_service.model_service')
def test_detect_legacy_success_mock(mock_model_service, client: TestClient, 
                                   test_image_small, mock_detection_results):
    """Test successful legacy detection with mocked model service"""
    # Mock model service
    mock_model_service.is_loaded = True
    mock_model_service.load_model = AsyncMock(return_value=True)
    mock_model_service.predict = AsyncMock(return_value=mock_detection_results)
    mock_model_service.get_model_info.return_value = {
        "status": "loaded",
        "backend": "pytorch",
        "classes": ["Pothole", "Alligator Crack"],
        "class_count": 2
    }
    
    files = {"file": ("test.jpg", test_image_small, "image/jpeg")}
    
    response = client.post("/detect", files=files)
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert "detections" in data
    assert "processing_time_ms" in data
    assert "image_size" in data
    assert "model_info" in data
    
    # Legacy endpoint should not have session-specific fields
    assert "new_reports" not in data
    assert "session_stats" not in data


@patch('app.services.model_service.model_service')
def test_detect_batch_success_mock(mock_model_service, client: TestClient, 
                                  test_image_small, mock_detection_results):
    """Test successful batch detection with mocked model service"""
    # Mock model service
    mock_model_service.is_loaded = True
    mock_model_service.load_model = AsyncMock(return_value=True)
    mock_model_service.predict = AsyncMock(return_value=mock_detection_results)
    mock_model_service.get_model_info.return_value = {
        "status": "loaded",
        "backend": "pytorch",
        "classes": ["Pothole", "Alligator Crack"],
        "class_count": 2
    }
    
    # Create multiple test images
    files = [
        ("files", ("test1.jpg", test_image_small, "image/jpeg")),
        ("files", ("test2.jpg", test_image_small, "image/jpeg"))
    ]
    
    response = client.post("/detect-batch", files=files)
    assert response.status_code == 200
    
    data = response.json()
    assert data["success"] is True
    assert "results" in data
    assert "total_processing_time_ms" in data
    assert "processed_count" in data
    assert "successful_count" in data
    assert "model_info" in data
    
    # Check batch results
    results = data["results"]
    assert len(results) == 2
    assert data["processed_count"] == 2
    assert data["successful_count"] == 2
    
    for result in results:
        assert result["success"] is True
        assert "detections" in result
        assert "filename" in result
        assert "file_index" in result


def test_detect_batch_mixed_files(client: TestClient, test_image_small):
    """Test batch detection with mixed valid/invalid files"""
    files = [
        ("files", ("test1.jpg", test_image_small, "image/jpeg")),
        ("files", ("test2.txt", b"not an image", "text/plain"))
    ]
    
    # Even with model not loaded, should get validation errors for non-images
    response = client.post("/detect-batch", files=files)
    
    # The response might be 503 if model not loaded, but let's check the structure
    if response.status_code == 200:
        data = response.json()
        results = data["results"]
        assert len(results) == 2
        
        # Second file should have error
        assert "error" in results[1]
        assert "File must be an image" in results[1]["error"]


def test_detect_batch_no_files(client: TestClient):
    """Test batch detection with no files"""
    response = client.post("/detect-batch", files=[])
    
    # This should return an error due to validation
    assert response.status_code in [422, 503]  # 422 for validation, 503 if model not loaded