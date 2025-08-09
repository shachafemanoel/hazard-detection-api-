"""
Tests for detection API endpoints
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.services.model_service import DetectionResult


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


@patch("app.services.model_service.model_service")
def test_detect_model_not_loaded(
    mock_model_service, client: TestClient, sample_session_id, test_image_small
):
    """Test detection when model is not loaded"""
    # Mock model service to indicate not loaded
    mock_model_service.is_loaded = False
    mock_model_service.load_model = AsyncMock(
        side_effect=Exception("Model loading failed")
    )

    files = {"file": ("test.jpg", test_image_small, "image/jpeg")}

    response = client.post(f"/detect/{sample_session_id}", files=files)
    assert response.status_code == 500


@patch("app.services.model_service.model_service")
def test_detect_success_mock(
    mock_model_service,
    client: TestClient,
    sample_session_id,
    test_image_small,
    mock_detection_results,
):
    """Test successful detection with mocked model service"""
    # Mock model service
    mock_model_service.is_loaded = True
    mock_model_service.load_model = AsyncMock(return_value=True)
    mock_model_service.predict = AsyncMock(return_value=mock_detection_results)
    mock_model_service.get_model_info.return_value = {
        "status": "loaded",
        "backend": "pytorch",
        "classes": ["Pothole", "Alligator Crack"],
        "class_count": 2,
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


@patch("app.services.model_service.model_service")
def test_detect_legacy_success_mock(
    mock_model_service, client: TestClient, test_image_small, mock_detection_results
):
    """Test successful legacy detection with mocked model service"""
    # Mock model service
    mock_model_service.is_loaded = True
    mock_model_service.load_model = AsyncMock(return_value=True)
    mock_model_service.predict = AsyncMock(return_value=mock_detection_results)
    mock_model_service.get_model_info.return_value = {
        "status": "loaded",
        "backend": "pytorch",
        "classes": ["Pothole", "Alligator Crack"],
        "class_count": 2,
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


@patch("app.services.model_service.model_service")
def test_detect_batch_success_mock(
    mock_model_service, client: TestClient, test_image_small, mock_detection_results
):
    """Test successful batch detection with mocked model service"""
    # Mock model service
    mock_model_service.is_loaded = True
    mock_model_service.load_model = AsyncMock(return_value=True)
    mock_model_service.predict = AsyncMock(return_value=mock_detection_results)
    mock_model_service.get_model_info.return_value = {
        "status": "loaded",
        "backend": "pytorch",
        "classes": ["Pothole", "Alligator Crack"],
        "class_count": 2,
    }

    # Create multiple test images
    files = [
        ("files", ("test1.jpg", test_image_small, "image/jpeg")),
        ("files", ("test2.jpg", test_image_small, "image/jpeg")),
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
        ("files", ("test2.txt", b"not an image", "text/plain")),
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
    assert response.status_code in [
        422,
        503,
    ]  # 422 for validation, 503 if model not loaded


@patch("app.services.model_service.model_service")
def test_detect_best0408_openvino_model(
    mock_model_service, client: TestClient, sample_session_id, test_image_small
):
    """Test detection specifically with best0408_openvino_model backend"""
    # Mock OpenVINO backend with specific model characteristics
    mock_model_service.is_loaded = True
    mock_model_service.backend = "openvino"
    mock_model_service.load_model = AsyncMock(return_value=True)

    # Create mock detection results specific to road hazard detection
    openvino_detections = [
        DetectionResult(
            bbox=[120, 150, 220, 250],
            confidence=0.94,
            class_id=7,  # Pothole
            class_name="Pothole",
        ),
        DetectionResult(
            bbox=[350, 200, 450, 300],
            confidence=0.87,
            class_id=0,  # Alligator Crack
            class_name="Alligator Crack",
        ),
        DetectionResult(
            bbox=[100, 400, 180, 480],
            confidence=0.76,
            class_id=5,  # Manhole
            class_name="Manhole",
        ),
    ]

    mock_model_service.predict = AsyncMock(return_value=openvino_detections)
    mock_model_service.get_model_info.return_value = {
        "status": "loaded",
        "backend": "openvino",
        "classes": [
            "Alligator Crack",
            "Block Crack",
            "Crosswalk Blur",
            "Lane Blur",
            "Longitudinal Crack",
            "Manhole",
            "Patch Repair",
            "Pothole",
            "Transverse Crack",
            "Wheel Mark Crack",
        ],
        "class_count": 10,
        "input_shape": [1, 3, 480, 480],
        "output_shape": [1, 25200, 15],
        "device": "AUTO",
        "performance_mode": "LATENCY",
        "async_inference": True,
    }

    files = {"file": ("road_damage.jpg", test_image_small, "image/jpeg")}

    response = client.post(f"/detect/{sample_session_id}", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["model_info"]["backend"] == "openvino"
    assert "input_shape" in data["model_info"]
    assert "output_shape" in data["model_info"]

    # Verify all expected detections
    detections = data["detections"]
    assert len(detections) == 3

    class_names = [det["class_name"] for det in detections]
    assert "Pothole" in class_names
    assert "Alligator Crack" in class_names
    assert "Manhole" in class_names

    # Verify high confidence detection creates reports
    high_conf_detections = [det for det in detections if det["confidence"] > 0.9]
    assert len(high_conf_detections) >= 1


@patch("app.services.model_service.model_service")
def test_detect_pytorch_fallback(
    mock_model_service, client: TestClient, sample_session_id, test_image_small
):
    """Test detection with PyTorch fallback when OpenVINO fails"""
    # Mock PyTorch fallback scenario
    mock_model_service.is_loaded = True
    mock_model_service.backend = "pytorch"
    mock_model_service.load_model = AsyncMock(return_value=True)

    pytorch_detections = [
        DetectionResult(
            bbox=[100, 100, 200, 200], confidence=0.82, class_id=7, class_name="Pothole"
        )
    ]

    mock_model_service.predict = AsyncMock(return_value=pytorch_detections)
    mock_model_service.get_model_info.return_value = {
        "status": "loaded",
        "backend": "pytorch",
        "classes": [
            "Alligator Crack",
            "Block Crack",
            "Crosswalk Blur",
            "Lane Blur",
            "Longitudinal Crack",
            "Manhole",
            "Patch Repair",
            "Pothole",
            "Transverse Crack",
            "Wheel Mark Crack",
        ],
        "class_count": 10,
    }

    files = {"file": ("test.jpg", test_image_small, "image/jpeg")}

    response = client.post(f"/detect/{sample_session_id}", files=files)
    assert response.status_code == 200

    data = response.json()
    assert data["model_info"]["backend"] == "pytorch"
    assert "input_shape" not in data["model_info"]  # PyTorch doesn't expose this


def test_detect_performance_tracking(
    client: TestClient, sample_session_id, test_image_small
):
    """Test that detection performance metrics are tracked"""
    with patch("app.services.model_service.model_service") as mock_model_service:
        mock_model_service.is_loaded = True
        mock_model_service.predict = AsyncMock(return_value=[])

        files = {"file": ("test.jpg", test_image_small, "image/jpeg")}

        response = client.post(f"/detect/{sample_session_id}", files=files)
        assert response.status_code == 200

        data = response.json()
        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], (int, float))
        assert data["processing_time_ms"] >= 0


@patch("app.services.model_service.model_service")
def test_detect_timeout_handling(
    mock_model_service, client: TestClient, sample_session_id, test_image_small
):
    """Test detection endpoint timeout handling"""
    # Mock model service to simulate timeout
    mock_model_service.is_loaded = True
    mock_model_service.load_model = AsyncMock(return_value=True)
    
    # Create a mock that raises TimeoutError
    async def timeout_predict(*args):
        raise asyncio.TimeoutError("Model inference timed out")
    
    mock_model_service.predict = timeout_predict

    files = {"file": ("test.jpg", test_image_small, "image/jpeg")}

    response = client.post(f"/detect/{sample_session_id}", files=files)
    assert response.status_code == 504
    
    data = response.json()
    assert data["detail"]["error"] == "inference_timeout"
    assert "timeout" in data["detail"]["message"].lower()
    assert "timeout_ms" in data["detail"]
