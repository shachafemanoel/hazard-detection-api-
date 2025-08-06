"""
Pytest configuration and fixtures for testing
"""

import os
import pytest
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

# Set test environment before importing app modules
os.environ["ENVIRONMENT"] = "testing"
os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["MODEL_DIR"] = "/tmp/test_models"

from app.main import app
from app.core.config import settings
from app.services.model_service import model_service, DetectionResult
from app.services.session_service import session_service


@pytest.fixture
def client():
    """Test client for FastAPI app"""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_image():
    """Create a test image"""
    # Create a simple 640x640 RGB test image
    img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    
    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer


@pytest.fixture
def test_image_small():
    """Create a small test image for faster tests"""
    # Create a simple 10x10 RGB test image
    img_array = np.zeros((10, 10, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    
    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer


@pytest.fixture
def mock_detection_results():
    """Mock detection results"""
    return [
        DetectionResult(
            bbox=[100, 100, 200, 200],
            confidence=0.85,
            class_id=0,
            class_name="Pothole"
        ),
        DetectionResult(
            bbox=[300, 300, 400, 400],
            confidence=0.75,
            class_id=1,
            class_name="Alligator Crack"
        )
    ]


@pytest.fixture
def mock_model_service():
    """Mock model service for testing"""
    mock = MagicMock()
    mock.is_loaded = True
    mock.backend = "pytorch"
    mock.load_model = AsyncMock(return_value=True)
    mock.predict = AsyncMock()
    mock.get_model_info.return_value = {
        "status": "loaded",
        "backend": "pytorch",
        "classes": ["Pothole", "Alligator Crack"],
        "class_count": 2
    }
    return mock


@pytest.fixture
def clean_session_service():
    """Clean session service for each test"""
    # Clear any existing sessions
    session_service.sessions.clear()
    session_service.active_detections.clear()
    yield session_service
    # Clean up after test
    session_service.sessions.clear()
    session_service.active_detections.clear()


@pytest.fixture(autouse=True)
def reset_model_service():
    """Reset model service state before each test"""
    # Store original state
    original_is_loaded = model_service.is_loaded
    original_backend = model_service.backend
    original_model = model_service.model
    
    yield
    
    # Restore original state
    model_service.is_loaded = original_is_loaded
    model_service.backend = original_backend
    model_service.model = original_model


@pytest.fixture
def sample_session_id(clean_session_service):
    """Create a sample session for testing"""
    return clean_session_service.create_session()


class MockOpenVinoModel:
    """Mock OpenVINO model for testing"""
    
    def __init__(self):
        self.input_shape = [1, 3, 512, 512]
        self.output_shape = [1, 25200, 15]  # YOLO output format
    
    def predict(self, image):
        # Mock prediction returning random detections
        predictions = np.random.rand(1, 100, 15)  # Mock YOLO format
        predictions[:, :, 4] = 0.8  # Set confidence
        return predictions


@pytest.fixture
def mock_openvino_model():
    """Mock OpenVINO model"""
    return MockOpenVinoModel()


# Test data constants
TEST_IMAGE_WIDTH = 640
TEST_IMAGE_HEIGHT = 640
TEST_CONFIDENCE_THRESHOLD = 0.5
TEST_IOU_THRESHOLD = 0.45