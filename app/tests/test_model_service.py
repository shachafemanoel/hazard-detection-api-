"""
Tests for model service - OpenVINO and PyTorch model loading and inference
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from PIL import Image

from app.services.model_service import ModelService, DetectionResult
from app.core.exceptions import ModelLoadingException, InferenceException


class TestModelService:
    """Test model service functionality"""

    def test_detection_result_creation(self):
        """Test DetectionResult class functionality"""
        bbox = [100.0, 150.0, 200.0, 250.0]
        confidence = 0.85
        class_id = 7
        class_name = "Pothole"

        result = DetectionResult(bbox, confidence, class_id, class_name)

        assert result.bbox == bbox
        assert result.confidence == confidence
        assert result.class_id == class_id
        assert result.class_name == class_name
        assert result.center_x == 150.0  # (100 + 200) / 2
        assert result.center_y == 200.0  # (150 + 250) / 2
        assert result.width == 100.0  # 200 - 100
        assert result.height == 100.0  # 250 - 150
        assert result.area == 10000.0  # 100 * 100

    def test_detection_result_to_dict(self):
        """Test DetectionResult serialization"""
        result = DetectionResult([100, 150, 200, 250], 0.85, 7, "Pothole")
        data = result.to_dict()

        assert data["bbox"] == [100.0, 150.0, 200.0, 250.0]
        assert data["confidence"] == 0.85
        assert data["class_id"] == 7
        assert data["class_name"] == "Pothole"
        assert data["center_x"] == 150.0
        assert data["center_y"] == 200.0
        assert data["width"] == 100.0
        assert data["height"] == 100.0
        assert data["area"] == 10000.0

    def test_model_service_initialization(self):
        """Test ModelService initialization"""
        service = ModelService()

        assert not service.is_loaded
        assert service.backend is None
        assert service.model is None
        assert service.compiled_model is None
        assert service.input_layer is None
        assert service.output_layer is None
        assert service.infer_request is None

    @pytest.mark.asyncio
    async def test_load_model_already_loaded(self):
        """Test load_model when model is already loaded"""
        service = ModelService()
        service.is_loaded = True

        result = await service.load_model()
        assert result is True

    @pytest.mark.asyncio
    @patch("app.services.model_service.ov")
    @patch("app.services.model_service.model_config")
    async def test_load_openvino_model_success(self, mock_model_config, mock_ov):
        """Test successful OpenVINO model loading"""
        service = ModelService()

        # Mock model config paths
        mock_xml_path = MagicMock()
        mock_xml_path.exists.return_value = True
        mock_xml_path.with_suffix.return_value.exists.return_value = True
        mock_model_config.openvino_model_paths = [mock_xml_path]

        # Mock OpenVINO components
        mock_core = MagicMock()
        mock_model = MagicMock()
        mock_compiled_model = MagicMock()
        mock_input_layer = MagicMock()
        mock_output_layer = MagicMock()

        mock_ov.Core.return_value = mock_core
        mock_core.available_devices = ["CPU", "GPU"]
        mock_core.read_model.return_value = mock_model
        mock_core.compile_model.return_value = mock_compiled_model

        # Mock model properties
        mock_model.inputs = [mock_input_layer]
        mock_model.outputs = [mock_output_layer]
        mock_input_layer.partial_shape.is_dynamic = False
        mock_output_layer.partial_shape.is_dynamic = False
        mock_input_layer.shape = [1, 3, 480, 480]
        mock_output_layer.shape = [1, 25200, 15]

        mock_compiled_model.input.return_value = mock_input_layer
        mock_compiled_model.output.return_value = mock_output_layer

        # Test load_model
        result = await service.load_model()

        assert result is True
        assert service.is_loaded is True
        assert service.backend == "openvino"
        assert service.compiled_model == mock_compiled_model

    @pytest.mark.asyncio
    @patch("app.services.model_service.YOLO")
    @patch("app.services.model_service.model_config")
    async def test_load_pytorch_model_success(self, mock_model_config, mock_yolo):
        """Test successful PyTorch model loading"""
        service = ModelService()

        # Mock model config paths
        mock_pt_path = MagicMock()
        mock_pt_path.exists.return_value = True
        mock_model_config.pytorch_model_paths = [mock_pt_path]

        # Mock YOLO model
        mock_model = MagicMock()
        mock_yolo.return_value = mock_model

        # Test load_model with pytorch preference
        with patch("app.services.model_service.settings.model_backend", "pytorch"):
            result = await service.load_model()

        assert result is True
        assert service.is_loaded is True
        assert service.backend == "pytorch"
        assert service.model == mock_model

    @pytest.mark.asyncio
    async def test_load_model_all_failed(self):
        """Test load_model when all loading attempts fail"""
        service = ModelService()

        # Mock all dependencies as unavailable
        with (
            patch("app.services.model_service.ov", None),
            patch("app.services.model_service.YOLO", None),
        ):

            with pytest.raises(ModelLoadingException):
                await service.load_model()

    @pytest.mark.asyncio
    async def test_predict_model_not_loaded(self):
        """Test predict when model is not loaded"""
        service = ModelService()
        test_image = Image.new("RGB", (100, 100))

        with pytest.raises(Exception):  # Should raise ModelNotLoadedException
            await service.predict(test_image)

    @pytest.mark.asyncio
    async def test_predict_pytorch_success(self):
        """Test successful PyTorch prediction"""
        service = ModelService()
        service.is_loaded = True
        service.backend = "pytorch"

        # Create mock YOLO model
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_box = MagicMock()

        # Mock box properties
        mock_box.xyxy = [[100, 150, 200, 250]]
        mock_box.conf = [0.85]
        mock_box.cls = [7]

        mock_result.boxes = [mock_box]
        mock_model.predict.return_value = [mock_result]
        service.model = mock_model

        test_image = Image.new("RGB", (480, 480))

        with patch("app.services.model_service.model_config") as mock_config:
            mock_config.class_names = [
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
            ]

            results = await service.predict(test_image)

        assert len(results) == 1
        assert isinstance(results[0], DetectionResult)
        assert results[0].class_name == "Pothole"
        assert results[0].confidence == 0.85

    def test_calculate_iou(self):
        """Test IoU calculation"""
        service = ModelService()

        # Test overlapping boxes
        box1 = [100, 100, 200, 200]  # 100x100 box
        box2 = [150, 150, 250, 250]  # 100x100 box, 50% overlap

        iou = service._calculate_iou(box1, box2)
        expected_iou = 2500 / 17500  # intersection / union
        assert abs(iou - expected_iou) < 0.01

        # Test non-overlapping boxes
        box3 = [300, 300, 400, 400]
        iou_no_overlap = service._calculate_iou(box1, box3)
        assert iou_no_overlap == 0.0

        # Test identical boxes
        iou_identical = service._calculate_iou(box1, box1)
        assert iou_identical == 1.0

    def test_apply_nms(self):
        """Test Non-Maximum Suppression"""
        service = ModelService()

        detections = [
            DetectionResult([100, 100, 200, 200], 0.9, 0, "Class1"),
            DetectionResult(
                [150, 150, 250, 250], 0.8, 0, "Class1"
            ),  # Overlaps with first
            DetectionResult(
                [300, 300, 400, 400], 0.7, 1, "Class2"
            ),  # Different location
        ]

        with patch("app.services.model_service.settings.iou_threshold", 0.5):
            filtered = service._apply_nms(detections)

        # Should keep highest confidence detection and non-overlapping one
        assert len(filtered) == 2
        assert filtered[0].confidence == 0.9  # Highest confidence kept
        assert filtered[1].confidence == 0.7  # Non-overlapping kept

    def test_get_model_info_not_loaded(self):
        """Test get_model_info when model is not loaded"""
        service = ModelService()
        info = service.get_model_info()

        assert info["status"] == "not_loaded"

    def test_get_model_info_openvino_loaded(self):
        """Test get_model_info when OpenVINO model is loaded"""
        service = ModelService()
        service.is_loaded = True
        service.backend = "openvino"

        # Mock OpenVINO components
        mock_input_layer = MagicMock()
        mock_output_layer = MagicMock()
        mock_input_layer.shape = [1, 3, 480, 480]
        mock_output_layer.shape = [1, 25200, 15]
        service.input_layer = mock_input_layer
        service.output_layer = mock_output_layer

        with (
            patch("app.services.model_service.model_config") as mock_config,
            patch("app.services.model_service.settings") as mock_settings,
        ):
            mock_config.class_names = ["Class1", "Class2"]
            mock_settings.openvino_device = "AUTO"
            mock_settings.openvino_performance_mode = "LATENCY"
            mock_settings.openvino_async_inference = True

            info = service.get_model_info()

        assert info["status"] == "loaded"
        assert info["backend"] == "openvino"
        assert info["input_shape"] == [1, 3, 480, 480]
        assert info["output_shape"] == [1, 25200, 15]
        assert info["device"] == "AUTO"
        assert info["performance_mode"] == "LATENCY"
        assert info["async_inference"] is True

    def test_get_model_info_pytorch_loaded(self):
        """Test get_model_info when PyTorch model is loaded"""
        service = ModelService()
        service.is_loaded = True
        service.backend = "pytorch"

        with patch("app.services.model_service.model_config") as mock_config:
            mock_config.class_names = ["Class1", "Class2"]

            info = service.get_model_info()

        assert info["status"] == "loaded"
        assert info["backend"] == "pytorch"
        assert info["class_count"] == 2
        assert "input_shape" not in info  # PyTorch doesn't expose this
