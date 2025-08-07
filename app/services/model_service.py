"""
Model service for loading and managing OpenVINO and PyTorch models
Handles intelligent backend selection and model operations
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from PIL import Image

from ..core.config import settings, model_config
from ..core.logging_config import get_logger
from ..core.exceptions import (
    ModelNotLoadedException,
    ModelLoadingException,
    InferenceException,
)

logger = get_logger("model_service")

# Optional imports with graceful fallback
try:
    import cv2
except ImportError:
    cv2 = None
    logger.info("OpenCV not available - using PIL fallback")

try:
    import openvino as ov
    import openvino.properties as props
except ImportError:
    ov = None
    props = None
    logger.info("OpenVINO not available")

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    YOLO = None
    torch = None
    logger.info("Ultralytics/PyTorch not available")

try:
    from cpuinfo import get_cpu_info
except ImportError:
    get_cpu_info = None
    logger.info("CPU info not available")


class DetectionResult:
    """Structured detection result"""

    def __init__(
        self, bbox: List[float], confidence: float, class_id: int, class_name: str
    ):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name

        # Calculate additional properties
        self.center_x = (bbox[0] + bbox[2]) / 2
        self.center_y = (bbox[1] + bbox[3]) / 2
        self.width = bbox[2] - bbox[0]
        self.height = bbox[3] - bbox[1]
        self.area = self.width * self.height

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "bbox": [float(x) for x in self.bbox],
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "center_x": float(self.center_x),
            "center_y": float(self.center_y),
            "width": float(self.width),
            "height": float(self.height),
            "area": float(self.area),
        }


class ModelService:
    """Service for model loading and inference operations"""

    def __init__(self):
        self.is_loaded = False
        self.backend = None
        self.model = None
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None
        self.infer_request = None
        self._load_start_time = None

    async def load_model(self) -> bool:
        """Load model with intelligent backend selection"""
        if self.is_loaded:
            return True

        self._load_start_time = time.time()
        logger.info("🚀 Starting intelligent model loading...")
        logger.info(f"🎯 Selected backend: {settings.model_backend}")
        logger.info(f"📁 Model directory: {settings.model_dir}")

        # Try loading based on backend preference
        if settings.model_backend in ["openvino", "auto"]:
            if await self._try_load_openvino():
                return True
            elif settings.model_backend == "openvino":
                raise ModelLoadingException(
                    "OpenVINO backend requested but loading failed"
                )

        if settings.model_backend in ["pytorch", "auto"]:
            if await self._try_load_pytorch():
                return True
            elif settings.model_backend == "pytorch":
                raise ModelLoadingException(
                    "PyTorch backend requested but loading failed"
                )

        # Try fallback locations
        await self._try_fallback_locations()

        if not self.is_loaded:
            raise ModelLoadingException("All model loading attempts failed")

        return True

    async def _try_load_openvino(self) -> bool:
        """Try to load OpenVINO model"""
        if ov is None:
            logger.info("OpenVINO not available - skipping")
            return False

        try:
            # Check CPU compatibility
            if get_cpu_info is not None:
                flags = get_cpu_info().get("flags", [])
                if not any(flag in flags for flag in ["sse4_2", "sse4.2", "avx"]):
                    logger.info(
                        "CPU lacks OpenVINO-optimal instruction sets - skipping"
                    )
                    return False

            logger.info("🔄 Attempting OpenVINO model loading...")

            # Initialize OpenVINO Core
            core = ov.Core()
            devices = core.available_devices
            logger.info(f"Available OpenVINO devices: {devices}")

            # Find model file
            model_path = None
            for path in model_config.openvino_model_paths:
                if path.exists():
                    model_path = path
                    logger.info(f"📄 Found OpenVINO model at: {model_path}")
                    break

            if not model_path:
                logger.info("No OpenVINO model files (.xml) found")
                return False

            # Verify .bin file exists
            bin_path = model_path.with_suffix(".bin")
            if not bin_path.exists():
                logger.warning(f"Missing .bin file: {bin_path}")
                return False

            # Load and configure model
            model = core.read_model(model=str(model_path))

            # Handle dynamic shapes
            input_info = model.inputs[0]
            if input_info.partial_shape.is_dynamic:
                logger.info("🔧 Reshaping dynamic model to static shape...")
                new_shape = ov.PartialShape(
                    [1, 3, settings.model_input_size, settings.model_input_size]
                )
                model.reshape({input_info.any_name: new_shape})

            # Configure compilation
            config = self._get_openvino_config()

            # Compile model
            logger.info(f"⚙️ Compiling model for {settings.openvino_device} device...")
            compiled_model = core.compile_model(
                model=model, device_name=settings.openvino_device, config=config
            )

            # Set up inference components
            self.compiled_model = compiled_model
            self.input_layer = compiled_model.input(0)
            self.output_layer = compiled_model.output(0)

            if settings.openvino_async_inference:
                self.infer_request = compiled_model.create_infer_request()

            # Mark as loaded
            self.backend = "openvino"
            self.is_loaded = True

            load_time = time.time() - self._load_start_time
            logger.info(f"✅ OpenVINO model loaded successfully in {load_time:.2f}s")
            logger.info(f"📊 Input shape: {list(self.input_layer.shape)}")
            logger.info(f"📊 Output shape: {list(self.output_layer.shape)}")

            # Record performance metrics
            try:
                from .performance_monitor import performance_monitor

                performance_monitor.record_model_load(load_time, "openvino")
            except ImportError:
                pass  # Performance monitoring is optional

            return True

        except Exception as e:
            logger.error(f"OpenVINO model loading failed: {e}")
            return False

    async def _try_load_pytorch(self) -> bool:
        """Try to load PyTorch model"""
        if YOLO is None:
            logger.info("Ultralytics/YOLO not available - skipping PyTorch")
            return False

        try:
            logger.info("🔄 Attempting PyTorch model loading...")

            # Find model file
            model_path = None
            for path in model_config.pytorch_model_paths:
                if path.exists():
                    model_path = path
                    logger.info(f"📄 Found PyTorch model at: {model_path}")
                    break

            if not model_path:
                logger.info("No PyTorch model files found")
                return False

            # Load PyTorch model
            os.environ["YOLO_VERBOSE"] = "False"
            self.model = YOLO(str(model_path))

            # Mark as loaded
            self.backend = "pytorch"
            self.is_loaded = True

            load_time = time.time() - self._load_start_time
            logger.info(f"✅ PyTorch model loaded successfully in {load_time:.2f}s")
            logger.info(f"📊 Model type: {type(self.model)}")

            # Record performance metrics
            try:
                from .performance_monitor import performance_monitor

                performance_monitor.record_model_load(load_time, "pytorch")
            except ImportError:
                pass  # Performance monitoring is optional

            return True

        except Exception as e:
            logger.error(f"PyTorch model loading failed: {e}")
            return False

    async def _try_fallback_locations(self):
        """Try fallback locations for model files"""
        logger.info("Trying fallback locations...")

        for fallback_dir in model_config.fallback_directories:
            if fallback_dir.exists():
                logger.info(f"🔄 Trying fallback location: {fallback_dir}")

                # Update paths temporarily
                old_model_dir = settings.model_dir
                settings.model_dir = str(fallback_dir)
                temp_config = ModelConfig(settings)

                # Try PyTorch first in fallback
                for path in temp_config.pytorch_model_paths:
                    if path.exists():
                        try:
                            self.model = YOLO(str(path))
                            self.backend = "pytorch"
                            self.is_loaded = True
                            logger.info(
                                "🚀 Using PyTorch backend from fallback location"
                            )
                            return
                        except Exception:
                            continue

                # Try OpenVINO in fallback
                for path in temp_config.openvino_model_paths:
                    if path.exists() and path.with_suffix(".bin").exists():
                        if await self._try_load_openvino():
                            return

                # Restore original model dir
                settings.model_dir = old_model_dir

    def _get_openvino_config(self) -> Dict[str, str]:
        """Get OpenVINO compilation configuration"""
        config = {}

        # Model caching
        if settings.openvino_cache_enabled:
            cache_dir = Path(settings.model_dir) / "openvino_cache"
            cache_dir.mkdir(exist_ok=True)
            config["CACHE_DIR"] = str(cache_dir)

        # Performance hints
        if settings.openvino_performance_mode:
            config["PERFORMANCE_HINT"] = settings.openvino_performance_mode

        # CPU optimizations
        if settings.openvino_device in ["CPU", "AUTO"]:
            config["CPU_THREADS_NUM"] = str(min(4, os.cpu_count() or 4))
            config["CPU_BIND_THREAD"] = "YES"

        return config

    async def predict(self, image: Image.Image) -> List[DetectionResult]:
        """Run inference on an image"""
        if not self.is_loaded:
            raise ModelNotLoadedException("Model not loaded. Call load_model() first.")

        try:
            inference_start = time.time()

            if self.backend == "openvino":
                result = await self._predict_openvino(image)
            elif self.backend == "pytorch":
                result = await self._predict_pytorch(image)
            else:
                raise InferenceException(f"Unknown backend: {self.backend}")

            # Record inference performance
            inference_time = time.time() - inference_start
            try:
                from .performance_monitor import performance_monitor

                performance_monitor.record_inference(inference_time, self.backend)
            except ImportError:
                pass  # Performance monitoring is optional

            return result
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise InferenceException(f"Inference failed: {str(e)}")

    async def _predict_openvino(self, image: Image.Image) -> List[DetectionResult]:
        """Run OpenVINO inference"""
        # Preprocess image
        input_shape = list(self.input_layer.shape)
        processed_image, scale, paste_x, paste_y = self._preprocess_image(
            image, input_shape
        )

        # Run inference
        if settings.openvino_async_inference and self.infer_request:
            self.infer_request.infer(
                inputs={self.input_layer.any_name: processed_image}
            )
            result = self.infer_request.get_output_tensor(self.output_layer.index).data
        else:
            result = self.compiled_model({self.input_layer.any_name: processed_image})[
                self.output_layer
            ]

        # Postprocess results
        return self._postprocess_predictions(
            result,
            image.width,
            image.height,
            input_shape[3],
            input_shape[2],
            scale,
            paste_x,
            paste_y,
        )

    async def _predict_pytorch(self, image: Image.Image) -> List[DetectionResult]:
        """Run PyTorch inference"""
        results = self.model.predict(image, imgsz=settings.model_input_size)
        detections = []

        for r in results:
            for box in r.boxes:
                bbox = box.xyxy[0]
                if hasattr(bbox, "tolist"):
                    bbox = bbox.tolist()
                x1, y1, x2, y2 = [float(x) for x in bbox]

                conf_val = box.conf[0]
                conf = (
                    float(conf_val.item()) if hasattr(conf_val, "item") else float(conf_val)
                )
                cls_val = box.cls[0]
                cls_id = int(cls_val.item()) if hasattr(cls_val, "item") else int(cls_val)

                if cls_id < len(model_config.class_names):
                    class_name = model_config.class_names[cls_id]
                else:
                    class_name = f"cls_{cls_id}"

                detection = DetectionResult(
                    bbox=[x1, y1, x2, y2],
                    confidence=conf,
                    class_id=cls_id,
                    class_name=class_name,
                )
                detections.append(detection)

        return detections

    def _preprocess_image(
        self, image: Image.Image, input_shape: List[int]
    ) -> Tuple[np.ndarray, float, int, int]:
        """Preprocess image for OpenVINO inference with letterbox padding"""
        N, C, H, W = input_shape
        target_height, target_width = H, W

        # Try OpenCV first, fall back to PIL if it fails
        opencv_failed = False
        
        if cv2 is not None and not getattr(self, '_opencv_disabled', False):
            try:
                # Use OpenCV for better performance
                # Ensure image is properly converted to numpy array
                img_array = np.array(image)
                if len(img_array.shape) != 3 or img_array.shape[2] != 3:
                    # Force RGB conversion if image format is unexpected
                    image = image.convert("RGB")
                    img_array = np.array(image)
                
                # Validate array before OpenCV processing
                if not isinstance(img_array, np.ndarray) or img_array.size == 0:
                    raise ValueError("Invalid numpy array for OpenCV")
                
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                # Some OpenCV builds may return non-ndarray types (e.g. UMat)
                # which cause cv2.resize to raise a type error. Convert any
                # such result explicitly to a numpy array so resizing always
                # receives a valid src.
                if img_cv is None:
                    raise ValueError("cv2.cvtColor returned None")
                if not isinstance(img_cv, np.ndarray):
                    img_cv = np.asarray(img_cv)
                original_height, original_width = img_cv.shape[:2]

                # Calculate letterbox scale
                scale = min(target_width / original_width, target_height / original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)

                # Resize and pad
                resized_img = cv2.resize(
                    img_cv, (new_width, new_height), interpolation=cv2.INTER_LINEAR
                )
                letterbox_img = np.full(
                    (target_height, target_width, 3), 114, dtype=np.uint8
                )

                paste_x = (target_width - new_width) // 2
                paste_y = (target_height - new_height) // 2
                letterbox_img[
                    paste_y : paste_y + new_height, paste_x : paste_x + new_width
                ] = resized_img

                # Convert back to RGB
                letterbox_img = cv2.cvtColor(letterbox_img, cv2.COLOR_BGR2RGB)
                
                logger.info("OpenCV image processing successful")
                
            except Exception as e:
                logger.warning(f"OpenCV processing failed ({e}), falling back to PIL")
                opencv_failed = True
                # Disable OpenCV for future calls in this instance
                self._opencv_disabled = True

        # PIL fallback (either OpenCV unavailable or failed)
        if cv2 is None or opencv_failed or getattr(self, '_opencv_disabled', False):
            logger.info("Using PIL-only image processing")
            img_rgb = image.convert("RGB")
            original_width, original_height = img_rgb.size
            scale = min(target_width / original_width, target_height / original_height)
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            resized_img = img_rgb.resize((new_width, new_height), Image.Resampling.LANCZOS)
            letterbox_img = Image.new(
                "RGB", (target_width, target_height), (114, 114, 114)
            )
            paste_x = (target_width - new_width) // 2
            paste_y = (target_height - new_height) // 2
            letterbox_img.paste(resized_img, (paste_x, paste_y))

            letterbox_img = np.array(letterbox_img)

        # Normalize and transpose
        img_array = letterbox_img.astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        return img_array, scale, paste_x, paste_y

    def _postprocess_predictions(
        self,
        predictions: np.ndarray,
        original_width: int,
        original_height: int,
        input_width: int,
        input_height: int,
        scale: float,
        paste_x: int,
        paste_y: int,
    ) -> List[DetectionResult]:
        """Postprocess OpenVINO predictions with letterbox coordinate adjustment"""
        detections = []

        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension

        for pred in predictions:
            x_center, y_center, width, height = pred[:4]
            confidence = pred[4]

            if confidence < settings.confidence_threshold:
                continue

            # Get class scores
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]

            # Final confidence
            final_confidence = confidence * class_confidence
            if final_confidence < settings.confidence_threshold:
                continue

            # Convert coordinates
            x1_input = x_center - width / 2
            y1_input = y_center - height / 2
            x2_input = x_center + width / 2
            y2_input = y_center + height / 2

            # Adjust for letterbox padding
            x1 = max(0, min((x1_input - paste_x) / scale, original_width))
            y1 = max(0, min((y1_input - paste_y) / scale, original_height))
            x2 = max(0, min((x2_input - paste_x) / scale, original_width))
            y2 = max(0, min((y2_input - paste_y) / scale, original_height))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            class_name = (
                model_config.class_names[class_id]
                if class_id < len(model_config.class_names)
                else f"unknown_{class_id}"
            )

            detection = DetectionResult(
                bbox=[x1, y1, x2, y2],
                confidence=final_confidence,
                class_id=int(class_id),
                class_name=class_name,
            )
            detections.append(detection)

        # Apply Non-Maximum Suppression
        return self._apply_nms(detections)

    def _apply_nms(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []

        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)

        selected = []
        while detections:
            best = detections.pop(0)
            selected.append(best)

            threshold = min(settings.iou_threshold, 0.1)
            # Remove overlapping detections based on IoU threshold
            detections = [
                det
                for det in detections
                if self._calculate_iou(best.bbox, det.bbox) < threshold
            ]

        return selected

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        # Calculate areas
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_loaded:
            return {"status": "not_loaded"}

        info = {
            "status": "loaded",
            "backend": self.backend,
            "classes": model_config.class_names,
            "class_count": len(model_config.class_names),
        }

        if self.backend == "openvino":
            info.update(
                {
                    "input_shape": list(self.input_layer.shape),
                    "output_shape": list(self.output_layer.shape),
                    "device": settings.openvino_device,
                    "performance_mode": settings.openvino_performance_mode,
                    "async_inference": settings.openvino_async_inference,
                }
            )

        return info


# Global model service instance
model_service = ModelService()
