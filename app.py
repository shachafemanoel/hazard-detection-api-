from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import time
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union
import math
from collections import defaultdict
import base64
import os
from pathlib import Path

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define placeholder API connector functions
class MockApiResponse:
    def __init__(self, success=False, data=None, error="Service not implemented"):
        self.success = success
        self.data = data
        self.error = error

async def mock_api_manager_health_check():
    """Mock API manager health check"""
    return {
        "status": "disabled",
        "message": "API connectors not configured",
        "timestamp": datetime.now().isoformat()
    }

async def mock_geocode_location(address: str):
    """Mock geocoding function"""
    return MockApiResponse(
        success=False, 
        error="Geocoding service not configured. Set up Google Maps API or external geocoding service."
    )

async def mock_cache_detection_result(detection_id: str, detection_data: dict):
    """Mock cache function"""
    return MockApiResponse(
        success=False,
        error="Caching service not configured. Set up Redis or external caching service."
    )

async def mock_upload_detection_image(image_data: str):
    """Mock image upload function"""
    return MockApiResponse(
        success=False,
        error="Image upload service not configured. Set up Cloudinary or external storage service."
    )

async def mock_reverse_geocode_location(lat: float, lng: float):
    """Mock reverse geocoding function"""
    return MockApiResponse(
        success=False,
        error="Reverse geocoding service not configured. Set up Google Maps API or external geocoding service."
    )

# Mock API manager class
class MockApiManager:
    async def health_check(self):
        return await mock_api_manager_health_check()
    
    def __init__(self):
        self.render = None

# Safe imports with fallbacks
ov = None
props = None
YOLO = None
torch = None
get_cpu_info = None

# Initialize basic service first, then try optional dependencies
logger.info("🚀 Initializing FastAPI service...")

try:
    import openvino as ov
    import openvino.properties as props
    logger.info("✅ OpenVINO runtime available")
except Exception as e:
    logger.info(f"⚠️ OpenVINO not available: {e}")

try:
    from ultralytics import YOLO
    import torch
    logger.info("✅ Ultralytics and PyTorch available")
except Exception as e:
    logger.info(f"⚠️ Ultralytics/PyTorch not available: {e}")

try:
    from cpuinfo import get_cpu_info
    logger.info("✅ CPU info available")
except Exception as e:
    logger.info(f"⚠️ CPU info not available: {e}")

# Use mock functions by default to prevent import failures
api_manager = MockApiManager()
geocode_location = mock_geocode_location
upload_detection_image = mock_upload_detection_image
cache_detection_result = mock_cache_detection_result
reverse_geocode_location = mock_reverse_geocode_location

# Try to import real API connectors but don't fail if missing
try:
    try:
        from api_connectors import (
            api_manager as real_api_manager,
            geocode_location as real_geocode,
            reverse_geocode_location as real_reverse_geocode,
        )
    except ImportError:
        from api.api_connectors import (  # type: ignore[reportMissingImports]
            api_manager as real_api_manager,
            geocode_location as real_geocode,
            reverse_geocode_location as real_reverse_geocode,
        )
    api_manager = real_api_manager
    geocode_location = real_geocode
    reverse_geocode_location = real_reverse_geocode
    logger.info("Real API connectors loaded successfully")
except ImportError:
    logger.info("Using mock API connectors (api_connectors module not found)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Hazard Detection Backend", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://<node-service>.railway.app"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

# Global variables for OpenVINO model and class names
core = None
compiled_model = None
input_layer = None
output_layer = None
torch_model = None
USE_OPENVINO = False
infer_request = None  # For asynchronous inference
async_inference_queue = None  # Queue for async requests
class_names = [
    'Alligator Crack', 'Block Crack', 'Crosswalk Blur', 'Lane Blur',
    'Longitudinal Crack', 'Manhole', 'Patch Repair', 'Pothole',
    'Transverse Crack', 'Wheel Mark Crack'
]

# Model configuration
MODEL_INPUT_SIZE = 512  # Updated to match new model input size
DEVICE_NAME = "AUTO"  # Use AUTO for intelligent device selection (OpenVINO 2024 best practice)
CACHE_ENABLED = False
PERFORMANCE_MODE = "LATENCY"  # Options: "LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"
ENABLE_ASYNC_INFERENCE = True  # Enable asynchronous inference for better performance

# Session management
sessions = {}
active_detections = defaultdict(list)  # session_id -> list of detections

# Detection tracking settings
TRACKING_DISTANCE_THRESHOLD = 50  # pixels
TRACKING_TIME_THRESHOLD = 2.0  # seconds
MIN_CONFIDENCE_FOR_REPORT = 0.6

# Enhanced model loading with intelligent backend selection
async def load_model():
    global core, compiled_model, input_layer, output_layer, torch_model, USE_OPENVINO
    logger.info("🚀 FASTAPI STARTUP - Intelligent model loading begins...")
    logger.info(f"🔍 Current working directory: {os.getcwd()}")
    logger.info(f"🌍 MODEL_DIR environment: {os.getenv('MODEL_DIR', 'NOT SET')}")
    logger.info(f"🧠 MODEL_BACKEND environment: {os.getenv('MODEL_BACKEND', 'NOT SET')}")
    
    # Get the intelligent model selection from environment
    selected_backend = os.getenv('MODEL_BACKEND', 'auto').lower()
    model_dir = os.getenv('MODEL_DIR', '/app/models')
    
    logger.info(f"🎯 Selected backend: {selected_backend}")
    logger.info(f"📁 Model directory: {model_dir}")
    
    # Try to load configuration file if available
    config_file = '/app/model-config.json'
    if os.path.exists(config_file):
        try:
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            logger.info("📋 Loaded intelligent model configuration")
            backend_config = config_data.get('selection_config', {})
            logger.info(f"🔧 Configuration reasons: {', '.join(backend_config.get('reasons', []))}")
        except Exception as e:
            logger.warning(f"Could not load model config: {e}")
    
    # Backend selection logic - prioritize OpenVINO for better performance
    if selected_backend in ['openvino', 'auto']:
        logger.info("🎯 Prioritizing OpenVINO backend for optimal performance")
        success = await try_load_openvino_model(model_dir)
        if success:
            logger.info("🚀 Using OpenVINO backend for inference")
            return
        else:
            logger.warning("⚠️ OpenVINO model loading failed, trying PyTorch fallback")
    
    if selected_backend in ['pytorch', 'auto']:
        logger.info("🔄 Attempting PyTorch backend as fallback")
        success = await try_load_pytorch_model(model_dir)
        if success:
            logger.info("🚀 Using YOLO PyTorch backend for inference")
            logger.warning("⚠️ Note: OpenVINO would provide better performance if available")
            return
        else:
            logger.warning("⚠️ PyTorch model loading also failed")
    
    # If both fail, try fallback locations
    logger.warning("Primary model loading failed, trying fallback locations...")
    fallback_locations = [
        '/app',  # Root directory (where best.pt and best_openvino_model/ are located)
        '/app/models',  # Standard models directory
        '/app/best_openvino_model',  # Specific OpenVINO directory
        '/app/models/pytorch',  # PyTorch subdirectory
        '/app/models/openvino',  # OpenVINO subdirectory
        '/app/public/object_detection_model'  # Legacy location
    ]
    
    for fallback_dir in fallback_locations:
        if os.path.exists(fallback_dir):
            logger.info(f"🔄 Trying fallback location: {fallback_dir}")
            if await try_load_pytorch_model(fallback_dir):
                logger.info("🚀 Using YOLO PyTorch backend for inference")
                return
            if await try_load_openvino_model(fallback_dir):
                logger.info("🚀 Using OpenVINO backend for inference")
                return
    
    logger.error("❌ All model loading attempts failed!")


async def ensure_model_loaded():
    if compiled_model is None and torch_model is None:
        await load_model()


async def try_load_openvino_model(model_dir):
    """Try to load OpenVINO model from specified directory using OpenVINO 2024 best practices"""
    global core, compiled_model, input_layer, output_layer, USE_OPENVINO, infer_request
    
    if ov is None:
        logger.info("OpenVINO not available - skipping")
        return False
        
    try:
        # Check CPU compatibility if cpuinfo is available
        if get_cpu_info is not None:
            flags = get_cpu_info().get("flags", [])
            if not any(flag in flags for flag in ["sse4_2", "sse4.2", "avx"]):
                logger.info("CPU lacks OpenVINO-optimal instruction sets - skipping")
                return False
        
        logger.info("🔄 Attempting OpenVINO model loading...")
        
        # Initialize OpenVINO Core
        core = ov.Core()
        devices = core.available_devices
        logger.info(f"Available OpenVINO devices: {devices}")
        
        # Show device information
        for device in devices:
            try:
                device_name = core.get_property(device, props.device.full_name)
                logger.info(f"{device}: {device_name}")
            except Exception:
                logger.info(f"{device}: (device info unavailable)")
        
        # Look for OpenVINO model files - specific paths for this project
        model_xml_paths = [
            os.path.join(model_dir, 'best_openvino_model', 'last_model_train12052025.xml'),  # New model
            os.path.join(model_dir, 'best_openvino_model', 'best.xml'),  # Legacy model
            os.path.join(model_dir, 'last_model_train12052025.xml'),  # Direct path to new model
            os.path.join(model_dir, 'best.xml'),  # Fallback
            os.path.join(model_dir, 'openvino', 'best.xml'),  # Alternative structure
            os.path.join(model_dir, 'model.xml')  # Generic fallback
        ]
        
        model_path = None
        for path in model_xml_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"📄 Found OpenVINO model at: {model_path}")
                break
        
        if not model_path:
            logger.info("No OpenVINO model files (.xml) found")
            return False
        
        # Verify .bin file exists
        bin_path = Path(model_path).with_suffix('.bin')
        if not bin_path.exists():
            logger.warning(f"Missing .bin file: {bin_path}")
            return False
            
        # Read the model using OpenVINO Core
        logger.info("📖 Reading OpenVINO model...")
        model = core.read_model(model=model_path)
        
        # Get model input/output info
        input_info = model.inputs[0]
        output_info = model.outputs[0]
        
        # Handle dynamic shapes safely
        if input_info.partial_shape.is_dynamic:
            logger.info(f"📊 Original input shape: {input_info.partial_shape} (dynamic)")
            logger.info("🔧 Reshaping dynamic model to static shape...")
            new_shape = ov.PartialShape([1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE])
            model.reshape({input_info.any_name: new_shape})
            logger.info(f"📊 Reshaped input to: {new_shape}")
            # Update input_info after reshape
            input_info = model.inputs[0]
        else:
            logger.info(f"📊 Original input shape: {input_info.shape}")
        
        if output_info.partial_shape.is_dynamic:
            logger.info(f"📊 Original output shape: {output_info.partial_shape} (dynamic)")
        else:
            logger.info(f"📊 Original output shape: {output_info.shape}")

        # Configure compilation with OpenVINO 2024 best practices
        config = {}
        
        # Enable model caching for faster subsequent loads
        if CACHE_ENABLED:
            cache_dir = os.path.join(os.path.dirname(model_path), 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            config['CACHE_DIR'] = str(cache_dir)
            logger.info(f"💾 Model caching enabled: {cache_dir}")
        
        # Configure performance hints (OpenVINO 2024 feature)
        if PERFORMANCE_MODE == "LATENCY":
            config['PERFORMANCE_HINT'] = "LATENCY"
            logger.info("🚀 Performance hint: LATENCY (optimized for low latency)")
        elif PERFORMANCE_MODE == "THROUGHPUT":
            config['PERFORMANCE_HINT'] = "THROUGHPUT"
            logger.info("🚀 Performance hint: THROUGHPUT (optimized for high throughput)")
        elif PERFORMANCE_MODE == "CUMULATIVE_THROUGHPUT":
            config['PERFORMANCE_HINT'] = "CUMULATIVE_THROUGHPUT"
            logger.info("🚀 Performance hint: CUMULATIVE_THROUGHPUT (optimized for multiple streams)")
        
        # Additional optimizations for CPU
        if DEVICE_NAME == "CPU" or DEVICE_NAME == "AUTO":
            config['CPU_THREADS_NUM'] = str(min(4, os.cpu_count() or 4))  # Optimize thread usage
            config['CPU_BIND_THREAD'] = "YES"  # Enable thread binding for better performance
            logger.info(f"🧠 CPU optimization: {config['CPU_THREADS_NUM']} threads with binding")
        
        # Compile the model for the target device
        logger.info(f"⚙️ Compiling model for {DEVICE_NAME} device with performance optimizations...")
        compiled_model = core.compile_model(model=model, device_name=DEVICE_NAME, config=config)
        
        # Get compiled model input/output layers
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        
        # Initialize asynchronous inference request for better performance
        if ENABLE_ASYNC_INFERENCE:
            infer_request = compiled_model.create_infer_request()
            logger.info("🔄 Asynchronous inference request created")
        
        # Log model details
        logger.info("✅ OpenVINO model compiled successfully")
        logger.info(f"📊 Compiled input shape: {input_layer.shape}")
        logger.info(f"📊 Compiled output shape: {output_layer.shape}")
        logger.info(f"🎯 Input precision: {input_layer.element_type}")
        logger.info(f"🎯 Output precision: {output_layer.element_type}")
        logger.info(f"🏷️ Input name: {input_layer.any_name}")

        # Safely retrieve and log output tensor name
        output_names = output_layer.get_tensor().get_names()
        if output_names:
            logger.info(f"🏷️ Output name: {next(iter(output_names))}")
        else:
            logger.info("🏷️ Output name: (unnamed)")
        
        # Log performance configuration
        logger.info(f"⚡ Performance mode: {PERFORMANCE_MODE}")
        logger.info(f"🔄 Async inference: {'Enabled' if ENABLE_ASYNC_INFERENCE else 'Disabled'}")
        logger.info(f"🎯 Device selection: {DEVICE_NAME}")
        
        USE_OPENVINO = True
        return True
        
    except Exception as e:
        logger.error(f"OpenVINO model loading failed: {e}")
        import traceback
        logger.error(f"Full error trace: {traceback.format_exc()}")
        return False


async def try_load_pytorch_model(model_dir):
    """Try to load PyTorch model from specified directory with enhanced error handling"""
    global torch_model, USE_OPENVINO
    
    if YOLO is None:
        logger.info("Ultralytics/YOLO not available - skipping PyTorch")
        return False
        
    try:
        logger.info("🔄 Attempting PyTorch model loading...")
        
        # Look for PyTorch model files - specific paths for this project
        model_pt_paths = [
            os.path.join(model_dir, 'best.pt'),  # Primary location
            os.path.join(model_dir, 'pytorch', 'best.pt'),  # Alternative structure
            os.path.join(model_dir, 'models', 'best.pt'),  # Another alternative
            os.path.join(model_dir, 'road_damage_detection_last_version.pt'),  # Legacy name
            os.path.join(model_dir, 'best_yolo12m.pt')  # Alternative model
        ]
        
        model_path = None
        for path in model_pt_paths:
            if os.path.exists(path):
                model_path = path
                logger.info(f"📄 Found PyTorch model at: {model_path}")
                break
        
        if not model_path:
            logger.info("No PyTorch model files found")
            return False
        
        # Load PyTorch model with error catching for ONNX issues
        try:
            # Disable ONNX export capabilities that cause issues
            os.environ['YOLO_VERBOSE'] = 'False'

            torch_model = YOLO(model_path)
            USE_OPENVINO = False

            # Test basic inference capability without ONNX export
            logger.info("🧪 Testing PyTorch model functionality...")

            logger.info("✅ PyTorch model loaded successfully")
            logger.info(f"📊 Model type: {type(torch_model)}")
            return True
            
        except ImportError as ie:
            if 'ExportOptions' in str(ie) or 'torch.onnx' in str(ie):
                logger.warning(f"PyTorch ONNX export issue detected: {ie}")
                logger.warning("This is likely a version compatibility issue, but model inference should still work")
                # Try to continue anyway, as the error might only affect export, not inference
                try:
                    torch_model = YOLO(model_path)
                    USE_OPENVINO = False
                    logger.info("✅ PyTorch model loaded despite ONNX export issues")
                    return True
                except Exception as e2:
                    logger.error(f"PyTorch model loading failed completely: {e2}")
                    return False
            else:
                raise ie
        
    except Exception as e:
        logger.warning(f"PyTorch model loading failed: {e}")
        logger.warning("This may be due to version compatibility issues. OpenVINO model should be preferred.")
        return False

# Helper functions for OpenVINO inference
def preprocess_image(image, input_shape):
    """
    Preprocess image for OpenVINO YOLO model - Following OpenVINO tutorial practices
    
    Args:
        image: PIL Image object
        input_shape: Model input shape [N, C, H, W]
    
    Returns:
        tuple: (processed_image, scale, paste_x, paste_y)
    """
    # Get target dimensions from input shape [N, C, H, W]
    N, C, H, W = input_shape
    target_height, target_width = H, W
    
    # Convert PIL to OpenCV format for consistent processing
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    original_height, original_width = img_cv.shape[:2]
    
    # Calculate scale to fit the image in target size (letterbox)
    scale = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize image using OpenCV for better performance
    resized_img = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create letterbox image with gray padding (YOLO standard)
    letterbox_img = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    
    # Calculate padding offsets
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    
    # Paste resized image onto letterbox
    letterbox_img[paste_y:paste_y + new_height, paste_x:paste_x + new_width] = resized_img
    
    # Convert BGR back to RGB for model inference
    letterbox_img = cv2.cvtColor(letterbox_img, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 and normalize to [0, 1] as expected by YOLO
    img_array = letterbox_img.astype(np.float32) / 255.0
    
    # Transpose from HWC to CHW format (NCHW layout)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension [1, C, H, W]
    img_array = np.expand_dims(img_array, axis=0)
    
    logger.debug(f"Image preprocessing: {original_width}x{original_height} -> {target_width}x{target_height}, scale: {scale:.3f}")
    
    return img_array, scale, paste_x, paste_y

def postprocess_predictions(predictions, original_width, original_height, input_width, input_height, conf_threshold=0.5, iou_threshold=0.45):
    """Postprocess OpenVINO model predictions"""
    detections = []
    
    # Scale factors to convert back to original image coordinates
    scale_x = original_width / input_width
    scale_y = original_height / input_height
    
    # predictions shape is typically [1, num_detections, 85] for YOLO
    # where each detection is [x, y, w, h, confidence, class_scores...]
    if len(predictions.shape) == 3:
        predictions = predictions[0]  # Remove batch dimension
    
    for pred in predictions:
        # Extract box coordinates (center format)
        x_center, y_center, width, height = pred[:4]
        confidence = pred[4]
        
        if confidence < conf_threshold:
            continue
            
        # Get class scores and find the class with highest score
        class_scores = pred[5:]
        class_id = np.argmax(class_scores)
        class_confidence = class_scores[class_id]
        
        # Final confidence is object confidence * class confidence
        final_confidence = confidence * class_confidence
        
        if final_confidence < conf_threshold:
            continue
        
        # Convert from center format to corner format and scale to original image
        x1 = (x_center - width / 2) * scale_x
        y1 = (y_center - height / 2) * scale_y
        x2 = (x_center + width / 2) * scale_x
        y2 = (y_center + height / 2) * scale_y
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, original_width))
        y1 = max(0, min(y1, original_height))
        x2 = max(0, min(x2, original_width))
        y2 = max(0, min(y2, original_height))
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': final_confidence,
            'class_id': int(class_id),
            'class_name': class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}"
        })
    
    # Apply Non-Maximum Suppression (simple version)
    detections = apply_nms(detections, iou_threshold)
    
    return detections

def apply_nms(detections, iou_threshold):
    """Apply Non-Maximum Suppression to remove overlapping detections"""
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    selected = []
    while detections:
        # Take the detection with highest confidence
        best = detections.pop(0)
        selected.append(best)
        
        # Remove overlapping detections
        detections = [det for det in detections if calculate_iou(best['bbox'], det['bbox']) < iou_threshold]
    
    return selected

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there's an intersection
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    # Calculate intersection area
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate areas of both boxes
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def run_pytorch_inference(image):
    """Run inference using the PyTorch model."""
    if torch_model is None:
        raise RuntimeError("PyTorch model not loaded")

    results = torch_model.predict(image, imgsz=MODEL_INPUT_SIZE)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class_id': cls_id,
                'class_name': class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
            })
    return detections

def run_openvino_inference_optimized(processed_image):
    """
    Run optimized OpenVINO inference using 2024 best practices
    Supports both synchronous and asynchronous inference
    """
    if compiled_model is None:
        raise RuntimeError("OpenVINO model not loaded")
    
    if ENABLE_ASYNC_INFERENCE and infer_request is not None:
        # Use asynchronous inference for better performance (OpenVINO 2024 best practice)
        infer_request.infer(inputs={input_layer.any_name: processed_image})
        result = infer_request.get_output_tensor(output_layer.index).data
    else:
        # Fallback to synchronous inference
        result = compiled_model({input_layer.any_name: processed_image})[output_layer]
    
    return result

def postprocess_predictions_letterbox(predictions, original_width, original_height, input_width, input_height, scale, paste_x, paste_y, conf_threshold=0.5, iou_threshold=0.45):
    """Postprocess OpenVINO model predictions with letterbox coordinate adjustment"""
    detections = []
    
    # predictions shape is typically [1, num_detections, 85] for YOLO
    # where each detection is [x, y, w, h, confidence, class_scores...]
    if len(predictions.shape) == 3:
        predictions = predictions[0]  # Remove batch dimension
    
    for pred in predictions:
        # Extract box coordinates (center format)
        x_center, y_center, width, height = pred[:4]
        confidence = pred[4]
        
        if confidence < conf_threshold:
            continue
            
        # Get class scores and find the class with highest score
        class_scores = pred[5:]
        class_id = np.argmax(class_scores)
        class_confidence = class_scores[class_id]
        
        # Final confidence is object confidence * class confidence
        final_confidence = confidence * class_confidence
        
        if final_confidence < conf_threshold:
            continue
        
        # Convert from center format to corner format in input space
        x1_input = x_center - width / 2
        y1_input = y_center - height / 2
        x2_input = x_center + width / 2
        y2_input = y_center + height / 2
        
        # Adjust for letterbox padding
        x1_adjusted = (x1_input - paste_x) / scale
        y1_adjusted = (y1_input - paste_y) / scale
        x2_adjusted = (x2_input - paste_x) / scale
        y2_adjusted = (y2_input - paste_y) / scale
        
        # Ensure coordinates are within original image bounds
        x1 = max(0, min(x1_adjusted, original_width))
        y1 = max(0, min(y1_adjusted, original_height))
        x2 = max(0, min(x2_adjusted, original_width))
        y2 = max(0, min(y2_adjusted, original_height))
        
        # Skip invalid boxes
        if x2 <= x1 or y2 <= y1:
            continue
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': final_confidence,
            'class_id': int(class_id),
            'class_name': class_names[class_id] if class_id < len(class_names) else f"unknown_{class_id}"
        })
    
    # Apply Non-Maximum Suppression
    detections = apply_nms(detections, iou_threshold)
    
    return detections

def calculate_distance(box1, box2):
    """Calculate Euclidean distance between box centers"""
    x1, y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    x2, y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def is_duplicate_detection(new_detection, existing_detections, time_threshold=TRACKING_TIME_THRESHOLD):
    """Check if detection is duplicate based on location and time"""
    current_time = time.time()
    new_bbox = new_detection['bbox']
    
    for existing in existing_detections:
        # Check if same class and within time threshold
        if (existing['class_id'] == new_detection['class_id'] and 
            current_time - existing['timestamp'] < time_threshold):
            
            # Check spatial proximity
            distance = calculate_distance(new_bbox, existing['bbox'])
            if distance < TRACKING_DISTANCE_THRESHOLD:
                return True, existing['report_id']
    
    return False, None

def create_report(detection, session_id, image_data=None):
    """Create a new report for a detection"""
    report_id = str(uuid.uuid4())
    report = {
        'report_id': report_id,
        'session_id': session_id,
        'detection': detection,
        'timestamp': datetime.now().isoformat(),
        'status': 'pending',  # pending, confirmed, dismissed
        'image_data': image_data,  # base64 encoded frame image
        'thumbnail': None,  # Will store thumbnail version
        'location': {
            'bbox': detection['bbox'],
            'center': [detection['center_x'], detection['center_y']]
        },
        'frame_info': {
            'has_image': image_data is not None,
            'image_size': len(image_data) if image_data else 0
        }
    }
    return report

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/status")
async def get_status():
    """Provides detailed status information about the service."""
    try:
        # Determine model status based on which backend is loaded
        if USE_OPENVINO and compiled_model is not None:
            model_status = "loaded_openvino"
            model_backend = "openvino"
        elif not USE_OPENVINO and torch_model is not None:
            model_status = "loaded_pytorch"
            model_backend = "pytorch"
        else:
            model_status = "loading"  # More optimistic status during startup
            model_backend = "auto"

        device_info = None

        # Get device information based on loaded model
        if USE_OPENVINO and compiled_model is not None:
            try:
                device_info = {
                    "device": DEVICE_NAME,
                    "input_shape": list(input_layer.shape),
                    "output_shape": list(output_layer.shape),
                    "model_path": "best_openvino_model/best.xml",
                    "cache_enabled": CACHE_ENABLED,
                    "backend": "openvino",
                    "performance_mode": PERFORMANCE_MODE,
                    "async_inference": ENABLE_ASYNC_INFERENCE,
                    "openvino_version": "2024_optimized"
                }
            except Exception as e:
                logger.warning(f"Could not get OpenVINO model info: {e}")
        elif torch_model is not None:
            try:
                device_info = {
                    "model_path": "best.pt",
                    "model_type": str(type(torch_model)),
                    "backend": "pytorch",
                    "device": "cpu"  # YOLO typically uses CPU for this setup
                }
            except Exception as e:
                logger.warning(f"Could not get PyTorch model info: {e}")

        # Get server environment info for mobile debugging
        import platform

        env_info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
            "deployment_env": os.getenv("RENDER", "local") or os.getenv("DEPLOYMENT_ENV", "unknown"),
            "port": os.getenv("PORT", "8000"),
            "cors_enabled": True,
            "mobile_friendly": True
        }

        # Check external API connectivity if available
        api_health = None
        if api_manager is not None:
            try:
                api_health = await api_manager.health_check()
            except Exception as e:
                logger.warning(f"API health check failed: {e}")
                api_health = {"status": "error", "message": str(e)}

        return {
            "status": "healthy",
            "model_status": model_status,
            "backend_inference": model_status.startswith("loaded"),
            "backend_type": model_backend,
            "active_sessions": len(sessions),
            "device_info": device_info,
            "environment": env_info,
            "api_connectors": api_health,
            "model_files": {
                "openvino_model": "/app/best_openvino_model/last_model_train12052025.xml",
                "pytorch_model": "/app/best.pt",
                "current_backend": model_backend,
                "model_classes": len(class_names),
                "input_size": MODEL_INPUT_SIZE
            },
            "endpoints": {
                "health": "/health",
                "status": "/status",
                "session_start": "/session/start",
                "session_detect": "/detect/{session_id}",
                "legacy_detect": "/detect",
                "batch_detect": "/detect-batch",
                "api_health": "/api/health",
                "geocode": "/api/geocode",
                "reverse_geocode": "/api/reverse-geocode"
            }
        }
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@app.get("/")
async def root():
    """Simple root endpoint for Railway gateway"""
    return {
        "status": "ok", 
        "service": "hazard-detection-api", 
        "version": "1.0",
        "message": "FastAPI service is running",
        "endpoints": ["/health", "/session/start", "/detect"]
    }

@app.post("/session/start")
async def start_session():
    """Start a new detection session"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        'id': session_id,
        'start_time': datetime.now().isoformat(),
        'reports': [],
        'detection_count': 0,
        'unique_hazards': 0
    }
    active_detections[session_id] = []

    return {"session_id": session_id}

@app.post("/session/{session_id}/end")
async def end_session(session_id: str):
    """End a detection session and return summary"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    sessions.pop(session_id, None)
    active_detections.pop(session_id, None)
    return {"message": "Session ended"}

@app.get("/session/{session_id}/summary")
async def get_session_summary(session_id: str):
    """Get session summary with all reports"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return sessions[session_id]

@app.post("/session/{session_id}/report/{report_id}/confirm")
async def confirm_report(session_id: str, report_id: str):
    """Confirm a report for submission"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    for report in session['reports']:
        if report['report_id'] == report_id:
            report['status'] = 'confirmed'
            return {"message": "Report confirmed", "report_id": report_id}
    
    raise HTTPException(status_code=404, detail="Report not found")

@app.post("/session/{session_id}/report/{report_id}/dismiss")
async def dismiss_report(session_id: str, report_id: str):
    """Dismiss a report"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    for report in session['reports']:
        if report['report_id'] == report_id:
            report['status'] = 'dismissed'
            return {"message": "Report dismissed", "report_id": report_id}
    
    raise HTTPException(status_code=404, detail="Report not found")

@app.post("/detect/{session_id}")
async def detect_hazards(session_id: str, file: UploadFile | None = File(None)):
    """
    Enhanced detection endpoint with object tracking and report generation
    Returns detections and creates reports for new unique hazards
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Start a session first.")

    if file is None or not file.filename or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="File field required")

    await ensure_model_loaded()

    if USE_OPENVINO and compiled_model is None:
        raise HTTPException(status_code=503, detail="OpenVINO model not loaded. Service may still be starting up.")
    if not USE_OPENVINO and torch_model is None:
        raise HTTPException(status_code=503, detail="PyTorch model not loaded. Service may still be starting up.")
    if compiled_model is None and torch_model is None:
        raise HTTPException(status_code=503, detail="No models available. Please check server logs for loading errors.")

    try:
        start_time = time.time()
        
        # Read and process image
        contents = await file.read()
        image_stream = BytesIO(contents)
        image_stream.seek(0)
        image = Image.open(image_stream).convert("RGB")
        
        # Store original image data for reports
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        # Convert to numpy array for consistency
        # img_array = np.array(image)  # Commented out as not currently used
        
        if USE_OPENVINO:
            input_shape = input_layer.shape
            processed_image, scale, paste_x, paste_y = preprocess_image(image, input_shape)
            
            # Use OpenVINO 2024 optimized inference
            result = run_openvino_inference_optimized(processed_image)
            
            raw_detections = postprocess_predictions_letterbox(
                result,
                image.width,
                image.height,
                input_shape[3],
                input_shape[2],
                scale,
                paste_x,
                paste_y,
                conf_threshold=0.5,
                iou_threshold=0.45
            )
        else:
            raw_detections = run_pytorch_inference(image)
        
        # Process results with tracking and report generation
        detections = []
        new_reports = []
        session = sessions[session_id]
        current_time = time.time()
        
        for raw_detection in raw_detections:
            x1, y1, x2, y2 = raw_detection['bbox']
            confidence = raw_detection['confidence']
            class_id = raw_detection['class_id']
            hazard_type = raw_detection['class_name']
            
            detection = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": hazard_type,
                "center_x": float((x1 + x2) / 2),
                "center_y": float((y1 + y2) / 2),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "area": float((x2 - x1) * (y2 - y1)),
                "timestamp": current_time
            }
            
            # Check for duplicates only for high-confidence detections
            if confidence >= MIN_CONFIDENCE_FOR_REPORT:
                is_duplicate, existing_report_id = is_duplicate_detection(
                    detection, active_detections[session_id]
                )
                
                if not is_duplicate:
                    # Create new report for unique detection with frame image
                    report = create_report(detection, session_id, image_base64)
                    session['reports'].append(report)
                    new_reports.append(report)
                    session['unique_hazards'] += 1
                    
                    # Add to active detections for tracking
                    detection['report_id'] = report['report_id']
                    active_detections[session_id].append(detection)
                    
                    # Mark as new detection
                    detection['is_new'] = True
                    detection['report_id'] = report['report_id']
                else:
                    # Mark as existing detection
                    detection['is_new'] = False
                    detection['report_id'] = existing_report_id
            else:
                # Low confidence detections are not tracked
                detection['is_new'] = False
                detection['report_id'] = None
            
            detections.append(detection)
            session['detection_count'] += 1
        
        processing_time = round((time.time() - start_time) * 1000, 2)  # in milliseconds
        
        logger.info(f"Processed image: {len(detections)} detections in {processing_time}ms")
        
        return {
            "success": True,
            "detections": detections,
            "new_reports": new_reports,
            "session_stats": {
                "total_detections": session['detection_count'],
                "unique_hazards": session['unique_hazards'],
                "pending_reports": len([r for r in session['reports'] if r['status'] == 'pending'])
            },
            "processing_time_ms": processing_time,
            "image_size": {
                "width": image.width,
                "height": image.height
            },
            "model_info": {
                "backend": "openvino" if USE_OPENVINO else "pytorch",
                "classes": class_names,
                "confidence_threshold": MIN_CONFIDENCE_FOR_REPORT,
                "tracking_enabled": True,
                **({
                    "input_shape": list(input_layer.shape),
                    "output_shape": list(output_layer.shape)
                } if USE_OPENVINO else {})
            }
        }
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

@app.post("/detect-batch")
async def detect_batch(files: list[UploadFile] = File(...)):
    """
    Batch detection endpoint for processing multiple images
    """
    if USE_OPENVINO and compiled_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not USE_OPENVINO and torch_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    start_time = time.time()
    
    for i, file in enumerate(files):
        if not file.content_type.startswith("image/"):
            results.append({
                "file_index": i,
                "filename": file.filename,
                "error": "File must be an image"
            })
            continue
        
        try:
            # Process each image
            contents = await file.read()
            image_stream = BytesIO(contents)
            image_stream.seek(0)
            image = Image.open(image_stream).convert("RGB")
            
            if USE_OPENVINO:
                input_shape = input_layer.shape
                processed_image, scale, paste_x, paste_y = preprocess_image(image, input_shape)
                
                # Use OpenVINO 2024 optimized inference
                result = run_openvino_inference_optimized(processed_image)
                
                detections = postprocess_predictions_letterbox(
                    result,
                    image.width,
                    image.height,
                    input_shape[3],
                    input_shape[2],
                    scale,
                    paste_x,
                    paste_y,
                    conf_threshold=0.5,
                    iou_threshold=0.45
                )
            else:
                detections = run_pytorch_inference(image)
            
            # Format detections for API response
            formatted_detections = []
            for detection in detections:
                formatted_detections.append({
                    "bbox": detection['bbox'],
                    "confidence": detection['confidence'],
                    "class_id": detection['class_id'],
                    "class_name": detection['class_name']
                })
            detections = formatted_detections
            
            results.append({
                "file_index": i,
                "filename": file.filename,
                "success": True,
                "detections": detections
            })
            
        except Exception as e:
            results.append({
                "file_index": i,
                "filename": file.filename,
                "error": str(e)
            })
    
    total_time = round((time.time() - start_time) * 1000, 2)
    
    return {
        "success": True,
        "results": results,
        "total_processing_time_ms": total_time,
        "processed_count": len(files)
    }
# Legacy detect endpoint to add to app.py

@app.post("/detect")
async def detect_hazards_legacy(file: UploadFile = File(...)):
    """
    Legacy detection endpoint for backward compatibility
    """
    if USE_OPENVINO and compiled_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not USE_OPENVINO and torch_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        start_time = time.time()
        
        # Read and process image
        contents = await file.read()
        image_stream = BytesIO(contents)
        image_stream.seek(0)
        image = Image.open(image_stream).convert("RGB")
        
        # Store original image data for reports
        image_base64 = base64.b64encode(contents).decode('utf-8')
        
        if USE_OPENVINO:
            input_shape = input_layer.shape
            processed_image, scale, paste_x, paste_y = preprocess_image(image, input_shape)
            
            # Use OpenVINO 2024 optimized inference
            result = run_openvino_inference_optimized(processed_image)
            
            raw_detections = postprocess_predictions_letterbox(
                result,
                image.width,
                image.height,
                input_shape[3],
                input_shape[2],
                scale,
                paste_x,
                paste_y,
                conf_threshold=0.5,
                iou_threshold=0.45
            )
        else:
            raw_detections = run_pytorch_inference(image)
        
        # Format detections for API response
        detections = []
        for raw_detection in raw_detections:
            x1, y1, x2, y2 = raw_detection['bbox']
            confidence = raw_detection['confidence']
            class_id = raw_detection['class_id']
            hazard_type = raw_detection['class_name']
            
            detection = {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": hazard_type,
                "center_x": float((x1 + x2) / 2),
                "center_y": float((y1 + y2) / 2),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "area": float((x2 - x1) * (y2 - y1))
            }
            detections.append(detection)
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        return {
            "success": True,
            "detections": detections,
            "processing_time_ms": processing_time,
            "image_size": {"width": image.width, "height": image.height},
            "model_info": {"backend": "openvino" if USE_OPENVINO else "pytorch", "classes": class_names}
        }
        
    except Exception as e:
        logger.error(f"Legacy detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# API Connector Endpoints
@app.get("/api/health")
async def api_health_check():
    """Check health of all external API services"""
    if api_manager is None:
        raise HTTPException(status_code=503, detail="api_manager not configured")

    try:
        health_status = await api_manager.health_check()
        return {
            "success": True,
            "services": health_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/api/geocode")
async def geocode_address_endpoint(address: str):
    """Geocode an address to coordinates"""
    if geocode_location is None:
        raise HTTPException(status_code=503, detail="Geocoding service unavailable")

    try:
        response = await geocode_location(address)
        if response.success:
            return {
                "success": True,
                "data": response.data,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail=response.error)
    except Exception as e:
        logger.error(f"Geocoding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Geocoding failed: {str(e)}")

@app.post("/api/reverse-geocode")
async def reverse_geocode_endpoint(lat: float, lng: float):
    """Reverse geocode coordinates to address"""
    try:
        response = await reverse_geocode_location(lat, lng)
        if response.success:
            return {
                "success": True,
                "data": response.data,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail=response.error)
    except Exception as e:
        logger.error(f"Reverse geocoding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reverse geocoding failed: {str(e)}")

@app.post("/api/cache-detection")
async def cache_detection_endpoint(detection_id: str, detection_data: dict):
    """Cache detection result"""
    if cache_detection_result is None:
        raise HTTPException(status_code=503, detail="Caching service unavailable")

    try:
        response = await cache_detection_result(detection_id, detection_data)
        if response.success:
            return {
                "success": True,
                "message": "Detection cached successfully",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail=response.error)
    except Exception as e:
        logger.error(f"Caching failed: {e}")
        raise HTTPException(status_code=500, detail=f"Caching failed: {str(e)}")

@app.get("/api/render/status")
async def render_status():
    """Get Render deployment status"""
    if api_manager is None or getattr(api_manager, "render", None) is None:
        raise HTTPException(status_code=503, detail="Render service unavailable")

    try:
        response = await api_manager.render.get_services()
        if response.success:
            return {
                "success": True,
                "services": response.data,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail=response.error)
    except Exception as e:
        logger.error(f"Render status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Add main entry point for Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    print("🔒 OpenVINO API runtime listening privately on port 8080")
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info", access_log=True)
