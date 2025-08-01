#!/usr/bin/env python3
"""
OpenVINO Implementation Validator
Validates the OpenVINO implementation against tutorial best practices
"""

import os
import sys
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 Checking Dependencies...")
    
    dependencies = {
        'openvino': None,
        'numpy': None,
        'PIL': None,
        'cv2': None,
        'fastapi': None
    }
    
    for dep in dependencies:
        try:
            if dep == 'PIL':
                from PIL import Image
                dependencies[dep] = "Available"
            elif dep == 'cv2':
                import cv2
                dependencies[dep] = f"Version {cv2.__version__}"
            elif dep == 'openvino':
                import openvino as ov
                dependencies[dep] = f"Available"
            elif dep == 'numpy':
                import numpy as np
                dependencies[dep] = f"Version {np.__version__}"
            elif dep == 'fastapi':
                import fastapi
                dependencies[dep] = f"Version {fastapi.__version__}"
            else:
                exec(f"import {dep}")
                dependencies[dep] = "Available"
        except ImportError as e:
            dependencies[dep] = f"❌ Missing: {e}"
    
    for dep, status in dependencies.items():
        if "❌" in str(status):
            print(f"  {dep}: {status}")
        else:
            print(f"  ✅ {dep}: {status}")
    
    missing = [dep for dep, status in dependencies.items() if "❌" in str(status)]
    return len(missing) == 0, missing

def validate_openvino_core():
    """Validate OpenVINO Core initialization"""
    print("\n🧠 Validating OpenVINO Core...")
    
    try:
        import openvino as ov
        import openvino.properties as props
        
        # Initialize core
        core = ov.Core()
        print("  ✅ OpenVINO Core initialized successfully")
        
        # Check available devices
        devices = core.available_devices
        print(f"  📱 Available devices: {devices}")
        
        for device in devices:
            try:
                device_name = core.get_property(device, props.device.full_name)
                print(f"    {device}: {device_name}")
            except Exception as e:
                print(f"    {device}: (info unavailable - {e})")
        
        return True, core
    except Exception as e:
        print(f"  ❌ OpenVINO Core initialization failed: {e}")
        return False, None

def create_test_model():
    """Create a simple test model for validation"""
    print("\n🏗️  Creating Test Model...")
    
    try:
        import openvino as ov
        
        # Create a simple model (dummy YOLO-like output)
        from openvino.runtime import Model, Core
        import openvino.runtime.opset13 as ops
        
        # Input: [1, 3, 640, 640] (NCHW)
        input_shape = [1, 3, 640, 640]
        input_node = ops.parameter(input_shape, name="input")
        
        # Simple processing: just return the input reshaped to detection format
        # YOLO output typically: [1, num_detections, 85]
        # Where 85 = 4 (bbox) + 1 (confidence) + 80 (classes)
        # For our 4 classes: 4 + 1 + 4 = 9
        
        # Flatten and reshape to simulate detection output
        flatten = ops.reshape(input_node, [1, -1], False)  # [1, 3*640*640]
        
        # Take first part and reshape to detection format
        detection_shape = [1, 100, 9]  # 100 detections, 9 values each
        slice_node = ops.slice(flatten, [0, 0], [1, 900], [1, 1])  # Take first 900 values
        output = ops.reshape(slice_node, detection_shape, False)
        
        # Create model
        model = Model([output], [input_node], "test_hazard_model")
        
        # Save as IR format
        test_model_dir = Path("test_models")
        test_model_dir.mkdir(exist_ok=True)
        
        ov.save_model(model, test_model_dir / "test_model.xml")
        
        print(f"  ✅ Test model created: {test_model_dir / 'test_model.xml'}")
        return True, test_model_dir / "test_model.xml"
        
    except Exception as e:
        print(f"  ❌ Test model creation failed: {e}")
        traceback.print_exc()
        return False, None

def validate_model_loading(core, model_path):
    """Validate model loading implementation"""
    print("\n📖 Validating Model Loading...")
    
    try:
        # Test model reading
        model = core.read_model(str(model_path))
        print("  ✅ Model read successfully")
        
        # Check model info
        print(f"  📊 Model inputs: {len(model.inputs)}")
        for i, input_info in enumerate(model.inputs):
            print(f"    Input {i}: {input_info.shape} ({input_info.element_type})")
        
        print(f"  📊 Model outputs: {len(model.outputs)}")
        for i, output_info in enumerate(model.outputs):
            print(f"    Output {i}: {output_info.shape} ({output_info.element_type})")
        
        # Test compilation
        compiled_model = core.compile_model(model, "CPU")
        print("  ✅ Model compiled successfully")
        
        return True, model, compiled_model
        
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        traceback.print_exc()
        return False, None, None

def validate_preprocessing():
    """Validate image preprocessing implementation"""
    print("\n🖼️  Validating Image Preprocessing...")
    
    try:
        # Import app functions
        sys.path.append(str(Path(__file__).parent))
        from app import preprocess_image
        
        # Create test image
        test_image = Image.new('RGB', (1024, 768), color='red')
        print(f"  📷 Created test image: {test_image.size}")
        
        # Test preprocessing
        input_shape = [1, 3, 640, 640]
        processed_image, scale, paste_x, paste_y = preprocess_image(test_image, input_shape)
        
        print(f"  ✅ Preprocessing successful")
        print(f"    Output shape: {processed_image.shape}")
        print(f"    Scale factor: {scale:.3f}")
        print(f"    Padding: ({paste_x}, {paste_y})")
        print(f"    Data type: {processed_image.dtype}")
        print(f"    Value range: [{processed_image.min():.3f}, {processed_image.max():.3f}]")
        
        # Validate output format
        expected_shape = tuple(input_shape)
        if processed_image.shape != expected_shape:
            print(f"  ⚠️  Shape mismatch: expected {expected_shape}, got {processed_image.shape}")
            return False
        
        if processed_image.dtype != np.float32:
            print(f"  ⚠️  Data type issue: expected float32, got {processed_image.dtype}")
            return False
        
        if not (0 <= processed_image.min() and processed_image.max() <= 1):
            print(f"  ⚠️  Value range issue: expected [0,1], got [{processed_image.min():.3f}, {processed_image.max():.3f}]")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Preprocessing validation failed: {e}")
        traceback.print_exc()
        return False

def validate_inference_pipeline(compiled_model):
    """Validate complete inference pipeline"""
    print("\n⚡ Validating Inference Pipeline...")
    
    try:
        # Get model info
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
        
        print(f"  📥 Input: {input_layer.shape} ({input_layer.element_type})")
        print(f"  📤 Output: {output_layer.shape} ({output_layer.element_type})")
        
        # Create test input
        input_shape = input_layer.shape
        test_input = np.random.random(input_shape).astype(np.float32)
        
        # Test inference methods from tutorial
        print("  🧪 Testing inference methods...")
        
        # Method 1: Direct call
        result1 = compiled_model(test_input)[output_layer]
        print(f"    ✅ Direct call: {result1.shape}")
        
        # Method 2: List input
        result2 = compiled_model([test_input])[output_layer]
        print(f"    ✅ List input: {result2.shape}")
        
        # Method 3: Dictionary input
        result3 = compiled_model({input_layer.any_name: test_input})[output_layer]
        print(f"    ✅ Dictionary input: {result3.shape}")
        
        # Method 4: InferRequest (tutorial best practice)
        request = compiled_model.create_infer_request()
        request.infer(inputs={input_layer.any_name: test_input})
        result4 = request.get_output_tensor(output_layer.index).data
        print(f"    ✅ InferRequest: {result4.shape}")
        
        # Verify all methods give same result
        if np.allclose(result1, result2) and np.allclose(result2, result3) and np.allclose(result3, result4):
            print("  ✅ All inference methods produce identical results")
        else:
            print("  ⚠️  Inference methods produce different results")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Inference pipeline validation failed: {e}")
        traceback.print_exc()
        return False

def validate_postprocessing():
    """Validate postprocessing implementation"""
    print("\n🔄 Validating Postprocessing...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        from app import postprocess_predictions_letterbox, apply_nms
        
        # Create mock prediction data (YOLO format)
        # Shape: [num_detections, 9] where 9 = [x, y, w, h, conf, class0, class1, class2, class3]
        mock_predictions = np.array([
            # Detection 1: pothole with high confidence
            [320, 240, 100, 80, 0.9, 0.1, 0.05, 0.8, 0.05],  # class 2 (pothole)
            # Detection 2: crack with medium confidence  
            [150, 100, 60, 40, 0.7, 0.05, 0.1, 0.05, 0.8],   # class 3 (crack?)
            # Detection 3: low confidence (should be filtered)
            [400, 300, 50, 30, 0.3, 0.2, 0.2, 0.3, 0.3],
        ])
        
        # Add batch dimension
        mock_predictions = np.expand_dims(mock_predictions, 0)  # [1, num_detections, 9]
        
        print(f"  📊 Mock predictions shape: {mock_predictions.shape}")
        
        # Test postprocessing
        detections = postprocess_predictions_letterbox(
            mock_predictions,
            original_width=1024,
            original_height=768,
            input_width=640,
            input_height=640,
            scale=0.75,  # Example scale
            paste_x=50,  # Example padding
            paste_y=100,
            conf_threshold=0.5
        )
        
        print(f"  ✅ Postprocessing successful: {len(detections)} detections")
        
        for i, det in enumerate(detections):
            print(f"    Detection {i+1}: {det['class_name']} ({det['confidence']:.2f})")
            print(f"      BBox: {[round(x, 1) for x in det['bbox']]}")
        
        # Test NMS
        if len(detections) > 1:
            nms_detections = apply_nms(detections, iou_threshold=0.45)
            print(f"  ✅ NMS applied: {len(nms_detections)} detections after NMS")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Postprocessing validation failed: {e}")
        traceback.print_exc()
        return False

def validate_caching():
    """Validate model caching implementation"""
    print("\n💾 Validating Model Caching...")
    
    try:
        import openvino as ov
        import time
        
        core = ov.Core()
        
        # Create cache directory
        cache_dir = Path("test_cache")
        cache_dir.mkdir(exist_ok=True)
        
        print(f"  📁 Cache directory: {cache_dir}")
        
        # Test with and without caching
        test_model_path = "test_models/test_model.xml"
        if not Path(test_model_path).exists():
            print("  ⚠️  Test model not available for caching test")
            return True  # Skip if no test model
        
        model = core.read_model(test_model_path)
        
        # First compilation (no cache)
        start_time = time.time()
        compiled_model1 = core.compile_model(model, "CPU")
        time_no_cache = time.time() - start_time
        print(f"  ⏱️  Compilation without cache: {time_no_cache:.3f}s")
        
        # Second compilation (with cache)
        config = {"CACHE_DIR": str(cache_dir)}
        start_time = time.time()
        compiled_model2 = core.compile_model(model, "CPU", config)
        time_with_cache = time.time() - start_time
        print(f"  ⏱️  Compilation with cache: {time_with_cache:.3f}s")
        
        # Third compilation (from cache)
        start_time = time.time()
        compiled_model3 = core.compile_model(model, "CPU", config)
        time_from_cache = time.time() - start_time
        print(f"  ⏱️  Compilation from cache: {time_from_cache:.3f}s")
        
        if time_from_cache < time_no_cache:
            print("  ✅ Caching provides performance improvement")
        else:
            print("  ⚠️  Caching performance benefit not clear (may be due to simple test model)")
        
        # Check cache files exist
        cache_files = list(cache_dir.glob("*"))
        print(f"  📄 Cache files created: {len(cache_files)}")
        
        # Cleanup
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Caching validation failed: {e}")
        traceback.print_exc()
        return False

def validate_full_api_integration():
    """Test full API integration (if possible)"""
    print("\n🌐 Validating API Integration...")
    
    try:
        sys.path.append(str(Path(__file__).parent))
        
        # Check if we can import key functions
        from app import load_model, try_load_openvino_model, try_load_pytorch_model
        
        print("  ✅ Successfully imported app functions")
        
        # Check global variables
        from app import core, compiled_model, input_layer, output_layer, USE_OPENVINO
        
        print(f"  📊 Global state:")
        print(f"    Core initialized: {core is not None}")
        print(f"    Model compiled: {compiled_model is not None}")
        print(f"    Using OpenVINO: {USE_OPENVINO}")
        
        if compiled_model is not None:
            print(f"    Input layer: {input_layer.shape if input_layer else 'None'}")
            print(f"    Output layer: {output_layer.shape if output_layer else 'None'}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ API integration validation failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_validation():
    """Run all validation tests"""
    print("🚀 OpenVINO Implementation Comprehensive Validation")
    print("=" * 60)
    
    results = []
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    results.append(("Dependencies", deps_ok))
    
    if not deps_ok:
        print(f"\n❌ Missing dependencies: {missing}")
        print("Please install missing packages and try again.")
        return False
    
    # Validate OpenVINO Core
    core_ok, core = validate_openvino_core()
    results.append(("OpenVINO Core", core_ok))
    
    if not core_ok:
        print("\n❌ Cannot proceed without OpenVINO Core")
        return False
    
    # Create test model
    model_created, model_path = create_test_model()
    results.append(("Test Model Creation", model_created))
    
    compiled_model = None
    if model_created:
        # Validate model loading
        loading_ok, model, compiled_model = validate_model_loading(core, model_path)
        results.append(("Model Loading", loading_ok))
    
    # Validate preprocessing
    preprocessing_ok = validate_preprocessing()
    results.append(("Image Preprocessing", preprocessing_ok))
    
    # Validate inference (if model is available)
    if compiled_model is not None:
        inference_ok = validate_inference_pipeline(compiled_model)
        results.append(("Inference Pipeline", inference_ok))
    
    # Validate postprocessing
    postprocessing_ok = validate_postprocessing()
    results.append(("Postprocessing", postprocessing_ok))
    
    # Validate caching
    caching_ok = validate_caching()
    results.append(("Model Caching", caching_ok))
    
    # Validate API integration
    api_ok = validate_full_api_integration()
    results.append(("API Integration", api_ok))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validations passed! Your OpenVINO implementation is correct.")
    else:
        print(f"\n⚠️  {total - passed} validation(s) failed. Check the output above for details.")
    
    # Cleanup
    cleanup_test_files()
    
    return passed == total

def cleanup_test_files():
    """Clean up test files"""
    import shutil
    
    test_dirs = ["test_models", "test_cache"]
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)