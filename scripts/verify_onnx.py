#!/usr/bin/env python3
"""
OpenVINO Model Verification Script
Verifies best0608.onnx model loading and performs smoke test inference

Usage:
    python scripts/verify_onnx.py [MODEL_PATH]
    
Example:
    python scripts/verify_onnx.py /app/best0608.onnx
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

# Add app to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def verify_openvino_installation():
    """Verify OpenVINO is available"""
    try:
        import openvino as ov
        print(f"✅ OpenVINO {ov.__version__} available")
        return ov
    except ImportError as e:
        print(f"❌ OpenVINO not available: {e}")
        return None

def get_model_path(provided_path: Optional[str] = None) -> Optional[Path]:
    """Get model path from argument or environment"""
    if provided_path:
        model_path = Path(provided_path)
        if model_path.exists():
            return model_path
        else:
            print(f"❌ Model not found at: {provided_path}")
            return None
    
    # Try environment variable
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        model_path = Path(env_path)
        if model_path.exists():
            return model_path
        else:
            print(f"❌ MODEL_PATH not found: {env_path}")
    
    # Try default locations
    search_paths = [
        Path("/app/best0608.onnx"),
        Path("./best0608.onnx"),
        Path("./best0408_openvino_model/best0408.xml"),
        Path("./best.pt"),
    ]
    
    for path in search_paths:
        if path.exists():
            print(f"📄 Found model at: {path}")
            return path
    
    print("❌ No model found in default locations")
    return None

def verify_model_loading(ov_core, model_path: Path) -> Optional[Any]:
    """Verify model can be loaded"""
    try:
        print(f"🔄 Loading model from: {model_path}")
        start_time = time.time()
        
        # Load model
        model = ov_core.read_model(str(model_path))
        load_time = time.time() - start_time
        
        print(f"✅ Model loaded successfully in {load_time:.2f}s")
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def inspect_model(model) -> Dict[str, Any]:
    """Inspect model input/output shapes"""
    try:
        input_info = model.inputs[0]
        output_info = model.outputs[0]
        
        input_shape = list(input_info.shape)
        output_shape = list(output_info.shape)
        
        info = {
            "input_shape": input_shape,
            "output_shape": output_shape,
            "input_name": input_info.any_name,
            "output_name": output_info.any_name,
        }
        
        print(f"📊 Input shape: {input_shape}")
        print(f"📊 Output shape: {output_shape}")
        
        # Verify expected shapes for best0608
        if len(input_shape) == 4 and input_shape[1:] == [3, 480, 480]:
            print("✅ Input shape matches expected 1×3×480×480")
        else:
            print(f"⚠️ Input shape {input_shape} doesn't match expected 1×3×480×480")
        
        if len(output_shape) == 3 and output_shape[1] == 300:
            if output_shape[2] == 6:
                print("✅ Output shape matches expected (1,300,6) for best0608")
            elif output_shape[2] == 9:  # 4 classes + 5 = 9 for best0408
                print("✅ Output shape matches expected (1,300,9) for best0408")
            else:
                print(f"⚠️ Output classes {output_shape[2]} doesn't match expected 6 or 9")
        else:
            print(f"⚠️ Output shape {output_shape} doesn't match expected (1,300,6)")
        
        return info
        
    except Exception as e:
        print(f"❌ Model inspection failed: {e}")
        return {}

def compile_model(ov_core, model, device="CPU") -> Optional[Any]:
    """Compile model for inference"""
    try:
        print(f"⚙️ Compiling model for {device} device...")
        start_time = time.time()
        
        compiled_model = ov_core.compile_model(model, device)
        compile_time = time.time() - start_time
        
        print(f"✅ Model compiled successfully in {compile_time:.2f}s")
        return compiled_model
        
    except Exception as e:
        print(f"❌ Model compilation failed: {e}")
        return None

def create_dummy_input(input_shape) -> np.ndarray:
    """Create dummy input tensor"""
    try:
        # Create dummy image input (1, 3, 480, 480)
        dummy_input = np.random.rand(*input_shape).astype(np.float32)
        print(f"📦 Created dummy input: {dummy_input.shape}")
        return dummy_input
        
    except Exception as e:
        print(f"❌ Dummy input creation failed: {e}")
        return None

def run_inference_test(compiled_model, dummy_input) -> bool:
    """Run smoke test inference"""
    try:
        print("🔄 Running inference smoke test...")
        start_time = time.time()
        
        # Run inference
        result = compiled_model(dummy_input)
        inference_time = time.time() - start_time
        
        # Get output
        output_key = list(result.keys())[0]
        output = result[output_key]
        
        print(f"✅ Inference successful in {inference_time*1000:.1f}ms")
        print(f"📊 Output shape: {output.shape}")
        print(f"📊 Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

def main():
    print("🚀 OpenVINO Model Verification Script")
    print("=" * 50)
    
    # Get model path
    model_path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    model_path = get_model_path(model_path_arg)
    
    if not model_path:
        print("❌ No valid model found")
        sys.exit(1)
    
    # Verify OpenVINO
    ov = verify_openvino_installation()
    if not ov:
        sys.exit(1)
    
    # Create OpenVINO core
    ov_core = ov.Core()
    print(f"🔧 Available devices: {ov_core.available_devices}")
    
    # Load model
    model = verify_model_loading(ov_core, model_path)
    if not model:
        sys.exit(1)
    
    # Inspect model
    model_info = inspect_model(model)
    if not model_info:
        sys.exit(1)
    
    # Compile model
    compiled_model = compile_model(ov_core, model)
    if not compiled_model:
        sys.exit(1)
    
    # Create dummy input
    dummy_input = create_dummy_input(model_info["input_shape"])
    if dummy_input is None:
        sys.exit(1)
    
    # Run inference test
    if not run_inference_test(compiled_model, dummy_input):
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✅ All verification tests passed!")
    print(f"📄 Model: {model_path}")
    print(f"📊 Input: {model_info['input_shape']}")
    print(f"📊 Output: {model_info['output_shape']}")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n❌ Verification interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Verification failed with unexpected error: {e}")
        sys.exit(1)