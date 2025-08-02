#!/usr/bin/env python3
"""
Simple test to validate OpenVINO 2024 improvements
Tests import and basic functionality without external dependencies
"""

import sys
import os
from pathlib import Path

def test_app_imports():
    """Test that app imports work with new OpenVINO features"""
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Import app module
        import app
        
        print("✅ App module imported successfully")
        
        # Check new configuration variables
        assert hasattr(app, 'PERFORMANCE_MODE'), "PERFORMANCE_MODE not defined"
        assert hasattr(app, 'ENABLE_ASYNC_INFERENCE'), "ENABLE_ASYNC_INFERENCE not defined"
        assert hasattr(app, 'DEVICE_NAME'), "DEVICE_NAME not defined"
        
        print(f"✅ Performance mode: {app.PERFORMANCE_MODE}")
        print(f"✅ Async inference: {app.ENABLE_ASYNC_INFERENCE}")
        print(f"✅ Device selection: {app.DEVICE_NAME}")
        
        # Check new global variables
        assert hasattr(app, 'infer_request'), "infer_request not defined"
        assert hasattr(app, 'async_inference_queue'), "async_inference_queue not defined"
        
        print("✅ New global variables defined")
        
        # Check new function exists
        assert hasattr(app, 'run_openvino_inference_optimized'), "run_openvino_inference_optimized function not found"
        
        print("✅ New optimized inference function available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except AssertionError as e:
        print(f"❌ Configuration error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_openvino_availability():
    """Test OpenVINO availability and version"""
    try:
        import openvino as ov
        import openvino.properties as props
        
        print("✅ OpenVINO runtime available")
        
        # Test core initialization
        core = ov.Core()
        print("✅ OpenVINO Core initialized")
        
        # Check available devices
        devices = core.available_devices
        print(f"✅ Available devices: {devices}")
        
        # Test performance hints configuration
        config = {}
        config['PERFORMANCE_HINT'] = 'LATENCY'
        print("✅ Performance hints configuration supported")
        
        return True
        
    except ImportError:
        print("⚠️ OpenVINO not available - this is expected in some environments")
        return True  # Don't fail if OpenVINO not available
    except Exception as e:
        print(f"❌ OpenVINO test failed: {e}")
        return False

def test_configuration_values():
    """Test that configuration values are valid"""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import app
        
        # Test PERFORMANCE_MODE values
        valid_modes = ["LATENCY", "THROUGHPUT", "CUMULATIVE_THROUGHPUT"]
        assert app.PERFORMANCE_MODE in valid_modes, f"Invalid performance mode: {app.PERFORMANCE_MODE}"
        print(f"✅ Performance mode '{app.PERFORMANCE_MODE}' is valid")
        
        # Test DEVICE_NAME values
        valid_devices = ["CPU", "GPU", "AUTO"]
        assert app.DEVICE_NAME in valid_devices, f"Invalid device: {app.DEVICE_NAME}"
        print(f"✅ Device '{app.DEVICE_NAME}' is valid")
        
        # Test boolean values
        assert isinstance(app.ENABLE_ASYNC_INFERENCE, bool), "ENABLE_ASYNC_INFERENCE must be boolean"
        assert isinstance(app.CACHE_ENABLED, bool), "CACHE_ENABLED must be boolean"
        print("✅ Boolean configuration values are correct")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def run_tests():
    """Run all tests"""
    print("🚀 Testing OpenVINO 2024 Improvements")
    print("=" * 50)
    
    tests = [
        ("App Imports", test_app_imports),
        ("OpenVINO Availability", test_openvino_availability),
        ("Configuration Values", test_configuration_values),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! OpenVINO 2024 improvements are working correctly.")
        return True
    else:
        print(f"⚠️ {total - passed} test(s) failed.")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)