#!/usr/bin/env python3
"""
Simple test script for the Hazard Detection API
Tests basic functionality without requiring model files
"""

import requests
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# API configuration
API_BASE_URL = "http://localhost:8000"

def create_test_image():
    """Create a simple test image"""
    # Create a simple 640x640 RGB image
    img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return buffer.getvalue()

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Root endpoint test failed: {e}")
        return False

def test_session_management():
    """Test session start and management"""
    print("\nTesting session management...")
    try:
        # Start session
        response = requests.post(f"{API_BASE_URL}/session/start")
        print(f"Start session status: {response.status_code}")
        
        if response.status_code == 200:
            session_data = response.json()
            session_id = session_data['session_id']
            print(f"Session ID: {session_id}")
            
            # Get session summary
            summary_response = requests.get(f"{API_BASE_URL}/session/{session_id}/summary")
            print(f"Session summary status: {summary_response.status_code}")
            
            if summary_response.status_code == 200:
                print(f"Summary: {json.dumps(summary_response.json(), indent=2)}")
            
            # End session
            end_response = requests.post(f"{API_BASE_URL}/session/{session_id}/end")
            print(f"End session status: {end_response.status_code}")
            
            return True
    except Exception as e:
        print(f"Session management test failed: {e}")
        return False

def test_legacy_detection():
    """Test the legacy detection endpoint (without model)"""
    print("\nTesting legacy detection endpoint...")
    try:
        # Create test image
        test_image_bytes = create_test_image()
        
        files = {'file': ('test.jpg', test_image_bytes, 'image/jpeg')}
        response = requests.post(f"{API_BASE_URL}/detect", files=files)
        
        print(f"Detection status: {response.status_code}")
        
        if response.status_code == 503:
            print("Expected: Model not loaded (this is normal without model files)")
            return True
        elif response.status_code == 200:
            print(f"Detection response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"Unexpected status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Legacy detection test failed: {e}")
        return False

def test_api_connectors():
    """Test API connector endpoints"""
    print("\nTesting API connector endpoints...")
    try:
        # Test API health
        response = requests.get(f"{API_BASE_URL}/api/health")
        print(f"API health status: {response.status_code}")
        
        if response.status_code in [200, 503]:  # 503 is expected if not configured
            if response.status_code == 200:
                print(f"API health response: {json.dumps(response.json(), indent=2)}")
            else:
                print("Expected: API connectors not configured")
            return True
        else:
            print(f"Unexpected status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"API connector test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("HAZARD DETECTION API TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint),
        ("Session Management", test_session_management),
        ("Legacy Detection", test_legacy_detection),
        ("API Connectors", test_api_connectors)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        result = test_func()
        results.append((test_name, result))
        print(f"Result: {'PASS' if result else 'FAIL'}")
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")

if __name__ == "__main__":
    run_all_tests()