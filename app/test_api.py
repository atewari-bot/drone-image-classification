"""
Test script for Drone Detection API
Tests the Flask API endpoints to ensure they work correctly.
"""

import requests
import json
import base64
import os
from PIL import Image
import io
import numpy as np

API_BASE_URL = "http://localhost:5000"

def create_test_image():
    """Create a simple test image"""
    # Create a simple 224x224 RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    return img

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ Health check passed")
            print(f"   Status: {result.get('status')}")
            print(f"   Models loaded: {result.get('models_loaded')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is the Flask server running?")
        return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_classes_endpoint():
    """Test the classes endpoint"""
    print("\n🔍 Testing classes endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/classes", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ Classes endpoint passed")
            print(f"   Available classes: {result.get('classes')}")
            return True
        else:
            print(f"❌ Classes endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Classes endpoint error: {str(e)}")
        return False

def test_classification_endpoint():
    """Test the classification endpoint"""
    print("\n🔍 Testing classification endpoint...")
    try:
        # Create test image
        test_image = create_test_image()
        img_base64 = image_to_base64(test_image)
        
        # Test with base64 input
        payload = {"image_base64": img_base64}
        response = requests.post(
            f"{API_BASE_URL}/classify",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Classification endpoint passed")
            print(f"   Predicted class: {result.get('predicted_class')}")
            print(f"   Confidence: {result.get('confidence'):.3f}")
            print(f"   All probabilities: {result.get('class_probabilities')}")
            return True
        else:
            print(f"❌ Classification endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Classification endpoint error: {str(e)}")
        return False

def test_detection_endpoint():
    """Test the detection endpoint"""
    print("\n🔍 Testing detection endpoint...")
    try:
        # Create test image
        test_image = create_test_image()
        img_base64 = image_to_base64(test_image)
        
        # Test with base64 input
        payload = {"image_base64": img_base64}
        response = requests.post(
            f"{API_BASE_URL}/detect",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Detection endpoint passed")
            detections = result.get('detections', [])
            print(f"   Number of detections: {len(detections)}")
            for i, detection in enumerate(detections):
                print(f"   Detection {i+1}: {detection.get('class')} ({detection.get('confidence'):.3f})")
            return True
        else:
            print(f"❌ Detection endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Detection endpoint error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚁 Drone Detection API Test Suite")
    print("==================================")
    
    tests = [
        test_health_endpoint,
        test_classes_endpoint,
        test_classification_endpoint,
        test_detection_endpoint
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the API server and models.")
        
    print("\n💡 To start the API server, run:")
    print("   python app/drone_detection.py")
    print("\n💡 To start the Streamlit UI, run:")
    print("   streamlit run app/streamlit_ui.py")

if __name__ == "__main__":
    main()
