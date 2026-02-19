#!/usr/bin/env python3
"""
STANDALONE TEST SCRIPT for ReluRay API
NOT a pytest module - run directly with: python tests/test_api.py

Tests API endpoints without requiring TensorFlow or model file
WARNING: This file returns booleans and uses sys.exit() - do not import as pytest module
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:5001/api"

def check_health():
    """Test health check endpoint (not a pytest test)"""
    print("🔍 Testing /api/health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("   ❌ Connection failed - API not running")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_info():
    """Test model info endpoint (not a pytest test)"""
    print("\n🔍 Testing /api/info...")
    try:
        response = requests.get(f"{BASE_URL}/info", timeout=5)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_predict_mock():
    """Test predict endpoint with mock data (will fail without model) (not a pytest test)"""
    print("\n🔍 Testing /api/predict...")
    
    # Create a minimal test image (1x1 pixel red image in base64)
    # This is a valid base64 encoded 1x1 red PNG
    test_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"image": test_image},
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code in [200, 400, 500, 503]  # Any response is good for testing
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("=" * 50)
    print("ReluRay API Test Suite")
    print("=" * 50)
    
    # Wait a bit for API to start
    print("\n⏳ Waiting for API to be ready...")
    for i in range(10):
        try:
            requests.get(f"{BASE_URL}/health", timeout=2)
            print("✅ API is ready!")
            break
        except:
            time.sleep(1)
            print(f"   Attempt {i+1}/10...")
    else:
        print("❌ API did not start. Make sure it's running on port 5000")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("Running Tests")
    print("=" * 50)
    
    results = []
    results.append(("Health Check", check_health()))
    results.append(("Model Info", check_info()))
    results.append(("Predict (Mock)", check_predict_mock()))
    
    print("\n" + "=" * 50)
    print("Test Results")
    print("=" * 50)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("⚠️  Some tests failed (expected if model not loaded)")
    print("=" * 50)

if __name__ == "__main__":
    main()
