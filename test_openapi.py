#!/usr/bin/env python3
"""
Test script to verify OpenAPI documentation generation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from fastapi.testclient import TestClient
import json

# Import the app
try:
    from app import app
    print("✅ Successfully imported FastAPI app")
except ImportError as e:
    print(f"❌ Failed to import app: {e}")
    sys.exit(1)

# Create test client
client = TestClient(app)

def test_openapi_schema():
    """Test that OpenAPI schema is generated correctly"""
    print("\n🔍 Testing OpenAPI schema generation...")
    
    response = client.get("/openapi.json")
    
    if response.status_code == 200:
        print("✅ OpenAPI schema generated successfully")
        
        schema = response.json()
        
        # Check basic OpenAPI structure
        required_keys = ["openapi", "info", "paths"]
        for key in required_keys:
            if key in schema:
                print(f"  ✅ Contains '{key}'")
            else:
                print(f"  ❌ Missing '{key}'")
        
        # Check API info
        info = schema.get("info", {})
        print(f"  📝 API Title: {info.get('title', 'Missing')}")
        print(f"  📝 API Version: {info.get('version', 'Missing')}")
        print(f"  📝 API Description: {info.get('description', 'Missing')[:50]}...")
        
        # Check endpoints
        paths = schema.get("paths", {})
        print(f"  🔗 Number of documented endpoints: {len(paths)}")
        
        for path, methods in paths.items():
            print(f"    - {path}: {list(methods.keys())}")
        
        return True
    else:
        print(f"❌ Failed to get OpenAPI schema: {response.status_code}")
        return False

def test_docs_endpoints():
    """Test that documentation endpoints are accessible"""
    print("\n🔍 Testing documentation endpoints...")
    
    endpoints = [
        ("/api/docs", "Swagger UI"),
        ("/api/redoc", "ReDoc"),
    ]
    
    all_ok = True
    for endpoint, name in endpoints:
        response = client.get(endpoint)
        if response.status_code == 200:
            print(f"✅ {name} accessible at {endpoint}")
        else:
            print(f"❌ {name} not accessible: {response.status_code}")
            all_ok = False
    
    return all_ok

def test_api_endpoints():
    """Test that API endpoints are documented"""
    print("\n🔍 Checking API endpoint documentation...")
    
    response = client.get("/openapi.json")
    schema = response.json()
    paths = schema.get("paths", {})
    
    expected_endpoints = [
        "/api/health",
        "/api/metrics", 
        "/api/predict",
        "/api/info"
    ]
    
    all_documented = True
    for endpoint in expected_endpoints:
        if endpoint in paths:
            print(f"✅ {endpoint} is documented")
        else:
            print(f"❌ {endpoint} is NOT documented")
            all_documented = False
    
    return all_documented

def test_response_models():
    """Test that response models are properly documented"""
    print("\n🔍 Checking response model documentation...")
    
    response = client.get("/openapi.json")
    schema = response.json()
    
    components = schema.get("components", {}).get("schemas", {})
    
    expected_models = [
        "HealthResponse",
        "PredictResponse", 
        "ErrorResponse",
        "ModelInfoResponse",
        "PredictRequest"
    ]
    
    all_documented = True
    for model in expected_models:
        if model in components:
            print(f"✅ {model} schema is documented")
        else:
            print(f"❌ {model} schema is NOT documented")
            all_documented = False
    
    return all_documented

def main():
    """Run all tests"""
    print("=" * 60)
    print("ReluRay OpenAPI Documentation Test")
    print("=" * 60)
    
    tests = [
        test_openapi_schema,
        test_docs_endpoints,
        test_api_endpoints,
        test_response_models
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! OpenAPI documentation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())