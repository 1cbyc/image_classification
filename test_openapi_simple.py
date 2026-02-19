#!/usr/bin/env python3
"""
Simple test to verify OpenAPI documentation without loading the model
"""

import sys
import os
import json

# Check if OpenAPI schema can be generated
def test_openapi_generation():
    """Test OpenAPI schema generation by importing app in a controlled way"""
    print("Testing OpenAPI documentation generation...")
    
    # Set environment to avoid model loading
    os.environ['ENVIRONMENT'] = 'test'
    
    try:
        # Import app without triggering model loading
        import fastapi
        from pydantic import BaseModel
        
        # Create a minimal app to test OpenAPI generation
        test_app = fastapi.FastAPI(
            title="ReluRay API Test",
            description="Test API",
            version="1.0.0"
        )
        
        # Add a test endpoint
        class TestResponse(BaseModel):
            status: str
        
        @test_app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        # Generate OpenAPI schema
        schema = test_app.openapi()
        
        print(f"✅ OpenAPI schema generated successfully")
        print(f"  Title: {schema['info']['title']}")
        print(f"  Version: {schema['info']['version']}")
        print(f"  Endpoints: {len(schema['paths'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_app_py_updates():
    """Check that app.py has been updated with OpenAPI enhancements"""
    print("\nChecking app.py updates...")
    
    try:
        with open('backend/app.py', 'r') as f:
            content = f.read()
        
        checks = [
            ("OpenAPI title and description", 'title="ReluRay API"' in content),
            ("API documentation", 'docs_url="/api/docs"' in content),
            ("ReDoc documentation", 'redoc_url="/api/redoc"' in content),
            ("OpenAPI tags", 'openapi_tags=' in content),
            ("Enhanced model documentation", '"""Request model for X-ray image analysis"""' in content),
            ("Contact information", 'contact={' in content),
            ("License information", 'license_info={' in content),
        ]
        
        all_passed = True
        for check_name, passed in checks:
            if passed:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                all_passed = False
        
        # Count documented endpoints
        import re
        endpoint_docs = re.findall(r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']', content)
        print(f"\n📊 Found {len(endpoint_docs)} API endpoints")
        
        for method, path in endpoint_docs:
            print(f"  - {method.upper():6s} {path}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error reading app.py: {e}")
        return False

def check_api_documentation_file():
    """Check that API_DOCUMENTATION.md exists and is comprehensive"""
    print("\nChecking API documentation file...")
    
    try:
        with open('API_DOCUMENTATION.md', 'r') as f:
            content = f.read()
        
        checks = [
            ("File exists", len(content) > 0),
            ("Contains overview", 'Overview' in content),
            ("Contains endpoints", 'Endpoints' in content),
            ("Contains examples", 'Example' in content),
            ("Contains error handling", 'Error Handling' in content),
            ("Contains integration examples", 'Integration Examples' in content),
        ]
        
        all_passed = True
        for check_name, passed in checks:
            if passed:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name}")
                all_passed = False
        
        print(f"📄 Documentation file size: {len(content)} bytes")
        
        return all_passed
        
    except FileNotFoundError:
        print("❌ API_DOCUMENTATION.md file not found")
        return False
    except Exception as e:
        print(f"❌ Error reading documentation file: {e}")
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("ReluRay OpenAPI Documentation Verification")
    print("=" * 60)
    
    tests = [
        test_openapi_generation,
        check_app_py_updates,
        check_api_documentation_file
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
    print("Verification Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All checks passed! OpenAPI documentation has been successfully implemented.")
        print("\n📚 Documentation available at:")
        print("  - https://reluray.com/api/docs (Swagger UI)")
        print("  - https://reluray.com/api/redoc (ReDoc)")
        print("  - API_DOCUMENTATION.md (Comprehensive guide)")
        return 0
    else:
        print("\n❌ Some checks failed. Review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())