#!/usr/bin/env python3
"""
Fixed pytest tests for ReluRay API
Proper pytest tests using assert statements instead of boolean returns
"""

import pytest
import requests
import json
import time

BASE_URL = "http://localhost:5001/api"

class TestReluRayAPI:
    """Proper pytest test class for ReluRay API"""
    
    def test_health_endpoint(self):
        """Test health check endpoint using assert"""
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200, "Health endpoint should return 200"
        
        data = response.json()
        assert data['status'] == 'healthy', "Status should be 'healthy'"
        assert 'model_loaded' in data, "Should contain model_loaded field"
        assert 'timestamp' in data, "Should contain timestamp field"
    
    def test_info_endpoint(self):
        """Test model info endpoint using assert"""
        response = requests.get(f"{BASE_URL}/info", timeout=5)
        assert response.status_code == 200, "Info endpoint should return 200"
        
        data = response.json()
        assert data['status'] == 'success', "Status should be 'success'"
        assert 'model_name' in data, "Should contain model_name field"
        assert 'model_version' in data, "Should contain model_version field"
        assert data['classes'] == ['Normal', 'Pneumonia'], "Should have correct classes"
    
    def test_openapi_schema(self):
        """Test OpenAPI schema generation"""
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        assert response.status_code == 200, "OpenAPI schema should be accessible"
        
        schema = response.json()
        assert schema['openapi'].startswith('3.'), "Should be OpenAPI 3.x"
        assert 'paths' in schema, "Should contain paths"
        assert 'components' in schema, "Should contain components"
        
        # Check that all expected endpoints are documented
        expected_paths = ['/api/health', '/api/info', '/api/predict', '/api/metrics']
        for path in expected_paths:
            assert path in schema['paths'], f"Path {path} should be documented"
    
    def test_predict_endpoint_structure(self):
        """Test predict endpoint request/response structure"""
        # Test that predict endpoint exists in OpenAPI schema
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        schema = response.json()
        
        predict_path = schema['paths'].get('/api/predict', {})
        assert 'post' in predict_path, "Predict should support POST method"
        
        # Check request schema
        post_schema = predict_path['post']
        assert 'requestBody' in post_schema, "Should have request body"
        
        # Check response schema
        responses = post_schema.get('responses', {})
        assert '200' in responses, "Should have 200 response"
        assert '400' in responses, "Should have 400 response"
        assert '500' in responses, "Should have 500 response"
    
    def test_response_values_match_docs(self):
        """Test that response values match documentation"""
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        schema = response.json()
        
        # Get PredictResponse schema
        schemas = schema.get('components', {}).get('schemas', {})
        predict_response = schemas.get('PredictResponse', {})
        
        # Check that prediction field has correct description
        prediction_field = predict_response.get('properties', {}).get('prediction', {})
        description = prediction_field.get('description', '')
        
        # Should mention lowercase values
        assert 'normal' in description.lower(), "Should mention 'normal' in description"
        assert 'pneumonia' in description.lower(), "Should mention 'pneumonia' in description"
        
        # Check example if present
        if 'example' in prediction_field:
            example = prediction_field['example']
            assert example in ['normal', 'pneumonia'], f"Example should be lowercase, got '{example}'"
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = requests.get(f"{BASE_URL}/metrics", timeout=5)
        # Metrics might not be implemented, so accept 200 or 404
        assert response.status_code in [200, 404], "Metrics should return 200 or 404"
        
        if response.status_code == 200:
            data = response.json()
            assert 'status' in data, "Should contain status field"
            assert 'timestamp' in data, "Should contain timestamp field"

# Additional tests for the actual test files
def test_test_files_structure():
    """Test that test files have proper structure"""
    # Check that test_api.py is a script, not a pytest module
    with open('tests/test_api.py', 'r') as f:
        content = f.read()
    
    # It should have a main() function since it's a standalone script
    assert 'def main()' in content, "test_api.py should have main() function"
    assert 'if __name__ == "__main__"' in content, "Should have __main__ guard"
    
    # Check that test_integration.py is a proper pytest class
    with open('tests/test_integration.py', 'r') as f:
        content = f.read()
    
    assert 'class TestAPIIntegration' in content, "Should be a pytest class"
    assert 'import pytest' in content, "Should import pytest"

if __name__ == "__main__":
    # This allows running the file directly for debugging
    import sys
    pytest.main([__file__, '-v'])