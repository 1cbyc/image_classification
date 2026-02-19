#!/usr/bin/env python3
"""
Unit tests for ReluRay API
These tests don't require a running API server
"""

import pytest
import json
import sys
import os

# Add backend to path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

def test_openapi_schema_structure():
    """Test OpenAPI schema structure without requiring API"""
    # This test checks that the OpenAPI schema would be valid
    # without actually making HTTP requests
    
    # Check that app.py has correct OpenAPI configuration
    with open('backend/app.py', 'r') as f:
        content = f.read()
    
    assert 'openapi_url="/api/openapi.json"' in content, "OpenAPI URL should be configured"
    assert 'docs_url="/api/docs"' in content, "Docs URL should be configured"
    
    # Check that prediction returns lowercase
    assert "result = 'normal'" in content, "Should return lowercase 'normal'"
    assert "result = 'pneumonia'" in content, "Should return lowercase 'pneumonia'"
    
    # Check no false rate limiting claims
    assert '10 requests per minute' not in content.lower(), "Should not claim rate limiting"

def test_pm2_config():
    """Test PM2 configuration"""
    with open('ecosystem.config.js', 'r') as f:
        content = f.read()
    
    # Should use relative paths, not hardcoded paths
    assert 'path.join(__dirname' in content, "Should use path.join for relative paths"
    assert '/home/isaac/reluray' not in content, "Should not have hardcoded paths"

def test_nginx_config():
    """Test nginx configuration for HTTPS"""
    if os.path.exists('nginx-https.conf'):
        with open('nginx-https.conf', 'r') as f:
            content = f.read()
        
        # Should have SSL configuration
        assert 'ssl_certificate' in content, "Should have SSL certificate config"
        assert 'listen 443 ssl' in content, "Should listen on HTTPS port"
        assert 'return 301 https://' in content, "Should redirect HTTP to HTTPS"

def test_test_structure():
    """Test that test files have correct structure"""
    # Check test_api.py is properly marked as standalone
    with open('tests/test_api.py', 'r') as f:
        content = f.read()
    
    assert 'STANDALONE TEST SCRIPT' in content, "Should be marked as standalone"
    assert 'def test_' not in content, "Should not have test_* functions (use check_*)"
    assert 'def check_' in content, "Should have check_* functions"
    
    # Check test_integration.py is a proper pytest class
    with open('tests/test_integration.py', 'r') as f:
        content = f.read()
    
    assert 'class TestAPIIntegration' in content, "Should be a pytest class"
    assert 'import pytest' in content, "Should import pytest"

def test_api_response_format():
    """Test API response format in code"""
    with open('backend/app.py', 'r') as f:
        content = f.read()
    
    # Find the predict function
    lines = content.split('\n')
    in_predict_function = False
    has_lowercase_normal = False
    has_lowercase_pneumonia = False
    
    for line in lines:
        if 'def predict(' in line:
            in_predict_function = True
        elif in_predict_function and 'def ' in line and 'def predict(' not in line:
            in_predict_function = False
        
        if in_predict_function:
            if "result = 'normal'" in line:
                has_lowercase_normal = True
            if "result = 'pneumonia'" in line:
                has_lowercase_pneumonia = True
    
    assert has_lowercase_normal, "Predict function should return 'normal' (lowercase)"
    assert has_lowercase_pneumonia, "Predict function should return 'pneumonia' (lowercase)"

def test_requirements_file():
    """Test that requirements.txt exists and has necessary packages"""
    assert os.path.exists('backend/requirements.txt'), "requirements.txt should exist"
    
    with open('backend/requirements.txt', 'r') as f:
        content = f.read()
    
    # Check for essential packages
    assert 'fastapi' in content.lower(), "Should have FastAPI"
    assert 'uvicorn' in content.lower(), "Should have uvicorn"
    assert 'tensorflow' in content.lower() or 'pytorch' in content.lower(), "Should have ML framework"

def test_documentation_files():
    """Test that documentation files exist"""
    assert os.path.exists('README.md'), "README.md should exist"
    assert os.path.exists('API_DOCUMENTATION.md'), "API_DOCUMENTATION.md should exist"
    
    # Check API documentation has correct info
    with open('API_DOCUMENTATION.md', 'r') as f:
        content = f.read()
    
    assert 'ReluRay API' in content, "Should mention ReluRay API"
    # Check for either OpenAPI or Swagger (both are acceptable)
    assert 'OpenAPI' in content or 'Swagger' in content, "Should mention OpenAPI or Swagger"

def test_frontend_structure():
    """Test frontend basic structure"""
    assert os.path.exists('frontend/package.json'), "frontend/package.json should exist"
    assert os.path.exists('frontend/app/layout.tsx'), "frontend app structure should exist"

if __name__ == "__main__":
    # Allow running as standalone script
    pytest.main([__file__, '-v'])