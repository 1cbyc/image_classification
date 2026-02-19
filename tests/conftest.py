"""
Pytest configuration for ReluRay tests
"""

import pytest
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

def pytest_collection_modifyitems(items):
    """Modify test items to ensure certain tests run properly"""
    for item in items:
        # Mark tests that require the API to be running
        if 'integration' in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark tests that require the model to be loaded
        if 'predict' in item.nodeid.lower() and 'mock' not in item.nodeid.lower():
            item.add_marker(pytest.mark.model_required)

def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires API running)"
    )
    config.addinivalue_line(
        "markers", "model_required: mark test as requiring ML model to be loaded"
    )
    
    # Skip test_api.py since it's a standalone script, not a pytest module
    config.addinivalue_line(
        "norecursedirs", "*.py"
    )

def pytest_runtest_setup(item):
    """Setup before each test"""
    # Skip tests that require model if we're in CI environment
    if 'model_required' in [mark.name for mark in item.iter_markers()]:
        if os.environ.get('CI') == 'true':
            pytest.skip("Skipping model-required test in CI environment")