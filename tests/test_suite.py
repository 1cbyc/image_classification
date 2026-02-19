#!/usr/bin/env python3
"""
Main test suite for CI/CD pipelines
Runs all tests that don't require a running API server
"""

import sys
import os

# Add the virtual environment to the path if it exists
venv_python = os.path.join(os.path.dirname(__file__), '..', 'backend', 'venv', 'bin', 'python')
if os.path.exists(venv_python):
    # We'll run pytest through the virtual environment
    import subprocess
    
    test_files = [
        "tests/test_unit.py",
        "tests/test_ci_integration.py",
    ]
    
    # Build the command
    cmd = [venv_python, "-m", "pytest"] + test_files + ["-v", "--tb=short"]
    
    # Run the command
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
else:
    # Try to import pytest directly
    try:
        import pytest
        
        test_files = [
            "tests/test_unit.py",
            "tests/test_ci_integration.py",
        ]
        
        pytest_args = test_files + ["-v", "--tb=short"]
        exit_code = pytest.main(pytest_args)
        sys.exit(exit_code)
    except ImportError:
        print("Error: pytest not installed. Install with: pip install pytest")
        sys.exit(1)