#!/usr/bin/env python3
"""
Main test suite for CI/CD pipelines
Runs all tests that don't require a running API server

This is a test runner script, NOT a test module.
Do not import this as a pytest module.
"""

import sys
import os

def main():
    """Run all CI-friendly tests"""
    # Get the project root directory (where this script is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Add the virtual environment to the path if it exists
    venv_python = os.path.join(project_root, 'backend', 'venv', 'bin', 'python')
    
    # Test files are relative to project root
    test_files = [
        os.path.join(script_dir, "test_unit.py"),
        os.path.join(script_dir, "test_ci_integration.py"),
    ]
    
    if os.path.exists(venv_python):
        # We'll run pytest through the virtual environment
        import subprocess
        
        # Build the command
        cmd = [venv_python, "-m", "pytest"] + test_files + ["-v", "--tb=short"]
        
        # Run the command from project root
        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode
    else:
        # Try to import pytest directly
        try:
            import pytest
            
            pytest_args = test_files + ["-v", "--tb=short"]
            exit_code = pytest.main(pytest_args)
            return exit_code
        except ImportError:
            print("Error: pytest not installed. Install with: pip install pytest")
            return 1

if __name__ == "__main__":
    # Only run when executed directly, not when imported
    sys.exit(main())