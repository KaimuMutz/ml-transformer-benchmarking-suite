#!/usr/bin/env python3
"""
Health check script for Docker container
Verifies that the ML application is functioning correctly
"""

import sys
import os
import shutil
from pathlib import Path

def check_python_environment():
    """Check if Python environment is set up correctly."""
    try:
        import torch
        import transformers
        import pandas as pd
        import numpy as np
        return True, "Python environment OK"
    except ImportError as e:
        return False, f"Missing dependency: {e}"

def check_disk_space():
    """Check if there's sufficient disk space."""
    try:
        total, used, free = shutil.disk_usage('.')
        free_gb = free / (1024**3)
        if free_gb < 1.0:  # Less than 1GB
            return False, f"Low disk space: {free_gb:.2f}GB available"
        return True, f"Disk space OK: {free_gb:.2f}GB available"
    except Exception as e:
        return False, f"Disk check failed: {e}"

def check_directories():
    """Check if required directories exist and are writable."""
    required_dirs = ['benchmark_results', 'logs', 'data', 'models']
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return False, f"Cannot create directory {dir_name}: {e}"
        
        # Test write permission
        test_file = dir_path / '.write_test'
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            return False, f"Cannot write to directory {dir_name}: {e}"
    
    return True, "Directories OK"

def main():
    """Main health check function."""
    print("Docker Health Check - ML Transformer Benchmarking Suite")
    print("=" * 60)
    
    checks = [
        ("Python Environment", check_python_environment),
        ("Disk Space", check_disk_space),
        ("Directories", check_directories),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            passed, message = check_func()
            status = "PASS" if passed else "FAIL"
            print(f"{status} {check_name}: {message}")
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"FAIL {check_name}: Unexpected error - {e}")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All health checks passed!")
        return 0
    else:
        print("Some health checks failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
