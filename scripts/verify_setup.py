#!/usr/bin/env python3
"""Setup verification script"""

import sys
import importlib
from pathlib import Path

def check_dependencies():
    required = [
        'torch', 'transformers', 'sklearn', 'pandas', 
        'numpy', 'matplotlib', 'seaborn', 'rich', 'kagglehub'
    ]
    
    missing = []
    for package in required:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        return False
    
    print("All dependencies available!")
    return True

def check_files():
    required_files = ['ml_benchmark_suite.py', 'requirements.txt', 'README.md']
    missing = [f for f in required_files if not Path(f).exists()]
    
    if missing:
        print(f"Missing files: {', '.join(missing)}")
        return False
    
    print("All required files present!")
    return True

def main():
    print("ML Benchmarking Suite - Setup Verification")
    print("=" * 50)
    
    if check_files() and check_dependencies():
        print("Setup verification PASSED!")
        return 0
    else:
        print("Setup verification FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
