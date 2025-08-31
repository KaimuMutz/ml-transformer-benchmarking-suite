#!/usr/bin/env python3
"""Final project validation script"""

import json
import os
from pathlib import Path

def validate_project_structure():
    """Validate that all project files are in place."""
    
    required_files = [
        'ml_benchmark_suite.py',
        'requirements.txt', 
        'README.md',
        'Dockerfile',
        'docker-compose.yml',
        'docker-compose.dev.yml',
        'Makefile',
        'DOCKER_USAGE.md',
        '.env.docker',
        'benchmark_results/benchmark_results.json',
        'reports/BENCHMARK_SUMMARY.md',
        'reports/PERFORMANCE_ANALYSIS.md',
        'DEPLOYMENT.md',
        'CHANGELOG.md',
        '.github/workflows/ci.yml',
        'scripts/verify_setup.py',
        'scripts/generate_results_visualization.py',
        'scripts/healthcheck.py',
        'scripts/test_docker.sh'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing files: {missing_files}")
        return False
    
    print("All required files present!")
    return True

def validate_results():
    """Validate benchmark results format and content."""
    
    try:
        with open('benchmark_results/benchmark_results.json', 'r') as f:
            results = json.load(f)
        
        expected_models = ['distilbert', 'electra-small', 'bert-base']
        found_models = list(results.keys())
        
        if not all(model in found_models for model in expected_models):
            print(f"Missing models in results. Expected: {expected_models}, Found: {found_models}")
            return False
        
        # Check result structure
        for model, data in results.items():
            required_keys = ['accuracy', 'f1_score', 'training_time_seconds', 'throughput_samples_per_second', 'parameter_count']
            if not all(key in data for key in required_keys):
                print(f"Missing keys in {model} results")
                return False
        
        print("Benchmark results validation passed!")
        return True
        
    except Exception as e:
        print(f"Results validation failed: {e}")
        return False

def validate_docker_setup():
    """Validate Docker configuration files."""
    
    docker_files = [
        'Dockerfile',
        'docker-compose.yml', 
        'docker-compose.dev.yml',
        '.dockerignore',
        'Makefile',
        'DOCKER_USAGE.md'
    ]
    
    for file_path in docker_files:
        if not Path(file_path).exists():
            print(f"Missing Docker file: {file_path}")
            return False
    
    print("Docker setup validation passed!")
    return True

def main():
    """Run complete project validation."""
    
    print("ML Benchmarking Suite - Final Validation (with Docker)")
    print("=" * 60)
    
    structure_valid = validate_project_structure()
    results_valid = validate_results()
    docker_valid = validate_docker_setup()
    
    if structure_valid and results_valid and docker_valid:
        print("\nPROJECT VALIDATION PASSED!")
        print("Repository is ready for commit and push.")
        
        # Display summary
        with open('benchmark_results/benchmark_results.json', 'r') as f:
            results = json.load(f)
        
        print("\nBenchmark Results Summary:")
        print("-" * 30)
        for model, data in results.items():
            print(f"{data['model_short_name']}: {data['accuracy']:.3f} accuracy, {data['throughput_samples_per_second']:.0f} samples/s")
        
        print("\nDocker Setup Confirmed:")
        print("-" * 30)
        print("Dockerfile configured")
        print("Docker Compose files ready")
        print("Development environment configured")
        print("Makefile with common commands")
        print("Health checks and testing scripts")
        
        return True
    else:
        print("\nPROJECT VALIDATION FAILED!")
        print("Please check missing files or invalid results.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
