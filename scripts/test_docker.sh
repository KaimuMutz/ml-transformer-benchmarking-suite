#!/bin/bash

# Docker Testing Script for ML Benchmarking Suite
# Tests Docker setup without running full benchmark

set -e

echo "Testing Docker Setup for ML Benchmarking Suite"
echo "==============================================="

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "Docker is not running. Please start Docker Desktop."
    exit 1
fi

echo "Docker is available and running"

# Build the image
echo "Building Docker image..."
if docker build -t ml-benchmark-test . > build.log 2>&1; then
    echo "Docker image built successfully"
else
    echo "Docker build failed. Check build.log for details."
    exit 1
fi

# Test basic functionality
echo "Testing container functionality..."
if docker run --rm ml-benchmark-test python ml_benchmark_suite.py --mode validate > test.log 2>&1; then
    echo "Container validation passed"
else
    echo "Container validation failed. Check test.log for details."
    exit 1
fi

# Test with small sample
echo "Testing with small sample..."
if docker run --rm ml-benchmark-test python ml_benchmark_suite.py --mode test --sample-size 500 > sample_test.log 2>&1; then
    echo "Sample test passed"
else
    echo "Sample test failed. Check sample_test.log for details."
    exit 1
fi

# Clean up test image
docker rmi ml-benchmark-test > /dev/null 2>&1

echo ""
echo "All Docker tests passed!"
echo "Your Docker setup is ready for use."
echo ""
echo "Next steps:"
echo "1. Run: docker build -t ml-benchmark ."
echo "2. Run: make docker-test"
echo "3. Run: make docker-run"
