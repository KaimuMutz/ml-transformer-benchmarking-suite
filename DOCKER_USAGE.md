# Docker Usage Guide for ML Transformer Benchmarking Suite

## Quick Start Commands

### Building the Image
```bash
# Build the Docker image
docker build -t ml-benchmark .

# Build with no cache (if you made changes)
docker build -t ml-benchmark . --no-cache
```

### Running with Docker Commands

```bash
# Quick validation test
docker run --rm ml-benchmark python ml_benchmark_suite.py --mode validate

# Quick functionality test
docker run --rm ml-benchmark python ml_benchmark_suite.py --mode test --sample-size 1000

# Full benchmark with persistent results
mkdir -p docker-results docker-logs
docker run \
  --name ml-benchmark-run \
  -v $(pwd)/docker-results:/app/benchmark_results \
  -v $(pwd)/docker-logs:/app/logs \
  ml-benchmark

# Interactive container for debugging
docker run -it --name ml-debug ml-benchmark bash
```

### Running with Docker Compose (Recommended)

```bash
# Production deployment
docker-compose up -d

# View logs
docker-compose logs -f

# Stop everything
docker-compose down

# Rebuild and restart
docker-compose up --build -d
```

### Development Workflow

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Access development container
docker-compose -f docker-compose.dev.yml exec ml-benchmark-dev bash

# Inside the container, run your tests:
python ml_benchmark_suite.py --mode validate
python ml_benchmark_suite.py --mode test --sample-size 1000

# Stop development environment
docker-compose -f docker-compose.dev.yml down
```

## Checking Results

After running benchmarks in Docker:

```bash
# Check what files were created
ls -la benchmark_results/
ls -la docker-results/

# View benchmark results
cat benchmark_results/benchmark_results.json
cat docker-results/benchmark_results.json

# View logs
tail logs/benchmark.log
tail docker-logs/benchmark.log
```

## Useful Docker Commands

```bash
# List all images
docker images

# List running containers
docker ps

# List all containers
docker ps -a

# Remove container
docker rm container-name

# Remove image
docker rmi image-name

# Clean up unused resources
docker system prune

# View container resource usage
docker stats
```

## Troubleshooting

### Build Issues
- Make sure you're in the project directory with Dockerfile
- Check that requirements.txt exists
- Try building with `--no-cache` flag

### Memory Issues
- Increase Docker Desktop memory allocation
- Add memory limits to docker-compose.yml
- Monitor usage with `docker stats`

### Permission Issues
- Ensure Docker is running
- On Linux, make sure user is in docker group: `sudo usermod -aG docker $USER`
