.PHONY: help docker-build docker-test docker-run docker-dev docker-clean docker-logs

# Default target
help:  ## Show this help message
	@echo "Docker commands for ML Transformer Benchmarking Suite"
	@echo "======================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $1, $2}'

# Docker operations
docker-build:  ## Build Docker image
	docker build -t ml-benchmark .

docker-rebuild:  ## Rebuild Docker image without cache
	docker build -t ml-benchmark . --no-cache

docker-test:  ## Run quick tests in Docker
	docker run --rm ml-benchmark python ml_benchmark_suite.py --mode validate
	docker run --rm ml-benchmark python ml_benchmark_suite.py --mode test --sample-size 1000

docker-run:  ## Run benchmark with persistent storage
	@mkdir -p docker-results docker-logs
	docker run --name ml-benchmark-run \
		-v $(pwd)/docker-results:/app/benchmark_results \
		-v $(pwd)/docker-logs:/app/logs \
		ml-benchmark

docker-run-bg:  ## Run benchmark in background
	@mkdir -p docker-results docker-logs
	docker run -d --name ml-benchmark-bg \
		-v $(pwd)/docker-results:/app/benchmark_results \
		-v $(pwd)/docker-logs:/app/logs \
		ml-benchmark

docker-dev:  ## Start development environment
	docker-compose -f docker-compose.dev.yml up -d

docker-dev-shell:  ## Access development container shell
	docker-compose -f docker-compose.dev.yml exec ml-benchmark-dev bash

docker-dev-stop:  ## Stop development environment
	docker-compose -f docker-compose.dev.yml down

docker-prod:  ## Start production environment
	docker-compose up -d

docker-prod-stop:  ## Stop production environment
	docker-compose down

docker-logs:  ## View container logs
	docker logs ml-benchmark-run || docker-compose logs -f

docker-clean:  ## Clean up Docker resources
	docker container prune -f
	docker image prune -f
	docker volume prune -f

docker-clean-all:  ## Remove all Docker resources (use carefully!)
	docker system prune -a -f

# Status commands
docker-status:  ## Show running containers
	docker ps

docker-images:  ## Show Docker images
	docker images

docker-stats:  ## Show container resource usage
	docker stats --no-stream
