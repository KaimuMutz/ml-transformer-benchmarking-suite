.PHONY: help install test benchmark validate clean docker-build docker-run

help:  ## Show help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $1, $2}'

install:  ## Install dependencies
	pip install -r requirements.txt

validate:  ## Validate environment
	python ml_benchmark_suite.py --mode validate

test:  ## Run quick test
	python ml_benchmark_suite.py --mode test --sample-size 1000

benchmark:  ## Run full benchmark
	python ml_benchmark_suite.py --mode benchmark --sample-size 5000

clean:  ## Clean generated files
	rm -rf benchmark_results/ logs/ __pycache__/ *.log

docker-build:  ## Build Docker image
	docker build -t ml-benchmark-suite .

docker-run:  ## Run Docker container
	docker run -v $(PWD)/benchmark_results:/app/benchmark_results ml-benchmark-suite

setup-check:  ## Verify setup
	python scripts/verify_setup.py
