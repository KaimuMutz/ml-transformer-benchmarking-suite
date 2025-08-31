FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy main application
COPY ml_benchmark_suite.py .

# Create output directories
RUN mkdir -p benchmark_results logs data models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/.cache/torch

# Default command
CMD ["python", "ml_benchmark_suite.py", "--mode", "benchmark", "--sample-size", "3000"]
