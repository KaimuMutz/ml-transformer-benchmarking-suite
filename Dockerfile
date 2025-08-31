# Use Python 3.10 on lightweight Linux
FROM python:3.10-slim

# Environment variables for better Python behavior in containers
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WANDB_DISABLED=true \
    WANDB_MODE=disabled \
    PIP_NO_CACHE_DIR=1

# Create non-root user for security best practice
RUN groupadd --gid 1000 mluser && \
    useradd --uid 1000 --gid mluser --shell /bin/bash --create-home mluser

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code with proper ownership
COPY --chown=mluser:mluser . .

# Create necessary directories with proper permissions
RUN mkdir -p benchmark_results logs data models .cache && \
    chown -R mluser:mluser /app

# Switch to non-root user
USER mluser

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python scripts/verify_setup.py || exit 1

# Default command when container starts
CMD ["python", "ml_benchmark_suite.py", "--mode", "benchmark", "--sample-size", "3000"]
