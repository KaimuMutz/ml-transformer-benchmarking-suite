# Deployment Guide

## Environment Setup

### Local Development
```bash
git clone https://github.com/KaimuMutz/ml-transformer-benchmarking-suite.git
cd ml-transformer-benchmarking-suite
pip install -r requirements.txt
```

### Google Colab
```python
!git clone https://github.com/KaimuMutz/ml-transformer-benchmarking-suite.git
%cd ml-transformer-benchmarking-suite
!pip install -r requirements.txt --quiet
```

### Docker Container
```bash
docker build -t ml-benchmark-suite .
docker run -v $(pwd)/results:/app/benchmark_results ml-benchmark-suite
```

## Usage Examples

### Quick Validation
```bash
python ml_benchmark_suite.py --mode validate
```

### Single Model Test
```bash
python ml_benchmark_suite.py --mode test --model-name distilbert-base-uncased --sample-size 1000
```

### Production Benchmark
```bash
python ml_benchmark_suite.py --mode benchmark --sample-size 5000 --output-dir production_results
```

## Hardware Requirements

### Minimum Configuration
- CPU: 2+ cores
- RAM: 4GB
- Storage: 2GB free space
- Network: Internet connection for model downloads

### Recommended Configuration
- CPU: 4+ cores
- RAM: 8GB
- GPU: 6GB+ VRAM (optional but recommended)
- Storage: 10GB free space

### Tested Environments
- Google Colab (Tesla T4)
- Local development (various configurations)
- AWS EC2 instances
- Docker containers

## Expected Execution Times

Based on Tesla T4 GPU performance:

| Sample Size | Expected Duration | Memory Usage |
|-------------|-------------------|--------------|
| 1,000 | 3-5 minutes | 2-3GB |
| 3,000 | 8-12 minutes | 4-6GB |
| 5,000 | 12-16 minutes | 6-8GB |
| 10,000 | 25-35 minutes | 10-12GB |

## Troubleshooting

### Common Issues

**CUDA Out of Memory**: Reduce batch size or sample size
**Model Download Fails**: Check internet connection and retry
**Permission Errors**: Ensure write access to output directories

### Performance Optimization

**GPU Utilization**: Enable mixed precision training for faster processing
**Memory Management**: Use gradient checkpointing for large models
**Data Loading**: Implement parallel data loading for faster preprocessing

## Integration

### API Integration
The benchmark results can be integrated into model selection pipelines:

```python
import json
with open('benchmark_results/benchmark_results.json', 'r') as f:
    results = json.load(f)

# Select model based on criteria
best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
best_speed = max(results.items(), key=lambda x: x[1]['throughput_samples_per_second'])
```

### Monitoring Integration
Results include metrics suitable for production monitoring systems and can be exported to various monitoring platforms.
