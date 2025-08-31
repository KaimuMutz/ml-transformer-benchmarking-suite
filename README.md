# ML Transformer Benchmarking Suite

A comprehensive benchmarking framework for evaluating transformer models on text classification tasks, designed to address real-world business requirements for automated text analysis systems.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/framework-pytorch-orange.svg)
![Transformers](https://img.shields.io/badge/huggingface-transformers-yellow.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

**Author**: Eric Mutembei Mwathi  
**Contact**: mutzintl@gmail.com  
**Repository**: [ml-transformer-benchmarking-suite](https://github.com/KaimuMutz/ml-transformer-benchmarking-suite)

## Problem Statement

Organizations processing large volumes of text data need reliable metrics to select appropriate transformer models for their specific use cases. This benchmarking suite provides standardized evaluation across multiple dimensions: accuracy, processing speed, resource requirements, and deployment constraints.

## Solution Architecture

### Single-File Implementation
The entire benchmarking pipeline is implemented in one comprehensive Python file (`ml_benchmark_suite.py`) for maximum portability and ease of deployment across different environments.

### Evaluation Framework
- **Performance Metrics**: Accuracy, F1-score, precision, recall
- **Efficiency Metrics**: Training time, inference throughput, parameter count
- **Business Metrics**: Resource utilization, confidence scoring, confusion analysis

## Benchmark Results

The following results were obtained using a 5,000-sample Amazon Reviews dataset on Google Colab with Tesla T4 GPU. Total benchmark execution time: approximately 14 minutes.

### Performance Summary

| Model | Accuracy | F1 Score | Training Time | Throughput | Parameters | Confidence |
|-------|----------|----------|---------------|------------|------------|------------|
| **BERT-Base** | 92.6% | 92.6% | 178s | 1,049/s | 109.5M | 98.3% |
| **DistilBERT** | 92.4% | 92.4% | 143s | 2,559/s | 67.0M | 97.0% |
| **ELECTRA-Small** | 88.7% | 88.7% | 45s | 1,053/s | 13.5M | 84.6% |

### Key Findings

**BERT-Base** achieves the highest accuracy (92.6%) but requires the most computational resources (109.5M parameters) and longest training time (178 seconds).

**DistilBERT** provides the optimal balance with 92.4% accuracy while delivering 2.4x faster inference speed than BERT-Base, making it suitable for high-throughput production environments.

**ELECTRA-Small** offers the fastest training (45 seconds) and smallest footprint (13.5M parameters) with acceptable accuracy (88.7%) for resource-constrained deployments.

### Business Applications

#### High-Accuracy Requirements
- **Model**: BERT-Base
- **Use Cases**: Legal document analysis, medical text classification, financial compliance
- **Trade-offs**: Higher computational cost for maximum accuracy

#### Balanced Production Systems
- **Model**: DistilBERT
- **Use Cases**: Customer support automation, content moderation, general sentiment analysis
- **Advantages**: 92.4% accuracy with 2,559 samples/second throughput

#### Resource-Constrained Environments
- **Model**: ELECTRA-Small
- **Use Cases**: Edge deployment, mobile applications, rapid prototyping
- **Benefits**: 5x fewer parameters with 88.7% accuracy

## Quick Start

### Google Colab (No Setup Required)
```python
# Clone and install
!git clone https://github.com/KaimuMutz/ml-transformer-benchmarking-suite.git
%cd ml-transformer-benchmarking-suite
!pip install -r requirements.txt --quiet

# Validate environment
!python ml_benchmark_suite.py --mode validate

# Run benchmark (adjust sample size as needed)
!python ml_benchmark_suite.py --mode benchmark --sample-size 2000
```

### Local Development
```bash
git clone https://github.com/KaimuMutz/ml-transformer-benchmarking-suite.git
cd ml-transformer-benchmarking-suite
pip install -r requirements.txt

# Environment validation
python ml_benchmark_suite.py --mode validate

# Single model test
python ml_benchmark_suite.py --mode test --model-name distilbert-base-uncased --sample-size 1000

# Full benchmark
python ml_benchmark_suite.py --mode benchmark --sample-size 5000
```

### Docker Deployment
```bash
docker build -t ml-benchmark-suite .
docker run -v $(pwd)/results:/app/benchmark_results ml-benchmark-suite
```

## Technical Implementation

### Core Components
- **Data Pipeline**: Automated Amazon Reviews dataset processing with stratified splitting
- **Model Management**: Dynamic model loading with error handling and device optimization
- **Training Framework**: HuggingFace Trainer integration with early stopping and metric tracking
- **Evaluation Engine**: Comprehensive metrics calculation including business-relevant measurements

### Architecture Decisions
- **Single-file design** for deployment simplicity and portability
- **Memory-efficient processing** with configurable batch sizes and sample limits
- **Cross-platform compatibility** supporting local, cloud, and container environments
- **Comprehensive error handling** with detailed logging and graceful failure recovery

## Deployment Options

### Production Environments
The benchmarking suite supports multiple deployment patterns:

- **Research & Development**: Direct execution in Jupyter/Colab environments
- **CI/CD Integration**: Automated model evaluation in deployment pipelines
- **Cloud Platforms**: Compatible with AWS SageMaker, Google AI Platform, Azure ML
- **Edge Computing**: Optimized configurations for resource-constrained environments

### Scalability Features
- Configurable sample sizes (1K to 50K+ samples tested)
- GPU acceleration with automatic fallback to CPU
- Parallel processing support for multiple model evaluation
- Memory-optimized data loading for large datasets

## Results Analysis

### Accuracy vs Speed Trade-offs
The benchmark reveals clear patterns for different deployment scenarios:

1. **Maximum Accuracy**: BERT-Base provides 92.6% accuracy but processes 1,049 samples/second
2. **Optimal Balance**: DistilBERT achieves 92.4% accuracy with 2.4x faster processing
3. **Speed Priority**: ELECTRA-Small delivers 1,053 samples/second with 88.7% accuracy

### Resource Utilization
Training efficiency varies significantly:
- ELECTRA-Small: 45 seconds training time (13.5M parameters)
- DistilBERT: 143 seconds training time (67.0M parameters)  
- BERT-Base: 178 seconds training time (109.5M parameters)

### Confidence Analysis
Model confidence correlates with accuracy:
- BERT-Base: 98.3% average confidence
- DistilBERT: 97.0% average confidence
- ELECTRA-Small: 84.6% average confidence

## Configuration Options

### Sample Size Recommendations
- **Development/Testing**: 1,000-2,000 samples (5-10 minutes)
- **Model Selection**: 3,000-5,000 samples (10-15 minutes)
- **Production Validation**: 10,000+ samples (30+ minutes)

### Hardware Requirements
- **Minimum**: 4GB RAM, CPU-only (slower processing)
- **Recommended**: 8GB RAM, GPU with 6GB+ VRAM
- **Optimal**: 16GB RAM, Tesla T4 or equivalent GPU

## Usage Commands

```bash
# Environment validation
python ml_benchmark_suite.py --mode validate

# Quick functionality test
python ml_benchmark_suite.py --mode test --model-name distilbert-base-uncased --sample-size 1000

# Standard benchmark
python ml_benchmark_suite.py --mode benchmark --sample-size 5000

# Custom configuration
python ml_benchmark_suite.py --mode benchmark --sample-size 3000 --output-dir custom_results
```

## Project Structure

```
ml-transformer-benchmarking-suite/
├── ml_benchmark_suite.py          # Complete implementation
├── requirements.txt               # Python dependencies
├── README.md                      # Documentation
├── Dockerfile                     # Container configuration
├── .github/workflows/ci.yml       # CI/CD pipeline
├── scripts/
│   ├── verify_setup.py           # Environment validation
│   └── generate_results_visualization.py  # Results plotting
├── benchmark_results/
│   ├── benchmark_results.json    # Actual benchmark data
│   └── benchmark_comparison.png  # Performance visualizations
├── reports/figures/               # Generated charts and graphs
├── data/                         # Dataset storage (gitignored)
├── models/                       # Model checkpoints (gitignored)
└── logs/                         # Application logs
```

## Applications

### Content Moderation Systems
Automated classification of user-generated content with configurable accuracy thresholds and processing speed requirements.

### Customer Support Automation
Intelligent ticket routing and priority classification with confidence scoring for escalation decisions.

### Market Intelligence
Real-time sentiment analysis of product reviews, social media, and news content for business intelligence systems.

### Document Processing
Classification and routing of business documents, contracts, and regulatory filings in enterprise environments.

## Technical Specifications

### Supported Models
- BERT variants (BERT-Base, DistilBERT, RoBERTa)
- ELECTRA models (Small, Base, Large)
- Custom transformer architectures via HuggingFace model hub

### Data Formats
- FastText format (.txt, .bz2)
- CSV with text/label columns
- JSON with structured text data
- Custom data loaders for proprietary formats

### Output Formats
- JSON results for programmatic access
- PNG visualizations for reporting
- CSV exports for spreadsheet analysis
- Markdown reports for documentation

## Dependencies

Core requirements:
- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- scikit-learn 1.3+
- pandas 2.0+

Visualization and reporting:
- matplotlib 3.7+
- seaborn 0.12+
- rich 13.0+

Data access:
- kagglehub 0.3+

## Docker Deployment

### Prerequisites
- Docker installed on your system
- Docker Compose (included with Docker Desktop)

### Quick Docker Setup

```bash
# Build the Docker image
docker build -t ml-benchmark .

# Run quick validation
docker run --rm ml-benchmark python ml_benchmark_suite.py --mode validate

# Run benchmark with persistent results
mkdir -p docker-results docker-logs
docker run \
  --name ml-benchmark-run \
  -v $(pwd)/docker-results:/app/benchmark_results \
  -v $(pwd)/docker-logs:/app/logs \
  ml-benchmark
```

### Using Docker Compose (Recommended)

```bash
# Production deployment
docker-compose up -d

# Development environment
docker-compose -f docker-compose.dev.yml up -d

# Access development container
docker-compose -f docker-compose.dev.yml exec ml-benchmark-dev bash

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Using Makefile Commands

```bash
# Build image
make docker-build

# Run tests
make docker-test

# Start development environment
make docker-dev

# Run production benchmark
make docker-prod

# View all available commands
make help
```

### Docker Benefits

- **Consistent Environment**: Same behavior across all systems
- **Easy Deployment**: Single command deployment anywhere
- **Isolation**: No conflicts with host system dependencies
- **Scalability**: Easy to scale across multiple containers
- **Reproducibility**: Identical results regardless of host system

### Container Resource Usage

The Docker setup includes resource limits:
- Memory: 8GB limit, 4GB reserved
- CPU: 4 cores limit, 2 cores reserved
- Adjust these in docker-compose.yml based on your system

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Contact

**Eric Mutembei Mwathi**  
**Email**: mutzintl@gmail.com  
**Location**: Nairobi, Kenya

For questions about implementation, deployment, or collaboration opportunities, please open an issue or contact directly.

---

*This benchmarking suite provides standardized evaluation metrics for transformer model selection in production text classification systems.*
