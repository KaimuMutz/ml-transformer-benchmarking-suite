# Google Colab Notebook Templates

## Quick Start Notebook

Create a new notebook in Google Colab with these cells:

### Cell 1: Environment Setup
```python
# Clone repository and install dependencies
!git clone https://github.com/KaimuMutz/ml-transformer-benchmarking-suite.git
%cd ml-transformer-benchmarking-suite
!pip install -r requirements.txt --quiet

# Verify setup
!python ml_benchmark_suite.py --mode validate
```

### Cell 2: Quick Test
```python
# Test single model functionality
!python ml_benchmark_suite.py --mode test --sample-size 1000

# Check GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
```

### Cell 3: Mini Benchmark
```python
# Run lightweight benchmark
!python ml_benchmark_suite.py --mode benchmark --sample-size 2000

# Check results
!ls -la benchmark_results/
```

### Cell 4: Results Visualization
```python
# Display results
from IPython.display import Image, display
import json

# Show comparison chart
try:
    display(Image('benchmark_results/benchmark_comparison.png'))
except:
    print("Run benchmark first to generate visualizations")

# Show summary
try:
    with open('benchmark_results/benchmark_results.json', 'r') as f:
        results = json.load(f)
    
    for model, metrics in results.items():
        print(f"{model}: {metrics['accuracy']:.3f} accuracy, {metrics['f1_score']:.3f} F1")
except:
    print("No results file found")
```

## Full Analysis Notebook

For comprehensive analysis, create additional cells:

### Advanced Analysis Cell
```python
# Import for custom analysis
from ml_benchmark_suite import *
import matplotlib.pyplot as plt
import pandas as pd

# Load and analyze results
with open('benchmark_results/benchmark_results.json', 'r') as f:
    results = json.load(f)

# Create custom visualizations
models = list(results.keys())
accuracies = [results[m]['accuracy'] for m in models]
throughputs = [results[m]['throughput_samples_per_second'] for m in models]

plt.figure(figsize=(10, 6))
plt.scatter(throughputs, accuracies, s=100)
for i, model in enumerate(models):
    plt.annotate(model, (throughputs[i], accuracies[i]))
plt.xlabel('Throughput (samples/sec)')
plt.ylabel('Accuracy')
plt.title('Speed vs Accuracy Trade-off Analysis')
plt.grid(True, alpha=0.3)
plt.show()
```
