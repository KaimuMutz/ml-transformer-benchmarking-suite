#!/usr/bin/env python3
"""Generate benchmark results visualization from actual data"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_benchmark_visualization():
    """Create professional benchmark visualization from actual results."""
    
    # Load actual results
    with open('benchmark_results/benchmark_results.json', 'r') as f:
        results = json.load(f)
    
    # Extract data
    models = list(results.keys())
    model_names = [results[m]['model_short_name'] for m in models]
    accuracies = [results[m]['accuracy'] for m in models]
    f1_scores = [results[m]['f1_score'] for m in models]
    throughputs = [results[m]['throughput_samples_per_second'] for m in models]
    training_times = [results[m]['training_time_seconds'] for m in models]
    param_counts = [results[m]['parameter_count'] / 1e6 for m in models]
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Benchmarking Results', fontsize=16, fontweight='bold')
    
    # Accuracy vs F1
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    ax1.scatter(accuracies, f1_scores, s=150, alpha=0.8, c=colors[:len(models)])
    for i, model in enumerate(model_names):
        ax1.annotate(model, (accuracies[i], f1_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax1.set_xlabel('Accuracy')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Accuracy vs F1 Score')
    ax1.grid(True, alpha=0.3)
    
    # Performance vs Speed
    ax2.scatter(throughputs, f1_scores, s=150, alpha=0.8, c=colors[:len(models)])
    for i, model in enumerate(model_names):
        ax2.annotate(model, (throughputs[i], f1_scores[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax2.set_xlabel('Throughput (samples/sec)')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Performance vs Speed')
    ax2.grid(True, alpha=0.3)
    
    # Training time comparison
    bars = ax3.bar(model_names, training_times, color=colors[:len(models)], alpha=0.8)
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Training Time Comparison')
    
    # Add value labels on bars
    for bar, time in zip(bars, training_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{time:.1f}s', ha='center', va='bottom', fontsize=10)
    
    # Model size vs performance
    ax4.scatter(param_counts, f1_scores, s=150, alpha=0.8, c=colors[:len(models)])
    for i, model in enumerate(model_names):
        ax4.annotate(model, (param_counts[i], f1_scores[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax4.set_xlabel('Parameters (Millions)')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('Model Size vs Performance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    output_path = Path('reports/figures/benchmark_comparison.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")
    return str(output_path)

if __name__ == "__main__":
    create_benchmark_visualization()
