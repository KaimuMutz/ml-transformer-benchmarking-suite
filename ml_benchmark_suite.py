# ============================================================================
# ML Transformer Benchmarking Suite - Complete Implementation (Colab Fixed)
# Author: Eric Mutembei Mwathi (mutzintl@gmail.com)
# Professional ML Engineering Portfolio Project
# ============================================================================

import os
import sys
import json
import time
import logging
import warnings
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import bz2

# Disable wandb before importing transformers
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

# Core ML libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)

# Deep learning libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader

# Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)

# Visualization and UI
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
console = Console()

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class ModelConfig:
    """Model configuration following professional ML standards."""
    name: str
    model_name: str
    num_labels: int = 2
    max_length: int = 256
    dropout: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults."""
    batch_size: int = 16
    num_epochs: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    evaluation_strategy: str = "steps"
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 50
    early_stopping_patience: int = 2
    output_dir: str = "models"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"

# ============================================================================
# Data Loading and Processing
# ============================================================================

class CustomDataLoader:
    """Professional data loading with error handling and caching."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "ml_benchmark"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def download_amazon_reviews(self) -> str:
        """Download Amazon reviews dataset using kagglehub."""
        try:
            import kagglehub
            console.print("Downloading Amazon Reviews dataset...")
            path = kagglehub.dataset_download("bittlingmayer/amazonreviews")
            console.print(f"Dataset downloaded to: {path}")
            return path
        except ImportError:
            raise ImportError("kagglehub required. Install with: pip install kagglehub")
        except Exception as e:
            console.print(f"Download failed: {e}")
            raise
    
    def load_fasttext_data(self, file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load FastText format data with proper parsing."""
        console.print(f"Loading data from: {os.path.basename(file_path)}")
        
        try:
            # Handle compressed files
            if file_path.endswith('.bz2'):
                with bz2.open(file_path, 'rt', encoding='utf-8') as f:
                    lines = f.readlines()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
        except Exception as e:
            raise FileNotFoundError(f"Could not read file {file_path}: {e}")
        
        # Parse lines
        data = []
        for line in lines[:sample_size] if sample_size else lines:
            line = line.strip()
            if line.startswith('__label__'):
                try:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        label = int(parts[0].replace('__label__', '')) - 1  # Convert to 0-based
                        text = parts[1]
                        if text.strip():  # Only add non-empty text
                            data.append({'text': text, 'label': label})
                except (ValueError, IndexError):
                    continue
        
        if not data:
            raise ValueError("No valid data found in file")
        
        df = pd.DataFrame(data)
        console.print(f"Loaded {len(df):,} samples")
        return df
    
    def prepare_data_splits(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data with stratification."""
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=df['label']
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, random_state=42, stratify=train_val_df['label']
        )
        
        console.print(f"Data splits - Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
        return train_df, val_df, test_df

# ============================================================================
# Dataset Implementation
# ============================================================================

class TextClassificationDataset(Dataset):
    """PyTorch dataset for text classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# ============================================================================
# Model Implementation
# ============================================================================

class TransformerClassifier:
    """Professional transformer classifier with proper error handling."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer with error handling."""
        try:
            console.print(f"Loading {self.config.model_name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_config = AutoConfig.from_pretrained(self.config.model_name)
            model_config.num_labels = self.config.num_labels
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                config=model_config,
            )
            
            self.model.to(self.device)
            console.print(f"Model loaded on {self.device}")
            
        except Exception as e:
            console.print(f"Failed to load model: {e}")
            raise
    
    def get_parameter_count(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

# ============================================================================
# Training Implementation
# ============================================================================

class ModelTrainer:
    """Professional model trainer with metrics tracking."""
    
    def __init__(self, model: TransformerClassifier, config: TrainingConfig):
        self.model = model
        self.config = config
        self.trainer = None
        
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics for training."""
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
        }
    
    def train(self, train_dataset: Dataset, eval_dataset: Dataset) -> Dict[str, Any]:
        """Train the model with comprehensive tracking."""
        
        # Create output directory
        output_dir = Path(self.config.output_dir) / self.model.config.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup training arguments - FIXED: Disable all external logging
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            eval_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            save_total_limit=self.config.save_total_limit,
            # CRITICAL FIX: Disable all external reporting
            report_to=[],  # Empty list instead of None
            logging_dir=None,
            disable_tqdm=False,
            dataloader_drop_last=False,
            fp16=torch.cuda.is_available(),
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)],
        )
        
        # Train model
        start_time = time.time()
        train_result = self.trainer.train()
        training_time = time.time() - start_time
        
        return {
            'metrics': train_result.metrics,
            'training_time_seconds': training_time,
        }

# ============================================================================
# Model Evaluation
# ============================================================================

class ModelEvaluator:
    """Comprehensive model evaluation with business metrics."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate_model(self, model: TransformerClassifier, test_dataloader: TorchDataLoader, 
                      class_names: List[str] = None) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        
        model.model.eval()
        all_predictions = []
        all_labels = []
        all_logits = []
        inference_times = []
        
        console.print("Evaluating model performance...")
        
        with torch.no_grad():
            for batch in test_dataloader:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Measure inference time
                start_time = time.time()
                outputs = model.model(**batch)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Collect results
                predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                logits = outputs.logits.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                all_logits.extend(logits)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Performance metrics
        total_time = sum(inference_times)
        throughput = len(all_predictions) / total_time
        
        # Confidence analysis
        confidences = []
        for logit in all_logits:
            probs = torch.softmax(torch.tensor(logit), dim=0)
            confidences.append(torch.max(probs).item())
        
        avg_confidence = np.mean(confidences)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'throughput_samples_per_second': throughput,
            'avg_confidence': avg_confidence,
            'confusion_matrix': cm.tolist(),
            'inference_time_total': total_time,
        }

# ============================================================================
# Benchmarking Suite
# ============================================================================

class BenchmarkSuite:
    """Professional benchmarking suite for transformer models."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def get_model_configs(self, sample_size: int) -> List[ModelConfig]:
        """Get model configurations based on sample size."""
        configs = [
            ModelConfig(
                name="distilbert",
                model_name="distilbert-base-uncased",
                max_length=256,
            ),
            ModelConfig(
                name="electra-small",
                model_name="google/electra-small-discriminator",
                max_length=256,
            ),
        ]
        
        # Add larger models for bigger datasets
        if sample_size >= 3000:
            configs.append(ModelConfig(
                name="bert-base",
                model_name="bert-base-uncased",
                max_length=256,
            ))
        
        return configs
    
    def benchmark_model(self, model_config: ModelConfig, train_df: pd.DataFrame, 
                       val_df: pd.DataFrame, test_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Benchmark a single model."""
        
        console.print(Panel(f"Benchmarking: {model_config.model_name}", style="bold blue"))
        
        try:
            # Initialize model
            model = TransformerClassifier(model_config)
            
            # Create datasets
            train_dataset = TextClassificationDataset(
                texts=train_df['text'].tolist(),
                labels=train_df['label'].tolist(),
                tokenizer=model.tokenizer,
                max_length=model_config.max_length,
            )
            
            val_dataset = TextClassificationDataset(
                texts=val_df['text'].tolist(),
                labels=val_df['label'].tolist(),
                tokenizer=model.tokenizer,
                max_length=model_config.max_length,
            )
            
            test_dataset = TextClassificationDataset(
                texts=test_df['text'].tolist(),
                labels=test_df['label'].tolist(),
                tokenizer=model.tokenizer,
                max_length=model_config.max_length,
            )
            
            # Training configuration
            training_config = TrainingConfig(
                batch_size=16 if len(train_df) > 2000 else 32,
                output_dir=str(self.output_dir / "models" / model_config.name),
            )
            
            # Train model
            trainer = ModelTrainer(model, training_config)
            train_results = trainer.train(train_dataset, val_dataset)
            
            # Evaluate model
            test_dataloader = TorchDataLoader(
                test_dataset, 
                batch_size=training_config.batch_size, 
                shuffle=False
            )
            
            evaluator = ModelEvaluator()
            eval_results = evaluator.evaluate_model(
                model, test_dataloader, ['Negative', 'Positive']
            )
            
            # Combine results
            result = {
                'model_name': model_config.model_name,
                'model_short_name': model_config.name,
                'parameter_count': model.get_parameter_count(),
                'training_time_seconds': train_results['training_time_seconds'],
                **eval_results
            }
            
            console.print(f"✅ {model_config.name}: Acc={result['accuracy']:.4f}, F1={result['f1_score']:.4f}")
            return result
            
        except Exception as e:
            console.print(f"❌ {model_config.name} failed: {str(e)}")
            return None
    
    def run_benchmark(self, sample_size: int = 5000) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        
        console.print(Panel("ML Transformer Benchmarking Suite", style="bold green"))
        console.print(f"Sample size: {sample_size:,}")
        console.print("=" * 60)
        
        # Load data
        data_loader = CustomDataLoader()
        dataset_path = data_loader.download_amazon_reviews()
        
        # Find training file
        files = os.listdir(dataset_path)
        train_file = None
        for file in files:
            if 'train' in file.lower():
                train_file = os.path.join(dataset_path, file)
                break
        
        if not train_file:
            train_file = os.path.join(dataset_path, files[0])
        
        # Load and split data
        df = data_loader.load_fasttext_data(train_file, sample_size)
        train_df, val_df, test_df = data_loader.prepare_data_splits(df)
        
        # Get model configurations
        model_configs = self.get_model_configs(sample_size)
        
        # Benchmark each model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            for config in model_configs:
                task = progress.add_task(f"Benchmarking {config.name}...", total=None)
                
                result = self.benchmark_model(config, train_df, val_df, test_df)
                if result:
                    self.results[config.name] = result
                
                progress.update(task, completed=True)
        
        # Generate reports
        self._save_results()
        self._create_visualizations()
        self._generate_report()
        
        return self.results
    
    def _save_results(self):
        """Save results to JSON."""
        results_file = self.output_dir / "benchmark_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in self.results.items():
            serializable_results[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    serializable_results[key][k] = v.tolist()
                else:
                    serializable_results[key][k] = v
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        console.print(f"Results saved to: {results_file}")
    
    def _create_visualizations(self):
        """Create benchmark comparison visualizations."""
        if not self.results:
            return
        
        # Prepare data for plotting
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        f1_scores = [self.results[m]['f1_score'] for m in models]
        throughputs = [self.results[m]['throughput_samples_per_second'] for m in models]
        training_times = [self.results[m]['training_time_seconds'] for m in models]
        param_counts = [self.results[m]['parameter_count'] / 1e6 for m in models]  # In millions
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Benchmarking Results', fontsize=16, fontweight='bold')
        
        # Accuracy vs F1
        ax1.scatter(accuracies, f1_scores, s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax1.annotate(model, (accuracies[i], f1_scores[i]), 
                        xytext=(5, 5), textcoords='offset points')
        ax1.set_xlabel('Accuracy')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('Accuracy vs F1 Score')
        ax1.grid(True, alpha=0.3)
        
        # Performance vs Speed
        ax2.scatter(throughputs, f1_scores, s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax2.annotate(model, (throughputs[i], f1_scores[i]),
                        xytext=(5, 5), textcoords='offset points')
        ax2.set_xlabel('Throughput (samples/sec)')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Performance vs Speed')
        ax2.grid(True, alpha=0.3)
        
        # Training time comparison
        ax3.bar(models, training_times, color='skyblue', alpha=0.8)
        ax3.set_ylabel('Training Time (seconds)')
        ax3.set_title('Training Time Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # Model size vs performance
        ax4.scatter(param_counts, f1_scores, s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax4.annotate(model, (param_counts[i], f1_scores[i]),
                        xytext=(5, 5), textcoords='offset points')
        ax4.set_xlabel('Parameters (Millions)')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('Model Size vs Performance')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / "benchmark_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"Visualizations saved to: {plot_file}")
    
    def _generate_report(self):
        """Generate professional benchmark report."""
        if not self.results:
            return
        
        # Create results table
        table = Table(title="Model Benchmarking Results")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Accuracy", justify="right")
        table.add_column("F1 Score", justify="right")
        table.add_column("Training Time", justify="right")
        table.add_column("Throughput", justify="right")
        table.add_column("Parameters", justify="right")
        
        for name, result in self.results.items():
            table.add_row(
                result['model_short_name'],
                f"{result['accuracy']:.4f}",
                f"{result['f1_score']:.4f}",
                f"{result['training_time_seconds']:.1f}s",
                f"{result['throughput_samples_per_second']:.1f}/s",
                f"{result['parameter_count']/1e6:.1f}M",
            )
        
        console.print(table)
        
        # Find best models
        best_accuracy = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        best_f1 = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        fastest_inference = max(self.results.items(), key=lambda x: x[1]['throughput_samples_per_second'])
        fastest_training = min(self.results.items(), key=lambda x: x[1]['training_time_seconds'])
        
        # Business recommendations
        console.print("\nBusiness Recommendations:")
        console.print(f"• Production Deployment: {best_f1[1]['model_short_name']} (F1: {best_f1[1]['f1_score']:.4f})")
        console.print(f"• Real-time Processing: {fastest_inference[1]['model_short_name']} ({fastest_inference[1]['throughput_samples_per_second']:.1f} samples/s)")
        console.print(f"• Resource-Constrained Training: {fastest_training[1]['model_short_name']} ({fastest_training[1]['training_time_seconds']:.1f}s)")

# ============================================================================
# Utility Functions
# ============================================================================

def setup_logging():
    """Setup professional logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('benchmark.log')
        ]
    )
    
    # Reduce transformer library noise
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

def validate_environment():
    """Validate that all dependencies are available."""
    
    console.print("Validating environment...")
    
    required_packages = [
        'torch', 'transformers', 'sklearn', 'pandas', 
        'numpy', 'matplotlib', 'seaborn', 'rich'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        console.print(f"Missing packages: {', '.join(missing)}")
        console.print("Install with: pip install -r requirements.txt")
        return False
    
    console.print("All dependencies available!")
    return True

def run_single_model_test(model_name: str = "distilbert-base-uncased", sample_size: int = 1000):
    """Test a single model quickly."""
    
    console.print(Panel(f"Quick Model Test: {model_name}", style="bold yellow"))
    
    try:
        # Load data
        data_loader = CustomDataLoader()
        dataset_path = data_loader.download_amazon_reviews()
        
        files = os.listdir(dataset_path)
        train_file = next((f for f in files if 'train' in f.lower()), files[0])
        train_file = os.path.join(dataset_path, train_file)
        
        df = data_loader.load_fasttext_data(train_file, sample_size)
        train_df, val_df, test_df = data_loader.prepare_data_splits(df)
        
        # Test model
        model_config = ModelConfig(name="test", model_name=model_name)
        model = TransformerClassifier(model_config)
        
        # Quick training test
        train_dataset = TextClassificationDataset(
            texts=train_df['text'].tolist()[:100],  # Very small for quick test
            labels=train_df['label'].tolist()[:100],
            tokenizer=model.tokenizer,
            max_length=128,
        )
        
        console.print("Model loads successfully!")
        console.print(f"Model has {model.get_parameter_count():,} parameters")
        console.print(f"Dataset created with {len(train_dataset)} samples")
        
        return True
        
    except Exception as e:
        console.print(f"Test failed: {e}")
        return False

def run_benchmark(sample_size: int = 5000, output_dir: str = "benchmark_results"):
    """Main benchmarking function."""
    
    # Setup
    setup_logging()
    
    # Run benchmark
    benchmark = BenchmarkSuite(output_dir)
    results = benchmark.run_benchmark(sample_size)
    
    if results:
        console.print("Benchmarking completed successfully!")
        console.print(f"Results available in: {output_dir}")
        return results
    else:
        console.print("Benchmarking failed")
        return None

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    # CRITICAL FIX: Disable wandb at the very start
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    
    parser = argparse.ArgumentParser(description="ML Transformer Benchmarking Suite")
    parser.add_argument("--mode", choices=["benchmark", "test", "validate"], 
                       default="benchmark", help="Execution mode")
    parser.add_argument("--sample-size", type=int, default=5000,
                       help="Number of samples for benchmarking")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased",
                       help="Model name for single model test")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == "validate":
        if validate_environment():
            console.print("Environment validation passed!")
            sys.exit(0)
        else:
            console.print("Environment validation failed!")
            sys.exit(1)
    
    elif args.mode == "test":
        if run_single_model_test(args.model_name, args.sample_size):
            console.print("Single model test passed!")
            sys.exit(0)
        else:
            console.print("Single model test failed!")
            sys.exit(1)
    
    elif args.mode == "benchmark":
        results = run_benchmark(args.sample_size, args.output_dir)
        if results:
            console.print("Benchmark completed successfully!")
            sys.exit(0)
        else:
            console.print("Benchmark failed!")
            sys.exit(1)
    
    else:
        parser.print_help()
