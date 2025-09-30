# Detailed Usage Guide for Spectral OOD Detection

This guide provides comprehensive usage examples for the spectral OOD detection framework with caching and argparse support.

## 🚀 Quick Start

### 1. Basic Demo
```bash
# Run interactive demo with synthetic data
python demo_notebook.py --data_dir ./data --cache_dir ./cache

# Skip real vision data demo
python demo_notebook.py --skip_real_data

# Show cache information
python demo_notebook.py --cache_info
```

### 2. Core Spectral OOD Detection
```bash
# Quick demo with limited samples
python spectral_ood_vision.py --quick_demo --max_samples 1000

# Full evaluation with custom directories
python spectral_ood_vision.py --full_eval --data_dir /path/to/datasets --cache_dir /path/to/cache

# Force recomputation of all features
python spectral_ood_vision.py --full_eval --force_recompute

# Clear cache before running
python spectral_ood_vision.py --clear_cache --full_eval
```

### 3. Advanced Hybrid Methods
```bash
# Run advanced spectral methods with caching
python advanced_spectral_vision.py --data_dir ./data --cache_dir ./cache --max_samples 3000

# Clear cache and show information
python advanced_spectral_vision.py --clear_cache --cache_info
```

### 4. Comprehensive Experiments
```bash
# Quick demo with limited configurations
python run_comprehensive_experiment.py --quick_demo

# Run specific architectures and methods
python run_comprehensive_experiment.py \
    --architectures resnet18 resnet50 vit_base \
    --methods hybrid_advanced unified_spectral \
    --max_experiments 20 \
    --max_samples 2000

# Full evaluation with all configurations
python run_comprehensive_experiment.py \
    --data_dir ./data \
    --cache_dir ./cache \
    --results_dir ./results \
    --max_samples 5000

# Analyze existing results only
python run_comprehensive_experiment.py --analysis_only results/comprehensive_results.json
```

## 🗂️ Cache Management

### Cache Operations
```bash
# Show detailed cache information
python spectral_ood_vision.py --cache_info

# Clear all cached features
python spectral_ood_vision.py --clear_cache

# Set custom cache size limit (10GB)
export CACHE_SIZE_GB=10
python spectral_ood_vision.py --full_eval
```

### Programmatic Cache Usage
```python
from feature_cache import FeatureCache, CachedFeatureExtractor

# Initialize cache with size limit
cache = FeatureCache(cache_dir='./cache', max_cache_size_gb=5.0)

# Check cache status
info = cache.get_cache_info()
print(f"Cache usage: {info['total_size_gb']:.2f} GB / {info['max_size_gb']:.1f} GB")

# Clear specific features
cache.clear_cache(dataset_name='cifar10')
cache.clear_cache(architecture='resnet50')

# Use with feature extractor
cached_extractor = CachedFeatureExtractor(cache_dir='./cache')
features, labels = cached_extractor.extract_features_with_cache(
    dataloader, 'cifar10', 'resnet50', max_samples=1000
)
```

## 🔬 Experimental Configurations

### Dataset Combinations
```bash
# CIFAR-10 as ID, various OOD datasets
python run_comprehensive_experiment.py \
    --data_dir ./data \
    --architectures resnet18 resnet50 \
    --methods unified_spectral hybrid_advanced

# Focus on specific dataset pairs
# Modify the script or use custom configuration files
```

### Architecture Filtering
```bash
# CNN architectures only
python run_comprehensive_experiment.py \
    --architectures resnet18 resnet50 vgg16 densenet121

# Transformer architectures only  
python run_comprehensive_experiment.py \
    --architectures vit_base vit_large swin_base

# Mixed architectures
python run_comprehensive_experiment.py \
    --architectures resnet50 efficientnet_b0 vit_base convnext_base
```

### Method Filtering
```bash
# Basic spectral methods
python run_comprehensive_experiment.py \
    --methods basic_spectral multiscale_spectral unified_spectral

# Advanced hybrid methods
python run_comprehensive_experiment.py \
    --methods hybrid_advanced hybrid_full

# Compare all methods
python run_comprehensive_experiment.py \
    --methods basic_spectral multiscale_spectral unified_spectral hybrid_advanced hybrid_full
```

## 📊 Analysis and Visualization

### Result Analysis
```bash
# Analyze results with visualizations
python run_comprehensive_experiment.py --analysis_only comprehensive_results.json

# Generate custom analysis
python -c "
from run_comprehensive_experiment import ExperimentOrchestrator
orchestrator = ExperimentOrchestrator()
df = orchestrator.analyze_results('comprehensive_results.json')
print(df.describe())
"
```

### Custom Analysis Scripts
```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load results
with open('results/comprehensive_results.json', 'r') as f:
    results = json.load(f)

# Filter successful results
successful = [r for r in results if r['success']]
df = pd.DataFrame(successful)

# Custom analysis
print("Top 10 configurations:")
top10 = df.nlargest(10, 'auc')[['id_dataset', 'ood_dataset', 'architecture', 'method_name', 'auc']]
print(top10)

# Create custom plots
fig, ax = plt.subplots()
df.boxplot(column='auc', by='method_name', ax=ax)
plt.title('AUC by Method')
plt.show()
```

## 🔧 Configuration Files

### Environment Variables
```bash
# Set default directories
export SPECTRAL_OOD_DATA_DIR=/path/to/datasets
export SPECTRAL_OOD_CACHE_DIR=/path/to/cache
export SPECTRAL_OOD_RESULTS_DIR=/path/to/results

# Cache configuration
export CACHE_SIZE_GB=10
export CACHE_CLEANUP_THRESHOLD=0.8

# Run experiments
python spectral_ood_vision.py --full_eval
```

### Batch Processing Scripts
```bash
#!/bin/bash
# batch_experiments.sh

# Clear cache
python spectral_ood_vision.py --clear_cache

# Run CNN architectures
python run_comprehensive_experiment.py \
    --architectures resnet18 resnet50 vgg16 densenet121 \
    --max_experiments 50 \
    --results_dir ./results/cnn_results

# Run transformer architectures  
python run_comprehensive_experiment.py \
    --architectures vit_base vit_large swin_base \
    --max_experiments 30 \
    --results_dir ./results/transformer_results

# Analyze all results
python run_comprehensive_experiment.py --analysis_only ./results/cnn_results/comprehensive_results.json
python run_comprehensive_experiment.py --analysis_only ./results/transformer_results/comprehensive_results.json
```

## 🐛 Debugging and Monitoring

### Verbose Logging
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logs
from spectral_ood_vision import main
main()
```

### Memory Monitoring
```bash
# Monitor memory usage
python -c "
import psutil
import time
from spectral_ood_vision import main

process = psutil.Process()
print(f'Initial memory: {process.memory_info().rss / 1024**2:.1f} MB')

main()

print(f'Final memory: {process.memory_info().rss / 1024**2:.1f} MB')
"
```

### Progress Tracking
```python
from tqdm import tqdm
import time

# Custom progress tracking
def run_with_progress():
    configurations = ['config1', 'config2', 'config3']  # Your configs
    
    with tqdm(total=len(configurations), desc="Running experiments") as pbar:
        for config in configurations:
            # Run experiment
            time.sleep(1)  # Placeholder
            pbar.update(1)
            pbar.set_postfix({'current': config})

run_with_progress()
```

## 📈 Performance Optimization

### Memory Optimization
```bash
# Reduce batch size for memory-constrained systems
python spectral_ood_vision.py --full_eval --max_samples 1000

# Use incremental PCA for large features
python -c "
from spectral_ood_vision import ImageSpectralOODDetector
detector = ImageSpectralOODDetector(pca_dim=256)  # Reduce PCA dimensions
"
```

### Speed Optimization
```bash
# Use GPU if available (automatic detection)
CUDA_VISIBLE_DEVICES=0 python spectral_ood_vision.py --full_eval

# Parallel processing (modify scripts as needed)
# Currently uses single GPU, can be extended for multi-GPU
```

### Cache Optimization
```bash
# Set optimal cache size (80% of available memory)
python -c "
import psutil
cache_gb = int(psutil.virtual_memory().available / (1024**3) * 0.8)
print(f'Recommended cache size: {cache_gb} GB')
"

# Use the recommended size
python spectral_ood_vision.py --cache_dir ./cache --full_eval
# (Note: cache size is set in the FeatureCache constructor)
```

## 🔄 Integration Examples

### Jupyter Notebook Integration
```python
# In Jupyter notebook
%load_ext autoreload
%autoreload 2

from spectral_ood_vision import *
from feature_cache import *

# Interactive cache management
cache = FeatureCache('./cache')
cache.print_cache_info()

# Quick experiment
loader = VisionDatasetLoader('./data')
extractor = AdvancedFeatureExtractor('resnet18')
detector = ImageSpectralOODDetector('unified')

# Run experiment interactively
id_loader = loader.get_cifar10(False)
features, labels = extractor.extract_features_with_cache(id_loader, 'cifar10', 1000)
detector.fit(features[:500])
scores = detector.predict_score(features[500:])
```

### Custom Dataset Integration
```python
from torch.utils.data import Dataset, DataLoader
from spectral_ood_vision import AdvancedFeatureExtractor

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, 0  # Dummy label

# Use with extractor
custom_data = CustomDataset(your_data, transform=your_transform)
custom_loader = DataLoader(custom_data, batch_size=32)

extractor = AdvancedFeatureExtractor('resnet50')
features, _ = extractor.extract_features_with_cache(
    custom_loader, 'custom_dataset', 1000, cache_dir='./cache'
)
```

This comprehensive usage guide should help you effectively use all the features of the spectral OOD detection framework with caching and command-line support!