# Spectral Out-of-Distribution Detection for Computer Vision

A comprehensive implementation of spectral graph theory-based methods for out-of-distribution (OOD) detection in computer vision, covering multiple datasets, architectures, and advanced spectral techniques.

## 🎯 Overview

This repository implements three cohesive research questions on spectral features for OOD detection:

1. **RQ1**: How do eigenvalue concentration phenomena and spectral gaps distinguish between ID/OOD samples?
2. **RQ2**: How can multi-scale spectral signatures provide hierarchical OOD detection across different structural resolutions?
3. **RQ3**: Can we develop a unified spectral framework integrating eigenvalue perturbation theory with manifold learning?

## 🔬 Key Features

### Mathematical Foundations
- **Spectral Gap Analysis**: Leverages Cheeger's inequality and Davis-Kahan perturbation theory
- **Multi-scale Wavelets**: Graph wavelets for hierarchical anomaly detection
- **Heat Kernel Signatures**: Multi-scale diffusion analysis
- **Persistent Homology**: Topological features for manifold structure
- **Unified Framework**: Combines all approaches with theoretical guarantees

### Computational Efficiency
- **Intelligent Caching**: Automatic feature caching with size management and LRU eviction
- **Argparse Support**: Command-line interface for all scripts with configurable parameters
- **Batch Processing**: Memory-efficient processing for large-scale experiments
- **Progress Tracking**: Real-time progress updates and cache statistics

### Supported Datasets
- CIFAR-10/100
- SVHN  
- ImageNet (subset)
- Describable Textures Dataset (DTD)
- Gaussian Noise (synthetic OOD)

### Supported Architectures
- **CNNs**: ResNet18/50/101, VGG16, DenseNet121, EfficientNet-B0/B4
- **Vision Transformers**: ViT-Base/Large, Swin Transformer, ConvNeXt
- **Multi-modal**: CLIP (ResNet backbone)

## 📁 Repository Structure

```
Graph_OOD/
├── spectral_ood_vision.py          # Core spectral OOD detection implementation
├── advanced_spectral_vision.py     # Advanced hybrid methods with topology
├── run_comprehensive_experiment.py # Experiment orchestration and analysis
├── feature_cache.py                # Intelligent feature caching system
├── demo_notebook.py                # Interactive demonstration
├── README.md                       # This file
├── requirements.txt                # Dependencies
├── cache/                          # Cached features (auto-created)
└── results/                        # Experiment results and visualizations
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repo_url>
cd Graph_OOD
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm transformers scikit-learn scipy matplotlib seaborn pandas datasets
pip install networkx umap-learn
```

## 🎮 Quick Start

### Basic Demo
```python
from spectral_ood_vision import main
main()  # Runs quick demo with CIFAR-10 vs noise
```

### Comprehensive Experiments
```bash
# Quick demo (5 configurations)
python run_comprehensive_experiment.py --quick_demo

# Full evaluation (limited architectures)
python run_comprehensive_experiment.py --architectures resnet18 resnet50 --methods hybrid_advanced

# Full comprehensive evaluation (all configurations)
python run_comprehensive_experiment.py --max_samples 2000
```

### Analysis Only
```bash
# Analyze existing results
python run_comprehensive_experiment.py --analysis_only comprehensive_results.json
```

## 🔧 Usage Examples

### Command Line Usage

#### Basic Operations
```bash
# Quick demo
python spectral_ood_vision.py --quick_demo --data_dir ./data --cache_dir ./cache

# Full evaluation with custom settings
python spectral_ood_vision.py --full_eval --data_dir /path/to/data --max_samples 5000

# Show cache information
python spectral_ood_vision.py --cache_info

# Clear cache and run evaluation
python spectral_ood_vision.py --clear_cache --full_eval
```

#### Advanced Experiments
```bash
# Comprehensive experiments with caching
python run_comprehensive_experiment.py --data_dir ./data --cache_dir ./cache --max_experiments 50

# Quick demo with limited configurations
python run_comprehensive_experiment.py --quick_demo --architectures resnet18 vit_base

# Run with specific methods and architectures
python run_comprehensive_experiment.py --methods hybrid_advanced unified_spectral --architectures resnet50 efficientnet_b0
```

### 1. Basic Spectral Gap Detection with Caching
```python
from spectral_ood_vision import ImageSpectralOODDetector, VisionDatasetLoader, AdvancedFeatureExtractor

# Load data and extract features with caching
loader = VisionDatasetLoader('./data')
extractor = AdvancedFeatureExtractor(architecture='resnet50')

id_loader = loader.get_cifar10(train=False)
ood_loader = loader.get_noise_ood(size=1000)

# Extract features with automatic caching
id_features, _ = extractor.extract_features_with_cache(
    id_loader, 'cifar10', max_samples=2000, cache_dir='./cache'
)
ood_features, _ = extractor.extract_features_with_cache(
    ood_loader, 'noise', max_samples=1000, cache_dir='./cache'
)

# Train spectral detector
detector = ImageSpectralOODDetector(method='spectral_gap', pca_dim=256)
detector.fit(id_features[:1000])  # Use subset for training

# Predict OOD scores
test_features = np.vstack([id_features[1000:1500], ood_features])
scores = detector.predict_score(test_features)
```

### 2. Advanced Hybrid Detection
```python
from advanced_spectral_vision import HybridSpectralOODDetector

# Initialize hybrid detector with multiple methods
detector = HybridSpectralOODDetector(
    embedding_dim=64,
    pca_components=512,
    adaptive_k=True,
    ensemble_methods=['spectral_gap', 'heat_kernel', 'topology']
)

detector.fit(train_features)
scores = detector.predict_score(test_features)
```

### 3. Multi-Architecture Comparison with Caching
```python
from run_comprehensive_experiment import ExperimentOrchestrator
from feature_cache import CachedFeatureExtractor

# Initialize orchestrator with cache support
orchestrator = ExperimentOrchestrator(
    data_dir='./data', 
    cache_dir='./cache',
    results_dir='./results'
)

# Check cache status
cache_system = CachedFeatureExtractor(cache_dir='./cache')
cache_system.print_cache_info()

# Run experiments with specific configurations
results = orchestrator.run_comprehensive_experiments(
    max_experiments=20,
    architecture_filter=['resnet18', 'vit_base', 'efficientnet_b0'],
    method_filter=['hybrid_advanced', 'unified_spectral']
)

# Analyze results
df = orchestrator.analyze_results()
```

### 4. Feature Caching Management
```python
from feature_cache import FeatureCache, CachedFeatureExtractor

# Initialize cache system
cache = FeatureCache(cache_dir='./cache', max_cache_size_gb=5.0)

# Check what's cached
cache.print_cache_info()

# Clear specific cached features
cache.clear_cache(dataset_name='cifar10', architecture='resnet50')

# Clear all cache
cache.clear_cache()

# Use cached feature extractor
cached_extractor = CachedFeatureExtractor(cache_dir='./cache')
features, labels = cached_extractor.extract_features_with_cache(
    dataloader, 'cifar10', 'resnet50', max_samples=1000
)
```

## 🧮 Mathematical Details

### Spectral Gap Theory (RQ1)
The spectral gap Δλ = λ₂ - λ₁ of the normalized Laplacian captures fundamental connectivity properties:

**Cheeger's Inequality**: λ₂/2 ≤ h(G) ≤ √(2λ₂)

**Concentration Bound**: ||L_n - L||_op = O(√(log n / n)) with high probability

### Multi-scale Analysis (RQ2)
Graph wavelets provide multi-resolution analysis:

**Wavelet Operator**: ψₜ = g(tL) where g is the kernel function (e.g., Mexican hat)

Different frequency bands capture distinct anomaly types:
- Low frequencies: Global structural changes
- Mid frequencies: Community-level anomalies  
- High frequencies: Local perturbations

### Unified Framework (RQ3)
Combines spectral gap analysis with manifold learning:

**Davis-Kahan Bound**: ||sin Θ(V̂, V)||_F ≤ 2||Δ||_op / gap

**Manifold Convergence**: Graph Laplacian → Laplace-Beltrami operator as n → ∞

## 📊 Experimental Results

The framework has been evaluated across:
- **5 datasets** (CIFAR-10/100, SVHN, DTD, Noise)
- **10+ architectures** (ResNet, ViT, EfficientNet, etc.)
- **5 spectral methods** (Basic, Multi-scale, Unified, Hybrid variants)
- **100+ configurations** total

### Key Findings
1. **Hybrid methods** consistently outperform individual approaches
2. **Transformer architectures** show strong performance with spectral methods
3. **Dataset difficulty**: CIFAR-10 vs CIFAR-100 most challenging
4. **Method robustness**: Spectral approaches maintain performance across architectures

## 📈 Performance Metrics

All methods are evaluated using:
- **AUROC**: Area under ROC curve
- **AUPRC**: Average precision (area under PR curve)  
- **FPR95**: False positive rate at 95% true positive rate
- **FPR80**: False positive rate at 80% true positive rate

## 🔍 Visualization Tools

The framework provides comprehensive visualizations:
- Spectral embedding comparisons
- Heat kernel evolution
- Multi-scale wavelet coefficients
- Eigenvalue spectrum analysis
- Performance heatmaps and correlation matrices

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{spectral_ood_2024,
  title={Spectral Out-of-Distribution Detection for Computer Vision: A Comprehensive Framework},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on PyTorch and scikit-learn
- Uses timm for state-of-the-art model architectures
- Spectral graph theory foundations from various mathematical libraries

## 💡 Future Work

- Integration with foundation models (CLIP, DINO)
- Real-time OOD detection for streaming data
- Theoretical analysis of transformer feature spaces
- Extension to other modalities (NLP, audio)

---

For questions or support, please open an issue or contact the maintainers.