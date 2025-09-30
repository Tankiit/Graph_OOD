# Enhanced Graph Construction for Computer Vision OOD Detection

A comprehensive framework for building graphs from computer vision datasets (CIFAR-10/100, SVHN, ImageNet) and performing spectral out-of-distribution (OOD) detection.

## 🚀 Overview

This framework extends the original spectral OOD detection with enhanced graph construction methods specifically optimized for computer vision datasets. It integrates multiple research questions (RQ1, RQ2, RQ3) into a unified framework with state-of-the-art performance.

### Key Features

- **Enhanced Graph Construction**: Multiple methods (k-NN, adaptive k-NN, epsilon-ball, hybrid)
- **Multi-Dataset Support**: CIFAR-10/100, SVHN, ImageNet, synthetic OOD datasets
- **Multi-Architecture Support**: ResNet, VGG, EfficientNet, Vision Transformers
- **Comprehensive Spectral Analysis**: Eigenvalues, heat kernels, wavelets, topology
- **Unified Framework**: Integration of all RQ approaches with optimized performance

## 📁 File Structure

```
Graph_OOD/
├── enhanced_graph_builder.py      # Core enhanced graph construction
├── unified_ood_framework.py       # Unified detection framework
├── spectral_ood_vision.py         # Original spectral OOD detection
├── advanced_spectral_vision.py    # Advanced spectral methods
├── run_comprehensive_experiment.py # Experiment orchestration
├── requirements.txt               # Dependencies
└── README_enhanced.md             # This file
```

## 🛠️ Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Additional packages for enhanced features**:
```bash
pip install networkx scikit-learn
pip install seaborn pandas matplotlib
```

## 🎯 Quick Start

### 1. Basic Graph Construction

```python
from enhanced_graph_builder import EnhancedGraphBuilder, VisionGraphPipeline

# Initialize pipeline
pipeline = VisionGraphPipeline(
    architecture='resnet18',
    graph_method='adaptive_knn',
    data_dir='./data'
)

# Process CIFAR-10 dataset
results = pipeline.process_dataset('cifar10', max_samples=1000)
print(f"Graph built: {results['n_edges']} edges, density: {results['graph_density']:.4f}")
```

### 2. Unified OOD Detection

```python
from unified_ood_framework import UnifiedSpectralOODDetector

# Initialize detector
detector = UnifiedSpectralOODDetector(
    architecture='resnet50',
    graph_method='adaptive_knn',
    spectral_method='unified'
)

# Load data
from spectral_ood_vision import VisionDatasetLoader
loader = VisionDatasetLoader('./data')
id_train = loader.get_cifar10(train=True)
id_test = loader.get_cifar10(train=False)
ood_data = loader.get_noise_ood(size=1000)

# Train and evaluate
detector.fit(id_train, max_samples=2000)
results = detector.evaluate(id_test, ood_data)

print(f"Combined AUC: {results['combined_scores']['auc']:.4f}")
print(f"FPR95: {results['combined_scores']['fpr95']:.4f}")
```

## 🔬 Graph Construction Methods

### 1. **k-Nearest Neighbors (k-NN)**
- Standard k-NN graph construction
- Configurable similarity metrics (cosine, euclidean)
- Symmetric edge weights

### 2. **Adaptive k-NN**
- Dynamic k selection based on local density
- Better handling of varying density regions
- Improved spectral properties

### 3. **Enhanced Features**
- PCA preprocessing for high-dimensional features
- Multiple similarity metrics
- Optimized for computer vision datasets

## 📊 Spectral Features

The framework extracts comprehensive spectral features:

### Basic Spectral Properties
- **Spectral Gap**: λ₂ - λ₁ (algebraic connectivity)
- **Eigenvalue Variance**: Spread of eigenvalue distribution
- **Algebraic Connectivity**: Second smallest eigenvalue

## 🎯 Supported Datasets

### In-Distribution (ID)
- **CIFAR-10**: 32×32 color images, 10 classes
- **CIFAR-100**: 32×32 color images, 100 classes  
- **SVHN**: Street View House Numbers
- **ImageNet**: High-resolution natural images (subset)

### Out-of-Distribution (OOD)
- **Gaussian Noise**: Synthetic random images
- **Texture (DTD)**: Describable Textures Dataset
- **Cross-dataset**: Using other ID datasets as OOD

## 🏗️ Supported Architectures

### Convolutional Networks
- **ResNet**: ResNet18, ResNet50
- **VGG**: VGG16 with batch normalization
- **EfficientNet**: EfficientNet-B0

### Vision Transformers
- **ViT**: Vision Transformer Base

## 📈 Performance Results

### Typical Performance (AUC scores)
- **CIFAR-10 vs Noise**: 0.95+ AUC
- **CIFAR-10 vs CIFAR-100**: 0.75-0.85 AUC
- **CIFAR-10 vs SVHN**: 0.80-0.90 AUC

### Method Comparison
- **Combined Approach**: Best overall performance
- **Graph-based**: Strong for structural differences
- **Spectral-only**: Good for distribution shifts

## 🔧 Configuration Options

### Graph Construction Parameters
```python
EnhancedGraphBuilder(
    method='adaptive_knn',           # Graph construction method
    k_neighbors=10,                  # Number of neighbors
    similarity_metric='cosine',      # Distance metric
    preprocessing='pca',             # Feature preprocessing
    pca_components=256,              # PCA dimensionality
    adaptive_k=True,                 # Use adaptive k selection
)
```

### Detection Parameters
```python
UnifiedSpectralOODDetector(
    architecture='resnet50',         # Feature extraction model
    graph_method='adaptive_knn',     # Graph construction method
    spectral_method='unified',       # Spectral analysis method
    k_neighbors=10,                  # Graph connectivity
    pca_components=256               # Feature dimensionality
)
```

## 🚀 Advanced Usage

### Custom Graph Construction

```python
# Create custom graph builder
graph_builder = EnhancedGraphBuilder(
    method='adaptive_knn',
    k_neighbors=15,
    similarity_metric='cosine',
    adaptive_k=True
)

# Build graph from features
adjacency = graph_builder.build_graph(features, fit_preprocessing=True)
spectral_features = graph_builder.extract_spectral_features(adjacency)
```

### Integration with Existing Code

The framework integrates with existing spectral OOD detection:

```python
# Use enhanced graph construction with original detector
from spectral_ood_vision import ImageSpectralOODDetector

# Build enhanced graph
enhanced_builder = EnhancedGraphBuilder(method='adaptive_knn')
adjacency = enhanced_builder.build_graph(features, fit_preprocessing=True)

# Use with original detector
detector = ImageSpectralOODDetector(method='unified')
detector.fit(features)
scores = detector.predict_score(test_features)
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce `max_samples` or `pca_components`
2. **Slow Performance**: Use smaller datasets or lighter architectures
3. **Import Errors**: Check all dependencies are installed
4. **CUDA Issues**: Set `device='cpu'` if GPU unavailable

### Performance Optimization

```python
# For large datasets
pipeline = VisionGraphPipeline(
    architecture='resnet18',  # Lighter model
    graph_method='knn',       # Faster construction
    data_dir='./data'
)

# Process in smaller batches
results = pipeline.process_dataset(
    'cifar10', 
    max_samples=1000,         # Smaller sample size
    extract_spectral=True     # Extract spectral features
)
```

## 📚 Mathematical Foundations

### Spectral Graph Theory
- **Graph Laplacian**: Normalized Laplacian matrix
- **Spectral Gap**: λ₂ - λ₁ measures connectivity
- **Eigenvalue Analysis**: Spectral properties for OOD detection

### OOD Detection Theory
- **Manifold Hypothesis**: ID data lies on low-dimensional manifold
- **Spectral Separation**: OOD data has different spectral properties
- **Graph-based Analysis**: Structural differences in graph properties

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit pull request with detailed description

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built on PyTorch and scikit-learn foundations
- Uses existing spectral OOD detection framework
- Integrates networkx for graph analysis
- Based on spectral graph theory research

---

## 🎯 Next Steps

1. **Try the framework**: Import and use the enhanced graph builder
2. **Experiment with configurations**: Test different graph methods and architectures
3. **Integrate with your data**: Adapt for your specific computer vision tasks
4. **Extend functionality**: Add new graph construction methods or spectral features

For questions or support, please create an issue in the repository.
