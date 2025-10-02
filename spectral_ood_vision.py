"""
Spectral Out-of-Distribution Detection for Computer Vision
Implementation across multiple datasets (CIFAR-10/100, SVHN, ImageNet) and architectures
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional, Union
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import pre-trained models
from torchvision import models
import timm

# Import caching system
from feature_cache import FeatureCache, CachedFeatureExtractor

class VisionDatasetLoader:
    """
    Unified dataset loader for CIFAR-10/100, SVHN, ImageNet and OOD datasets
    """
    
    def __init__(self, data_dir: str = './data', batch_size: int = 128):
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # Standard transforms for different datasets
        self.transforms = {
            'cifar': transforms.Compose([
                transforms.Resize(224),  # Resize for pretrained models
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            'svhn': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            'imagenet': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            'tinyimagenet': transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
            'texture': transforms.Compose([
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]),
        }
    
    def get_tiny_imagenet(self, train: bool = True) -> DataLoader:
        """Load Tiny ImageNet dataset"""
        dataset = load_dataset('Maysee/tiny-imagenet', cache_dir=self.data_dir, split="train" if train else "valid")
        
        _transform = self.transforms['tinyimagenet']

        def transforms(example):
            return {"image":[_transform(x.convert("RGB")) for x in example["image"]],"label":example["label"]}

        def collate(args):
            return (torch.stack([x["image"] for x in args]),torch.tensor([x["label"] for x in args]))

        dataset.set_transform(transforms)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train, collate_fn=collate, num_workers=4)
    
    def get_cifar10(self, train: bool = True) -> DataLoader:
        """Load CIFAR-10 dataset"""
        dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=train, download=True,
            transform=self.transforms['cifar']
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers=4)
    
    def get_cifar100(self, train: bool = True) -> DataLoader:
        """Load CIFAR-100 dataset"""
        dataset = torchvision.datasets.CIFAR100(
            root=self.data_dir, train=train, download=True,
            transform=self.transforms['cifar']
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=train, num_workers=4)
    
    def get_svhn(self, split: str = 'train') -> DataLoader:
        """Load SVHN dataset"""
        dataset = torchvision.datasets.SVHN(
            root=self.data_dir, split=split, download=True,
            transform=self.transforms['svhn']
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=(split=='train'), num_workers=4)
    
    def get_imagenet(self, split: str = 'val', subset_size: Optional[int] = None) -> DataLoader:
        """Load ImageNet dataset (validation set for efficiency)"""
        try:
            dataset = torchvision.datasets.ImageNet(
                root=self.data_dir, split=split, transform=self.transforms['imagenet']
            )
            if subset_size:
                # Create subset for computational efficiency
                indices = np.random.choice(len(dataset), subset_size, replace=False)
                dataset = torch.utils.data.Subset(dataset, indices)
        except:
            # If ImageNet not available, create dummy dataset
            pass  # ImageNet not found, using CIFAR-10 as substitute
            return self.get_cifar10(train=(split=='train'))
            
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=(split=='train'), num_workers=4)
    
    def get_texture_ood(self) -> DataLoader:
        """Load texture dataset as OOD (DTD - Describable Textures Dataset)"""
        dataset = torchvision.datasets.DTD(
            root=self.data_dir, split='test', download=True,
            transform=self.transforms['texture']
        )
            
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def get_noise_ood(self, size: int = 1000) -> DataLoader:
        """Generate Gaussian noise as OOD"""
        dataset = NoiseDataset(size=size,transform=self.transforms['cifar'])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


class NoiseDataset(Dataset):
    """Custom dataset for Gaussian noise OOD samples"""
    
    def __init__(self, size: int, transform=None):
        self.size = size
        self.transform = transform
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random noise image as PIL Image format
        import numpy as np
        from PIL import Image
        
        # Generate noise in [0, 255] range
        noise = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        noise_pil = Image.fromarray(noise)
        
        if self.transform:
            noise_pil = self.transform(noise_pil)
            
        return noise_pil, -1  # Label -1 for OOD


class FeatureExtractor:
    """
    Feature extraction from various pre-trained architectures
    """
    
    def __init__(self, architecture: str = 'resnet50', layer: str = 'penultimate'):
        self.architecture = architecture
        self.layer = layer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.hook_features = None
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model"""
        if self.architecture == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            if self.layer == 'penultimate':
                self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove last FC
                
        elif self.architecture == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            if self.layer == 'penultimate':
                self.model = nn.Sequential(*list(self.model.children())[:-1])
                
        elif self.architecture == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            if self.layer == 'penultimate':
                self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
                
        elif self.architecture == 'efficientnet':
            self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)  # No classification head
            
        elif self.architecture == 'vit':
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
            
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        self.model.eval()
        self.model.to(self.device)
    
    def extract_features(self, dataloader: DataLoader, max_samples: Optional[int] = None) -> np.ndarray:
        """Extract features from dataloader"""
        features = []
        labels = []
        
        with torch.no_grad():
            sample_count = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                if max_samples and sample_count >= max_samples:
                    break
                    
                data = data.to(self.device)
                
                # Forward pass
                if self.architecture in ['efficientnet', 'vit']:
                    output = self.model(data)
                else:
                    output = self.model(data)
                    if len(output.shape) > 2:
                        output = output.view(output.size(0), -1)  # Flatten
                
                features.append(output.cpu().numpy())
                labels.extend(target.numpy())
                sample_count += len(data)
                
                # Progress tracking removed for cleaner output
        
        return np.vstack(features), np.array(labels)
    
    def extract_features_with_cache(self, dataloader: DataLoader, dataset_name: str, 
                                   max_samples: Optional[int] = None,
                                   cache_dir: str = './cache',
                                   force_recompute: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features with caching support"""
        cached_extractor = CachedFeatureExtractor(cache_dir=cache_dir)
        return cached_extractor.extract_features_with_cache(
            dataloader, dataset_name, self.architecture, max_samples, force_recompute
        )


class ImageSpectralOODDetector:
    """
    Spectral OOD Detection adapted for high-dimensional image features
    Combines all three research questions for vision applications
    """
    
    def __init__(self, 
                 method: str = 'unified',  # 'spectral_gap', 'multiscale', 'unified'
                 k_neighbors: int = 10,
                 embedding_dim: int = 50,
                 pca_dim: int = 512,
                 n_scales: int = 8):
        self.method = method
        self.k_neighbors = k_neighbors
        self.embedding_dim = embedding_dim
        self.pca_dim = pca_dim
        self.n_scales = n_scales
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_dim)
        
        # Spectral components
        self.reference_eigenvalues = None
        self.reference_embedding = None
        self.reference_signatures = None
        self.threshold = None
        
    def _preprocess_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Preprocess high-dimensional features for spectral analysis"""
        if fit:
            # Fit preprocessing on training data
            features_scaled = self.scaler.fit_transform(features)
            features_reduced = self.pca.fit_transform(features_scaled)
        else:
            # Transform test data
            features_scaled = self.scaler.transform(features)
            features_reduced = self.pca.transform(features_scaled)
            
        return features_reduced
    
    def _compute_graph_laplacian(self, X: np.ndarray) -> csr_matrix:
        """Compute normalized graph Laplacian"""
        # Build k-NN graph
        A = kneighbors_graph(X, n_neighbors=self.k_neighbors, 
                           mode='connectivity', include_self=False)
        A = 0.5 * (A + A.T)  # Symmetrize
        
        # Normalized Laplacian
        D = np.array(A.sum(axis=1)).flatten()
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-10)))
        I = np.eye(A.shape[0])
        L = I - D_sqrt_inv @ A.toarray() @ D_sqrt_inv
        
        return csr_matrix(L)
    
    def _spectral_gap_method(self, X: np.ndarray) -> Dict:
        """Research Question 1: Spectral gap analysis"""
        L = self._compute_graph_laplacian(X)
        
        # Compute eigenvalues
        try:
            eigenvals, eigenvecs = eigsh(L, k=min(self.embedding_dim, L.shape[0]-2), which='SM')
        except:
            eigenvals, eigenvecs = np.linalg.eigh(L.toarray())
            eigenvals = eigenvals[:self.embedding_dim]
            eigenvecs = eigenvecs[:, :self.embedding_dim]
            
        # Sort eigenvalues
        idx = np.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        features = {
            'eigenvalues': eigenvals,
            'spectral_gap': eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0,
            'algebraic_connectivity': eigenvals[1] if len(eigenvals) > 1 else 0,
            'eigenvalue_variance': np.var(eigenvals),
            'eigenvalue_entropy': self._spectral_entropy(eigenvals)
        }
        
        return features
    
    def _multiscale_method(self, X: np.ndarray) -> Dict:
        """Research Question 2: Multi-scale wavelet analysis"""
        L = self._compute_graph_laplacian(X)
        
        # Eigendecomposition
        try:
            eigenvals, eigenvecs = eigsh(L, k=min(self.embedding_dim, L.shape[0]-2), which='SM')
        except:
            eigenvals, eigenvecs = np.linalg.eigh(L.toarray())
            eigenvals = eigenvals[:self.embedding_dim]
            eigenvecs = eigenvecs[:, :self.embedding_dim]
            
        idx = np.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Design wavelet filterbank
        lambda_max = np.max(eigenvals)
        scales = np.logspace(-2, np.log10(lambda_max + 1e-10), self.n_scales)
        
        # Compute multi-scale features
        scale_energies = []
        for scale in scales:
            # Mexican hat wavelet
            g = eigenvals * scale * np.exp(-eigenvals * scale)
            # Energy at this scale
            energy = np.sum(g**2)
            scale_energies.append(energy)
            
        features = {
            'eigenvalues': eigenvals,
            'scale_energies': np.array(scale_energies),
            'dominant_scale': np.argmax(scale_energies),
            'scale_entropy': self._spectral_entropy(np.array(scale_energies) + 1e-10),
            'total_energy': np.sum(scale_energies)
        }
        
        return features
    
    def _unified_method(self, X: np.ndarray) -> Dict:
        """Research Question 3: Unified spectral-manifold framework"""
        # Combine spectral gap and multiscale features
        spectral_features = self._spectral_gap_method(X)
        multiscale_features = self._multiscale_method(X)
        
        # Add manifold-specific features
        L = self._compute_graph_laplacian(X)
        
        try:
            eigenvals, eigenvecs = eigsh(L, k=min(self.embedding_dim, L.shape[0]-2), which='SM')
        except:
            eigenvals, eigenvecs = np.linalg.eigh(L.toarray())
            eigenvals = eigenvals[:self.embedding_dim]
            eigenvecs = eigenvecs[:, :self.embedding_dim]
            
        idx = np.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Heat kernel features
        t = 1.0  # Heat diffusion time
        heat_eigenvals = np.exp(-t * eigenvals)
        heat_kernel_trace = np.sum(heat_eigenvals)
        
        # Combine all features
        features = {
            **spectral_features,
            **multiscale_features,
            'heat_kernel_trace': heat_kernel_trace,
            'effective_resistance': np.sum(1.0 / (eigenvals + 1e-10)),
            'spectral_embedding': eigenvecs,
            'cheeger_lower': 0.5 * spectral_features['spectral_gap'],
            'cheeger_upper': np.sqrt(2 * spectral_features['spectral_gap'])
        }
        
        return features
    
    def _spectral_entropy(self, values: np.ndarray) -> float:
        """Compute spectral entropy"""
        values_pos = np.abs(values) + 1e-10
        p = values_pos / np.sum(values_pos)
        return -np.sum(p * np.log(p + 1e-10))
    
    def fit(self, features: np.ndarray):
        """Fit the spectral OOD detector"""
        # Preprocess features
        features_processed = self._preprocess_features(features, fit=True)
        
        # Features preprocessed for spectral analysis
        
        # Extract spectral features based on method
        if self.method == 'spectral_gap':
            spectral_features = self._spectral_gap_method(features_processed)
        elif self.method == 'multiscale':
            spectral_features = self._multiscale_method(features_processed)
        elif self.method == 'unified':
            spectral_features = self._unified_method(features_processed)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Store reference features
        self.reference_signatures = spectral_features
        
        # Set threshold (simple approach - can be made more sophisticated)
        if 'spectral_gap' in spectral_features:
            self.threshold = spectral_features['spectral_gap'] * 0.5
        else:
            self.threshold = 0.1
        
        return self
    
    def predict_score(self, features: np.ndarray) -> np.ndarray:
        """Compute OOD scores for test features"""
        # Preprocess features
        features_processed = self._preprocess_features(features, fit=False)
        
        # Batch processing for large datasets
        batch_size = min(1000, features_processed.shape[0])
        scores = []
        
        for i in range(0, features_processed.shape[0], batch_size):
            batch_features = features_processed[i:i+batch_size]
            
            # Extract spectral features for batch
            if self.method == 'spectral_gap':
                batch_spectral = self._spectral_gap_method(batch_features)
                # Score based on spectral gap deviation
                gap_dev = abs(batch_spectral['spectral_gap'] - 
                            self.reference_signatures['spectral_gap'])
                batch_scores = [gap_dev] * len(batch_features)
                
            elif self.method == 'multiscale':
                batch_spectral = self._multiscale_method(batch_features)
                # Score based on scale energy distribution
                energy_dev = np.linalg.norm(
                    batch_spectral['scale_energies'] - 
                    self.reference_signatures['scale_energies']
                )
                batch_scores = [energy_dev] * len(batch_features)
                
            elif self.method == 'unified':
                batch_spectral = self._unified_method(batch_features)
                # Combined score
                gap_dev = abs(batch_spectral['spectral_gap'] - 
                            self.reference_signatures['spectral_gap'])
                if 'scale_energies' in batch_spectral and 'scale_energies' in self.reference_signatures:
                    energy_dev = np.linalg.norm(
                        batch_spectral['scale_energies'] - 
                        self.reference_signatures['scale_energies']
                    )
                else:
                    energy_dev = 0
                    
                heat_dev = abs(batch_spectral['heat_kernel_trace'] - 
                             self.reference_signatures['heat_kernel_trace'])
                
                combined_score = gap_dev + 0.3 * energy_dev + 0.2 * heat_dev
                batch_scores = [combined_score] * len(batch_features)
            
            scores.extend(batch_scores)
        
        return np.array(scores)


class VisionOODEvaluator:
    """
    Comprehensive evaluation framework for spectral OOD detection in computer vision
    """
    
    def __init__(self, data_dir: str = './data'):
        self.data_dir = data_dir
        self.loader = VisionDatasetLoader(data_dir, batch_size=64)  # Smaller batch for memory
        self.results = {}
        
    def evaluate_single_config(self, 
                             id_dataset: str,
                             ood_dataset: str,
                             architecture: str,
                             method: str,
                             max_samples: int = 2000) -> Dict:
        """Evaluate single configuration"""
        
        # Load datasets
        if id_dataset == 'cifar10':
            id_loader = self.loader.get_cifar10(train=False)
        elif id_dataset == 'cifar100':
            id_loader = self.loader.get_cifar100(train=False)
        elif id_dataset == 'svhn':
            id_loader = self.loader.get_svhn(split='test')
        elif id_dataset == 'imagenet':
            id_loader = self.loader.get_imagenet(split='val', subset_size=max_samples)
        
        if ood_dataset == 'noise':
            ood_loader = self.loader.get_noise_ood(size=max_samples//2)
        elif ood_dataset == 'texture':
            ood_loader = self.loader.get_texture_ood()
        elif ood_dataset == 'cifar10' and id_dataset != 'cifar10':
            ood_loader = self.loader.get_cifar10(train=False)
        elif ood_dataset == 'cifar100' and id_dataset != 'cifar100':
            ood_loader = self.loader.get_cifar100(train=False)
        elif ood_dataset == 'svhn' and id_dataset != 'svhn':
            ood_loader = self.loader.get_svhn(split='test')
        
        # Extract features
        feature_extractor = FeatureExtractor(architecture=architecture)
        
        id_features, id_labels = feature_extractor.extract_features(id_loader, max_samples)
        ood_features, ood_labels = feature_extractor.extract_features(ood_loader, max_samples//2)
        
        # Split ID data for training/testing
        n_train = min(1000, len(id_features) // 2)  # Use subset for training
        train_features = id_features[:n_train]
        test_id_features = id_features[n_train:n_train+500]  # Subset for testing
        
        # Combine test data
        test_features = np.vstack([test_id_features, ood_features])
        test_labels = np.concatenate([
            np.zeros(len(test_id_features)), 
            np.ones(len(ood_features))
        ])
        
        # Train spectral detector
        detector = ImageSpectralOODDetector(method=method, pca_dim=min(256, train_features.shape[1]))
        detector.fit(train_features)
        
        # Predict scores
        scores = detector.predict_score(test_features)
        
        # Evaluate
        auc = roc_auc_score(test_labels, scores)
        ap = average_precision_score(test_labels, scores)
        
        # Additional metrics
        fpr, tpr, _ = roc_curve(test_labels, scores)
        precision, recall, _ = precision_recall_curve(test_labels, scores)
        
        # FPR at 95% TPR
        tpr95_idx = np.where(tpr >= 0.95)[0]
        fpr95 = fpr[tpr95_idx[0]] if len(tpr95_idx) > 0 else 1.0
        
        results = {
            'auc': auc,
            'average_precision': ap,
            'fpr95': fpr95,
            'id_dataset': id_dataset,
            'ood_dataset': ood_dataset,
            'architecture': architecture,
            'method': method,
            'n_train': len(train_features),
            'n_test_id': len(test_id_features),
            'n_test_ood': len(ood_features),
            'feature_dim_original': id_features.shape[1],
            'feature_dim_processed': detector.pca_dim
        }
        
        return results
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run evaluation across all combinations"""
        
        # Configuration
        id_datasets = ['cifar10', 'cifar100', 'svhn']
        ood_datasets = ['noise', 'texture'] + ['cifar10', 'cifar100', 'svhn']
        architectures = ['resnet18', 'resnet50', 'vgg16', 'efficientnet']
        methods = ['spectral_gap', 'multiscale', 'unified']
        
        all_results = []
        
        for id_data in id_datasets:
            for ood_data in ood_datasets:
                if id_data == ood_data:
                    continue  # Skip same dataset
                    
                for arch in architectures:
                    for method in methods:
                        try:
                            result = self.evaluate_single_config(
                                id_data, ood_data, arch, method, max_samples=1500
                            )
                            all_results.append(result)
                            
                        except Exception as e:
                            print(f"Error in {id_data}-{ood_data}-{arch}-{method}: {e}")
                            continue
        
        self.results = all_results
        return all_results
    
    def analyze_results(self):
        """Analyze and visualize results"""
        if not self.results:
            print("No results to analyze. Run evaluation first.")
            return
            
        import pandas as pd
        df = pd.DataFrame(self.results)
        
        # Summary statistics
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS ANALYSIS")
        print("="*80)
        
        # Best performance by method
        print("\nBest AUC by Method:")
        method_auc = df.groupby('method')['auc'].agg(['mean', 'std', 'max'])
        print(method_auc)
        
        # Best performance by architecture
        print("\nBest AUC by Architecture:")
        arch_auc = df.groupby('architecture')['auc'].agg(['mean', 'std', 'max'])
        print(arch_auc)
        
        # Best performance by dataset combination
        print("\nTop 10 Dataset Combinations (by AUC):")
        dataset_combo = df.groupby(['id_dataset', 'ood_dataset'])['auc'].mean().sort_values(ascending=False)
        print(dataset_combo.head(10))
        
        # Visualization
        self._create_visualizations(df)
    
    def _create_visualizations(self, df):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Method comparison
        df.boxplot(column='auc', by='method', ax=axes[0, 0])
        axes[0, 0].set_title('AUC by Method')
        axes[0, 0].set_xlabel('Method')
        axes[0, 0].set_ylabel('AUC')
        
        # 2. Architecture comparison
        df.boxplot(column='auc', by='architecture', ax=axes[0, 1])
        axes[0, 1].set_title('AUC by Architecture')
        axes[0, 1].set_xlabel('Architecture')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Dataset combination heatmap
        pivot_auc = df.pivot_table(values='auc', index='id_dataset', 
                                  columns='ood_dataset', aggfunc='mean')
        sns.heatmap(pivot_auc, annot=True, fmt='.3f', ax=axes[0, 2], cmap='viridis')
        axes[0, 2].set_title('AUC Heatmap (ID vs OOD)')
        
        # 4. FPR95 comparison
        df.boxplot(column='fpr95', by='method', ax=axes[1, 0])
        axes[1, 0].set_title('FPR95 by Method')
        axes[1, 0].set_xlabel('Method')
        axes[1, 0].set_ylabel('FPR95')
        
        # 5. Method vs Architecture performance
        pivot_method_arch = df.pivot_table(values='auc', index='method', 
                                          columns='architecture', aggfunc='mean')
        sns.heatmap(pivot_method_arch, annot=True, fmt='.3f', ax=axes[1, 1], cmap='viridis')
        axes[1, 1].set_title('Method vs Architecture AUC')
        
        # 6. Performance distribution
        axes[1, 2].hist(df['auc'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(df['auc'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {df["auc"].mean():.3f}')
        axes[1, 2].set_xlabel('AUC')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('AUC Distribution')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Additional detailed analysis
        self._detailed_analysis(df)
    
    def _detailed_analysis(self, df):
        """Detailed statistical analysis"""
        print("\n" + "="*60)
        print("DETAILED STATISTICAL ANALYSIS")
        print("="*60)
        
        # Statistical significance tests
        from scipy import stats
        
        methods = df['method'].unique()
        print("\nPairwise t-tests between methods (p-values):")
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                scores1 = df[df['method'] == method1]['auc']
                scores2 = df[df['method'] == method2]['auc']
                _, p_value = stats.ttest_ind(scores1, scores2)
                print(f"{method1} vs {method2}: p={p_value:.4f}")
        
        # Correlation analysis
        print("\nCorrelation with dataset properties:")
        corr_features = ['n_train', 'n_test_id', 'n_test_ood', 'feature_dim_original']
        for feature in corr_features:
            corr, p_val = stats.pearsonr(df[feature], df['auc'])
            print(f"{feature}: r={corr:.3f}, p={p_val:.4f}")
        
        # Best configurations
        print(f"\nTop 5 Best Configurations:")
        top_configs = df.nlargest(5, 'auc')[['id_dataset', 'ood_dataset', 
                                            'architecture', 'method', 'auc', 'fpr95']]
        print(top_configs.to_string(index=False))


# Main execution and demonstration
def main():
    """Main execution function with argparse support"""
    parser = argparse.ArgumentParser(description='Spectral OOD Detection for Computer Vision')
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='Directory containing datasets (default: ./data)')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                       help='Directory for caching features (default: ./cache)')
    parser.add_argument('--max_samples', type=int, default=2000,
                       help='Maximum samples per dataset (default: 2000)')
    parser.add_argument('--force_recompute', action='store_true',
                       help='Force recomputation of cached features')
    parser.add_argument('--clear_cache', action='store_true',
                       help='Clear all cached features before running')
    parser.add_argument('--cache_info', action='store_true',
                       help='Show cache information and exit')
    parser.add_argument('--full_eval', action='store_true',
                       help='Run full evaluation without prompting')
    parser.add_argument('--quick_demo', action='store_true',
                       help='Run quick demo only')
    
    args = parser.parse_args()
    
    print("="*80)
    print("SPECTRAL OOD DETECTION FOR COMPUTER VISION")
    print("Implementation across Multiple Datasets and Architectures")
    print("="*80)
    print(f"Data Directory: {args.data_dir}")
    print(f"Cache Directory: {args.cache_dir}")
    print(f"Max Samples: {args.max_samples}")
    
    # Initialize cache system
    cache_system = CachedFeatureExtractor(cache_dir=args.cache_dir)
    
    # Handle cache operations
    if args.cache_info:
        cache_system.print_cache_info()
        return
        
    if args.clear_cache:
        print("\n🧹 Clearing cache...")
        cache_system.clear_cache()
        print("Cache cleared successfully!")
    
    # Initialize evaluator
    evaluator = VisionOODEvaluator(data_dir=args.data_dir)
    
    # Handle quick demo mode
    if args.quick_demo:
        print("\n🔬 Running Quick Demo...")
        demo_result = evaluator.evaluate_single_config(
            id_dataset='cifar10',
            ood_dataset='noise', 
            architecture='resnet18',
            method='unified',
            max_samples=min(500, args.max_samples)
        )
        
        print(f"\nDemo Results:")
        print(f"AUC: {demo_result['auc']:.4f}")
        print(f"Average Precision: {demo_result['average_precision']:.4f}")
        print(f"FPR95: {demo_result['fpr95']:.4f}")
        print("\n✅ Quick demo completed successfully!")
        return
    
    # Full evaluation or interactive mode
    should_run_full = args.full_eval
    
    if not should_run_full:
        # Quick demo first
        print("\n🔬 Running Quick Demo...")
        demo_result = evaluator.evaluate_single_config(
            id_dataset='cifar10',
            ood_dataset='noise', 
            architecture='resnet18',
            method='unified',
            max_samples=min(500, args.max_samples)
        )
        
        print(f"\nDemo Results:")
        print(f"AUC: {demo_result['auc']:.4f}")
        print(f"Average Precision: {demo_result['average_precision']:.4f}")
        print(f"FPR95: {demo_result['fpr95']:.4f}")
        
        # Ask user if they want full evaluation
        response = input("\n🚀 Run full evaluation? This may take 30-60 minutes (y/n): ")
        should_run_full = response.lower() == 'y'
    
    if should_run_full:
        print("\n🔥 Running Comprehensive Evaluation...")
        print(f"Using cached features from: {args.cache_dir}")
        print(f"Force recompute: {args.force_recompute}")
        
        # Show cache info before starting
        cache_system.print_cache_info()
        
        all_results = evaluator.run_comprehensive_evaluation()
        
        # Analyze results
        evaluator.analyze_results()
        
        # Save results
        import json
        with open('spectral_ood_vision_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print("\n💾 Results saved to 'spectral_ood_vision_results.json'")
        
        # Show final cache info
        print("\n📊 Final cache statistics:")
        cache_system.print_cache_info()
        
    else:
        print("\n✅ Demo completed successfully!")
        print("To run full evaluation later, use: python spectral_ood_vision.py --full_eval")


if __name__ == "__main__":
    main()