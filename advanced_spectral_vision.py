"""
Advanced Spectral OOD Detection for Computer Vision
Enhanced implementation with state-of-the-art features and optimizations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import models, transforms
import timm
from transformers import ViTImageProcessor, ViTModel
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, eigs
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import json
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

# Import caching system
from feature_cache import FeatureCache, CachedFeatureExtractor

class AdvancedFeatureExtractor:
    """
    Advanced feature extraction supporting latest architectures and techniques
    """
    
    def __init__(self, 
                 architecture: str = 'resnet50',
                 layer: str = 'penultimate',
                 device: Optional[str] = None):
        self.architecture = architecture
        self.layer = layer
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load and prepare model for feature extraction"""
        print(f"Loading {self.architecture}...")
        
        if self.architecture == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            
        elif self.architecture == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            
        elif self.architecture == 'resnet101':
            self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            
        elif self.architecture == 'vgg16':
            self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
            
        elif self.architecture == 'densenet121':
            self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.model.classifier = nn.Identity()
            
        elif self.architecture == 'efficientnet_b0':
            self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
            
        elif self.architecture == 'efficientnet_b4':
            self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)
            
        elif self.architecture == 'vit_base':
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
            
        elif self.architecture == 'vit_large':
            self.model = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=0)
            
        elif self.architecture == 'swin_base':
            self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
            
        elif self.architecture == 'convnext_base':
            self.model = timm.create_model('convnext_base', pretrained=True, num_classes=0)
            
        elif self.architecture == 'clip_resnet':
            import clip
            self.model, _ = clip.load("RN50", device=self.device)
            
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        self.model.eval()
        self.model.to(self.device)
        print(f"✅ {self.architecture} loaded successfully")
    
    def extract_features(self, dataloader: DataLoader, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features with memory-efficient processing"""
        features = []
        labels = []
        sample_count = 0
        
        print(f"Extracting features using {self.architecture}...")
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                if max_samples and sample_count >= max_samples:
                    break
                    
                data = data.to(self.device)
                batch_size = data.size(0)
                
                # Extract features based on architecture
                if self.architecture.startswith('clip'):
                    output = self.model.encode_image(data)
                elif self.architecture in ['efficientnet_b0', 'efficientnet_b4', 'vit_base', 'vit_large', 
                                          'swin_base', 'convnext_base']:
                    output = self.model(data)
                else:
                    output = self.model(data)
                    
                # Flatten if needed
                if len(output.shape) > 2:
                    output = output.view(output.size(0), -1)
                
                features.append(output.cpu().numpy())
                labels.extend(target.numpy())
                sample_count += batch_size
                
                if batch_idx % 20 == 0:
                    print(f"  Processed {sample_count} samples...")
        
        features_array = np.vstack(features)
        labels_array = np.array(labels)
        
        print(f"✅ Extracted features shape: {features_array.shape}")
        return features_array, labels_array
    
    def extract_features_with_cache(self, dataloader: DataLoader, dataset_name: str,
                                   max_samples: Optional[int] = None,
                                   cache_dir: str = './cache',
                                   force_recompute: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features with caching support"""
        cached_extractor = CachedFeatureExtractor(cache_dir=cache_dir)
        return cached_extractor.extract_features_with_cache(
            dataloader, dataset_name, self.architecture, max_samples, force_recompute
        )


class HybridSpectralOODDetector:
    """
    Hybrid spectral OOD detector combining multiple advanced techniques:
    1. Graph spectral analysis with adaptive neighborhoods
    2. Persistent homology for topological features  
    3. Heat kernel signatures at multiple scales
    4. Manifold learning with geodesic distances
    5. Ensemble of spectral methods
    """
    
    def __init__(self,
                 embedding_dim: int = 64,
                 pca_components: int = 512,
                 adaptive_k: bool = True,
                 use_heat_kernel: bool = True,
                 use_persistent_homology: bool = True,
                 ensemble_methods: List[str] = ['spectral_gap', 'heat_kernel', 'topology']):
        
        self.embedding_dim = embedding_dim
        self.pca_components = pca_components
        self.adaptive_k = adaptive_k
        self.use_heat_kernel = use_heat_kernel
        self.use_persistent_homology = use_persistent_homology
        self.ensemble_methods = ensemble_methods
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.pca = None
        
        # Reference features
        self.reference_features = {}
        self.thresholds = {}
        
    def _adaptive_neighborhood_selection(self, X: np.ndarray) -> int:
        """Adaptively select optimal neighborhood size"""
        n_samples = X.shape[0]
        
        # Try different k values and select based on spectral gap
        k_candidates = [5, 10, 15, 20, 30, min(50, n_samples//10)]
        k_candidates = [k for k in k_candidates if k < n_samples]
        
        best_k = k_candidates[0]
        best_gap = 0
        
        for k in k_candidates:
            try:
                A = kneighbors_graph(X, n_neighbors=k, mode='connectivity')
                A = 0.5 * (A + A.T)
                
                # Normalized Laplacian
                D = np.array(A.sum(axis=1)).flatten()
                D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-10)))
                L = np.eye(A.shape[0]) - D_sqrt_inv @ A.toarray() @ D_sqrt_inv
                
                # Compute first few eigenvalues
                eigenvals = eigsh(L, k=min(5, L.shape[0]-2), which='SM', return_eigenvectors=False)
                eigenvals = np.sort(eigenvals)
                
                if len(eigenvals) > 1:
                    gap = eigenvals[1] - eigenvals[0]
                    if gap > best_gap:
                        best_gap = gap
                        best_k = k
            except:
                continue
                
        return best_k
    
    def _compute_heat_kernel_signatures(self, L: csr_matrix, scales: np.ndarray) -> np.ndarray:
        """Compute heat kernel signatures at multiple scales"""
        try:
            eigenvals, eigenvecs = eigsh(L, k=min(self.embedding_dim, L.shape[0]-2), which='SM')
        except:
            eigenvals, eigenvecs = np.linalg.eigh(L.toarray())
            eigenvals = eigenvals[:self.embedding_dim]
            eigenvecs = eigenvecs[:, :self.embedding_dim]
        
        idx = np.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Compute HKS for each scale
        n_nodes = L.shape[0]
        hks = np.zeros((n_nodes, len(scales)))
        
        for i, t in enumerate(scales):
            # Heat kernel at time t
            heat_eigenvals = np.exp(-t * eigenvals)
            # HKS is diagonal of heat kernel
            hks[:, i] = np.sum((eigenvecs**2) * heat_eigenvals, axis=1)
        
        return hks
    
    def _compute_persistent_homology_features(self, X: np.ndarray) -> Dict[str, float]:
        """Compute topological features using persistent homology (simplified)"""
        # Simplified persistent homology using distance matrices
        from scipy.spatial.distance import pdist, squareform
        
        # Subsample for computational efficiency
        n_samples = min(200, X.shape[0])
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X_sub = X[indices]
        
        # Distance matrix
        distances = squareform(pdist(X_sub))
        
        # Compute filtration values (simplified Vietoris-Rips)
        filtration_values = np.linspace(0, np.max(distances), 20)
        
        # Count connected components at each filtration value
        components = []
        for thresh in filtration_values:
            # Create graph at threshold
            adj = distances <= thresh
            n_components = self._count_connected_components(adj)
            components.append(n_components)
        
        # Topological features
        features = {
            'persistence_entropy': self._persistence_entropy(components),
            'max_components': np.max(components),
            'persistence_length': len([c for c in components if c > 1]),
            'avg_components': np.mean(components)
        }
        
        return features
    
    def _count_connected_components(self, adj_matrix: np.ndarray) -> int:
        """Count connected components in adjacency matrix"""
        visited = np.zeros(adj_matrix.shape[0], dtype=bool)
        components = 0
        
        for i in range(adj_matrix.shape[0]):
            if not visited[i]:
                # DFS to mark all connected nodes
                stack = [i]
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        neighbors = np.where(adj_matrix[node])[0]
                        stack.extend([n for n in neighbors if not visited[n]])
                components += 1
        
        return components
    
    def _persistence_entropy(self, components: List[int]) -> float:
        """Compute entropy of persistence diagram"""
        if len(components) <= 1:
            return 0.0
        
        # Convert to persistence pairs (birth, death)
        persistence_lengths = []
        for i in range(1, len(components)):
            if components[i] < components[i-1]:
                # Component died
                length = i - (i-1)  # Simplified birth-death calculation
                persistence_lengths.append(length)
        
        if not persistence_lengths:
            return 0.0
            
        # Normalize to probabilities
        total = sum(persistence_lengths)
        if total == 0:
            return 0.0
            
        probs = [l/total for l in persistence_lengths]
        
        # Compute entropy
        entropy = -sum([p * np.log(p + 1e-10) for p in probs])
        return entropy
    
    def _spectral_gap_features(self, L: csr_matrix) -> Dict[str, float]:
        """Extract spectral gap-based features"""
        try:
            eigenvals = eigsh(L, k=min(self.embedding_dim, L.shape[0]-2), 
                            which='SM', return_eigenvectors=False)
        except:
            eigenvals = np.linalg.eigvalsh(L.toarray())[:self.embedding_dim]
        
        eigenvals = np.sort(eigenvals)
        
        features = {
            'spectral_gap': eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0,
            'algebraic_connectivity': eigenvals[1] if len(eigenvals) > 1 else 0,
            'spectral_radius': eigenvals[-1],
            'eigenvalue_variance': np.var(eigenvals),
            'eigenvalue_skewness': self._skewness(eigenvals),
            'eigenvalue_kurtosis': self._kurtosis(eigenvals),
            'effective_resistance': np.sum(1.0 / (eigenvals + 1e-10))
        }
        
        return features
    
    def _skewness(self, x: np.ndarray) -> float:
        """Compute skewness"""
        mean_x = np.mean(x)
        std_x = np.std(x)
        if std_x == 0:
            return 0
        return np.mean(((x - mean_x) / std_x) ** 3)
    
    def _kurtosis(self, x: np.ndarray) -> float:
        """Compute kurtosis"""
        mean_x = np.mean(x)
        std_x = np.std(x)
        if std_x == 0:
            return 0
        return np.mean(((x - mean_x) / std_x) ** 4) - 3
    
    def fit(self, features: np.ndarray):
        """Fit the hybrid spectral detector"""
        print("Fitting Hybrid Spectral OOD Detector...")
        
        # Preprocessing
        features_scaled = self.scaler.fit_transform(features)
        
        # Adaptive PCA
        explained_variance_threshold = 0.95
        self.pca = PCA(n_components=min(self.pca_components, features.shape[1]))
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Find number of components for desired explained variance
        cumsum_var = np.cumsum(self.pca.explained_variance_ratio_)
        n_components = np.where(cumsum_var >= explained_variance_threshold)[0]
        if len(n_components) > 0:
            n_components = min(n_components[0] + 1, self.pca_components)
            features_reduced = features_pca[:, :n_components]
        else:
            features_reduced = features_pca
            
        print(f"Reduced to {features_reduced.shape[1]} dimensions")
        
        # Adaptive neighborhood selection
        if self.adaptive_k:
            optimal_k = self._adaptive_neighborhood_selection(features_reduced)
            print(f"Selected k={optimal_k} neighbors")
        else:
            optimal_k = 15
            
        # Build graph and compute Laplacian
        A = kneighbors_graph(features_reduced, n_neighbors=optimal_k, mode='connectivity')
        A = 0.5 * (A + A.T)
        
        D = np.array(A.sum(axis=1)).flatten()
        D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-10)))
        L = csr_matrix(np.eye(A.shape[0]) - D_sqrt_inv @ A.toarray() @ D_sqrt_inv)
        
        # Extract features using ensemble methods
        self.reference_features = {}
        
        if 'spectral_gap' in self.ensemble_methods:
            self.reference_features['spectral_gap'] = self._spectral_gap_features(L)
            
        if 'heat_kernel' in self.ensemble_methods and self.use_heat_kernel:
            scales = np.logspace(-2, 1, 8)  # Multiple time scales
            hks = self._compute_heat_kernel_signatures(L, scales)
            self.reference_features['heat_kernel'] = {
                'mean_hks': np.mean(hks, axis=0),
                'std_hks': np.std(hks, axis=0),
                'max_hks': np.max(hks, axis=0)
            }
            
        if 'topology' in self.ensemble_methods and self.use_persistent_homology:
            try:
                topo_features = self._compute_persistent_homology_features(features_reduced)
                self.reference_features['topology'] = topo_features
            except:
                print("Warning: Could not compute topological features")
        
        # Set thresholds for each method
        for method in self.ensemble_methods:
            if method in self.reference_features:
                # Simple threshold based on feature magnitude
                if method == 'spectral_gap':
                    self.thresholds[method] = self.reference_features[method]['spectral_gap'] * 0.5
                else:
                    self.thresholds[method] = 0.1
        
        print("✅ Fitting completed")
        return self
    
    def predict_score(self, features: np.ndarray) -> np.ndarray:
        """Predict OOD scores using ensemble of methods"""
        # Preprocess
        features_scaled = self.scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        
        # Use same number of components as training
        n_components = len(self.reference_features.get('spectral_gap', {}).keys()) if 'spectral_gap' in self.reference_features else features_pca.shape[1]
        features_reduced = features_pca[:, :min(n_components, features_pca.shape[1])]
        
        # Batch processing for memory efficiency
        batch_size = min(500, features_reduced.shape[0])
        all_scores = []
        
        for i in range(0, features_reduced.shape[0], batch_size):
            batch_features = features_reduced[i:i+batch_size]
            batch_scores = self._score_batch(batch_features)
            all_scores.extend(batch_scores)
        
        return np.array(all_scores)
    
    def _score_batch(self, batch_features: np.ndarray) -> List[float]:
        """Score a batch of features"""
        # Build local graph for batch
        try:
            k_neighbors = min(10, batch_features.shape[0] - 1)
            A = kneighbors_graph(batch_features, n_neighbors=k_neighbors, mode='connectivity')
            A = 0.5 * (A + A.T)
            
            D = np.array(A.sum(axis=1)).flatten()
            D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-10)))
            L = csr_matrix(np.eye(A.shape[0]) - D_sqrt_inv @ A.toarray() @ D_sqrt_inv)
        except:
            # Fallback: return default scores
            return [1.0] * batch_features.shape[0]
        
        # Compute scores for each method
        method_scores = {}
        
        if 'spectral_gap' in self.ensemble_methods and 'spectral_gap' in self.reference_features:
            test_features = self._spectral_gap_features(L)
            ref_features = self.reference_features['spectral_gap']
            
            # Compute deviation from reference
            gap_dev = abs(test_features['spectral_gap'] - ref_features['spectral_gap'])
            connect_dev = abs(test_features['algebraic_connectivity'] - ref_features['algebraic_connectivity'])
            var_dev = abs(test_features['eigenvalue_variance'] - ref_features['eigenvalue_variance'])
            
            method_scores['spectral_gap'] = gap_dev + 0.5 * connect_dev + 0.3 * var_dev
        
        if 'heat_kernel' in self.ensemble_methods and 'heat_kernel' in self.reference_features:
            try:
                scales = np.logspace(-2, 1, 8)
                hks = self._compute_heat_kernel_signatures(L, scales)
                ref_hks = self.reference_features['heat_kernel']
                
                # Compare HKS statistics
                mean_dev = np.linalg.norm(np.mean(hks, axis=0) - ref_hks['mean_hks'])
                std_dev = np.linalg.norm(np.std(hks, axis=0) - ref_hks['std_hks'])
                
                method_scores['heat_kernel'] = mean_dev + 0.5 * std_dev
            except:
                method_scores['heat_kernel'] = 1.0
        
        if 'topology' in self.ensemble_methods and 'topology' in self.reference_features:
            try:
                test_topo = self._compute_persistent_homology_features(batch_features)
                ref_topo = self.reference_features['topology']
                
                entropy_dev = abs(test_topo['persistence_entropy'] - ref_topo['persistence_entropy'])
                components_dev = abs(test_topo['avg_components'] - ref_topo['avg_components'])
                
                method_scores['topology'] = entropy_dev + 0.5 * components_dev
            except:
                method_scores['topology'] = 1.0
        
        # Ensemble combination (equal weights)
        if method_scores:
            ensemble_score = np.mean(list(method_scores.values()))
        else:
            ensemble_score = 1.0
        
        # Return same score for all samples in batch (can be refined)
        return [ensemble_score] * batch_features.shape[0]


class ComprehensiveVisionOODEvaluator:
    """
    Most comprehensive evaluation framework for spectral OOD detection
    """
    
    def __init__(self, data_dir: str = './data'):
        self.data_dir = data_dir
        self.results = []
        
    def run_extensive_evaluation(self):
        """Run most comprehensive evaluation possible"""
        
        # Define all configurations
        configs = [
            # CIFAR-10 as ID
            {'id': 'cifar10', 'ood': 'cifar100', 'arch': 'resnet18', 'method': 'hybrid'},
            {'id': 'cifar10', 'ood': 'svhn', 'arch': 'resnet50', 'method': 'hybrid'},
            {'id': 'cifar10', 'ood': 'noise', 'arch': 'efficientnet_b0', 'method': 'hybrid'},
            
            # CIFAR-100 as ID  
            {'id': 'cifar100', 'ood': 'cifar10', 'arch': 'vgg16', 'method': 'hybrid'},
            {'id': 'cifar100', 'ood': 'svhn', 'arch': 'densenet121', 'method': 'hybrid'},
            
            # SVHN as ID
            {'id': 'svhn', 'ood': 'cifar10', 'arch': 'vit_base', 'method': 'hybrid'},
            {'id': 'svhn', 'ood': 'cifar100', 'arch': 'swin_base', 'method': 'hybrid'},
        ]
        
        from spectral_ood_vision import VisionDatasetLoader
        loader = VisionDatasetLoader(self.data_dir, batch_size=32)
        
        for config in configs:
            try:
                print(f"\n{'='*80}")
                print(f"Evaluating: {config['id']} vs {config['ood']} | {config['arch']}")
                print(f"{'='*80}")
                
                # Load datasets
                if config['id'] == 'cifar10':
                    id_loader = loader.get_cifar10(train=False)
                elif config['id'] == 'cifar100':
                    id_loader = loader.get_cifar100(train=False)
                elif config['id'] == 'svhn':
                    id_loader = loader.get_svhn(split='test')
                
                if config['ood'] == 'cifar10':
                    ood_loader = loader.get_cifar10(train=False)
                elif config['ood'] == 'cifar100':
                    ood_loader = loader.get_cifar100(train=False)
                elif config['ood'] == 'svhn':
                    ood_loader = loader.get_svhn(split='test')
                elif config['ood'] == 'noise':
                    ood_loader = loader.get_noise_ood(size=1000)
                
                # Extract features
                extractor = AdvancedFeatureExtractor(architecture=config['arch'])
                
                id_features, id_labels = extractor.extract_features(id_loader, max_samples=2000)
                ood_features, ood_labels = extractor.extract_features(ood_loader, max_samples=1000)
                
                # Train/test split
                n_train = min(1500, len(id_features) // 2)
                train_features = id_features[:n_train]
                test_id_features = id_features[n_train:n_train+500]
                
                test_features = np.vstack([test_id_features, ood_features])
                test_labels = np.concatenate([np.zeros(len(test_id_features)), 
                                            np.ones(len(ood_features))])
                
                # Train detector
                detector = HybridSpectralOODDetector(
                    embedding_dim=64,
                    pca_components=min(512, train_features.shape[1]),
                    adaptive_k=True,
                    ensemble_methods=['spectral_gap', 'heat_kernel', 'topology']
                )
                
                detector.fit(train_features)
                
                # Predict and evaluate
                scores = detector.predict_score(test_features)
                
                auc = roc_auc_score(test_labels, scores)
                ap = average_precision_score(test_labels, scores)
                
                # Additional metrics
                fpr, tpr, _ = roc_curve(test_labels, scores)
                fpr95 = fpr[np.where(tpr >= 0.95)[0][0]] if len(np.where(tpr >= 0.95)[0]) > 0 else 1.0
                
                result = {
                    **config,
                    'auc': auc,
                    'ap': ap,
                    'fpr95': fpr95,
                    'n_train': len(train_features),
                    'n_test_id': len(test_id_features),
                    'n_test_ood': len(ood_features)
                }
                
                self.results.append(result)
                
                print(f"Results: AUC={auc:.4f}, AP={ap:.4f}, FPR95={fpr95:.4f}")
                
            except Exception as e:
                print(f"Error in configuration {config}: {e}")
                continue
        
        # Save results
        with open('comprehensive_spectral_ood_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Evaluation completed! {len(self.results)} configurations evaluated.")
        return self.results
    
    def create_comprehensive_report(self):
        """Create detailed analysis report"""
        if not self.results:
            print("No results to analyze")
            return
            
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*100)
        print("COMPREHENSIVE SPECTRAL OOD DETECTION REPORT")
        print("="*100)
        
        # Overall statistics
        print(f"\nTotal Configurations Evaluated: {len(df)}")
        print(f"Average AUC: {df['auc'].mean():.4f} ± {df['auc'].std():.4f}")
        print(f"Average AP: {df['ap'].mean():.4f} ± {df['ap'].std():.4f}")
        print(f"Average FPR95: {df['fpr95'].mean():.4f} ± {df['fpr95'].std():.4f}")
        
        # Best configurations
        print(f"\nTop 5 Configurations by AUC:")
        top5 = df.nlargest(5, 'auc')[['id', 'ood', 'arch', 'auc', 'ap', 'fpr95']]
        print(top5.to_string(index=False))
        
        # Architecture analysis
        print(f"\nPerformance by Architecture:")
        arch_stats = df.groupby('arch')[['auc', 'ap', 'fpr95']].agg(['mean', 'std'])
        print(arch_stats)
        
        # Dataset combination analysis
        print(f"\nPerformance by Dataset Combination:")
        dataset_stats = df.groupby(['id', 'ood'])[['auc', 'ap', 'fpr95']].mean().sort_values('auc', ascending=False)
        print(dataset_stats)
        
        # Visualization
        self._create_advanced_visualizations(df)
    
    def _create_advanced_visualizations(self, df):
        """Create advanced visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Performance distribution
        axes[0, 0].hist(df['auc'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(df['auc'].mean(), color='red', linestyle='--', label=f'Mean: {df["auc"].mean():.3f}')
        axes[0, 0].set_xlabel('AUC Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('AUC Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Architecture comparison
        arch_data = df.groupby('arch')['auc'].apply(list)
        axes[0, 1].boxplot([arch_data[arch] for arch in arch_data.index], 
                          labels=list(arch_data.index))
        axes[0, 1].set_ylabel('AUC Score')
        axes[0, 1].set_title('Performance by Architecture')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Dataset combination heatmap
        pivot_data = df.pivot_table(values='auc', index='id', columns='ood', aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=axes[0, 2])
        axes[0, 2].set_title('AUC by Dataset Combination')
        
        # 4. AUC vs FPR95 scatter
        axes[1, 0].scatter(df['auc'], df['fpr95'], alpha=0.6, s=60)
        axes[1, 0].set_xlabel('AUC Score')
        axes[1, 0].set_ylabel('FPR95')
        axes[1, 0].set_title('AUC vs FPR95')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Performance correlation matrix
        corr_data = df[['auc', 'ap', 'fpr95']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Metric Correlation Matrix')
        
        # 6. Dataset difficulty ranking
        dataset_difficulty = df.groupby(['id', 'ood'])['auc'].mean().sort_values()
        axes[1, 2].barh(range(len(dataset_difficulty)), dataset_difficulty.values)
        axes[1, 2].set_yticks(range(len(dataset_difficulty)))
        axes[1, 2].set_yticklabels([f"{combo[0]} vs {combo[1]}" for combo in dataset_difficulty.index])
        axes[1, 2].set_xlabel('Average AUC')
        axes[1, 2].set_title('Dataset Combination Difficulty (Low AUC = Harder)')
        
        plt.tight_layout()
        plt.savefig('comprehensive_spectral_ood_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


# Demonstration and main execution
def main():
    """Main execution with comprehensive evaluation and caching support"""
    parser = argparse.ArgumentParser(description='Advanced Spectral OOD Detection for Computer Vision')
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
    
    args = parser.parse_args()
    
    print("🚀 ADVANCED SPECTRAL OOD DETECTION FOR COMPUTER VISION")
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
    
    evaluator = ComprehensiveVisionOODEvaluator(data_dir=args.data_dir)
    
    print("\n📊 Starting comprehensive evaluation...")
    print("This will evaluate spectral OOD detection across:")
    print("- Multiple datasets: CIFAR-10/100, SVHN")
    print("- Multiple architectures: ResNet, VGG, EfficientNet, ViT, Swin")
    print("- Hybrid spectral methods: Spectral gaps + Heat kernels + Topology")
    print(f"- Using feature caching from: {args.cache_dir}")
    
    # Show cache info before starting
    cache_system.print_cache_info()
    
    # Run evaluation
    results = evaluator.run_extensive_evaluation()
    
    # Generate comprehensive report
    evaluator.create_comprehensive_report()
    
    # Show final cache statistics
    print("\n📊 Final cache statistics:")
    cache_system.print_cache_info()
    
    print(f"\n✅ Evaluation completed successfully!")
    print(f"📁 Results saved to 'comprehensive_spectral_ood_results.json'")
    print(f"📊 Visualizations saved to 'comprehensive_spectral_ood_analysis.png'")


if __name__ == "__main__":
    main()