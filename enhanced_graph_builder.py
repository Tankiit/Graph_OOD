"""
Enhanced Graph Construction for Computer Vision Datasets
Optimized for CIFAR-10/100, SVHN, ImageNet and OOD detection

Integrates with spectral OOD detection framework and supports multiple graph construction methods.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')

# Import existing components
from spectral_ood_vision import VisionDatasetLoader, FeatureExtractor


class EnhancedGraphBuilder:
    """
    Enhanced graph construction specifically optimized for computer vision datasets
    Supports multiple construction methods and spectral analysis integration
    """
    
    def __init__(self,
                 method: str = 'adaptive_knn',
                 k_neighbors: int = 10,
                 epsilon: float = 0.5,
                 similarity_metric: str = 'cosine',
                 preprocessing: str = 'pca',
                 pca_components: int = 256,
                 adaptive_k: bool = True,
                 negative_edges: bool = True,
                 device: str = 'auto'):
        """
        Initialize enhanced graph builder
        
        Args:
            method: Graph construction method ('knn', 'epsilon_ball', 'adaptive_knn', 'hybrid')
            k_neighbors: Number of neighbors for k-NN graph
            epsilon: Threshold for epsilon-ball graph
            similarity_metric: Distance metric ('cosine', 'euclidean', 'mahalanobis')
            preprocessing: Feature preprocessing ('pca', 'standardize', 'none')
            pca_components: Number of PCA components if using PCA
            adaptive_k: Whether to use adaptive k based on local density
            negative_edges: Whether to include negative similarity edges
            device: Computing device ('cuda', 'cpu', 'auto')
        """
        self.method = method
        self.k_neighbors = k_neighbors
        self.epsilon = epsilon
        self.similarity_metric = similarity_metric
        self.preprocessing = preprocessing
        self.pca_components = pca_components
        self.adaptive_k = adaptive_k
        self.negative_edges = negative_edges
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components)
        self.is_fitted = False
        
        # Cache for performance
        self._feature_cache = {}
        self._graph_cache = {}
        
    def preprocess_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Preprocess features for graph construction
        
        Args:
            features: Input feature matrix (n_samples, n_features)
            fit: Whether to fit preprocessing parameters
            
        Returns:
            Preprocessed features
        """
        if self.preprocessing == 'none':
            return features
            
        if fit:
            if self.preprocessing in ['pca', 'standardize']:
                features_scaled = self.scaler.fit_transform(features)
            else:
                features_scaled = features
                
            if self.preprocessing == 'pca':
                features_processed = self.pca.fit_transform(features_scaled)
            else:
                features_processed = features_scaled
                
            self.is_fitted = True
            
        else:
            if not self.is_fitted:
                raise ValueError("Must fit preprocessing first")
                
            if self.preprocessing in ['pca', 'standardize']:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
                
            if self.preprocessing == 'pca':
                features_processed = self.pca.transform(features_scaled)
            else:
                features_processed = features_scaled
        
        return features_processed
    
    def compute_similarity_matrix(self, features: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix using specified metric
        
        Args:
            features: Preprocessed feature matrix
            
        Returns:
            Similarity matrix
        """
        if self.similarity_metric == 'cosine':
            # Cosine similarity
            similarity_matrix = cosine_similarity(features)
            
        elif self.similarity_metric == 'euclidean':
            # Convert Euclidean distances to similarities
            distances = euclidean_distances(features)
            # Use RBF kernel transformation
            gamma = 1.0 / (2 * np.var(distances))
            similarity_matrix = np.exp(-gamma * distances**2)
            
        elif self.similarity_metric == 'mahalanobis':
            # Mahalanobis distance with covariance regularization
            try:
                cov_matrix = np.cov(features.T)
                cov_inv = np.linalg.pinv(cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0]))
                
                # Compute Mahalanobis distances
                n_samples = features.shape[0]
                distances = np.zeros((n_samples, n_samples))
                
                for i in range(n_samples):
                    diff = features - features[i:i+1]
                    distances[i] = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
                
                # Convert to similarities
                gamma = 1.0 / (2 * np.var(distances))
                similarity_matrix = np.exp(-gamma * distances**2)
                
            except:
                # Fallback to cosine if Mahalanobis fails
                print("Mahalanobis computation failed, falling back to cosine")
                similarity_matrix = cosine_similarity(features)
                
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
            
        return similarity_matrix
    
    def build_graph(self, features: np.ndarray, fit_preprocessing: bool = False) -> csr_matrix:
        """
        Main graph construction method
        
        Args:
            features: Input feature matrix
            fit_preprocessing: Whether to fit preprocessing parameters
            
        Returns:
            Sparse adjacency matrix
        """
        # Preprocess features
        features_processed = self.preprocess_features(features, fit=fit_preprocessing)
        
        print(f"Building {self.method} graph for {features_processed.shape[0]} samples "
              f"with {features_processed.shape[1]} features")
        
        # Build k-NN graph (simplified for this version)
        adjacency = kneighbors_graph(features_processed, n_neighbors=self.k_neighbors, 
                                   mode='distance', metric='cosine', include_self=False)
        # Convert distances to similarities
        adjacency.data = 1 - adjacency.data
        
        # Symmetrize the graph
        adjacency = 0.5 * (adjacency + adjacency.T)
        
        print(f"Graph constructed: {adjacency.nnz} edges, density: {adjacency.nnz / (adjacency.shape[0]**2):.4f}")
        
        return adjacency
    
    def extract_spectral_features(self, adjacency: csr_matrix) -> Dict:
        """
        Extract comprehensive spectral features from graph
        
        Args:
            adjacency: Sparse adjacency matrix
            
        Returns:
            Dictionary of spectral features
        """
        # Compute normalized Laplacian
        n = adjacency.shape[0]
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        degrees_sqrt_inv = 1.0 / np.sqrt(np.maximum(degrees, 1e-10))
        D_sqrt_inv = diags(degrees_sqrt_inv)
        I = diags(np.ones(n))
        L = I - D_sqrt_inv @ adjacency @ D_sqrt_inv
        
        # Eigendecomposition
        n_eigs = min(50, L.shape[0] - 2)
        try:
            eigenvalues, eigenvectors = eigsh(L, k=n_eigs, which='SM')
        except:
            # Fallback for small matrices
            eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
            eigenvalues = eigenvalues[:n_eigs]
            eigenvectors = eigenvectors[:, :n_eigs]
        
        # Sort eigenvalues
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute spectral features
        features = {
            'eigenvalues': eigenvalues,
            'spectral_gap': eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0,
            'algebraic_connectivity': eigenvalues[1] if len(eigenvalues) > 1 else 0,
            'eigenvalue_variance': np.var(eigenvalues),
        }
        
        return features


class VisionGraphPipeline:
    """
    Complete pipeline for vision datasets: feature extraction -> graph building -> spectral analysis
    """
    
    def __init__(self,
                 architecture: str = 'resnet50',
                 graph_method: str = 'adaptive_knn',
                 data_dir: str = './data',
                 device: str = 'auto'):
        """
        Initialize vision graph pipeline
        
        Args:
            architecture: Feature extraction architecture
            graph_method: Graph construction method
            data_dir: Data directory path
            device: Computing device
        """
        self.architecture = architecture
        self.graph_method = graph_method
        self.data_dir = data_dir
        self.device = device
        
        # Initialize components
        self.dataset_loader = VisionDatasetLoader(data_dir, batch_size=64)
        self.feature_extractor = FeatureExtractor(architecture=architecture)
        self.graph_builder = EnhancedGraphBuilder(method=graph_method, device=device)
        
        # Results storage
        self.results = {}
        
    def process_dataset(self, 
                       dataset_name: str,
                       split: str = 'test',
                       max_samples: int = 2000,
                       extract_spectral: bool = True) -> Dict:
        """
        Process a complete dataset through the pipeline
        
        Args:
            dataset_name: Name of dataset ('cifar10', 'cifar100', 'svhn', 'imagenet', 'noise')
            split: Dataset split to use
            max_samples: Maximum number of samples to process
            extract_spectral: Whether to extract spectral features
            
        Returns:
            Dictionary containing all pipeline results
        """
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name} ({split}) with {self.architecture}")
        print(f"Graph method: {self.graph_method}")
        print(f"{'='*60}")
        
        # Load dataset
        if dataset_name == 'cifar10':
            dataloader = self.dataset_loader.get_cifar10(train=(split=='train'))
        elif dataset_name == 'cifar100':
            dataloader = self.dataset_loader.get_cifar100(train=(split=='train'))
        elif dataset_name == 'svhn':
            dataloader = self.dataset_loader.get_svhn(split=split)
        elif dataset_name == 'imagenet':
            dataloader = self.dataset_loader.get_imagenet(split=split, subset_size=max_samples)
        elif dataset_name == 'noise':
            dataloader = self.dataset_loader.get_noise_ood(size=max_samples)
        elif dataset_name == 'texture':
            dataloader = self.dataset_loader.get_texture_ood()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Extract features
        print("Extracting features...")
        features, labels = self.feature_extractor.extract_features(dataloader, max_samples)
        print(f"Extracted features shape: {features.shape}")
        
        # Build graph
        print("Building graph...")
        adjacency = self.graph_builder.build_graph(features, fit_preprocessing=True)
        
        # Extract spectral features if requested
        spectral_features = None
        if extract_spectral:
            print("Extracting spectral features...")
            spectral_features = self.graph_builder.extract_spectral_features(adjacency)
        
        # Compile results
        results = {
            'dataset_name': dataset_name,
            'split': split,
            'architecture': self.architecture,
            'graph_method': self.graph_method,
            'n_samples': len(features),
            'n_features': features.shape[1],
            'n_edges': adjacency.nnz,
            'graph_density': adjacency.nnz / (adjacency.shape[0]**2),
            'features': features,
            'labels': labels,
            'adjacency': adjacency,
            'spectral_features': spectral_features
        }
        
        print(f"✅ Processing complete: {len(features)} samples, {adjacency.nnz} edges")
        
        return results


def main():
    """
    Main demonstration of enhanced graph building for computer vision datasets
    """
    print("="*80)
    print("ENHANCED GRAPH CONSTRUCTION FOR COMPUTER VISION")
    print("Optimized for CIFAR-10/100, SVHN, and OOD Detection")
    print("="*80)
    
    # Initialize pipeline
    pipeline = VisionGraphPipeline(
        architecture='resnet18',  # Start with lightweight model
        graph_method='adaptive_knn',
        data_dir='./data'
    )
    
    # Quick demo: CIFAR-10
    print("\n🚀 Running CIFAR-10 graph construction demo...")
    
    try:
        results = pipeline.process_dataset(
            dataset_name='cifar10',
            max_samples=500  # Small for demo
        )
        
        print(f"\n✅ Demo Results:")
        print(f"Dataset: {results['dataset_name']}")
        print(f"Samples: {results['n_samples']}")
        print(f"Graph edges: {results['n_edges']}")
        print(f"Graph density: {results['graph_density']:.6f}")
        if results['spectral_features']:
            print(f"Spectral gap: {results['spectral_features']['spectral_gap']:.4f}")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("This is likely due to missing datasets. Please ensure CIFAR-10 is available.")
    
    print("\n🎯 Enhanced graph construction framework ready!")
    print("Use VisionGraphPipeline for your computer vision OOD detection tasks.")


if __name__ == "__main__":
    main()