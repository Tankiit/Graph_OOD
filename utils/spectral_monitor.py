"""
Spectral Monitor for Graph-based OOD Detection

This module provides comprehensive spectral analysis capabilities for monitoring
and analyzing the spectral properties of feature graphs in OOD detection.

Key features:
1. Graph construction from features using k-NN
2. Spectral analysis including eigenvalues, eigenvectors, and graph Laplacian
3. Cheeger constant computation for graph connectivity analysis
4. Spectral gap monitoring for OOD detection
5. Graph visualization and analysis tools
6. Support for different similarity metrics and graph construction methods
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, csgraph
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class SpectralMonitor:
    """
    Spectral monitoring and analysis for feature graphs
    """
    
    def __init__(self, k=10, similarity_metric='cosine', normalize=True, 
                 use_gpu=False, device='cuda'):
        """
        Args:
            k (int): Number of nearest neighbors for graph construction
            similarity_metric (str): Similarity metric ('cosine', 'euclidean', 'rbf')
            normalize (bool): Whether to normalize features
            use_gpu (bool): Whether to use GPU acceleration
            device (str): Device for computations
        """
        self.k = k
        self.similarity_metric = similarity_metric
        self.normalize = normalize
        self.use_gpu = use_gpu
        self.device = device
        
        # Graph properties
        self.adjacency_matrix = None
        self.laplacian_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.cheeger_constant = None
        
        # Feature properties
        self.features = None
        self.labels = None
        self.n_samples = 0
        self.n_features = 0
        
        # Analysis results
        self.spectral_gap = None
        self.connectivity_metrics = {}
        
    def build_graph(self, features, labels=None, k=None, rebuild=True):
        """
        Build k-NN graph from features
        
        Args:
            features (np.ndarray or torch.Tensor): Feature matrix (n_samples, n_features)
            labels (np.ndarray, optional): Class labels
            k (int, optional): Number of neighbors (overrides self.k)
            rebuild (bool): Whether to rebuild if graph already exists
        
        Returns:
            dict: Graph construction statistics
        """
        if self.adjacency_matrix is not None and not rebuild:
            return self._get_graph_stats()
        
        # Convert to numpy if needed
        if torch.is_tensor(features):
            features = features.cpu().numpy()
        
        if labels is not None and torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        # Normalize features
        if self.normalize:
            features = self._normalize_features(features)
        
        # Store features and labels
        self.features = features
        self.labels = labels
        self.n_samples, self.n_features = features.shape
        
        # Use provided k or default
        k = k if k is not None else self.k
        
        # Build adjacency matrix
        self.adjacency_matrix = self._build_adjacency_matrix(features, k)
        
        # Compute graph Laplacian
        self.laplacian_matrix = self._compute_laplacian()
        
        # Compute spectral properties
        self._compute_spectral_properties()
        
        # Compute connectivity metrics
        self._compute_connectivity_metrics()
        
        return self._get_graph_stats()
    
    def _normalize_features(self, features):
        """Normalize features to unit length"""
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return features / norms
    
    def _build_adjacency_matrix(self, features, k):
        """Build k-NN adjacency matrix"""
        if self.similarity_metric == 'cosine':
            # Use cosine similarity
            similarity_matrix = cosine_similarity(features)
            # Convert to distances (1 - similarity)
            distance_matrix = 1 - similarity_matrix
        elif self.similarity_metric == 'euclidean':
            distance_matrix = euclidean_distances(features)
        elif self.similarity_metric == 'rbf':
            # RBF kernel
            gamma = 1.0 / self.n_features
            distance_matrix = euclidean_distances(features)
            similarity_matrix = np.exp(-gamma * distance_matrix ** 2)
            distance_matrix = 1 - similarity_matrix
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Build k-NN graph
        knn = NearestNeighbors(n_neighbors=k+1, metric='precomputed')
        knn.fit(distance_matrix)
        
        # Get k-NN indices (excluding self)
        distances, indices = knn.kneighbors(distance_matrix)
        distances = distances[:, 1:]  # Remove self
        indices = indices[:, 1:]  # Remove self
        
        # Build adjacency matrix
        n = features.shape[0]
        adjacency = np.zeros((n, n))
        
        for i in range(n):
            for j, neighbor_idx in enumerate(indices[i]):
                adjacency[i, neighbor_idx] = 1.0
                adjacency[neighbor_idx, i] = 1.0  # Make symmetric
        
        return csr_matrix(adjacency)
    
    def _compute_laplacian(self):
        """Compute normalized graph Laplacian"""
        # Degree matrix
        degrees = np.array(self.adjacency_matrix.sum(axis=1)).flatten()
        
        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))  # Add small epsilon
        laplacian = np.eye(self.n_samples) - D_inv_sqrt @ self.adjacency_matrix.toarray() @ D_inv_sqrt
        
        return csr_matrix(laplacian)
    
    def _compute_spectral_properties(self):
        """Compute eigenvalues and eigenvectors of the Laplacian"""
        # Compute smallest eigenvalues
        n_eigenvals = min(50, self.n_samples - 1)  # Compute up to 50 eigenvalues
        
        try:
            eigenvals, eigenvecs = eigsh(self.laplacian_matrix, k=n_eigenvals, 
                                        sigma=0, which='LM', tol=1e-6)
            
            # Sort eigenvalues
            idx = np.argsort(eigenvals)
            self.eigenvalues = eigenvals[idx]
            self.eigenvectors = eigenvecs[:, idx]
            
            # Spectral gap (second smallest eigenvalue)
            if len(self.eigenvalues) > 1:
                self.spectral_gap = self.eigenvalues[1]
            else:
                self.spectral_gap = 0.0
                
        except Exception as e:
            print(f"Warning: Could not compute eigenvalues: {e}")
            self.eigenvalues = np.array([0.0])
            self.eigenvectors = np.eye(self.n_samples)
            self.spectral_gap = 0.0
    
    def _compute_connectivity_metrics(self):
        """Compute various connectivity metrics"""
        # Number of connected components
        n_components = csgraph.connected_components(self.adjacency_matrix, 
                                                   directed=False, return_labels=False)
        
        # Cheeger constant
        self.cheeger_constant = self._compute_cheeger_constant()
        
        # Store metrics
        self.connectivity_metrics = {
            'n_components': n_components,
            'cheeger_constant': self.cheeger_constant,
            'spectral_gap': self.spectral_gap,
            'n_nodes': self.n_samples,
            'n_edges': self.adjacency_matrix.nnz // 2,  # Undirected graph
            'avg_degree': self.adjacency_matrix.sum() / self.n_samples
        }
    
    def _compute_cheeger_constant(self):
        """Compute Cheeger constant using spectral approximation"""
        if self.spectral_gap is None or self.spectral_gap == 0:
            return 0.0
        
        # Cheeger's inequality: h(G) <= sqrt(2 * lambda_2)
        # where lambda_2 is the second smallest eigenvalue
        cheeger_upper = np.sqrt(2 * self.spectral_gap)
        
        # For a more accurate estimate, we can also use the Fiedler vector
        if self.eigenvectors is not None and self.eigenvectors.shape[1] > 1:
            fiedler_vector = self.eigenvectors[:, 1]  # Second eigenvector
            
            # Find the cut that minimizes the Cheeger ratio
            sorted_indices = np.argsort(fiedler_vector)
            n = len(fiedler_vector)
            
            min_ratio = float('inf')
            for i in range(1, n):
                S = sorted_indices[:i]
                S_complement = sorted_indices[i:]
                
                # Count edges between S and S_complement
                cut_edges = 0
                for node in S:
                    for neighbor in self.adjacency_matrix[node].indices:
                        if neighbor in S_complement:
                            cut_edges += 1
                
                # Compute Cheeger ratio
                if len(S) <= len(S_complement):
                    ratio = cut_edges / len(S)
                else:
                    ratio = cut_edges / len(S_complement)
                
                min_ratio = min(min_ratio, ratio)
            
            cheeger_constant = min_ratio
        else:
            cheeger_constant = cheeger_upper
        
        return cheeger_constant
    
    def _get_graph_stats(self):
        """Get graph construction statistics"""
        if self.adjacency_matrix is None:
            return {}
        
        return {
            'n_nodes': self.n_samples,
            'n_edges': self.adjacency_matrix.nnz // 2,
            'avg_degree': self.adjacency_matrix.sum() / self.n_samples,
            'density': (self.adjacency_matrix.nnz / 2) / (self.n_samples * (self.n_samples - 1) / 2),
            'spectral_gap': self.spectral_gap,
            'cheeger_constant': self.cheeger_constant
        }
    
    def compute_spectral_scores(self, features, normalize=True):
        """
        Compute spectral-based OOD scores for new features
        
        Args:
            features (np.ndarray or torch.Tensor): New features to score
            normalize (bool): Whether to normalize features
        
        Returns:
            np.ndarray: Spectral scores (higher = more OOD)
        """
        if self.laplacian_matrix is None or self.eigenvectors is None:
            raise ValueError("Graph not built. Call build_graph() first.")
        
        # Convert to numpy if needed
        if torch.is_tensor(features):
            features = features.cpu().numpy()
        
        # Normalize features
        if normalize:
            features = self._normalize_features(features)
        
        n_new_samples = features.shape[0]
        scores = np.zeros(n_new_samples)
        
        # For each new sample, compute its spectral embedding distance
        for i, feature in enumerate(features):
            # Compute distances to all training features
            if self.similarity_metric == 'cosine':
                distances = 1 - cosine_similarity([feature], self.features)[0]
            else:
                distances = euclidean_distances([feature], self.features)[0]
            
            # Find k nearest neighbors
            neighbor_indices = np.argsort(distances)[:self.k]
            
            # Compute spectral score based on Fiedler vector values
            if self.eigenvectors.shape[1] > 1:
                fiedler_values = self.eigenvectors[neighbor_indices, 1]
                # Score is the variance of Fiedler values of neighbors
                scores[i] = np.var(fiedler_values)
            else:
                scores[i] = 0.0
        
        return scores
    
    def analyze_ood_samples(self, ood_features, ood_labels=None, threshold=None):
        """
        Analyze OOD samples using spectral properties
        
        Args:
            ood_features (np.ndarray): OOD features
            ood_labels (np.ndarray, optional): OOD labels
            threshold (float, optional): Threshold for OOD detection
        
        Returns:
            dict: Analysis results
        """
        # Compute spectral scores
        scores = self.compute_spectral_scores(ood_features)
        
        # Compute statistics
        results = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'median_score': np.median(scores),
            'scores': scores
        }
        
        # Compare with ID distribution if available
        if self.features is not None:
            id_scores = self.compute_spectral_scores(self.features)
            results['id_mean_score'] = np.mean(id_scores)
            results['id_std_score'] = np.std(id_scores)
            results['score_separation'] = results['mean_score'] - results['id_mean_score']
        
        # Apply threshold if provided
        if threshold is not None:
            ood_predictions = scores > threshold
            results['ood_predictions'] = ood_predictions
            results['ood_ratio'] = np.mean(ood_predictions)
        
        return results
    
    def visualize_graph(self, max_nodes=1000, figsize=(12, 8)):
        """
        Visualize the graph structure
        
        Args:
            max_nodes (int): Maximum number of nodes to visualize
            figsize (tuple): Figure size
        """
        if self.adjacency_matrix is None:
            print("No graph to visualize. Call build_graph() first.")
            return
        
        # Subsample if too many nodes
        if self.n_samples > max_nodes:
            indices = np.random.choice(self.n_samples, max_nodes, replace=False)
            adj_sub = self.adjacency_matrix[np.ix_(indices, indices)].toarray()
            features_sub = self.features[indices]
            labels_sub = self.labels[indices] if self.labels is not None else None
        else:
            adj_sub = self.adjacency_matrix.toarray()
            features_sub = self.features
            labels_sub = self.labels
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Adjacency matrix heatmap
        sns.heatmap(adj_sub, cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Adjacency Matrix')
        axes[0, 0].set_xlabel('Node Index')
        axes[0, 0].set_ylabel('Node Index')
        
        # Degree distribution
        degrees = adj_sub.sum(axis=1)
        axes[0, 1].hist(degrees, bins=20, alpha=0.7)
        axes[0, 1].set_title('Degree Distribution')
        axes[0, 1].set_xlabel('Degree')
        axes[0, 1].set_ylabel('Frequency')
        
        # Feature visualization (if 2D or reduced)
        if features_sub.shape[1] == 2:
            if labels_sub is not None:
                scatter = axes[1, 0].scatter(features_sub[:, 0], features_sub[:, 1], 
                                           c=labels_sub, cmap='tab10', alpha=0.6)
                plt.colorbar(scatter, ax=axes[1, 0])
            else:
                axes[1, 0].scatter(features_sub[:, 0], features_sub[:, 1], alpha=0.6)
            axes[1, 0].set_title('Feature Space')
            axes[1, 0].set_xlabel('Feature 1')
            axes[1, 0].set_ylabel('Feature 2')
        else:
            # Show first two principal components
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features_sub)
            if labels_sub is not None:
                scatter = axes[1, 0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                           c=labels_sub, cmap='tab10', alpha=0.6)
                plt.colorbar(scatter, ax=axes[1, 0])
            else:
                axes[1, 0].scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6)
            axes[1, 0].set_title('PCA Projection')
            axes[1, 0].set_xlabel('PC1')
            axes[1, 0].set_ylabel('PC2')
        
        # Eigenvalue spectrum
        if self.eigenvalues is not None:
            axes[1, 1].plot(self.eigenvalues[:20], 'o-')
            axes[1, 1].set_title('Eigenvalue Spectrum')
            axes[1, 1].set_xlabel('Eigenvalue Index')
            axes[1, 1].set_ylabel('Eigenvalue')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_summary(self):
        """Get comprehensive summary of spectral analysis"""
        if self.adjacency_matrix is None:
            return "No graph built. Call build_graph() first."
        
        summary = {
            'graph_properties': self._get_graph_stats(),
            'connectivity_metrics': self.connectivity_metrics,
            'spectral_properties': {
                'spectral_gap': self.spectral_gap,
                'cheeger_constant': self.cheeger_constant,
                'n_eigenvalues_computed': len(self.eigenvalues) if self.eigenvalues is not None else 0
            }
        }
        
        return summary


def create_spectral_monitor(k=10, similarity_metric='cosine', normalize=True, 
                           use_gpu=False, device='cuda'):
    """
    Factory function to create SpectralMonitor
    
    Args:
        k (int): Number of nearest neighbors
        similarity_metric (str): Similarity metric
        normalize (bool): Whether to normalize features
        use_gpu (bool): Whether to use GPU
        device (str): Device for computations
    
    Returns:
        SpectralMonitor: Configured monitor
    """
    return SpectralMonitor(k=k, similarity_metric=similarity_metric, 
                          normalize=normalize, use_gpu=use_gpu, device=device)


if __name__ == '__main__':
    # Test the spectral monitor
    print("Testing SpectralMonitor...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    n_features = 50
    
    # Create two clusters
    cluster1 = np.random.randn(n_samples // 2, n_features) + np.array([2, 0] * (n_features // 2))
    cluster2 = np.random.randn(n_samples // 2, n_features) + np.array([0, 2] * (n_features // 2))
    features = np.vstack([cluster1, cluster2])
    labels = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)])
    
    # Create spectral monitor
    monitor = SpectralMonitor(k=10, similarity_metric='cosine')
    
    # Build graph
    stats = monitor.build_graph(features, labels)
    print(f"Graph built: {stats}")
    
    # Get summary
    summary = monitor.get_summary()
    print(f"Spectral analysis summary: {summary}")
    
    # Test OOD detection
    ood_features = np.random.randn(100, n_features) + np.array([5, 0] * (n_features // 2))
    ood_analysis = monitor.analyze_ood_samples(ood_features)
    print(f"OOD analysis: {ood_analysis}")
    
    print("SpectralMonitor test completed successfully!")
