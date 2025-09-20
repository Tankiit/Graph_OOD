"""
Unified OOD Detection Framework
Integrates enhanced graph construction with existing spectral OOD detection
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Import components
from enhanced_graph_builder import EnhancedGraphBuilder, VisionGraphPipeline
from spectral_ood_vision import ImageSpectralOODDetector, VisionDatasetLoader, FeatureExtractor
import json
from sklearn.metrics import roc_auc_score, average_precision_score


class UnifiedSpectralOODDetector:
    """
    Unified framework combining enhanced graph construction with spectral OOD detection
    """
    
    def __init__(self,
                 architecture: str = 'resnet50',
                 graph_method: str = 'adaptive_knn',
                 spectral_method: str = 'unified',
                 k_neighbors: int = 10,
                 pca_components: int = 256,
                 device: str = 'auto'):
        """
        Initialize unified spectral OOD detector
        
        Args:
            architecture: Feature extraction architecture
            graph_method: Graph construction method
            spectral_method: Spectral analysis method
            k_neighbors: Number of neighbors for graph construction
            pca_components: PCA dimensionality reduction
            device: Computing device
        """
        self.architecture = architecture
        self.graph_method = graph_method
        self.spectral_method = spectral_method
        self.k_neighbors = k_neighbors
        self.pca_components = pca_components
        self.device = device
        
        # Initialize components
        self.feature_extractor = FeatureExtractor(architecture=architecture)
        self.graph_builder = EnhancedGraphBuilder(
            method=graph_method,
            k_neighbors=k_neighbors,
            pca_components=pca_components,
            device=device
        )
        self.spectral_detector = ImageSpectralOODDetector(
            method=spectral_method,
            k_neighbors=k_neighbors,
            pca_dim=pca_components
        )
        
        # Training data storage
        self.reference_features = None
        self.reference_spectral = None
        self.reference_graph = None
        self.is_fitted = False
        
    def fit(self, dataloader, max_samples: int = 2000):
        """
        Fit the unified detector on in-distribution data
        
        Args:
            dataloader: Training data loader
            max_samples: Maximum samples to use for training
        """
        # Extract features
        features, labels = self.feature_extractor.extract_features(dataloader, max_samples)
        
        # Build graph and extract spectral features
        adjacency = self.graph_builder.build_graph(features, fit_preprocessing=True)
        spectral_features = self.graph_builder.extract_spectral_features(adjacency)
        
        # Fit spectral detector
        self.spectral_detector.fit(features)
        
        # Store reference data
        self.reference_features = features
        self.reference_spectral = spectral_features
        self.reference_graph = adjacency
        self.is_fitted = True
        
        return self
    
    def predict_score(self, dataloader, max_samples: int = 1000) -> Dict:
        """
        Predict OOD scores for test data
        
        Args:
            dataloader: Test data loader
            max_samples: Maximum samples to process
            
        Returns:
            Dictionary of OOD scores
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        # Extract test features
        test_features, test_labels = self.feature_extractor.extract_features(dataloader, max_samples)
        
        # Method 1: Standard spectral OOD scores
        spectral_scores = self.spectral_detector.predict_score(test_features)
        
        # Method 2: Enhanced graph-based scores
        graph_scores = self._compute_graph_ood_scores(test_features)
        
        # Method 3: Combined approach
        combined_scores = 0.6 * spectral_scores + 0.4 * graph_scores
        
        return {
            'spectral_scores': spectral_scores,
            'graph_scores': graph_scores,
            'combined_scores': combined_scores,
            'features': test_features,
            'labels': test_labels
        }
    
    def _compute_graph_ood_scores(self, test_features: np.ndarray) -> np.ndarray:
        """
        Compute OOD scores using enhanced graph analysis
        
        Args:
            test_features: Test feature matrix
            
        Returns:
            Array of graph-based OOD scores
        """
        # Build graph for test data
        test_adjacency = self.graph_builder.build_graph(test_features, fit_preprocessing=False)
        test_spectral = self.graph_builder.extract_spectral_features(test_adjacency)
        
        # Compare with reference spectral features
        gap_deviation = abs(test_spectral['spectral_gap'] - self.reference_spectral['spectral_gap'])
        
        # Return same score for all samples in batch (could be made sample-specific)
        return np.full(test_features.shape[0], gap_deviation)
    
    def evaluate(self, id_dataloader, ood_dataloader, 
                max_samples: int = 1000) -> Dict:
        """
        Comprehensive evaluation on ID and OOD data
        
        Args:
            id_dataloader: In-distribution test data
            ood_dataloader: Out-of-distribution test data
            max_samples: Maximum samples per dataset
            
        Returns:
            Evaluation results dictionary
        """
        # Get predictions for ID data
        id_results = self.predict_score(id_dataloader, max_samples)
        id_labels = np.zeros(len(id_results['combined_scores']))
        
        # Get predictions for OOD data
        ood_results = self.predict_score(ood_dataloader, max_samples // 2)
        ood_labels = np.ones(len(ood_results['combined_scores']))
        
        # Combine results
        all_labels = np.concatenate([id_labels, ood_labels])
        
        # Evaluate different scoring methods
        evaluation = {}
        
        for score_type in ['spectral_scores', 'graph_scores', 'combined_scores']:
            scores = np.concatenate([id_results[score_type], ood_results[score_type]])
            
            auc = roc_auc_score(all_labels, scores)
            ap = average_precision_score(all_labels, scores)
            
            # FPR at 95% TPR
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(all_labels, scores)
            tpr95_idx = np.where(tpr >= 0.95)[0]
            fpr95 = fpr[tpr95_idx[0]] if len(tpr95_idx) > 0 else 1.0
            
            evaluation[score_type] = {
                'auc': auc,
                'average_precision': ap,
                'fpr95': fpr95
            }
        
        # Additional metadata
        evaluation['metadata'] = {
            'architecture': self.architecture,
            'graph_method': self.graph_method,
            'spectral_method': self.spectral_method,
            'n_id_samples': len(id_labels),
            'n_ood_samples': len(ood_labels)
        }
        
        return evaluation


def main():
    """
    Main execution for unified OOD detection framework
    """
    try:
        # Initialize detector
        detector = UnifiedSpectralOODDetector(
            architecture='resnet18',
            graph_method='adaptive_knn',
            spectral_method='unified'
        )
        
        # Load data
        dataset_loader = VisionDatasetLoader('./data', batch_size=64)
        
        # Demo with small dataset
        id_train_loader = dataset_loader.get_cifar10(train=True)
        id_test_loader = dataset_loader.get_cifar10(train=False)
        ood_loader = dataset_loader.get_noise_ood(size=200)
        
        # Fit and evaluate
        detector.fit(id_train_loader, max_samples=400)
        results = detector.evaluate(id_test_loader, ood_loader, max_samples=200)
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Demo error: {e}. Check datasets and dependencies.")


if __name__ == "__main__":
    main()