"""
Spectral SchemaNet OOD Postprocessor

This module implements a spectral-based out-of-distribution detection method
that leverages graph spectral properties for OOD detection. It combines
spectral analysis with energy-based methods for robust OOD detection.

Key features:
1. Spectral-based OOD scoring using graph Laplacian properties
2. Integration with energy-based detection methods
3. Multi-scale spectral analysis
4. Graph construction from feature embeddings
5. Cheeger constant and spectral gap analysis
6. Adaptive thresholding based on spectral properties
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, csgraph
from scipy.sparse.linalg import eigsh
import logging
import warnings
warnings.filterwarnings('ignore')


class SpectralSchemaNetDetector:
    """
    Spectral SchemaNet OOD detector that uses graph spectral properties
    for out-of-distribution detection.
    
    The method constructs a k-NN graph from feature embeddings and uses
    spectral properties (eigenvalues, eigenvectors, Cheeger constant) to
    detect OOD samples.
    """
    
    def __init__(self, model, k=10, similarity_metric='cosine', 
                 spectral_method='cheeger', temperature=1.0, device='cuda'):
        """
        Args:
            model: Pre-trained neural network model
            k (int): Number of nearest neighbors for graph construction
            similarity_metric (str): Similarity metric for graph construction
            spectral_method (str): Spectral method to use ('cheeger', 'fiedler', 'spectral_gap')
            temperature (float): Temperature for energy computation
            device (str): Device for computations
        """
        self.model = model
        self.k = k
        self.similarity_metric = similarity_metric
        self.spectral_method = spectral_method
        self.temperature = temperature
        self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Spectral graph properties
        self.spectral_monitor = None
        self.feature_embeddings = None
        self.labels = None
        self.graph_properties = {}
        
        # OOD detection parameters
        self.threshold = None
        self.spectral_stats = {}
        
    def build_spectral_graph(self, train_dataloader, feature_layer=None):
        """
        Build spectral graph from training features
        
        Args:
            train_dataloader: DataLoader for training data
            feature_layer (str): Layer name for feature extraction
        
        Returns:
            dict: Graph construction statistics
        """
        # Extract features from training data
        self.feature_embeddings, self.labels = self._extract_features(
            train_dataloader, feature_layer
        )
        
        # Initialize spectral monitor
        from utils.spectral_monitor import SpectralMonitor
        self.spectral_monitor = SpectralMonitor(
            k=self.k,
            similarity_metric=self.similarity_metric,
            normalize=True,
            device=self.device
        )
        
        # Build graph
        graph_stats = self.spectral_monitor.build_graph(
            self.feature_embeddings, self.labels
        )
        
        # Store graph properties
        self.graph_properties = graph_stats
        
        # Compute spectral statistics for OOD detection
        self._compute_spectral_statistics()
        
        return graph_stats
    
    def _extract_features(self, dataloader, feature_layer=None):
        """Extract features from the specified layer"""
        self.model.eval()
        features = []
        labels = []
        
        # Hook for feature extraction
        hook_handle = None
        if feature_layer is not None:
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    features.append(output[0].detach().cpu())
                else:
                    features.append(output.detach().cpu())
            
            # Find and register hook
            for name, module in self.model.named_modules():
                if name == feature_layer:
                    hook_handle = module.register_forward_hook(hook_fn)
                    break
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    data, target = batch[0], batch[1]
                else:
                    data = batch
                    target = None
                
                data = data.to(self.device)
                
                # Forward pass
                if feature_layer is not None:
                    _ = self.model(data)
                else:
                    # Use logits as features
                    logits = self.model(data)
                    features.append(logits.cpu())
                
                if target is not None:
                    labels.extend(target.numpy())
        
        # Remove hook
        if hook_handle is not None:
            hook_handle.remove()
        
        # Concatenate features
        if features:
            features = torch.cat(features, dim=0).numpy()
        else:
            features = np.array([])
        
        labels = np.array(labels) if labels else None
        
        return features, labels
    
    def _compute_spectral_statistics(self):
        """Compute spectral statistics for OOD detection"""
        if self.spectral_monitor is None:
            return
        
        # Get spectral properties
        summary = self.spectral_monitor.get_summary()
        
        # Store key statistics
        self.spectral_stats = {
            'spectral_gap': summary['spectral_properties']['spectral_gap'],
            'cheeger_constant': summary['spectral_properties']['cheeger_constant'],
            'n_components': summary['connectivity_metrics']['n_components'],
            'avg_degree': summary['graph_properties']['avg_degree'],
            'density': summary['graph_properties']['density']
        }
        
        # Compute baseline spectral scores for training data
        train_scores = self.spectral_monitor.compute_spectral_scores(
            self.feature_embeddings
        )
        
        self.spectral_stats['train_mean_score'] = np.mean(train_scores)
        self.spectral_stats['train_std_score'] = np.std(train_scores)
        self.spectral_stats['train_scores'] = train_scores
    
    def compute_spectral_scores(self, dataloader, feature_layer=None):
        """
        Compute spectral-based OOD scores
        
        Args:
            dataloader: DataLoader containing test data
            feature_layer (str): Layer name for feature extraction
        
        Returns:
            np.ndarray: Spectral scores (higher = more OOD)
        """
        if self.spectral_monitor is None:
            raise ValueError("Spectral graph not built. Call build_spectral_graph() first.")
        
        # Extract features
        features, _ = self._extract_features(dataloader, feature_layer)
        
        # Compute spectral scores
        spectral_scores = self.spectral_monitor.compute_spectral_scores(features)
        
        return spectral_scores
    
    def compute_energy_scores(self, dataloader):
        """
        Compute energy-based scores as baseline
        
        Args:
            dataloader: DataLoader containing test data
        
        Returns:
            np.ndarray: Energy scores
        """
        self.model.eval()
        energy_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                
                data = data.to(self.device)
                logits = self.model(data)
                
                # Compute energy scores
                energy = -self.temperature * torch.logsumexp(
                    logits / self.temperature, dim=1
                )
                energy_scores.extend(energy.cpu().numpy())
        
        return np.array(energy_scores)
    
    def compute_combined_scores(self, dataloader, feature_layer=None, 
                               spectral_weight=0.7, energy_weight=0.3):
        """
        Compute combined spectral and energy scores
        
        Args:
            dataloader: DataLoader containing test data
            feature_layer (str): Layer name for feature extraction
            spectral_weight (float): Weight for spectral scores
            energy_weight (float): Weight for energy scores
        
        Returns:
            np.ndarray: Combined scores
        """
        # Compute individual scores
        spectral_scores = self.compute_spectral_scores(dataloader, feature_layer)
        energy_scores = self.compute_energy_scores(dataloader)
        
        # Normalize scores to [0, 1] range
        spectral_scores_norm = self._normalize_scores(spectral_scores)
        energy_scores_norm = self._normalize_scores(energy_scores)
        
        # Combine scores
        combined_scores = (spectral_weight * spectral_scores_norm + 
                          energy_weight * energy_scores_norm)
        
        return combined_scores
    
    def _normalize_scores(self, scores):
        """Normalize scores to [0, 1] range"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.zeros_like(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized
    
    def detect_ood(self, dataloader, method='combined', threshold=None, 
                   feature_layer=None, **kwargs):
        """
        Detect OOD samples using spectral methods
        
        Args:
            dataloader: DataLoader containing test data
            method (str): Detection method ('spectral', 'energy', 'combined')
            threshold (float): Detection threshold
            feature_layer (str): Layer name for feature extraction
            **kwargs: Additional arguments for combined method
        
        Returns:
            dict: Detection results
        """
        # Compute scores based on method
        if method == 'spectral':
            scores = self.compute_spectral_scores(dataloader, feature_layer)
        elif method == 'energy':
            scores = self.compute_energy_scores(dataloader)
        elif method == 'combined':
            scores = self.compute_combined_scores(dataloader, feature_layer, **kwargs)
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        # Determine threshold
        if threshold is None:
            threshold = self._compute_adaptive_threshold(scores, method)
        
        # Make predictions
        ood_predictions = scores > threshold
        
        results = {
            'scores': scores,
            'ood_predictions': ood_predictions,
            'threshold': threshold,
            'ood_ratio': np.mean(ood_predictions),
            'method': method
        }
        
        return results
    
    def _compute_adaptive_threshold(self, scores, method):
        """Compute adaptive threshold based on spectral properties"""
        if method == 'spectral' and 'train_scores' in self.spectral_stats:
            # Use training data statistics
            train_mean = self.spectral_stats['train_mean_score']
            train_std = self.spectral_stats['train_std_score']
            threshold = train_mean + 2 * train_std
        else:
            # Use percentile-based threshold
            threshold = np.percentile(scores, 90)
        
        return threshold
    
    def evaluate_ood_detection(self, id_dataloader, ood_dataloader, 
                              method='combined', feature_layer=None, **kwargs):
        """
        Evaluate OOD detection performance
        
        Args:
            id_dataloader: DataLoader for ID data
            ood_dataloader: DataLoader for OOD data
            method (str): Detection method
            feature_layer (str): Layer name for feature extraction
            **kwargs: Additional arguments
        
        Returns:
            dict: Evaluation metrics
        """
        # Compute scores for both datasets
        if method == 'spectral':
            id_scores = self.compute_spectral_scores(id_dataloader, feature_layer)
            ood_scores = self.compute_spectral_scores(ood_dataloader, feature_layer)
        elif method == 'energy':
            id_scores = self.compute_energy_scores(id_dataloader)
            ood_scores = self.compute_energy_scores(ood_dataloader)
        elif method == 'combined':
            id_scores = self.compute_combined_scores(id_dataloader, feature_layer, **kwargs)
            ood_scores = self.compute_combined_scores(ood_dataloader, feature_layer, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Compute AUC
        y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        y_scores = np.concatenate([id_scores, ood_scores])
        
        auc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
        
        # Compute optimal threshold
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else np.median(y_scores)
        
        # Compute statistics
        id_stats = {
            'mean': np.mean(id_scores),
            'std': np.std(id_scores),
            'median': np.median(id_scores)
        }
        
        ood_stats = {
            'mean': np.mean(ood_scores),
            'std': np.std(ood_scores),
            'median': np.median(ood_scores)
        }
        
        results = {
            'auc': auc,
            'aupr': aupr,
            'optimal_threshold': optimal_threshold,
            'id_scores': id_scores,
            'ood_scores': ood_scores,
            'id_statistics': id_stats,
            'ood_statistics': ood_stats,
            'method': method,
            'spectral_stats': self.spectral_stats
        }
        
        return results
    
    def analyze_spectral_properties(self, dataloader, feature_layer=None):
        """
        Analyze spectral properties of the dataset
        
        Args:
            dataloader: DataLoader containing data to analyze
            feature_layer (str): Layer name for feature extraction
        
        Returns:
            dict: Spectral analysis results
        """
        if self.spectral_monitor is None:
            raise ValueError("Spectral graph not built. Call build_spectral_graph() first.")
        
        # Extract features
        features, labels = self._extract_features(dataloader, feature_layer)
        
        # Compute spectral scores
        spectral_scores = self.spectral_monitor.compute_spectral_scores(features)
        
        # Analyze connectivity
        connectivity_analysis = self.spectral_monitor.analyze_ood_samples(features, labels)
        
        # Get graph summary
        graph_summary = self.spectral_monitor.get_summary()
        
        results = {
            'spectral_scores': spectral_scores,
            'connectivity_analysis': connectivity_analysis,
            'graph_summary': graph_summary,
            'n_samples': len(features),
            'feature_dim': features.shape[1] if len(features) > 0 else 0
        }
        
        return results
    
    def get_detector_info(self):
        """Get information about the detector"""
        info = {
            'k': self.k,
            'similarity_metric': self.similarity_metric,
            'spectral_method': self.spectral_method,
            'temperature': self.temperature,
            'device': self.device,
            'graph_properties': self.graph_properties,
            'spectral_stats': self.spectral_stats
        }
        
        return info


def create_spectral_detector(model, detector_type='spectral_schemanet', **kwargs):
    """
    Factory function to create spectral-based detectors
    
    Args:
        model: Pre-trained neural network model
        detector_type (str): Type of detector
        **kwargs: Additional arguments for the detector
    
    Returns:
        Spectral detector instance
    """
    if detector_type == 'spectral_schemanet':
        return SpectralSchemaNetDetector(model, **kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


if __name__ == '__main__':
    # Test the spectral detector
    print("Testing SpectralSchemaNetDetector...")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 3)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create test data
    model = TestModel()
    X_train = torch.randn(200, 10)
    y_train = torch.randint(0, 3, (200,))
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
    
    X_test = torch.randn(100, 10)
    y_test = torch.randint(0, 3, (100,))
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    # Create spectral detector
    detector = SpectralSchemaNetDetector(model, k=5, device='cpu')
    
    # Build spectral graph
    graph_stats = detector.build_spectral_graph(train_dataloader)
    print(f"Graph built: {graph_stats}")
    
    # Test spectral scoring
    spectral_scores = detector.compute_spectral_scores(test_dataloader)
    print(f"Spectral scores shape: {spectral_scores.shape}")
    print(f"Spectral scores range: [{spectral_scores.min():.3f}, {spectral_scores.max():.3f}]")
    
    # Test OOD detection
    detection_results = detector.detect_ood(test_dataloader, method='spectral')
    print(f"OOD detection: {detection_results['ood_ratio']:.3f} OOD ratio")
    
    # Test evaluation
    eval_results = detector.evaluate_ood_detection(train_dataloader, test_dataloader, method='spectral')
    print(f"Evaluation AUC: {eval_results['auc']:.3f}")
    
    print("SpectralSchemaNetDetector test completed successfully!")
