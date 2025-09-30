"""
Wang & Li Energy-based OOD Detection Implementation

This module implements the energy-based out-of-distribution detection method
from "Energy-based Out-of-distribution Detection" (NeurIPS 2020).

Key features:
1. Energy score computation for OOD detection
2. Temperature scaling for calibration
3. Feature extraction from pre-trained models
4. Batch processing for efficiency
5. Integration with spectral analysis
6. Multiple energy variants (standard, modified, etc.)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
import warnings
warnings.filterwarnings('ignore')


class WangLiEnergyDetector:
    """
    Energy-based OOD Detection following Wang & Li (2020)
    
    The energy function is defined as:
    E(x) = -T * log(sum(exp(f(x)/T)))
    
    where f(x) are the logits and T is the temperature parameter.
    Lower energy indicates higher confidence (more likely to be ID).
    """
    
    def __init__(self, model, temperature=1.0, use_softmax=False, 
                 feature_layer=None, device='cuda'):
        """
        Args:
            model: Pre-trained neural network model
            temperature (float): Temperature parameter for energy computation
            use_softmax (bool): Whether to use softmax probabilities instead of logits
            feature_layer (str): Name of layer to extract features from (optional)
            device (str): Device for computations
        """
        self.model = model
        self.temperature = temperature
        self.use_softmax = use_softmax
        self.feature_layer = feature_layer
        self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Feature extraction setup
        self.features = None
        self.labels = None
        self.hook_handle = None
        
        # Energy statistics
        self.energy_stats = {}
        
    def _register_hook(self):
        """Register forward hook for feature extraction"""
        if self.feature_layer is None:
            return
        
        def hook_fn(module, input, output):
            # Store the output of the specified layer
            if isinstance(output, tuple):
                self.features = output[0].detach().cpu()
            else:
                self.features = output.detach().cpu()
        
        # Find the layer by name
        for name, module in self.model.named_modules():
            if name == self.feature_layer:
                self.hook_handle = module.register_forward_hook(hook_fn)
                break
        
        if self.hook_handle is None:
            print(f"Warning: Layer '{self.feature_layer}' not found. Using logits instead.")
    
    def _remove_hook(self):
        """Remove the registered hook"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def compute_energy_scores(self, dataloader, return_logits=False):
        """
        Compute energy scores for a dataset
        
        Args:
            dataloader: DataLoader containing the dataset
            return_logits (bool): Whether to return logits along with scores
        
        Returns:
            np.ndarray: Energy scores (lower = more ID-like)
            np.ndarray: Logits (if return_logits=True)
        """
        self.model.eval()
        energy_scores = []
        all_logits = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                
                data = data.to(self.device)
                
                # Forward pass
                if self.feature_layer is not None:
                    self._register_hook()
                
                logits = self.model(data)
                
                # Compute energy scores
                if self.use_softmax:
                    # Use softmax probabilities
                    probs = F.softmax(logits / self.temperature, dim=1)
                    energy = -self.temperature * torch.log(probs.sum(dim=1) + 1e-8)
                else:
                    # Use logits directly
                    energy = -self.temperature * torch.logsumexp(
                        logits / self.temperature, dim=1
                    )
                
                energy_scores.extend(energy.cpu().numpy())
                
                if return_logits:
                    all_logits.extend(logits.cpu().numpy())
        
        # Remove hook if registered
        self._remove_hook()
        
        energy_scores = np.array(energy_scores)
        
        if return_logits:
            return energy_scores, np.array(all_logits)
        else:
            return energy_scores
    
    def extract_features(self, dataloader, layer_name=None):
        """
        Extract features from a specified layer
        
        Args:
            dataloader: DataLoader containing the dataset
            layer_name (str): Name of layer to extract features from
        
        Returns:
            tuple: (features, labels) as numpy arrays
        """
        self.model.eval()
        features = []
        labels = []
        
        # Set up feature extraction
        original_feature_layer = self.feature_layer
        if layer_name is not None:
            self.feature_layer = layer_name
        
        # Register hook for feature extraction
        if self.feature_layer is not None:
            self._register_hook()
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    data, target = batch[0], batch[1]
                else:
                    data = batch
                    target = None
                
                data = data.to(self.device)
                
                # Forward pass
                logits = self.model(data)
                
                # Extract features if hook is registered
                if self.features is not None:
                    features.extend(self.features.numpy())
                
                if target is not None:
                    labels.extend(target.numpy())
        
        # Clean up
        self._remove_hook()
        self.feature_layer = original_feature_layer
        
        features = np.array(features) if features else None
        labels = np.array(labels) if labels else None
        
        return features, labels
    
    def extract_logits(self, dataloader):
        """
        Extract logits from the model
        
        Args:
            dataloader: DataLoader containing the dataset
        
        Returns:
            tuple: (logits, labels) as numpy arrays
        """
        self.model.eval()
        logits_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    data, target = batch[0], batch[1]
                else:
                    data = batch
                    target = None
                
                data = data.to(self.device)
                logits = self.model(data)
                
                logits_list.extend(logits.cpu().numpy())
                
                if target is not None:
                    labels_list.extend(target.numpy())
        
        logits = np.array(logits_list)
        labels = np.array(labels_list) if labels_list else None
        
        return logits, labels
    
    def compute_energy_statistics(self, id_scores, ood_scores):
        """
        Compute statistics for energy scores
        
        Args:
            id_scores (np.ndarray): Energy scores for ID data
            ood_scores (np.ndarray): Energy scores for OOD data
        
        Returns:
            dict: Statistics including separation metrics
        """
        stats = {
            'id_mean': np.mean(id_scores),
            'id_std': np.std(id_scores),
            'id_median': np.median(id_scores),
            'ood_mean': np.mean(ood_scores),
            'ood_std': np.std(ood_scores),
            'ood_median': np.median(ood_scores),
            'separation': np.mean(ood_scores) - np.mean(id_scores),
            'id_min': np.min(id_scores),
            'id_max': np.max(id_scores),
            'ood_min': np.min(ood_scores),
            'ood_max': np.max(ood_scores)
        }
        
        # Compute overlap metrics
        id_95_percentile = np.percentile(id_scores, 95)
        ood_5_percentile = np.percentile(ood_scores, 5)
        stats['overlap_metric'] = max(0, id_95_percentile - ood_5_percentile)
        
        return stats
    
    def detect_ood(self, dataloader, threshold=None, id_stats=None):
        """
        Detect OOD samples using energy scores
        
        Args:
            dataloader: DataLoader containing test data
            threshold (float): Energy threshold for OOD detection
            id_stats (dict): Statistics from ID data for threshold computation
        
        Returns:
            dict: Detection results
        """
        # Compute energy scores
        energy_scores = self.compute_energy_scores(dataloader)
        
        # Determine threshold
        if threshold is None:
            if id_stats is not None:
                # Use mean + 2*std as threshold
                threshold = id_stats['id_mean'] + 2 * id_stats['id_std']
            else:
                # Use median as threshold
                threshold = np.median(energy_scores)
        
        # Make predictions
        ood_predictions = energy_scores > threshold
        
        results = {
            'energy_scores': energy_scores,
            'ood_predictions': ood_predictions,
            'threshold': threshold,
            'ood_ratio': np.mean(ood_predictions)
        }
        
        return results
    
    def evaluate_ood_detection(self, id_dataloader, ood_dataloader):
        """
        Evaluate OOD detection performance
        
        Args:
            id_dataloader: DataLoader for ID data
            ood_dataloader: DataLoader for OOD data
        
        Returns:
            dict: Evaluation metrics
        """
        # Compute energy scores
        id_energy_scores = self.compute_energy_scores(id_dataloader)
        ood_energy_scores = self.compute_energy_scores(ood_dataloader)
        
        # Compute statistics
        energy_stats = self.compute_energy_statistics(id_energy_scores, ood_energy_scores)
        
        # Compute AUC
        y_true = np.concatenate([np.zeros(len(id_energy_scores)), 
                                np.ones(len(ood_energy_scores))])
        y_scores = np.concatenate([id_energy_scores, ood_energy_scores])
        
        # For energy scores, higher values indicate OOD
        # So we don't need to reverse the scores
        auc = roc_auc_score(y_true, y_scores)
        
        # Compute AUPR
        aupr = average_precision_score(y_true, y_scores)
        
        # Find optimal threshold
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else np.median(y_scores)
        
        results = {
            'auc': auc,
            'aupr': aupr,
            'optimal_threshold': optimal_threshold,
            'energy_statistics': energy_stats,
            'id_scores': id_energy_scores,
            'ood_scores': ood_energy_scores
        }
        
        return results
    
    def calibrate_temperature(self, calibration_dataloader, target_confidence=0.95):
        """
        Calibrate temperature parameter for better OOD detection
        
        Args:
            calibration_dataloader: DataLoader for calibration data
            target_confidence (float): Target confidence level
        
        Returns:
            float: Calibrated temperature
        """
        self.model.eval()
        confidences = []
        
        with torch.no_grad():
            for batch in calibration_dataloader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                
                data = data.to(self.device)
                logits = self.model(data)
                
                # Compute confidence (max softmax probability)
                probs = F.softmax(logits, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                confidences.extend(max_probs.cpu().numpy())
        
        confidences = np.array(confidences)
        
        # Find temperature that achieves target confidence
        # This is a simplified calibration - in practice, you might want
        # to use more sophisticated methods like Platt scaling
        
        # For now, we'll use a simple heuristic
        current_confidence = np.mean(confidences)
        if current_confidence > target_confidence:
            # Increase temperature to reduce confidence
            calibrated_temp = self.temperature * (current_confidence / target_confidence)
        else:
            # Decrease temperature to increase confidence
            calibrated_temp = self.temperature * (current_confidence / target_confidence)
        
        self.temperature = calibrated_temp
        
        return calibrated_temp
    
    def get_model_info(self):
        """Get information about the model and detector"""
        info = {
            'temperature': self.temperature,
            'use_softmax': self.use_softmax,
            'feature_layer': self.feature_layer,
            'device': self.device,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        return info


class ModifiedEnergyDetector(WangLiEnergyDetector):
    """
    Modified energy-based detector with additional variants
    """
    
    def __init__(self, model, temperature=1.0, energy_type='standard', 
                 use_softmax=False, feature_layer=None, device='cuda'):
        """
        Args:
            energy_type (str): Type of energy computation
                - 'standard': Original Wang & Li energy
                - 'modified': Modified energy with different normalization
                - 'maha': Mahalanobis-like energy
        """
        super().__init__(model, temperature, use_softmax, feature_layer, device)
        self.energy_type = energy_type
    
    def compute_energy_scores(self, dataloader, return_logits=False):
        """Compute energy scores with different variants"""
        self.model.eval()
        energy_scores = []
        all_logits = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    data = batch[0]
                else:
                    data = batch
                
                data = data.to(self.device)
                
                # Forward pass
                if self.feature_layer is not None:
                    self._register_hook()
                
                logits = self.model(data)
                
                # Compute energy scores based on type
                if self.energy_type == 'standard':
                    energy = -self.temperature * torch.logsumexp(
                        logits / self.temperature, dim=1
                    )
                elif self.energy_type == 'modified':
                    # Modified energy: E(x) = -log(sum(exp(f(x))))
                    energy = -torch.logsumexp(logits, dim=1)
                elif self.energy_type == 'maha':
                    # Mahalanobis-like energy using logit statistics
                    mean_logits = torch.mean(logits, dim=1, keepdim=True)
                    centered_logits = logits - mean_logits
                    energy = torch.sum(centered_logits ** 2, dim=1)
                else:
                    raise ValueError(f"Unknown energy type: {self.energy_type}")
                
                energy_scores.extend(energy.cpu().numpy())
                
                if return_logits:
                    all_logits.extend(logits.cpu().numpy())
        
        # Remove hook if registered
        self._remove_hook()
        
        energy_scores = np.array(energy_scores)
        
        if return_logits:
            return energy_scores, np.array(all_logits)
        else:
            return energy_scores


def create_energy_detector(model, detector_type='wang_li', **kwargs):
    """
    Factory function to create energy-based detectors
    
    Args:
        model: Pre-trained neural network model
        detector_type (str): Type of detector ('wang_li', 'modified')
        **kwargs: Additional arguments for the detector
    
    Returns:
        Energy detector instance
    """
    if detector_type == 'wang_li':
        return WangLiEnergyDetector(model, **kwargs)
    elif detector_type == 'modified':
        return ModifiedEnergyDetector(model, **kwargs)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


if __name__ == '__main__':
    # Test the energy detector
    print("Testing WangLiEnergyDetector...")
    
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
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    
    # Create energy detector
    detector = WangLiEnergyDetector(model, temperature=1.0, device='cpu')
    
    # Test energy computation
    energy_scores = detector.compute_energy_scores(dataloader)
    print(f"Energy scores shape: {energy_scores.shape}")
    print(f"Energy scores range: [{energy_scores.min():.3f}, {energy_scores.max():.3f}]")
    
    # Test feature extraction
    features, labels = detector.extract_features(dataloader)
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Test OOD detection
    ood_results = detector.detect_ood(dataloader, threshold=np.median(energy_scores))
    print(f"OOD detection results: {ood_results['ood_ratio']:.3f} OOD ratio")
    
    print("WangLiEnergyDetector test completed successfully!")
