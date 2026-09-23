from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbe(nn.Module):
    """Linear classifier probe for hidden state representations."""

    def __init__(self, input_dim: int, num_classes: int = 2, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for input representations [B, D]."""
        return self.linear(x)

    def get_direction(self, class_idx: int = 1, normalize: bool = True) -> torch.Tensor:
        """Extract classification direction weight vector for steering.

        For binary classification (num_classes=2), returns W[1] - W[0] or W[class_idx].
        """
        w = self.linear.weight.data
        if self.num_classes == 2:
            direction = w[1] - w[0]
        else:
            direction = w[class_idx]

        if normalize:
            norm = torch.norm(direction, p=2)
            if norm > 1e-9:
                direction = direction / norm
        return direction


class MLPProbe(nn.Module):
    """Multi-layer non-linear probe for representation classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnergyProbe(nn.Module):
    """Energy-based OOD detector head computing Helmholtz free energy score."""

    def __init__(self, input_dim: int, num_classes: int, temperature: float = 1.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute free energy score: E(x) = -T * LogSumExp(logits / T).

        Higher energy indicates in-distribution; lower (more negative) indicates OOD.
        """
        logits = self.linear(x)
        # LogSumExp over classes
        energy = self.temperature * torch.logsumexp(logits / self.temperature, dim=-1)
        return energy


class MahalanobisDetector:
    """Empirical Gaussian covariance-based Mahalanobis OOD detector."""

    def __init__(self, num_classes: int, feature_dim: int):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_means: Optional[torch.Tensor] = None  # [C, D]
        self.precision: Optional[torch.Tensor] = None    # [D, D]

    def fit(self, features: torch.Tensor, labels: torch.Tensor, shrink_cov: float = 1e-4) -> None:
        """Fit empirical class-conditional means and tied covariance matrix.

        Args:
            features: [N, D] tensor of representations.
            labels: [N] tensor of class labels.
            shrink_cov: Ridge shrinkage parameter for covariance numerical stability.
        """
        device = features.device
        self.class_means = torch.zeros(self.num_classes, self.feature_dim, device=device)
        total_cov = torch.zeros(self.feature_dim, self.feature_dim, device=device)

        for c in range(self.num_classes):
            mask = (labels == c)
            class_feats = features[mask]
            if len(class_feats) == 0:
                continue
            mean_c = class_feats.mean(dim=0)
            self.class_means[c] = mean_c
            centered = class_feats - mean_c
            cov_c = torch.mm(centered.t(), centered)
            total_cov += cov_c

        total_cov /= max(1, len(features))
        # Add shrinkage / identity regularization
        identity = torch.eye(self.feature_dim, device=device)
        regularized_cov = total_cov + shrink_cov * identity
        self.precision = torch.linalg.pinv(regularized_cov)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis score = -min_c (x - mu_c)^T Sigma^-1 (x - mu_c).

        Higher score = more In-Distribution.
        """
        if self.class_means is None or self.precision is None:
            raise RuntimeError("MahalanobisDetector must be fitted before scoring.")

        device = features.device
        class_means = self.class_means.to(device)
        precision = self.precision.to(device)

        # features: [B, D]
        # class_means: [C, D]
        batch_size = features.shape[0]
        distances = torch.zeros(batch_size, self.num_classes, device=device)

        for c in range(self.num_classes):
            diff = features - class_means[c].unsqueeze(0)  # [B, D]
            # (diff @ precision) * diff summed over dim -1
            dist_c = torch.sum(torch.mm(diff, precision) * diff, dim=-1)
            distances[:, c] = dist_c

        min_distances, _ = torch.min(distances, dim=1)
        # Negative distance so that higher score is ID
        return -min_distances
