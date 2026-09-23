from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveRepresentationLoss(nn.Module):
    """Supervised / Contrastive InfoNCE loss for steering vectors and representation separation."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute SupCon loss for normalized features [N, D] and labels [N]."""
        device = features.device
        norm_features = F.normalize(features, p=2, dim=1)
        similarity = torch.matmul(norm_features, norm_features.t()) / self.temperature

        # Create positive mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(device)

        # Remove diagonal (self-similarity)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0], device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # Numerical stability
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Log prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp(min=1e-9))

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-9)
        loss = -mean_log_prob_pos.mean()
        return loss


class EnergyMarginLoss(nn.Module):
    """Energy margin loss for training OOD detectors to have low energy on ID and high on OOD.

    L = E(x_id) + max(0, margin - E(x_ood))
    """

    def __init__(self, margin: float = -5.0, temperature: float = 1.0):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        """Energy = -T * LogSumExp(logits / T)."""
        return -self.temperature * torch.logsumexp(logits / self.temperature, dim=-1)

    def forward(self, id_logits: torch.Tensor, ood_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        id_energy = self.compute_energy(id_logits)
        loss = id_energy.mean()

        if ood_logits is not None:
            ood_energy = self.compute_energy(ood_logits)
            ood_loss = torch.relu(ood_energy - self.margin).mean()
            loss = loss + ood_loss

        return loss
