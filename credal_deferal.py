# credal_deferral.py
"""
Credal deferral model for LLM answer correctness.

Takes frozen LLM hidden states, runs an ensemble of lightweight probes
trained to predict answer correctness, and constructs credible intervals
over the ensemble's predicted class probabilities as a credal
representation of model uncertainty.

The credal bounds feed into decision rules (gamma-maximin, maximality,
E-admissibility; see decision_rules.py) which produce typed deferral
decisions, and into conformal calibration (see conformal.py) which
provides coverage guarantees.

No concepts, no aleatoric head, no CBM machinery. This is a deliberately
lean model focused on one thing: producing meaningful credal bounds from
ensemble disagreement over frozen LLM features.

Author: Tanmoy
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# HEAD CONFIGURATION
# ============================================================================
# Diverse probe configurations drive ensemble disagreement (epistemic
# uncertainty). We vary three axes: hidden dimension (analogous to LoRA
# rank), dropout rate, and pooling strategy.

@dataclass
class HeadConfig:
    """Configuration for a single probe in the ensemble.

    The three diversity axes:
      hidden_dim:    capacity of the probe. Low cap -> smoother predictions;
                     high cap -> sharper. Heterogeneity here captures
                     capacity-dependent uncertainty.
      dropout_rate:  probability of dropout during training. Heterogeneity
                     here captures representational noise robustness.
      pooling:       which token position to pool from hidden states.
                     'cls', 'mean', 'last'. For decoder-only LLMs use 'last'.
    """
    hidden_dim: int = 256
    dropout_rate: float = 0.1
    pooling: str = "last"


def generate_head_configs(
    n_heads: int = 10,
    dropout_min: float = 0.05,
    dropout_max: float = 0.30,
    hidden_dims: Optional[List[int]] = None,
    pooling: str = "last",
    use_pooling_diversity: bool = False,
) -> List[HeadConfig]:
    """Construct diverse head configurations.

    Hidden dimensions follow a geometric progression analogous to CREDENCE's
    LoRA rank diversity (4, 8, 16, 32, 64 -> 128, 192, 256, 384, 512, ...).
    Dropout rates are geometrically spaced in keep-probability space, which
    CREDENCE's ablations show produces more functional diversity than linear
    spacing.

    Args:
        n_heads:               number of probes in the ensemble.
                               Default 10 to make quantile-based credible
                               intervals statistically meaningful.
        dropout_min/max:       range for dropout rates.
        hidden_dims:           optional override for hidden dimensions.
                               If None, defaults to a geometric progression.
        pooling:               default pooling strategy.
        use_pooling_diversity: if True, vary pooling across heads
                               (only applicable for encoder models).

    Returns:
        List of HeadConfig, length n_heads.
    """
    if hidden_dims is None:
        # Geometric progression, doubled from CREDENCE's 4,8,16,32,64 spirit.
        # For n_heads=10: [128, 160, 192, 224, 256, 320, 384, 448, 512, 640]
        # The precise values matter less than the spread.
        base_dims = [128, 160, 192, 224, 256, 320, 384, 448, 512, 640]
        if n_heads <= len(base_dims):
            hidden_dims = base_dims[:n_heads]
        else:
            # If more heads requested, repeat with small perturbations
            hidden_dims = base_dims + [
                base_dims[i % len(base_dims)] + (i // len(base_dims)) * 32
                for i in range(n_heads - len(base_dims))
            ]

    # Geometric dropout spacing in keep-probability space
    # d_h = 1 - exp(log(1-d_min) + h/(H-1) * [log(1-d_max) - log(1-d_min)])
    import math
    if n_heads == 1:
        dropouts = [(dropout_min + dropout_max) / 2]
    else:
        log_k_min = math.log(1 - dropout_min)
        log_k_max = math.log(1 - dropout_max)
        dropouts = [
            1 - math.exp(log_k_min + h / (n_heads - 1) * (log_k_max - log_k_min))
            for h in range(n_heads)
        ]

    # Pooling: either all same, or rotate through choices
    if use_pooling_diversity:
        pool_options = ["cls", "mean", "last"]
        poolings = [pool_options[h % len(pool_options)] for h in range(n_heads)]
    else:
        poolings = [pooling] * n_heads

    return [
        HeadConfig(
            hidden_dim=hidden_dims[h],
            dropout_rate=dropouts[h],
            pooling=poolings[h],
        )
        for h in range(n_heads)
    ]


# ============================================================================
# PROBE (LABEL HEAD)
# ============================================================================

class LabelHead(nn.Module):
    """Single probe predicting label logits from pooled hidden states.

    A probe is a two-layer MLP on top of pooled frozen-encoder features.
    Diverse probes (varying capacity and dropout) produce diverse
    predictions whose spread captures epistemic uncertainty.
    """

    def __init__(self, config: HeadConfig, input_dim: int, num_classes: int):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout_rate)
        self.fc1 = nn.Linear(input_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, num_classes)

    def pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool sequence of hidden states to a single vector per example."""
        if self.config.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.config.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(
                min=1e-9
            )
        elif self.config.pooling == "last":
            seq_lens = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            return hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                seq_lens,
            ]
        raise ValueError(f"Unknown pooling: {self.config.pooling}")

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, softmax_probs) for this probe.

        Returns logits in dtype of the probe parameters. Softmax is
        computed in FP32 for numerical stability, then cast back.
        """
        x = self.pool(hidden_states, attention_mask)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits.float(), dim=-1).to(logits.dtype)
        return logits, probs

class CredalEnsemble(nn.Module):
    """Ensemble of LabelHeads that computes credal intervals."""

    def __init__(self, configs: List[HeadConfig], input_dim: int, num_classes: int):
        super().__init__()
        self.heads = nn.ModuleList([
            LabelHead(config, input_dim, num_classes) for config in configs
        ])
        
    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for all heads.
        
        Returns:
            all_logits: Tensor of shape (batch, n_heads, num_classes)
            all_probs: Tensor of shape (batch, n_heads, num_classes)
        """
        logits_list = []
        probs_list = []
        
        for head in self.heads:
            l, p = head(hidden_states, attention_mask)
            logits_list.append(l)
            probs_list.append(p)
            
        all_logits = torch.stack(logits_list, dim=1)
        all_probs = torch.stack(probs_list, dim=1)
        
        return all_logits, all_probs
        
    def get_credal_bounds(
        self, all_probs: torch.Tensor, alpha: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute empirical credal intervals over the ensemble of probabilities.
        
        Args:
            all_probs: Tensor of shape (batch_size, n_heads, num_classes)
            alpha: Significance level for the interval (e.g. 0.1 for 90% bounds)
            
        Returns:
            lower_bound: Tensor of shape (batch_size, num_classes)
            upper_bound: Tensor of shape (batch_size, num_classes)
        """
        lower_bound = torch.quantile(all_probs, alpha / 2.0, dim=1)
        upper_bound = torch.quantile(all_probs, 1.0 - alpha / 2.0, dim=1)
        
        return lower_bound, upper_bound