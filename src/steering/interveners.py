from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class BaseIntervener(ABC):
    """Abstract base class for activation intervention operators."""

    @abstractmethod
    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply intervention to hidden state tensor."""
        pass


class AdditiveIntervener(BaseIntervener):
    """Additive steering: h' = h + coefficient * vector."""

    def __init__(
        self,
        vector: torch.Tensor,
        coefficient: float = 1.0,
        normalize: bool = False,
        token_indices: Optional[Union[int, List[int]]] = None
    ):
        self.vector = vector
        self.coefficient = coefficient
        self.normalize = normalize
        self.token_indices = token_indices

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Move vector to tensor device & dtype
        v = self.vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if self.normalize:
            norm = torch.norm(v, p=2)
            if norm > 1e-9:
                v = v / norm

        steer = self.coefficient * v

        if hidden_states.dim() == 3:
            # [batch_size, seq_len, hidden_dim]
            if self.token_indices is None:
                # Steer all tokens
                return hidden_states + steer.view(1, 1, -1)
            elif isinstance(self.token_indices, int):
                out = hidden_states.clone()
                out[:, self.token_indices, :] += steer.view(1, -1)
                return out
            else:
                out = hidden_states.clone()
                for idx in self.token_indices:
                    out[:, idx, :] += steer.view(1, -1)
                return out
        elif hidden_states.dim() == 2:
            # [batch_size, hidden_dim]
            return hidden_states + steer.view(1, -1)
        elif hidden_states.dim() == 4:
            # [batch_size, channels, H, W]
            return hidden_states + steer.view(1, -1, 1, 1)
        else:
            return hidden_states + steer


class OrthogonalProjectionIntervener(BaseIntervener):
    """Ablates or removes the steering direction component: h' = h - (h . v_norm) * v_norm."""

    def __init__(self, vector: torch.Tensor):
        self.vector = vector

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        v = self.vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        norm = torch.norm(v, p=2)
        if norm < 1e-9:
            return hidden_states
        v_unit = v / norm

        if hidden_states.dim() == 3:
            # [B, S, D]
            # Proj = sum(H * v_unit, dim=-1, keepdim=True) * v_unit
            proj = (hidden_states * v_unit.view(1, 1, -1)).sum(dim=-1, keepdim=True) * v_unit.view(1, 1, -1)
            return hidden_states - proj
        elif hidden_states.dim() == 2:
            proj = (hidden_states * v_unit.view(1, -1)).sum(dim=-1, keepdim=True) * v_unit.view(1, -1)
            return hidden_states - proj
        else:
            return hidden_states


class SubspaceClampingIntervener(BaseIntervener):
    """Clamps the projection magnitude along steering direction within [min_val, max_val]."""

    def __init__(self, vector: torch.Tensor, min_val: float = -2.0, max_val: float = 2.0):
        self.vector = vector
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        v = self.vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        norm = torch.norm(v, p=2)
        if norm < 1e-9:
            return hidden_states
        v_unit = v / norm

        if hidden_states.dim() == 3:
            dot = (hidden_states * v_unit.view(1, 1, -1)).sum(dim=-1, keepdim=True)
            clamped_dot = torch.clamp(dot, min=self.min_val, max=self.max_val)
            delta = (clamped_dot - dot) * v_unit.view(1, 1, -1)
            return hidden_states + delta
        elif hidden_states.dim() == 2:
            dot = (hidden_states * v_unit.view(1, -1)).sum(dim=-1, keepdim=True)
            clamped_dot = torch.clamp(dot, min=self.min_val, max=self.max_val)
            delta = (clamped_dot - dot) * v_unit.view(1, -1)
            return hidden_states + delta
        return hidden_states


class GatedIntervener(BaseIntervener):
    """Applies steering conditionally based on activation magnitude or cosine similarity."""

    def __init__(
        self,
        vector: torch.Tensor,
        coefficient: float = 1.0,
        threshold: float = 0.5
    ):
        self.vector = vector
        self.coefficient = coefficient
        self.threshold = threshold

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        v = self.vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        v_norm = torch.norm(v, p=2)
        if v_norm < 1e-9:
            return hidden_states
        v_unit = v / v_norm

        if hidden_states.dim() == 3:
            h_norm = torch.norm(hidden_states, p=2, dim=-1, keepdim=True).clamp(min=1e-9)
            cosine = (hidden_states * v_unit.view(1, 1, -1)).sum(dim=-1, keepdim=True) / h_norm
            # Gate is active when similarity exceeds threshold
            gate = torch.sigmoid((cosine - self.threshold) * 5.0)
            steer = self.coefficient * v_unit.view(1, 1, -1)
            return hidden_states + (gate * steer)
        elif hidden_states.dim() == 2:
            h_norm = torch.norm(hidden_states, p=2, dim=-1, keepdim=True).clamp(min=1e-9)
            cosine = (hidden_states * v_unit.view(1, -1)).sum(dim=-1, keepdim=True) / h_norm
            gate = torch.sigmoid((cosine - self.threshold) * 5.0)
            steer = self.coefficient * v_unit.view(1, -1)
            return hidden_states + (gate * steer)
        return hidden_states
