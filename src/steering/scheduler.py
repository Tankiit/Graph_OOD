import math
from typing import Dict, List, Optional, Union


class LayerScheduler:
    """Computes layer-wise steering strength multipliers."""

    def __init__(
        self,
        layer_names: List[str],
        schedule_type: str = "constant",
        base_coeff: float = 1.0,
        min_coeff: float = 0.0
    ):
        self.layer_names = layer_names
        self.schedule_type = schedule_type
        self.base_coeff = base_coeff
        self.min_coeff = min_coeff

    def get_coefficients(self) -> Dict[str, float]:
        """Compute multiplier dictionary mapping layer_name -> float coefficient."""
        n = len(self.layer_names)
        if n == 0:
            return {}

        coeffs = {}
        for idx, name in enumerate(self.layer_names):
            if self.schedule_type == "constant":
                val = self.base_coeff

            elif self.schedule_type == "linear_ramp":
                # Linear increase from min_coeff to base_coeff
                factor = idx / max(1, n - 1)
                val = self.min_coeff + factor * (self.base_coeff - self.min_coeff)

            elif self.schedule_type == "peak_middle":
                # Triangular peak in the middle layers
                mid = (n - 1) / 2.0
                dist = abs(idx - mid) / max(1.0, mid)
                factor = max(0.0, 1.0 - dist)
                val = self.min_coeff + factor * (self.base_coeff - self.min_coeff)

            elif self.schedule_type == "cosine":
                # Cosine bell curve peaking in middle
                angle = math.pi * (idx / max(1, n - 1))
                factor = math.sin(angle)
                val = self.min_coeff + factor * (self.base_coeff - self.min_coeff)

            else:
                val = self.base_coeff

            coeffs[name] = val

        return coeffs


class TokenPositionScheduler:
    """Manages token positions targeted during sequence steering."""

    @staticmethod
    def get_token_indices(
        mode: str = "all",
        seq_len: int = 1,
        prompt_len: Optional[int] = None
    ) -> Optional[List[int]]:
        """Resolve token indices for intervention.

        Args:
            mode: 'all', 'last', 'prompt', or 'generation'.
            seq_len: Total sequence length.
            prompt_len: Length of input prompt (if distinguishing prompt vs generated tokens).

        Returns:
            List of integer token indices or None for all tokens.
        """
        if mode == "all":
            return None
        elif mode == "last":
            return [max(0, seq_len - 1)]
        elif mode == "prompt":
            if prompt_len is None:
                return None
            return list(range(min(prompt_len, seq_len)))
        elif mode == "generation":
            if prompt_len is None:
                return None
            return list(range(prompt_len, seq_len))
        else:
            return None
