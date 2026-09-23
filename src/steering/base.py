from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import torch


@dataclass
class SteeringVector:
    """Represents one or more steering vectors across model layers with metadata."""

    vectors: Dict[str, torch.Tensor]  # layer_name -> 1D or 2D steering vector
    concept: str = "general"
    method: str = "mean_difference"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalize(self, p: float = 2.0) -> "SteeringVector":
        """Return a new SteeringVector where each layer's vector is normalized to unit norm."""
        normalized = {}
        for layer, vec in self.vectors.items():
            norm = torch.norm(vec.float(), p=p)
            if norm > 1e-9:
                normalized[layer] = vec / norm
            else:
                normalized[layer] = vec.clone()
        return SteeringVector(
            vectors=normalized,
            concept=self.concept,
            method=self.method,
            metadata=self.metadata.copy()
        )

    def scale(self, factor: float) -> "SteeringVector":
        """Return a new SteeringVector scaled by factor."""
        scaled = {layer: vec * factor for layer, vec in self.vectors.items()}
        return SteeringVector(
            vectors=scaled,
            concept=self.concept,
            method=self.method,
            metadata=self.metadata.copy()
        )

    def to(self, device: Union[torch.device, str]) -> "SteeringVector":
        """Move all underlying vector tensors to target device."""
        dev = torch.device(device) if isinstance(device, str) else device
        moved = {layer: vec.to(dev) for layer, vec in self.vectors.items()}
        return SteeringVector(
            vectors=moved,
            concept=self.concept,
            method=self.method,
            metadata=self.metadata.copy()
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save steering vectors and metadata to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Save as PyTorch checkpoint
        payload = {
            "vectors": {k: v.cpu() for k, v in self.vectors.items()},
            "concept": self.concept,
            "method": self.method,
            "metadata": self.metadata
        }
        torch.save(payload, p)

    @classmethod
    def load(cls, path: Union[str, Path], map_location: str = "cpu") -> "SteeringVector":
        """Load steering vectors from disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Steering vector file not found: {p}")
        payload = torch.load(p, map_location=map_location, weights_only=False)
        return cls(
            vectors=payload["vectors"],
            concept=payload.get("concept", "general"),
            method=payload.get("method", "unknown"),
            metadata=payload.get("metadata", {})
        )

    def __getitem__(self, layer_name: str) -> torch.Tensor:
        return self.vectors[layer_name]

    def __contains__(self, layer_name: str) -> bool:
        return layer_name in self.vectors

    @property
    def layer_names(self) -> List[str]:
        return list(self.vectors.keys())
