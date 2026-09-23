from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


class BaseModelWrapper(nn.Module, ABC):
    """Abstract base class for all neural model wrappers with steering and extraction support."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_layer_names(self) -> List[str]:
        """Return list of candidate layer names available for activation hooking."""
        pass

    @abstractmethod
    def get_layer_module(self, layer_name: str) -> nn.Module:
        """Retrieve submodule corresponding to given layer name."""
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through underlying model."""
        pass

    @abstractmethod
    def extract_representations(
        self,
        batch: Any,
        layer_names: Optional[List[str]] = None,
        pooling: str = "last",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Extract hidden representations across specified layers.

        Args:
            batch: Model input batch.
            layer_names: List of layer names to extract from (defaults to all candidate layers).
            pooling: Token/spatial pooling method ('last', 'mean', 'cls', 'none').

        Returns:
            Dictionary mapping layer_name -> extracted Tensor.
        """
        pass
