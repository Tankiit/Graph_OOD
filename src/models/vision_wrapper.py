from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .base import BaseModelWrapper
from .hook_manager import HookManager
from ..utils.device import get_device, to_device


class VisionWrapper(BaseModelWrapper):
    """Wrapper for Vision backbones (ResNet, ViT, ConvNeXt, MLP) with intermediate layer hooks."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[torch.device, str]] = None,
        layer_names: Optional[List[str]] = None
    ):
        target_device = get_device(device) if isinstance(device, str) or device is None else device
        model = model.to(target_device)
        super().__init__(model)
        self.device = target_device
        self.hook_manager = HookManager(self.model)
        self._layer_names = layer_names or self._discover_vision_layers()

    def _discover_vision_layers(self) -> List[str]:
        """Automatically identify major stage/block layers in vision backbones."""
        candidates = []
        for name, mod in self.model.named_modules():
            # Match standard blocks: layer1, layer2, layer3, layer4, blocks.0, etc.
            parts = name.split(".")
            if len(parts) == 1 and parts[0].startswith("layer"):
                candidates.append(name)
            elif len(parts) >= 2 and parts[-1].isdigit() and parts[-2] in ("blocks", "stages", "layers"):
                candidates.append(name)

        if not candidates:
            # Fallback to all top-level child modules
            candidates = [name for name, _ in self.model.named_children()]

        return candidates

    def get_layer_names(self) -> List[str]:
        return list(self._layer_names)

    def get_layer_module(self, layer_name: str) -> nn.Module:
        return self.hook_manager.get_submodule(layer_name)

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def extract_representations(
        self,
        batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
        layer_names: Optional[List[str]] = None,
        pooling: str = "mean",
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Extract hidden representations across specified layers.

        Args:
            batch: Tensor [B, C, H, W] or dict with 'images'.
            layer_names: Subset of layer names (defaults to discovered candidate layers).
            pooling: 'mean' (spatial average pool) or 'none'.

        Returns:
            Dict mapping layer_name -> Tensor.
        """
        self.model.eval()
        target_layers = layer_names or self.get_layer_names()

        if isinstance(batch, dict) and "images" in batch:
            images = to_device(batch["images"], self.device)
        elif isinstance(batch, torch.Tensor):
            images = to_device(batch, self.device)
        else:
            raise ValueError(f"Invalid vision batch type: {type(batch)}")

        with torch.no_grad():
            with self.hook_manager.capture_activations(target_layers) as storage:
                _ = self.model(images)

        representations: Dict[str, torch.Tensor] = {}
        for name in target_layers:
            acts = storage[name]
            tensor = acts[0] if len(acts) == 1 else torch.cat(acts, dim=0)

            # Spatial pooling if 4D tensor [B, C, H, W]
            if tensor.dim() == 4 and pooling == "mean":
                tensor = tensor.mean(dim=[-2, -1])  # [B, C]
            elif tensor.dim() == 3 and pooling in ("mean", "cls"):
                # ViT token sequences [B, N, D]
                tensor = tensor[:, 0, :] if pooling == "cls" else tensor.mean(dim=1)

            representations[name] = tensor

        return representations
