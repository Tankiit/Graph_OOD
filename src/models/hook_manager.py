from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
import torch
import torch.nn as nn


def _extract_tensor_from_output(output: Any) -> torch.Tensor:
    """Extract primary hidden state tensor from module output (handling tuples/custom objects)."""
    if isinstance(output, torch.Tensor):
        return output
    elif isinstance(output, (tuple, list)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
        return output[0]
    elif hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
        return output.last_hidden_state
    else:
        raise ValueError(f"Unable to extract hidden state tensor from module output of type {type(output)}")


def _replace_tensor_in_output(output: Any, new_tensor: torch.Tensor) -> Any:
    """Reconstruct module output structure replacing the primary hidden state tensor."""
    if isinstance(output, torch.Tensor):
        return new_tensor
    elif isinstance(output, tuple):
        return (new_tensor, *output[1:])
    elif isinstance(output, list):
        return [new_tensor, *output[1:]]
    else:
        return new_tensor


class HookManager:
    """Manages PyTorch forward hooks for activation caching and runtime steering interventions."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.active_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.cached_activations: Dict[str, List[torch.Tensor]] = {}

    def get_submodule(self, target_name: str) -> nn.Module:
        """Resolve a submodule by dot-separated path or named_modules lookup."""
        if not target_name or target_name == "":
            return self.model
        for name, mod in self.model.named_modules():
            if name == target_name:
                return mod
        raise KeyError(f"Submodule '{target_name}' not found in model hierarchy.")

    def clear(self) -> None:
        """Remove all active hooks and clear cached activations."""
        for handle in self.active_handles:
            handle.remove()
        self.active_handles.clear()
        self.cached_activations.clear()

    @contextmanager
    def capture_activations(
        self,
        layer_names: List[str],
        clone: bool = True,
        detach: bool = True
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """Context manager to record intermediate activations from specified layers.

        Args:
            layer_names: List of module names to hook.
            clone: Whether to clone tensors before storing.
            detach: Whether to detach tensors from computation graph.

        Yields:
            Dictionary mapping layer_name -> list or batched tensor of activations.
        """
        storage: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}
        handles: List[torch.utils.hooks.RemovableHandle] = []

        def make_hook(layer_name: str):
            def hook_fn(module: nn.Module, inputs: Any, output: Any):
                tensor = _extract_tensor_from_output(output)
                if detach:
                    tensor = tensor.detach()
                if clone:
                    tensor = tensor.clone()
                storage[layer_name].append(tensor)
            return hook_fn

        try:
            for name in layer_names:
                mod = self.get_submodule(name)
                h = mod.register_forward_hook(make_hook(name))
                handles.append(h)
                self.active_handles.append(h)
            yield storage
        finally:
            for h in handles:
                h.remove()
                if h in self.active_handles:
                    self.active_handles.remove(h)

    @contextmanager
    def apply_steering(
        self,
        interventions: Dict[str, Callable[[torch.Tensor], torch.Tensor]]
    ) -> Iterator[None]:
        """Context manager to apply runtime steering interventions on specified layers.

        Args:
            interventions: Dict mapping layer_name -> Callable intervention fn(tensor) -> steered_tensor.

        Yields:
            None
        """
        handles: List[torch.utils.hooks.RemovableHandle] = []

        def make_steering_hook(fn: Callable[[torch.Tensor], torch.Tensor]):
            def hook_fn(module: nn.Module, inputs: Any, output: Any):
                orig_tensor = _extract_tensor_from_output(output)
                steered_tensor = fn(orig_tensor)
                return _replace_tensor_in_output(output, steered_tensor)
            return hook_fn

        try:
            for layer_name, fn in interventions.items():
                mod = self.get_submodule(layer_name)
                h = mod.register_forward_hook(make_steering_hook(fn))
                handles.append(h)
                self.active_handles.append(h)
            yield
        finally:
            for h in handles:
                h.remove()
                if h in self.active_handles:
                    self.active_handles.remove(h)
