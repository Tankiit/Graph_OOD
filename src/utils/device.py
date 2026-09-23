from typing import Any, Dict, List, Optional, Tuple, Union
import torch


def get_device(preferred: Optional[Union[str, torch.device]] = "auto") -> torch.device:
    """Resolve the best available computing device (CUDA, MPS, or CPU).

    Args:
        preferred: 'auto', 'cuda', 'mps', 'cpu', None, or torch.device instance.

    Returns:
        torch.device instance.
    """
    if isinstance(preferred, torch.device):
        return preferred

    if preferred is not None and preferred != "auto":
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_device_cache(device: Union[torch.device, str]) -> None:
    """Clear memory cache for the specified device."""
    dev_str = str(device)
    if "cuda" in dev_str and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif "mps" in dev_str and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def to_device(data: Any, device: torch.device) -> Any:
    """Recursively move tensors/batches to the target device.

    Args:
        data: Tensor, dict, list, tuple, or arbitrary container.
        device: Target torch device.

    Returns:
        Data moved to device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    return data
