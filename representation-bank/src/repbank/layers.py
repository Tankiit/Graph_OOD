from __future__ import annotations

from torch import nn


def transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """Resolve common HF decoder layouts without relying on absolute layer names."""
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("model", "decoder", "layers"),
    )
    for path in candidates:
        node = model
        try:
            for part in path:
                node = getattr(node, part)
        except AttributeError:
            continue
        if isinstance(node, (nn.ModuleList, list)):
            return node
    raise ValueError("Cannot locate transformer blocks; add this architecture to transformer_blocks().")


def fraction_to_index(fraction: float, n_layers: int) -> int:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"depth fraction must be in [0,1], got {fraction}")
    return min(n_layers - 1, max(0, round(fraction * (n_layers - 1))))

