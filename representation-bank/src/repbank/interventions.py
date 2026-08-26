from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

from .layers import fraction_to_index, transformer_blocks

Primitive = Literal["add", "project_out", "clamp", "scale"]
TokenPolicy = Literal["last", "span", "all"]


@dataclass(frozen=True)
class Intervention:
    primitive: Primitive
    direction: Tensor
    depth_fraction: float
    token_policy: TokenPolicy = "last"
    strength: float = 1.0  # alpha, c, or gamma depending on primitive
    span: tuple[int, int] | None = None


def _selected_positions(x: Tensor, policy: TokenPolicy, span: tuple[int, int] | None) -> slice:
    if policy == "all":
        return slice(None)
    if policy == "last":
        return slice(x.shape[-2] - 1, x.shape[-2])
    if span is None:
        raise ValueError("token_policy='span' requires span=(start, stop)")
    start, stop = span
    return slice(max(0, start), min(x.shape[-2], stop))


def apply_intervention(x: Tensor, spec: Intervention) -> Tensor:
    """Apply an intervention to [batch, tokens, d] without mutating hook input."""
    v = spec.direction.to(device=x.device, dtype=x.dtype)
    v = v / v.norm().clamp_min(torch.finfo(x.dtype).eps)
    pos = _selected_positions(x, spec.token_policy, spec.span)
    out = x.clone()
    h = out[..., pos, :]
    coordinate = torch.einsum("...td,d->...t", h, v).unsqueeze(-1)
    if spec.primitive == "add":
        h = h + spec.strength * v
    elif spec.primitive == "project_out":
        h = h - coordinate * v
    elif spec.primitive == "clamp":
        h = h + (spec.strength - coordinate) * v
    elif spec.primitive == "scale":
        h = h + (spec.strength - 1.0) * coordinate * v
    else:
        raise ValueError(spec.primitive)
    out[..., pos, :] = h
    return out


class InterventionHarness(AbstractContextManager):
    """Forward hook that propagates the edited residual stream downstream."""

    def __init__(self, model: nn.Module, specs: list[Intervention]):
        self.model = model
        self.specs = specs
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self):
        blocks = transformer_blocks(self.model)
        grouped: dict[int, list[Intervention]] = {}
        for spec in self.specs:
            grouped.setdefault(fraction_to_index(spec.depth_fraction, len(blocks)), []).append(spec)
        for index, layer_specs in grouped.items():
            def hook(_module, _inputs, output, layer_specs=layer_specs):
                hidden = output[0] if isinstance(output, tuple) else output
                for spec in layer_specs:
                    hidden = apply_intervention(hidden, spec)
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
            self.handles.append(blocks[index].register_forward_hook(hook))
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

