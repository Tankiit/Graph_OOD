"""Steering vectors on a hypersphere, applied at one layer of a frozen classifier.

    split = SplitModel(model, "resnet18_scratch", "layer3.1")
    h = split.prefix(x)                       # activations after the layer, computed once
    swarm = SteeringSwarm(K=1024, dim=split.dim(h), radius=2.0)
    feat, logits = split.suffix(steer(h, swarm()))   # K*B steered forward passes

Layer names are those of models.ActivationRecorder. A vector is added to every spatial position
(ResNet: one value per channel) or every token (ViT), so its norm is the displacement per
position; at "penultimate" it is added to the pooled feature, just before the classifier head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitModel(nn.Module):
    """A classifier cut after `layer`: prefix(x) -> h, suffix(h) -> (penultimate feature, logits)."""

    def __init__(self, model: nn.Module, model_name: str, layer: str):
        super().__init__()
        self.model, self.layer = model, layer
        self.is_vit = not model_name.startswith("resnet")
        if self.is_vit:
            names = [f"blocks.{i}" for i in range(len(model.blocks))]
            blocks = list(model.blocks)
        else:
            names, blocks = [], []
            for lname in ["layer1", "layer2", "layer3", "layer4"]:
                for i, b in enumerate(getattr(model, lname)):
                    names.append(f"{lname}.{i}")
                    blocks.append(b)
        if layer != "penultimate" and layer not in names:
            raise ValueError(f"unknown layer {layer!r}; choose from {names + ['penultimate']}")
        cut = len(blocks) if layer == "penultimate" else names.index(layer) + 1
        self.pre_blocks, self.post_blocks = nn.Sequential(*blocks[:cut]), nn.Sequential(*blocks[cut:])
        self.layers = names + ["penultimate"]
        self.cut, self.block_names = cut, names

    def _stem(self, x):
        m = self.model
        if self.is_vit:
            return m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(x))))
        return m.maxpool(m.relu(m.bn1(m.conv1(x))))

    def _pool(self, h):
        m = self.model
        if self.is_vit:
            return m.forward_head(m.norm(h), pre_logits=True)
        return torch.flatten(m.avgpool(h), 1)

    def _head(self, feat):
        return self.model.head(feat) if self.is_vit else self.model.fc(feat)

    def prefix(self, x):
        h = self.pre_blocks(self._stem(x))
        return self._pool(h) if self.layer == "penultimate" else h

    def suffix(self, h):
        feat = h if self.layer == "penultimate" else self._pool(self.post_blocks(h))
        return feat, self._head(feat)

    def check_sim_layer(self, sim_layer: str):
        """`sim_layer` must be `layer` itself or downstream of it."""
        if sim_layer not in self.layers:
            raise ValueError(f"unknown sim_layer {sim_layer!r}; choose from {self.layers}")
        if self.layers.index(sim_layer) < self.layers.index(self.layer):
            raise ValueError(f"sim_layer {sim_layer!r} is upstream of the steered layer {self.layer!r}")

    def _reduce(self, x):
        """One vector per image at an intermediate block, as in models.ActivationRecorder:
        ResNet map -> mean over positions, ViT tokens -> CLS token."""
        return x[:, 0] if self.is_vit else x.mean(dim=(2, 3))

    def suffix_sim(self, h, sim_layer: str):
        """suffix(h) plus the activation at `sim_layer` (reduced to [B, d]): (sim, feat, logits)."""
        if sim_layer == "penultimate":
            feat, logits = self.suffix(h)
            return feat, feat, logits
        x = h
        sim = self._reduce(x) if sim_layer == self.layer else None
        for name, block in zip(self.block_names[self.cut:], self.post_blocks):
            x = block(x)
            if name == sim_layer:
                sim = self._reduce(x)
        feat = self._pool(x)
        return sim, feat, self._head(feat)

    @staticmethod
    def dim(h) -> int:
        """Size of a steering vector for activations h: channels (ResNet maps) or width (tokens, features)."""
        return h.shape[1] if h.dim() == 4 else h.shape[-1]

    @staticmethod
    def position_norms(h) -> torch.Tensor:
        """Norm of the activation at every position / token / sample, flattened."""
        return (h.norm(dim=1) if h.dim() == 4 else h.norm(dim=-1)).flatten()

    @staticmethod
    def position_mean(h) -> torch.Tensor:
        """Mean activation vector over samples and positions: [dim]."""
        if h.dim() == 4:
            return h.mean((0, 2, 3))
        return h.reshape(-1, h.shape[-1]).mean(0)


def steer(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """h [B, ...], v [K, dim] -> [K*B, ...]: every vector applied to every sample of the batch."""
    if h.dim() == 4:        # ResNet map [B, C, H, W]: same channel offset at every position
        out = h[None] + v[:, None, :, None, None]
    elif h.dim() == 3:      # ViT tokens [B, T, D]: added to every token
        out = h[None] + v[:, None, None, :]
    else:                   # pooled feature [B, D]
        out = h[None] + v[:, None, :]
    return out.flatten(0, 1)


class SteeringSwarm(nn.Module):
    """K vectors constrained to the sphere of radius r: v_k = r * u_k / |u_k|, u unconstrained."""

    def __init__(self, K: int, dim: int, radius: float, init: torch.Tensor | None = None):
        super().__init__()
        self.u = nn.Parameter(torch.randn(K, dim) if init is None else init.clone())
        self.register_buffer("radius", torch.tensor(float(radius)))

    def forward(self) -> torch.Tensor:
        return self.radius * F.normalize(self.u, dim=1)

    def directions(self) -> torch.Tensor:
        return F.normalize(self.u.detach(), dim=1)
