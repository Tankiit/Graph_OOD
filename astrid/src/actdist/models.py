"""Model builders and the layers whose activations we record.

A "neuron" is one channel of a ResNet block output (global-average-pooled over H, W)
or one hidden dimension of the ViT CLS token.
"""

import timm
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

MODELS = {
    "resnet18_scratch": dict(size=32, channels=1, pretrained=False),
    "resnet18_ft": dict(size=112, channels=3, pretrained=True),
    "vit_small_ft": dict(size=224, channels=3, pretrained=True),
}
VIT_NAME = "vit_small_patch16_224.augreg_in21k_ft_in1k"


def build_model(name: str, num_classes: int = 10, pretrained: bool | None = None) -> nn.Module:
    """`pretrained=None` uses the model's default; pass False when loading a checkpoint."""
    if pretrained is None:
        pretrained = MODELS[name]["pretrained"]
    if name == "resnet18_scratch":
        m = resnet18(weights=None, num_classes=num_classes)
        # CIFAR-style stem for small grayscale inputs
        m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m
    if name == "resnet18_ft":
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "vit_small_ft":
        return timm.create_model(VIT_NAME, pretrained=pretrained, num_classes=num_classes)
    raise ValueError(f"unknown model {name}")


def _gap(out: torch.Tensor) -> torch.Tensor:
    return out.mean(dim=(2, 3))


def _cls(out: torch.Tensor) -> torch.Tensor:
    return out[:, 0]


class ActivationRecorder:
    """Registers forward hooks and collects reduced activations, one [B, D] tensor per layer."""

    def __init__(self, model: nn.Module, model_name: str):
        self.acts: dict[str, torch.Tensor] = {}
        self.handles = []
        if model_name.startswith("resnet"):
            for lname in ["layer1", "layer2", "layer3", "layer4"]:
                for i, block in enumerate(getattr(model, lname)):
                    self._hook(block, f"{lname}.{i}", _gap)
            # layer4.1 GAP is exactly the penultimate (avgpool) feature
            self._pre_hook(model.fc, "penultimate")
        else:
            for i, block in enumerate(model.blocks):
                self._hook(block, f"blocks.{i}", _cls)
            self._pre_hook(model.head, "penultimate")

    def _hook(self, module, name, reduce):
        def fn(_m, _inp, out):
            self.acts[name] = reduce(out).detach().float().cpu()
        self.handles.append(module.register_forward_hook(fn))

    def _pre_hook(self, module, name):
        def fn(_m, inp):
            self.acts[name] = inp[0].detach().float().cpu()
        self.handles.append(module.register_forward_pre_hook(fn))

    def pop(self) -> dict[str, torch.Tensor]:
        acts, self.acts = self.acts, {}
        return acts

    def remove(self):
        for h in self.handles:
            h.remove()
