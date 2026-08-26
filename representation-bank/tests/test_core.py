import torch

from repbank.config import CaptureConfig
from repbank.interventions import Intervention, apply_intervention


def test_fractional_layers():
    assert CaptureConfig().model_layer_indices(36) == [7, 18, 28]
    assert CaptureConfig().model_layer_indices(32) == [6, 16, 25]


def test_project_out_removes_coordinate():
    h = torch.randn(2, 4, 8)
    v = torch.randn(8)
    y = apply_intervention(h, Intervention("project_out", v, 0.5, "all"))
    assert torch.allclose(y @ (v / v.norm()), torch.zeros(2, 4), atol=1e-5)


def test_clamp_last_only():
    h = torch.randn(1, 3, 5)
    v = torch.randn(5)
    y = apply_intervention(h, Intervention("clamp", v, 0.5, "last", 2.0))
    assert torch.equal(h[:, :-1], y[:, :-1])
    assert torch.allclose(y[:, -1] @ (v / v.norm()), torch.tensor([2.0]), atol=1e-5)

