import pytest
import torch
from src.steering.interveners import (
    AdditiveIntervener,
    OrthogonalProjectionIntervener,
    SubspaceClampingIntervener,
    GatedIntervener,
)
from src.steering.scheduler import LayerScheduler, TokenPositionScheduler


def test_additive_intervener():
    h = torch.zeros(2, 4, 8)  # [B, S, D]
    vec = torch.ones(8)
    intervener = AdditiveIntervener(vec, coefficient=2.0)
    out = intervener(h)
    assert out.shape == (2, 4, 8)
    assert torch.allclose(out, torch.full((2, 4, 8), 2.0))


def test_orthogonal_projection_intervener():
    # Vector along unit x0
    vec = torch.zeros(8)
    vec[0] = 1.0
    h = torch.ones(2, 8)  # [B, D]
    intervener = OrthogonalProjectionIntervener(vec)
    out = intervener(h)
    assert out[:, 0].sum().item() == 0.0  # x0 component erased
    assert out[:, 1:].sum().item() != 0.0  # Other components intact


def test_subspace_clamping_intervener():
    vec = torch.zeros(4)
    vec[0] = 1.0
    h = torch.tensor([[5.0, 1.0, 1.0, 1.0]])
    intervener = SubspaceClampingIntervener(vec, min_val=-2.0, max_val=2.0)
    out = intervener(h)
    assert out[0, 0].item() == 2.0  # Clamped down to 2.0
    assert out[0, 1].item() == 1.0


def test_schedulers():
    layers = ["layer0", "layer1", "layer2", "layer3"]
    sched = LayerScheduler(layers, schedule_type="linear_ramp", base_coeff=1.0, min_coeff=0.0)
    coeffs = sched.get_coefficients()
    assert coeffs["layer0"] == 0.0
    assert coeffs["layer3"] == 1.0

    pos = TokenPositionScheduler.get_token_indices(mode="last", seq_len=10)
    assert pos == [9]
