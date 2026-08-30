import numpy as np
import torch
from torch import nn

from repbank.jacobian import (
    _forward_captured,
    dispersion,
    fit_readout,
    jvp_diagnostics,
    paired_indices,
)


def test_pair_selection_and_readout():
    rng = np.random.default_rng(3)
    pair_ids = np.repeat(np.arange(10), 2).astype(str)
    roles = np.tile([0, 1], 10)
    states = rng.normal(size=(20, 8)).astype(np.float32)
    states[roles == 1, 0] += 3
    correct, wrong = paired_indices(pair_ids, roles)
    assert len(correct) == len(wrong) == 10
    fitted = fit_readout(states, pair_ids, roles, epochs=100)
    assert fitted.train_pairs == 8
    assert fitted.heldout_pairs == 2


def test_dispersion_constant_field():
    report = dispersion(np.tile(np.array([[1.0, 0.0]]), (5, 1)))
    assert report["median_degrees"] == 0
    assert report["resultant_length"] == 1


def test_captured_residual_is_gradient_leaf_with_frozen_model():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

        def forward(self, input_ids, use_cache=False):
            hidden = torch.nn.functional.one_hot(input_ids, 4).float()
            for layer in self.model.layers:
                hidden = layer(hidden)
            return hidden

    model = Toy()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    early, late = _forward_captured(model, torch.tensor([[0, 1]]), 0, 1)
    gradient, = torch.autograd.grad(late.sum(), early)
    assert early.requires_grad
    assert gradient.shape == (1, 2, 4)


def test_jvp_diagnostics_detects_class_gap():
    jvp = np.ones((4, 3, 2), dtype=np.float32)
    jvp[1::2, :, 0] = 2
    report = jvp_diagnostics(jvp, np.array([0, 1, 0, 1]))
    assert report["median_cosine_by_epsilon"] == [1, 1, 1]
    assert report["truth_wrong_mean_relative_gap"] > 0
