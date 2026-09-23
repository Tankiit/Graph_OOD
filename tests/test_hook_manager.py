import pytest
import torch
import torch.nn as nn
from src.models.hook_manager import HookManager
from src.steering.interveners import AdditiveIntervener


def test_hook_manager_capture():
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4)
    )
    hook_mgr = HookManager(model)
    x = torch.randn(2, 8)

    with hook_mgr.capture_activations(["0", "2"]) as storage:
        out = model(x)

    assert "0" in storage
    assert "2" in storage
    assert storage["0"][0].shape == (2, 16)
    assert storage["2"][0].shape == (2, 4)
    assert len(hook_mgr.active_handles) == 0  # Cleaned up


def test_hook_manager_steering_additive():
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.Linear(16, 4)
    )
    hook_mgr = HookManager(model)
    x = torch.randn(2, 8)

    orig_out = model(x).clone()

    steering_vec = torch.ones(16) * 5.0
    intervener = AdditiveIntervener(steering_vec, coefficient=1.0)

    with hook_mgr.apply_steering({"0": intervener}):
        steered_out = model(x)

    assert not torch.allclose(orig_out, steered_out)

    # Verify original output is restored after context exit
    restored_out = model(x)
    assert torch.allclose(orig_out, restored_out)
    assert len(hook_mgr.active_handles) == 0
