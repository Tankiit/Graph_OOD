import numpy as np
import pytest
from src.data.synth import make_world, draw, splits


def test_make_world_separation():
    d = 20
    sep = 3.5
    world = make_world(d=d, sep=sep, seed=42)

    mu = world["mu"]
    Sigma = world["Sigma"]
    Delta = world["Delta"]
    Sinv = world["Sinv"]

    assert mu.shape == (d,)
    assert Sigma.shape == (d, d)
    assert Delta.shape == (d,)

    # Verify Delta^T Sigma^-1 Delta == sep**2
    quad_form = float(Delta.T @ Sinv @ Delta)
    np.testing.assert_allclose(quad_form, sep**2, rtol=1e-5)


def test_draw_proportions():
    world = make_world(d=10, sep=2.0, seed=0)
    rng = np.random.default_rng(123)

    # Clean P (pi = 0.0)
    X_clean, z_clean = draw(world, n=1000, pi=0.0, rng=rng)
    assert X_clean.shape == (1000, 10)
    assert np.all(~z_clean)

    # Pure Q (pi = 1.0)
    X_ood, z_ood = draw(world, n=1000, pi=1.0, rng=rng)
    assert X_ood.shape == (1000, 10)
    assert np.all(z_ood)

    # Mixture (pi = 0.4)
    X_mix, z_mix = draw(world, n=2000, pi=0.4, rng=rng)
    assert X_mix.shape == (2000, 10)
    assert 0.35 <= z_mix.mean() <= 0.45


def test_splits_shapes():
    world = make_world(d=15, sep=3.0, seed=7)
    rng = np.random.default_rng(42)

    data = splits(world, rng, n_ref=100, n_calib=200, n_probe=50, n_dev=300, n_oracle=500, n_contam=150, pi_dev=0.3)

    assert data["ref"].shape == (100, 15)
    assert data["calib"].shape == (200, 15)
    assert data["probe"].shape == (50, 15)

    X_dev, z_dev = data["dev"]
    assert X_dev.shape == (300, 15)
    assert z_dev.shape == (300,)

    assert data["oracle"].shape == (500, 15)
    assert data["contam"].shape == (150, 15)
