"""Synthetic data generator for representation shift and OOD detection experiments."""

from typing import Any, Dict, Optional, Tuple
import numpy as np


def make_world(
    d: int = 20,
    sep: float = 3.0,
    seed: int = 0
) -> Dict[str, Any]:
    """Create synthetic Gaussian world parameters with specified separation (Mahalanobis distance).

    Args:
        d: Dimension of the feature space.
        sep: Mahalanobis distance separation between In-Distribution P and Shifted Q (Delta^T Sigma^-1 Delta == sep**2).
        seed: Random seed for direction vector generation.

    Returns:
        Dictionary with 'mu', 'Sigma', 'Delta', and 'Sinv'.
    """
    rng = np.random.default_rng(seed)
    mu = np.zeros(d)
    Sigma = np.diag(np.linspace(0.3, 3.0, d))  # anisotropic on purpose
    Sinv = np.linalg.inv(Sigma)

    u = rng.normal(size=d)
    # Scale u so that Delta^T Sigma^-1 Delta == sep**2
    u_quad = float(u.T @ Sinv @ u)
    if u_quad > 0:
        c = sep / np.sqrt(u_quad)
        Delta = c * u
    else:
        Delta = np.zeros(d)

    return dict(mu=mu, Sigma=Sigma, Delta=Delta, Sinv=Sinv)


def draw(
    world: Dict[str, Any],
    n: int,
    pi: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw n samples from mixture (1 - pi) * P + pi * Q.

    Args:
        world: World parameter dict from make_world.
        n: Number of samples to draw.
        pi: Mixture proportion of shifted distribution Q.
        rng: NumPy random generator.

    Returns:
        Tuple of (X, z) where X is [n, d] feature matrix and z is [n] boolean indicator (True -> from Q).
    """
    z = rng.random(n) < pi  # True -> from Q
    X = rng.multivariate_normal(world["mu"], world["Sigma"], n)
    X[z] += world["Delta"]
    return X, z  # z is for analysis only, never for fitting


def splits(
    world: Dict[str, Any],
    rng: np.random.Generator,
    n_ref: int = 500,
    n_calib: int = 2000,
    n_probe: int = 200,
    n_dev: int = 2000,
    n_oracle: int = 50_000,
    n_contam: int = 2000,
    pi_dev: float = 0.5
) -> Dict[str, Any]:
    """Generate reference, calibration, probe, dev pool, oracle, and contamination datasets.

    Args:
        world: World parameter dict from make_world.
        rng: NumPy random generator.
        n_ref: Reference In-Distribution sample count.
        n_calib: Calibration sample count (In-Distribution).
        n_probe: Probe sample count (In-Distribution).
        n_dev: Development/evaluation pool sample count (mixture).
        n_oracle: Oracle clean In-Distribution sample count.
        n_contam: Pure out-of-distribution contamination sample count.
        pi_dev: Proportion of OOD/Q in dev mixture pool (default: 0.5).

    Returns:
        Dict of splits with ref, calib, probe, dev (X, z), oracle, contam.
    """
    return dict(
        ref=draw(world, n_ref, 0.0, rng)[0],
        calib=draw(world, n_calib, 0.0, rng)[0],
        probe=draw(world, n_probe, 0.0, rng)[0],
        dev=draw(world, n_dev, pi_dev, rng),  # (X_dev, z_dev) where pi_dev is the OOD pool mixture
        oracle=draw(world, n_oracle, 0.0, rng)[0],
        contam=draw(world, n_contam, 1.0, rng)[0],  # pure Q, to be mixed into ref later
    )
