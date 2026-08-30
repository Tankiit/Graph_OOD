from pathlib import Path

import numpy as np

from repbank.geometry import (
    cosine,
    covariance_mismatch,
    solve_direction,
    whitened_eigenvalues,
)


def test_equal_covariance_has_unit_whitened_spectrum():
    rng = np.random.default_rng(0)
    left = rng.normal(size=(1000, 4))
    right = rng.normal(size=(1000, 4)) + 1
    values = whitened_eigenvalues(left, right, ridge_fraction=0.0)
    assert np.allclose(values, 1, atol=0.2)
    assert covariance_mismatch(left, right) < 0.2


def test_solve_direction_matches_raw_for_spherical_data():
    rng = np.random.default_rng(1)
    left = rng.normal(loc=[1, 0], size=(1000, 2))
    right = rng.normal(size=(1000, 2))
    eu, raw = solve_direction(left, right)
    assert cosine(eu, raw) > 0.98


def test_geometry_has_no_prevalence_parameter():
    source = Path(__file__).parents[1] / "src" / "repbank" / "geometry.py"
    assert "π" not in source.read_text()
