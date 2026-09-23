import csv
import json

import numpy as np
import pytest
from scipy.stats import chi2

from analysis_synth import _coverage, _percentile_bound
from analytic import (
    alpha_star_maha,
    alpha_star_maha_contaminated,
    contaminated_moments,
    contaminated_threshold,
)
from detectors import KNN, Linear, Maha, Spectral, calibrate, calibrate_on_pool
from paths import first_rejection, random_dirs
from run_synth import Config, run
from src.data.synth import contaminant_count, draw, make_world, paired_pool, splits


@pytest.mark.parametrize("d,sep", [(1, 0), (1, 3), (5, 2.5)])
def test_world_and_draw(d, sep):
    world = make_world(d, sep, 4)
    assert world.Delta @ np.linalg.solve(world.Sigma, world.Delta) == pytest.approx(
        sep**2
    )
    same = make_world(d, sep, 4)
    np.testing.assert_array_equal(world.Sigma, same.Sigma)
    for pi in (0, 0.3, 1):
        X, labels = draw(world, 40000, pi, np.random.default_rng(4))
        np.testing.assert_allclose(
            X.mean(axis=0), world.mu + pi * world.Delta, atol=0.03
        )
        assert labels.mean() == pytest.approx(pi, abs=0.01)


def test_independent_splits_and_paired_pools():
    rng = np.random.default_rng(1)
    sizes = {"dev": 20, "calib": 10, "probe": 4, "contaminant": 20, "oracle": 12}
    data = splits(make_world(2, 3, 1), sizes, rng)
    assert data["probe"].shape == (4, 2)
    clean, q = data["dev"], data["contaminant"]
    first, y1 = paired_pool(clean, q, 0.2)
    second, y2 = paired_pool(clean, q, 0.4)
    np.testing.assert_array_equal(first, second[: len(first)])
    np.testing.assert_array_equal(first[: len(clean)], clean)
    assert y1.mean() == 0.2
    assert y2.sum() == contaminant_count(20, 0.4)
    assert not np.array_equal(data["dev"][:10], data["calib"])


def test_detectors_and_calibration():
    rng = np.random.default_rng(4)
    D, calib = rng.normal(size=(100, 3)), rng.normal(size=(80, 3))
    maha, threshold = calibrate(Maha(ridge=0), D, calib, 0.9)
    expected = np.array(
        [(x - D.mean(0)) @ np.linalg.solve(np.cov(D.T), x - D.mean(0)) for x in calib]
    )
    np.testing.assert_allclose(maha.score(calib), expected)
    assert threshold == pytest.approx(np.quantile(expected, 0.9))
    knn, threshold_b = calibrate_on_pool(KNN(3), D)
    assert threshold_b == pytest.approx(np.quantile(knn.score(D), 0.95))
    reference = rng.normal(size=(30, 3)) + 2
    linear = Linear(reference, ridge=0).fit(D)
    np.testing.assert_allclose(
        linear.w, np.linalg.solve(np.cov(D.T), reference.mean(0) - D.mean(0))
    )
    with pytest.raises(ValueError, match="contaminant"):
        Linear().fit(D)


def test_spectral_exact_insertion_and_signed_scores():
    rng = np.random.default_rng(7)
    D, X = rng.normal(size=(8, 2)), rng.normal(size=(3, 2))
    fitted = Spectral(bandwidth=1.3).fit(D)
    expected = []
    for x in X:
        augmented = np.vstack((D, x))
        W = np.exp(
            -np.sum((augmented[:, None] - augmented[None, :]) ** 2, axis=-1)
            / (2 * 1.3**2)
        )
        np.fill_diagonal(W, 0)
        expected.append(np.linalg.eigvalsh(np.diag(W.sum(1)) - W)[1] - fitted.lambda2_)
    np.testing.assert_allclose(fitted.score_signed(X), expected, atol=1e-12)
    np.testing.assert_allclose(fitted.score(X), np.abs(expected), atol=1e-12)
    np.testing.assert_allclose(
        fitted.laplacian_ @ fitted.fiedler_,
        fitted.lambda2_ * fitted.fiedler_,
        atol=1e-12,
    )


def test_paths_recrossing_censoring_and_straight_directions():
    # Three crossings at 1,2,3: first rejection followed by two recrossings.
    score = lambda X: (X[:, 0] - 1) * (X[:, 0] - 2) * (X[:, 0] - 3)
    alpha, censored, recross = first_rejection(
        score, 0, [0], [1], np.linspace(0, 4, 17)
    )
    assert alpha == pytest.approx(1, abs=1e-8)
    assert not censored and recross == 2
    assert first_rejection(lambda X: np.zeros(len(X)), 1, [0], [1], [0, 1]) == (
        None,
        True,
        0,
    )
    assert first_rejection(lambda X: np.ones(len(X)), 0, [0], [1], [0, 1]) == (
        0,
        False,
        0,
    )
    dirs = random_dirs(np.zeros(4), 10000, np.random.default_rng(0))
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1)
    np.testing.assert_allclose(dirs.mean(0), 0, atol=0.02)
    assert random_dirs(np.zeros(4), 0, np.random.default_rng(0)).shape == (0, 4)
    with pytest.raises(ValueError, match="start at zero"):
        first_rejection(score, 0, [0], [1], [1, 2])


def test_maha_quadratic_matches_numerical_for_random_paths():
    rng = np.random.default_rng(51)
    world = make_world(4, 3, 8)
    model = Maha().fit(draw(world, 100, 0, rng)[0])
    for _ in range(30):
        x = draw(world, 1, 0, rng)[0][0]
        v = random_dirs(x, 1, rng)[0]
        exact = alpha_star_maha(x, v, model.mu, model.Sigma, 9)
        numerical, censored, _ = first_rejection(
            model.score, 9, x, v, np.linspace(0, 30, 40)
        )
        assert not censored
        assert numerical == pytest.approx(exact, abs=1e-8)


@pytest.mark.parametrize(
    "x,v,expected",
    [(0, 1, 1), (2, -1, 0), (1, -1, 2), (1, 1, 0), (0, 0, None), (1, 0, None)],
)
def test_quadratic_boundary_cases(x, v, expected):
    assert alpha_star_maha([x], [v], [0], [[1]], 1) == expected


@pytest.mark.parametrize("d", [1, 3])
@pytest.mark.parametrize("protocol", ["A", "B"])
def test_population_contaminated_quantile_against_independent_monte_carlo(d, protocol):
    world, pi, quantile = make_world(d, 3, 42), 0.3, 0.9
    mu, Sigma = contaminated_moments(world, pi)
    X = draw(world, 120000, pi if protocol == "B" else 0, np.random.default_rng(99))[0]
    centered = X - mu
    scores = np.einsum("ni,ij,nj->n", centered, np.linalg.inv(Sigma), centered)
    threshold = contaminated_threshold(world, pi, protocol, quantile)
    assert np.mean(scores <= threshold) == pytest.approx(quantile, abs=0.004)
    assert contaminated_threshold(world, 0, protocol, quantile) == pytest.approx(
        chi2.ppf(quantile, d)
    )
    x, v = world.mu, np.ones(d) / np.sqrt(d)
    expected = alpha_star_maha(x, v, mu, Sigma, threshold)
    assert (
        alpha_star_maha_contaminated(x, v, world, pi, protocol, quantile=quantile)
        == expected
    )


def test_censoring_is_not_silently_discarded():
    rows = [
        {"alpha_star": value, "censored": value is None} for value in (1, 2, 3, None)
    ]
    assert _percentile_bound(rows, 0.05) == 1
    assert _percentile_bound(rows, 0.95) is None
    assert _coverage(1, None, 3, 10) is True
    assert _coverage(1, None, 20, 10) is None
    assert _coverage(None, None, 3, 10) is False
    assert _coverage(1, 4, None, 10, target_censored=True) is False
    assert _coverage(1, None, None, 10, target_censored=True) is None


def test_end_to_end_all_detectors_and_protocols(tmp_path):
    config = Config(
        d=2,
        n=12,
        n_calib=12,
        n_probe=1,
        n_contaminant=12,
        n_oracle=6,
        B=3,
        R=2,
        m=1,
        pis=(0, 0.2),
        alpha_steps=9,
        refine_steps=20,
        k=3,
    )
    rows = run(config, tmp_path, plots=False)
    assert len(rows) == 2 * 3 * 2 * 2 * 4 * 2 * 2
    assert {row["protocol"] for row in rows} == {"A", "B"}
    assert {row["n_clean"] for row in rows} == {6, 12}
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["closed_form_max_abs_error"] < 2e-6
    with (tmp_path / "coverage.csv").open() as handle:
        coverage = list(csv.DictReader(handle))
    assert len(coverage) == 4 * 2 * 2 * 2 * 2
    assert all(int(row["R"]) == 2 for row in coverage)
    assert {row["oracle_kind"] for row in coverage} == {
        "population",
        "finite_reference",
    }
    assert {row["detector"] for row in coverage} == set(config.detectors)
    assert (tmp_path / "worlds.npz").is_file()
