from __future__ import annotations

import numpy as np

from repbank.gates import same_class_covariance_control, stratified_heldout_scores


def test_same_class_covariance_control_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    correct = rng.normal(size=(80, 12))
    wrong = rng.normal(size=(50, 12))
    first = same_class_covariance_control(correct, wrong, repeats=40, seed=11)
    second = same_class_covariance_control(correct, wrong, repeats=40, seed=11)

    assert first == second
    assert first["correct_half_split"]["rows_per_half"] == 40
    assert first["correct_bootstrap_matched_to_observed_class_sizes"]["second_rows"] == 50


def test_same_class_covariance_control_detects_large_shift() -> None:
    rng = np.random.default_rng(13)
    correct = rng.normal(size=(400, 8))
    wrong = rng.normal(size=(300, 8)) * np.array([5, 4, 3, 2, 1, 1, 1, 1])
    result = same_class_covariance_control(correct, wrong, repeats=200, seed=17)

    assert result["observed_correct_vs_wrong"] > result["correct_half_split"]["q975"]
    assert result["interpretation"] == "observed mismatch exceeds at least one same-class null"


def test_stratified_scores_return_crossfit_leverage() -> None:
    rng = np.random.default_rng(19)
    groups = np.repeat([f"q{i}" for i in range(30)], 2)
    labels = np.tile([0, 1], 30)
    strata = np.repeat(np.array(["a", "b", "c"] * 10), 2)
    states = rng.normal(size=(60, 8)) + labels[:, None] * 0.4

    eu, raw, leverage = stratified_heldout_scores(
        states, labels, groups, strata, return_leverage=True
    )

    assert np.isfinite(eu).all()
    assert np.isfinite(raw).all()
    assert np.isfinite(leverage).all()
    assert (leverage >= 0).all()
