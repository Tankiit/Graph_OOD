from __future__ import annotations

import numpy as np
from repbank.gates import same_class_covariance_control


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
