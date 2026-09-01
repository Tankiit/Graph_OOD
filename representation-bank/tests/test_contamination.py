from __future__ import annotations

import json

import numpy as np

from repbank.contamination import (
    ContaminationConfig,
    contamination_grid,
    merge_contamination_outputs,
    resolve_deflation,
)


def test_contamination_grid_is_deterministic_and_reports_fixed_evaluation() -> None:
    rng = np.random.default_rng(12)
    rows, layers, width = 120, 3, 10
    labels = np.repeat([1, 0], rows // 2).astype(np.int8)
    # A shared-covariance mean shift gives the contamination prediction a real
    # population target, while groups exercise the held-out split.
    states = rng.normal(size=(rows, layers, width)).astype(np.float32)
    states[labels == 1, :, 0] += 1.5
    groups = np.array([f"question-{index // 2}" for index in range(rows)])
    config = ContaminationConfig(
        contamination=(0.2, 0.5), dimensions=(4,), folds=3, repeats=3, seed=9,
    )

    first = contamination_grid(states, labels, groups, config)
    second = contamination_grid(states, labels, groups, config)

    assert first == second
    assert len(first["cells"]) == (layers - 1) * 2
    assert first["protocol"]["fixed_evaluation"].startswith("question-grouped held-out")
    assert first["cells"][0]["draws"] == 9
    assert first["warnings"]


def test_contamination_grid_rejects_unavailable_common_pool() -> None:
    rng = np.random.default_rng(3)
    states = rng.normal(size=(40, 2, 6)).astype(np.float32)
    labels = np.repeat([1, 0], 20).astype(np.int8)
    groups = np.array([f"q-{index // 2}" for index in range(40)])
    config = ContaminationConfig(contamination=(0.5,), dimensions=(3,), folds=2,
                                 repeats=1, pool_size=100)

    try:
        contamination_grid(states, labels, groups, config)
    except ValueError as error:
        assert "feasible" in str(error)
    else:
        raise AssertionError("expected infeasible pool size to fail")


def test_merge_contamination_outputs_joins_distinct_layer_chunks(tmp_path) -> None:
    rng = np.random.default_rng(5)
    states = rng.normal(size=(60, 3, 6)).astype(np.float32)
    labels = np.repeat([1, 0], 30).astype(np.int8)
    states[labels == 1, :, 0] += 1
    groups = np.array([f"q-{index // 2}" for index in range(60)])
    common = {"contamination": (0.5,), "dimensions": (3,), "folds": 3, "repeats": 1, "seed": 2}
    reports = []
    for block in (0, 1):
        report = contamination_grid(states, labels, groups, ContaminationConfig(**common, blocks=(block,)))
        report.update({"bank": "synthetic.npz", "position": "answer_last", "bank_metadata": {}})
        path = tmp_path / f"block-{block}.json"
        path.write_text(json.dumps(report))
        reports.append(path)

    merged = merge_contamination_outputs(reports)

    assert merged["config"]["blocks"] == [0, 1]
    assert len(merged["cells"]) == 2


def test_resolution_matches_rank_one_prediction_and_has_no_perp_change() -> None:
    clean = np.diag([2.0, 3.0])
    delta = np.array([1.0, -2.0])
    pi = 0.3
    mixed = clean + pi * (1 - pi) * np.outer(delta, delta)
    result = resolve_deflation(clean, mixed, delta, np.array([3.0, 1.0]), pi)

    assert np.isclose(result["along"], result["theory_upper"])
    assert np.isclose(result["perp"], 0.0, atol=1e-12)
    assert np.isclose(result["cross_contribution"], 0.0, atol=1e-12)
    assert np.isclose(result["reconstruction_error"], 0.0, atol=1e-12)
