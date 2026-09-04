from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .coupled_extract import verify_checksum
from .geometry import (
    cosine,
    covariance,
    covariance_mismatch,
    reduced_basis,
    solve_direction,
    whitened_eigenvalues,
)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    return float(((positive[:, None] > negative).sum() + 0.5 * (positive[:, None] == negative).sum())
                 / (len(positive) * len(negative)))


def load_bank(path: str | Path) -> tuple[dict, dict]:
    verify_checksum(path)
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    return data, metadata


def heldout_scores(states: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                   folds: int = 5, max_dim: int = 64) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    assignment = {group: index % folds for index, group in enumerate(unique)}
    eu_scores = np.full(len(labels), np.nan)
    raw_scores = np.full(len(labels), np.nan)
    for fold in range(folds):
        test = np.array([assignment[group] == fold for group in groups])
        train = ~test
        if len(np.unique(labels[train])) < 2:
            continue
        center, basis = reduced_basis(states[train], max_dim=max_dim)
        train_reduced = (states[train] - center) @ basis
        correct = train_reduced[labels[train] == 1]
        wrong = train_reduced[labels[train] == 0]
        eu, raw = solve_direction(correct, wrong)
        test_reduced = (states[test] - center) @ basis
        eu_scores[test] = test_reduced @ eu
        raw_scores[test] = test_reduced @ raw
    return eu_scores, raw_scores


def stratified_heldout_scores(states: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                              strata: np.ndarray, folds: int = 5,
                              max_dim: int = 64, seed: int = 501_693,
                              return_leverage: bool = False):
    """Question-grouped scores with folds balanced within categorical strata."""
    unique = np.unique(groups)
    group_stratum = {}
    for group in unique:
        values = np.unique(strata[groups == group])
        if len(values) != 1:
            raise ValueError(f"group {group!r} spans multiple strata")
        group_stratum[group] = values[0]
    rng = np.random.default_rng(seed)
    assignment = {}
    offset = 0
    for stratum in sorted(np.unique(strata).tolist()):
        members = np.array(sorted(group for group in unique if group_stratum[group] == stratum))
        rng.shuffle(members)
        for index, group in enumerate(members):
            assignment[group] = (offset + index) % folds
        offset = (offset + len(members)) % folds

    eu_scores = np.full(len(labels), np.nan)
    raw_scores = np.full(len(labels), np.nan)
    leverage = np.full(len(labels), np.nan)
    for fold in range(folds):
        test = np.array([assignment[group] == fold for group in groups])
        train = ~test
        if len(np.unique(labels[train])) < 2:
            continue
        center, basis = reduced_basis(states[train], max_dim=max_dim)
        train_reduced = (states[train] - center) @ basis
        correct = train_reduced[labels[train] == 1]
        wrong = train_reduced[labels[train] == 0]
        eu, raw = solve_direction(correct, wrong)
        test_reduced = (states[test] - center) @ basis
        eu_scores[test] = test_reduced @ eu
        raw_scores[test] = test_reduced @ raw
        if return_leverage:
            correct_center = correct.mean(0)
            correct_covariance = covariance(correct)
            ridge = 0.1 * np.trace(correct_covariance) / correct_covariance.shape[0]
            centered = test_reduced - correct_center
            solved = np.linalg.solve(
                correct_covariance + ridge * np.eye(correct_covariance.shape[0]), centered.T
            ).T
            leverage[test] = np.sum(centered * solved, axis=1)
    if return_leverage:
        return eu_scores, raw_scores, leverage
    return eu_scores, raw_scores


def _fit_logistic(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    weights = np.zeros(features.shape[1])
    for _ in range(200):
        probability = 1 / (1 + np.exp(-(features @ weights)))
        weights -= 0.1 * (features.T @ (probability - labels) / len(labels))
    return weights


def nested_combined_scores(states: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                           confidence: np.ndarray, folds: int = 5,
                           max_dim: int = 64) -> tuple[np.ndarray, list[float]]:
    """Strict outer-fold confidence+EU predictions with inner-fold EU features."""
    unique = np.unique(groups)
    assignment = {group: index % folds for index, group in enumerate(unique)}
    output = np.full(len(labels), np.nan)
    eu_coefficients = []
    for fold in range(folds):
        test = np.array([assignment[group] == fold for group in groups])
        train = ~test
        inner_eu, _ = heldout_scores(
            states[train], labels[train], groups[train], folds=min(4, len(np.unique(groups[train]))),
            max_dim=max_dim,
        )
        inner_valid = np.isfinite(inner_eu) & np.isfinite(confidence[train])
        train_confidence = confidence[train][inner_valid]
        train_eu = inner_eu[inner_valid]
        conf_mean, conf_std = train_confidence.mean(), train_confidence.std()
        eu_mean, eu_std = train_eu.mean(), train_eu.std()
        conf_std, eu_std = max(conf_std, 1e-8), max(eu_std, 1e-8)
        features = np.column_stack([
            np.ones(inner_valid.sum()),
            (train_confidence - conf_mean) / conf_std,
            (train_eu - eu_mean) / eu_std,
        ])
        weights = _fit_logistic(features, labels[train][inner_valid].astype(np.float64))
        center, basis = reduced_basis(states[train], max_dim=max_dim)
        reduced_train = (states[train] - center) @ basis
        eu, _ = solve_direction(
            reduced_train[labels[train] == 1], reduced_train[labels[train] == 0]
        )
        outer_eu = ((states[test] - center) @ basis) @ eu
        test_features = np.column_stack([
            np.ones(test.sum()),
            (confidence[test] - conf_mean) / conf_std,
            (outer_eu - eu_mean) / eu_std,
        ])
        output[test] = test_features @ weights
        eu_coefficients.append(float(weights[2]))
    return output, eu_coefficients


def same_class_covariance_control(correct: np.ndarray, wrong: np.ndarray,
                                  repeats: int = 10_000, seed: int = 20_260_831) -> dict:
    """Calibrate class-covariance mismatch against resampled correct-only nulls."""
    if len(correct) < 4:
        raise ValueError("same-class covariance control needs at least four correct rows")
    rng = np.random.default_rng(seed)
    observed = covariance_mismatch(correct, wrong)
    half = len(correct) // 2
    split_null = np.empty(repeats, dtype=np.float64)
    matched_null = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        permutation = rng.permutation(len(correct))
        split_null[index] = covariance_mismatch(
            correct[permutation[:half]], correct[permutation[half:2 * half]]
        )
        matched_null[index] = covariance_mismatch(
            correct[rng.integers(0, len(correct), len(correct))],
            correct[rng.integers(0, len(correct), len(wrong))],
        )

    def summarize(values: np.ndarray) -> dict:
        return {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "p_null_ge_observed": float((1 + np.sum(values >= observed)) / (repeats + 1)),
        }

    split_summary = summarize(split_null)
    matched_summary = summarize(matched_null)
    return {
        "observed_correct_vs_wrong": observed,
        "correct_rows": len(correct),
        "wrong_rows": len(wrong),
        "dimension": int(correct.shape[1]),
        "repeats": repeats,
        "seed": seed,
        "correct_half_split": {
            "rows_per_half": half,
            **split_summary,
        },
        "correct_bootstrap_matched_to_observed_class_sizes": {
            "first_rows": len(correct),
            "second_rows": len(wrong),
            **matched_summary,
        },
        "interpretation": "no detectable heteroscedasticity" if (
            split_summary["p_null_ge_observed"] >= 0.05
            and matched_summary["p_null_ge_observed"] >= 0.05
        ) else "observed mismatch exceeds at least one same-class null",
    }


def gate_g2_g3(bank_path: str | Path, depth_fraction: float = 0.8) -> dict:
    bank, metadata = load_bank(bank_path)
    layers = metadata["n_layers"]
    block = min(layers - 1, max(0, round(depth_fraction * (layers - 1))))
    states = bank["h_last"][:, block + 1].astype(np.float32)
    labels = bank["label"].astype(np.int8)
    groups = bank["question_id"]
    eu_scores, raw_scores = heldout_scores(states, labels, groups)
    valid = np.isfinite(eu_scores)
    confidence = np.nanmean(bank["logprobs"], axis=1)

    center, basis = reduced_basis(states, max_dim=64)
    reduced = (states - center) @ basis
    correct, wrong = reduced[labels == 1], reduced[labels == 0]
    eu, raw = solve_direction(correct, wrong)
    eigenvalues = whitened_eigenvalues(correct, wrong)
    same_class_control = same_class_covariance_control(correct, wrong)
    combined, eu_coefficients = nested_combined_scores(
        states, labels, groups, confidence, folds=5, max_dim=64
    )
    combined_valid = valid & np.isfinite(combined)
    return {
        "bank": str(bank_path),
        "depth_fraction": depth_fraction,
        "block_index": block,
        "heldout_rows": int(valid.sum()),
        "g2": {
            "eu_auc": auc(labels[valid], eu_scores[valid]),
            "raw_auc": auc(labels[valid], raw_scores[valid]),
            "eu_raw_cosine_reduced": cosine(eu, raw),
            "covariance_mismatch_relative_fro": covariance_mismatch(correct, wrong),
            "same_class_covariance_control": same_class_control,
            "whitened_eigen_fraction_0.9_1.1": float(np.mean((eigenvalues >= 0.9) & (eigenvalues <= 1.1))),
        },
        "g3": {
            "confidence_auc": auc(labels[valid], confidence[valid]),
            "combined_auc_nested": auc(labels[combined_valid], combined[combined_valid]),
            "incremental_auc_nested": (
                auc(labels[combined_valid], combined[combined_valid])
                - auc(labels[combined_valid], confidence[combined_valid])
            ),
            "eu_coefficient_standardized_mean": float(np.mean(eu_coefficients)),
            "evaluation_protocol": "outer question-grouped 5-fold; inner grouped 4-fold EU feature",
        },
    }


def gate_g4(true_bank_path: str | Path, hal_bank_path: str | Path) -> dict:
    true_bank, true_meta = load_bank(true_bank_path)
    hal_bank, hal_meta = load_bank(hal_bank_path)
    if true_meta["generation_set_checksum"] != hal_meta["generation_set_checksum"]:
        raise ValueError("G4 banks use different generation sets")
    if not np.array_equal(true_bank["role"], hal_bank["role"]):
        raise ValueError("G4 bank row order differs")
    true_score = np.nansum(true_bank["logprobs"], axis=1)
    hal_score = np.nansum(hal_bank["logprobs"], axis=1)
    ratio = true_score - hal_score
    correct = (true_bank["role"] == 0).astype(np.int8)
    return {
        "rows": len(correct),
        "ratio_auc_correct": auc(correct, ratio),
        "mean_ratio_correct": float(ratio[correct == 1].mean()),
        "mean_ratio_wrong": float(ratio[correct == 0].mean()),
        "separation": float(ratio[correct == 1].mean() - ratio[correct == 0].mean()),
    }


def compare_model_axis(base_path: str | Path, adapter_path: str | Path,
                       depth_fraction: float = 0.8) -> dict:
    """Compare a controlled adapter displacement with truth/wrong directions."""
    base, base_meta = load_bank(base_path)
    adapter, adapter_meta = load_bank(adapter_path)
    for field in ("generation_set_checksum", "tokenizer_sha256", "n_layers", "d_model"):
        if base_meta[field] != adapter_meta[field]:
            raise ValueError(f"controlled banks differ on {field}")
    if not np.array_equal(base["pair_id"], adapter["pair_id"]):
        raise ValueError("controlled banks have different row order")
    block = min(base_meta["n_layers"] - 1,
                max(0, round(depth_fraction * (base_meta["n_layers"] - 1))))
    states = base["h_last"][:, block + 1].astype(np.float32)
    adapted = adapter["h_last"][:, block + 1].astype(np.float32)
    labels = base["label"].astype(np.int8)
    center, basis = reduced_basis(states, max_dim=64)
    reduced = (states - center) @ basis
    correct, wrong = reduced[labels == 1], reduced[labels == 0]
    eu, raw = solve_direction(correct, wrong)
    displacement = (adapted - states).mean(axis=0) @ basis
    return {
        "base": str(base_path), "adapter": str(adapter_path),
        "depth_fraction": depth_fraction, "block_index": block,
        "mean_displacement_norm_reduced": float(np.linalg.norm(displacement)),
        "cosine_adapter_vs_eu": cosine(displacement, eu),
        "cosine_adapter_vs_raw_delta": cosine(displacement, raw),
        "generation_set_checksum": base_meta["generation_set_checksum"],
    }


def rank_geometry_curve(base_path: str | Path, adapter_paths: list[str | Path],
                        depth_fraction: float = 0.8) -> dict:
    """Measure controlled geometry changes in one primary-model coordinate system."""
    base, base_meta = load_bank(base_path)
    block = min(base_meta["n_layers"] - 1,
                max(0, round(depth_fraction * (base_meta["n_layers"] - 1))))
    base_states = base["h_last"][:, block + 1].astype(np.float32)
    labels = base["label"].astype(np.int8)
    center, basis = reduced_basis(base_states, max_dim=64)
    base_reduced = (base_states - center) @ basis
    base_correct = base_reduced[labels == 1]
    base_wrong = base_reduced[labels == 0]
    base_eu, base_delta_unit = solve_direction(base_correct, base_wrong)
    base_delta = base_correct.mean(0) - base_wrong.mean(0)
    base_covariances = (covariance(base_correct), covariance(base_wrong))
    rows = []
    for path in adapter_paths:
        bank, metadata = load_bank(path)
        for field in ("generation_set_checksum", "tokenizer_sha256", "n_layers", "d_model"):
            if metadata[field] != base_meta[field]:
                raise ValueError(f"rank bank {path} differs from primary on {field}")
        if not np.array_equal(base["pair_id"], bank["pair_id"]):
            raise ValueError(f"rank bank {path} has different row order")
        states = bank["h_last"][:, block + 1].astype(np.float32)
        reduced = (states - center) @ basis
        correct, wrong = reduced[labels == 1], reduced[labels == 0]
        eu, delta_unit = solve_direction(correct, wrong)
        delta = correct.mean(0) - wrong.mean(0)
        class_covariances = (covariance(correct), covariance(wrong))
        displacement = (states - base_states).mean(0) @ basis
        eu_raw = cosine(eu, delta_unit)
        rows.append({
            "adapter_id": metadata["adapter_id"], "rank": metadata["rank"],
            "delta_norm_ratio_to_primary": float(np.linalg.norm(delta) / np.linalg.norm(base_delta)),
            "delta_cosine_to_primary": cosine(delta, base_delta),
            "eu_cosine_to_primary": cosine(eu, base_eu),
            "eu_raw_cosine": eu_raw,
            "eu_orthogonal_fraction": float(np.sqrt(max(0.0, 1 - eu_raw ** 2))),
            "covariance_mismatch_relative_fro": covariance_mismatch(correct, wrong),
            "correct_covariance_change_from_primary": float(
                np.linalg.norm(class_covariances[0] - base_covariances[0], ord="fro")
                / np.linalg.norm(base_covariances[0], ord="fro")
            ),
            "wrong_covariance_change_from_primary": float(
                np.linalg.norm(class_covariances[1] - base_covariances[1], ord="fro")
                / np.linalg.norm(base_covariances[1], ord="fro")
            ),
            "mean_displacement_norm": float(np.linalg.norm(displacement)),
            "displacement_cosine_to_primary_delta": cosine(displacement, base_delta_unit),
            "displacement_cosine_to_primary_eu": cosine(displacement, base_eu),
        })
    return {
        "base": str(base_path), "depth_fraction": depth_fraction, "block_index": block,
        "analysis_dimension": int(basis.shape[1]),
        "isotropic_random_cosine_sd": float(1 / np.sqrt(basis.shape[1])),
        "generation_set_checksum": base_meta["generation_set_checksum"],
        "rows": sorted(rows, key=lambda item: item["rank"]),
    }
