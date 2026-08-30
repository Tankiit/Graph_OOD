from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .coupled_extract import verify_checksum
from .geometry import (
    cosine,
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
    standardized = lambda value: (value - np.nanmean(value)) / np.nanstd(value)
    x = np.column_stack([np.ones(valid.sum()), standardized(confidence[valid]),
                         standardized(eu_scores[valid])])
    y = labels[valid].astype(np.float64)
    weights = np.zeros(x.shape[1])
    for _ in range(100):
        probability = 1 / (1 + np.exp(-(x @ weights)))
        gradient = x.T @ (probability - y) / len(y)
        weights -= 0.1 * gradient
    combined = x @ weights
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
            "whitened_eigen_fraction_0.9_1.1": float(np.mean((eigenvalues >= 0.9) & (eigenvalues <= 1.1))),
        },
        "g3": {
            "confidence_auc": auc(labels[valid], confidence[valid]),
            "combined_auc": auc(labels[valid], combined),
            "incremental_auc": auc(labels[valid], combined) - auc(labels[valid], confidence[valid]),
            "eu_coefficient_standardized": float(weights[2]),
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
