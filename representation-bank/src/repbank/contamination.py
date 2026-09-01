"""Held-out covariance-contamination experiment.

The experiment intentionally changes *only* the class composition of the
covariance-fitting pool.  Class means used to measure the Mahalanobis signal
come from a fixed, question-grouped held-out set.  This makes the outcome an
estimator-bias measurement, rather than an AUC/prevalence measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coupled_extract import verify_checksum


@dataclass(frozen=True)
class ContaminationConfig:
    """Configuration for the layer x contamination covariance experiment.

    ``contamination`` is the fraction of label-0 (hallucinated) rows in the
    covariance fit.  Label-1 rows form the reference distribution.  The
    ridge is estimated from the clean fit and then held fixed within a draw,
    preserving the rank-one covariance prediction under regularization.
    """

    contamination: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    dimensions: tuple[int, ...] = (32, 64, 128)
    folds: int = 5
    repeats: int = 20
    ridge_fraction: float = 0.1
    shrinkage: str = "ridge"
    seed: int = 20_260_901
    pool_size: int | None = None
    blocks: tuple[int, ...] | None = None

    def validate(self) -> None:
        if not self.contamination:
            raise ValueError("at least one contamination value is required")
        if any(not 0 < value < 1 for value in self.contamination):
            raise ValueError("contamination values must lie strictly between zero and one")
        if not self.dimensions or any(value < 1 for value in self.dimensions):
            raise ValueError("dimensions must be positive")
        if self.folds < 2 or self.repeats < 1:
            raise ValueError("folds must be at least two and repeats at least one")
        if self.ridge_fraction <= 0:
            raise ValueError("ridge_fraction must be positive")
        if self.shrinkage not in {"ridge", "oas"}:
            raise ValueError("shrinkage must be 'ridge' or 'oas'")
        if self.pool_size is not None and self.pool_size < 4:
            raise ValueError("pool_size must be at least four")
        if self.blocks is not None and (not self.blocks or any(value < 0 for value in self.blocks)):
            raise ValueError("blocks must be a nonempty sequence of nonnegative indices")


def load_feature_bank(path: str | Path, position: str = "answer_last") -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load one all-layer token-position tensor from a checksumed NPZ bank.

    Legacy banks call their final answer-token representation ``h_last``.
    New extraction banks additionally expose ``h_prompt_last`` and
    ``h_answer_mean``.  All choices have shape ``[rows, layers + 1, width]``.
    """

    path = Path(path)
    verify_checksum(path)
    bank = np.load(path, allow_pickle=False)
    names = {
        "answer_last": "h_last",
        "prompt_last": "h_prompt_last",
        "answer_mean": "h_answer_mean",
    }
    if position not in names:
        raise ValueError(f"unknown position {position!r}; choose {sorted(names)}")
    name = names[position]
    if name not in bank:
        available = sorted(key for key in names if names[key] in bank)
        raise ValueError(
            f"bank {path} does not contain {name}; available positions are {available}. "
            "Re-extract to compare prompt_last or answer_mean."
        )
    metadata = json.loads(str(bank["metadata"]))
    states = bank[name].astype(np.float32)
    labels = bank["label"].astype(np.int8)
    groups = bank["question_id"]
    if states.ndim != 3 or len(states) != len(labels) or len(groups) != len(labels):
        raise ValueError("bank features, labels, and question groups do not align")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("contamination experiment requires both binary labels 0 and 1")
    return states, labels, groups, metadata


def _grouped_folds(groups: np.ndarray, labels: np.ndarray, folds: int) -> np.ndarray:
    """Deterministically balance label counts while keeping question groups intact."""

    unique = np.unique(groups)
    if len(unique) < folds:
        raise ValueError(f"only {len(unique)} question groups for {folds} folds")
    counts = []
    for group in unique:
        selected = labels[groups == group]
        counts.append((group, int((selected == 1).sum()), int((selected == 0).sum())))
    # Large and label-skewed groups first gives the greedy allocation room to balance.
    counts.sort(key=lambda value: (-(value[1] + value[2]), -abs(value[1] - value[2]), str(value[0])))
    total_positive = int((labels == 1).sum())
    total_negative = int((labels == 0).sum())
    fold_positive = np.zeros(folds, dtype=np.int64)
    fold_negative = np.zeros(folds, dtype=np.int64)
    assignment: dict[object, int] = {}
    for group, positive, negative in counts:
        penalties = []
        for fold in range(folds):
            next_positive = fold_positive.copy(); next_positive[fold] += positive
            next_negative = fold_negative.copy(); next_negative[fold] += negative
            penalties.append(
                ((next_positive - total_positive / folds).var()
                 + (next_negative - total_negative / folds).var(),
                 int(fold_positive[fold] + fold_negative[fold]), fold)
            )
        selected_fold = min(penalties)[2]
        assignment[group] = selected_fold
        fold_positive[selected_fold] += positive
        fold_negative[selected_fold] += negative
    return np.array([assignment[group] for group in groups], dtype=np.int16)


def _pca_basis(clean_train: np.ndarray, maximum_dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA only on clean *training* rows, never on a contaminated pool."""

    center = clean_train.mean(axis=0)
    centered = clean_train - center
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    usable = min(maximum_dimension, len(clean_train) - 1, right.shape[0])
    if usable < 1:
        raise ValueError("not enough clean training rows to fit a PCA basis")
    return center, right[:usable].T


def _covariance(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    return centered.T @ centered / (len(values) - 1)


def _quadratic(vector: np.ndarray, precision_covariance: np.ndarray) -> float:
    return float(vector @ np.linalg.solve(precision_covariance, vector))


def _oas_shrinkage(covariance: np.ndarray, rows: int) -> float:
    """Oracle-approximating shrinkage intensity for a centered covariance."""

    dimension = covariance.shape[0]
    # The OAS expression is scale-invariant, so the unbiased (n - 1)
    # covariance here gives the same intensity as the MLE covariance.
    mean_variance = float(np.trace(covariance) / dimension)
    alpha = float(np.mean(covariance ** 2))
    denominator = (rows + 1) * (alpha - mean_variance ** 2 / dimension)
    if denominator <= 0:
        return 1.0
    return float(min(1.0, (alpha + mean_variance ** 2) / denominator))


def _regularize_clean(covariance: np.ndarray, rows: int, config: ContaminationConfig) -> tuple[np.ndarray, dict]:
    dimension = covariance.shape[0]
    if config.shrinkage == "ridge":
        ridge = config.ridge_fraction * float(np.trace(covariance) / dimension)
        return covariance + ridge * np.eye(dimension, dtype=covariance.dtype), {
            "method": "ridge", "ridge": ridge, "shrinkage": 0.0, "rank_one_scale": 1.0,
        }
    shrinkage = _oas_shrinkage(covariance, rows)
    target_variance = float(np.trace(covariance) / dimension)
    return (1 - shrinkage) * covariance + shrinkage * target_variance * np.eye(
        dimension, dtype=covariance.dtype
    ), {
        "method": "oas", "ridge": 0.0, "shrinkage": shrinkage,
        "target_variance": target_variance, "rank_one_scale": 1 - shrinkage,
    }


def _regularize_mixed(covariance: np.ndarray, clean_regularization: dict) -> np.ndarray:
    dimension = covariance.shape[0]
    if clean_regularization["method"] == "ridge":
        return covariance + clean_regularization["ridge"] * np.eye(
            dimension, dtype=covariance.dtype
        )
    return ((1 - clean_regularization["shrinkage"]) * covariance
            + clean_regularization["shrinkage"] * clean_regularization["target_variance"]
            * np.eye(dimension, dtype=covariance.dtype))


def resolve_deflation(
    sigma_clean: np.ndarray,
    sigma_mixed: np.ndarray,
    delta: np.ndarray,
    x_eval: np.ndarray,
    pi: float,
    rank_one_scale: float = 1.0,
) -> dict[str, float]:
    """Resolve covariance change along the fitted Delta direction and its complement.

    ``delta`` is estimated only from the training fold; ``x_eval`` is the
    independent held-out class-mean difference.  Under equal class
    covariances, the mixed covariance is a rank-one update and the ``perp``
    and ``cross`` terms are zero.  The top-level theory prediction is an
    upper bound for a general x; ``alignment_adjusted_prediction`` exposes
    the necessary squared Mahalanobis cosine rather than hiding it.
    """

    m_delta = _quadratic(delta, sigma_clean)
    clean_total = _quadratic(x_eval, sigma_clean)
    if min(m_delta, clean_total) <= 0:
        raise ValueError("clean covariance produced a nonpositive quadratic form")
    coefficient = float(pi * (1 - pi) * rank_one_scale)
    theory_upper = coefficient * m_delta / (1 + coefficient * m_delta)
    projection_weight = float(x_eval @ np.linalg.solve(sigma_clean, delta) / m_delta)
    x_along = projection_weight * delta
    x_perp = x_eval - x_along
    clean_along = _quadratic(x_along, sigma_clean)
    clean_perp = _quadratic(x_perp, sigma_clean)
    mixed_total = _quadratic(x_eval, sigma_mixed)
    mixed_along = _quadratic(x_along, sigma_mixed)
    mixed_perp = _quadratic(x_perp, sigma_mixed)
    mixed_cross = 2 * float(x_along @ np.linalg.solve(sigma_mixed, x_perp))
    rho2 = clean_along / clean_total
    along = 1 - mixed_along / clean_along if clean_along > 1e-12 else float("nan")
    perp = 1 - mixed_perp / clean_perp if clean_perp > 1e-12 else float("nan")
    total = 1 - mixed_total / clean_total
    cross_contribution = -mixed_cross / clean_total
    reconstructed = rho2 * along + (1 - rho2) * perp + cross_contribution
    return {
        "m_delta": m_delta,
        "m_eval": clean_total,
        "rho2": rho2,
        "theory_upper": theory_upper,
        "alignment_adjusted_prediction": rho2 * theory_upper,
        "total": total,
        "along": along,
        "perp": perp,
        "cross_contribution": cross_contribution,
        "reconstruction_error": total - reconstructed,
    }


def _condition_number(covariance: np.ndarray) -> float:
    values = np.linalg.eigvalsh(covariance)
    largest = float(values[-1])
    smallest = float(values[0])
    if smallest <= largest * np.finfo(values.dtype).eps:
        return float("inf")
    return largest / smallest


def _summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {
            "mean": None, "median": None, "q025": None, "q975": None,
            "nonfinite_draws": len(array),
        }
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "q025": float(np.quantile(finite, 0.025)),
        "q975": float(np.quantile(finite, 0.975)),
        "nonfinite_draws": int(len(array) - len(finite)),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def contamination_grid(
    states: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: ContaminationConfig | None = None,
) -> dict:
    """Run the fixed-evaluation layer x contamination experiment.

    The reported outcome resolves the held-out quadratic-form change into the
    fitted-Delta direction and its clean-Mahalanobis orthogonal complement.
    For ``Sigma + pi(1-pi) Delta Delta^T``, Sherman--Morrison predicts the
    upper-bound along-direction deflation ``c / (1 + c)`` and zero in the
    complement.  Training and held-out estimates of Delta are deliberately
    kept distinct.
    """

    config = config or ContaminationConfig()
    config.validate()
    if states.ndim != 3:
        raise ValueError("states must have shape [rows, layers + 1, width]")
    if len(states) != len(labels) or len(labels) != len(groups):
        raise ValueError("states, labels, and groups must have the same row count")
    folds = _grouped_folds(groups, labels, config.folds)
    max_dimension = max(config.dimensions)
    block_indices = config.blocks or tuple(range(states.shape[1] - 1))
    if any(block >= states.shape[1] - 1 for block in block_indices):
        raise ValueError(f"block indices must be below {states.shape[1] - 1}")
    per_fold_counts = []
    for fold in range(config.folds):
        train = folds != fold
        clean = int(np.sum(labels[train] == 1))
        contaminant = int(np.sum(labels[train] == 0))
        test_positive = int(np.sum(labels[~train] == 1))
        test_negative = int(np.sum(labels[~train] == 0))
        if min(clean, contaminant, test_positive, test_negative) < 2:
            raise ValueError("each train and held-out fold needs both classes")
        per_fold_counts.append({"fold": fold, "train_clean": clean,
                                "train_contaminant": contaminant,
                                "test_clean": test_positive, "test_contaminant": test_negative})
    feasible_pool = min(
        count["train_clean"] for count in per_fold_counts
    )
    for pi in config.contamination:
        feasible_pool = min(
            feasible_pool,
            *(int(count["train_clean"] // (1 - pi)) for count in per_fold_counts),
            *(int(count["train_contaminant"] // pi) for count in per_fold_counts),
        )
    pool_size = config.pool_size or feasible_pool
    if pool_size > feasible_pool:
        raise ValueError(f"pool_size={pool_size} exceeds the common feasible maximum {feasible_pool}")

    # A cell stores draws across outer folds and independent subsampling draws.
    cells: dict[tuple[int, int, float], dict[str, list[float]]] = {}
    for block in block_indices:
        for dimension in config.dimensions:
            for pi in config.contamination:
                cells[(block, dimension, pi)] = {
                    "m_delta": [], "m_eval": [], "rho2": [], "theory_upper": [],
                    "alignment_adjusted_prediction": [], "total": [], "along": [], "perp": [],
                    "cross_contribution": [], "reconstruction_error": [],
                    "clean_condition": [], "mixed_condition": [],
                    "clean_regularized_condition": [], "mixed_regularized_condition": [],
                    "ridge": [], "shrinkage": [],
                }

    rng = np.random.default_rng(config.seed)
    for fold in range(config.folds):
        train = folds != fold
        test = ~train
        clean_indices = np.flatnonzero(train & (labels == 1))
        contaminant_indices = np.flatnonzero(train & (labels == 0))
        layer_contexts = []
        for block in block_indices:
            clean_train = states[clean_indices, block + 1]
            contaminant_train = states[contaminant_indices, block + 1]
            center, basis = _pca_basis(clean_train, max_dimension)
            # This projection is fit once per outer fold and is identical for
            # every pi and resample; contamination cannot select its basis.
            reduced_all = (states[:, block + 1] - center) @ basis
            heldout = reduced_all[test]
            heldout_labels = labels[test]
            delta_train = (clean_train.mean(axis=0) - contaminant_train.mean(axis=0)) @ basis
            x_eval = heldout[heldout_labels == 1].mean(axis=0) - heldout[heldout_labels == 0].mean(axis=0)
            layer_contexts.append((reduced_all, delta_train, x_eval))
        for repeat in range(config.repeats):
            clean_draw = rng.choice(clean_indices, size=pool_size, replace=False)
            mixtures: dict[float, np.ndarray] = {}
            for pi in config.contamination:
                contaminated_count = round(pool_size * pi)
                clean_count = pool_size - contaminated_count
                # The exact realized pi is reported below; non-integer grid points
                # are inevitable for a finite, without-replacement pool.
                mixtures[pi] = np.concatenate([
                    rng.choice(clean_indices, size=clean_count, replace=False),
                    rng.choice(contaminant_indices, size=contaminated_count, replace=False),
                ])
            for context_index, block in enumerate(block_indices):
                reduced_all, delta_train, x_eval = layer_contexts[context_index]
                clean_reduced = reduced_all[clean_draw]
                clean_covariance = _covariance(clean_reduced)
                for dimension in config.dimensions:
                    selected_delta = delta_train[:dimension]
                    selected_x_eval = x_eval[:dimension]
                    selected_clean_covariance = clean_covariance[:dimension, :dimension]
                    regularized_clean, regularization = _regularize_clean(
                        selected_clean_covariance, len(clean_draw), config
                    )
                    clean_condition = _condition_number(selected_clean_covariance)
                    for requested_pi, mixed_draw in mixtures.items():
                        mixed_reduced = reduced_all[mixed_draw, :dimension]
                        mixed_covariance = _covariance(mixed_reduced)
                        regularized_mixed = _regularize_mixed(mixed_covariance, regularization)
                        realized_pi = round(pool_size * requested_pi) / pool_size
                        resolved = resolve_deflation(
                            regularized_clean, regularized_mixed, selected_delta, selected_x_eval,
                            realized_pi, regularization["rank_one_scale"],
                        )
                        cell = cells[(block, dimension, requested_pi)]
                        for key in (
                            "m_delta", "m_eval", "rho2", "theory_upper",
                            "alignment_adjusted_prediction", "total", "along", "perp",
                            "cross_contribution", "reconstruction_error",
                        ):
                            cell[key].append(resolved[key])
                        cell["clean_condition"].append(clean_condition)
                        cell["mixed_condition"].append(_condition_number(mixed_covariance))
                        cell["clean_regularized_condition"].append(_condition_number(regularized_clean))
                        cell["mixed_regularized_condition"].append(_condition_number(regularized_mixed))
                        cell["ridge"].append(regularization["ridge"])
                        cell["shrinkage"].append(regularization["shrinkage"])

    rows = []
    for (block, dimension, pi), values in sorted(cells.items()):
        rows.append({
            "block_index": block,
            "fractional_depth": block / max(1, states.shape[1] - 2),
            "dimension": dimension,
            "requested_pi": pi,
            "realized_pi": round(pool_size * pi) / pool_size,
            "draws": len(values["total"]),
            "m_delta_train": _summary(values["m_delta"]),
            "m_eval_heldout": _summary(values["m_eval"]),
            "rho2_alignment": _summary(values["rho2"]),
            "theory_upper_deflation": _summary(values["theory_upper"]),
            "alignment_adjusted_prediction": _summary(values["alignment_adjusted_prediction"]),
            "observed_total_deflation": _summary(values["total"]),
            "observed_along_deflation": _summary(values["along"]),
            "observed_perp_deflation": _summary(values["perp"]),
            "cross_contribution_to_total_deflation": _summary(values["cross_contribution"]),
            "resolution_reconstruction_error": _summary(values["reconstruction_error"]),
            "clean_covariance_condition_number": _summary(values["clean_condition"]),
            "mixed_covariance_condition_number": _summary(values["mixed_condition"]),
            "clean_regularized_covariance_condition_number": _summary(values["clean_regularized_condition"]),
            "mixed_regularized_covariance_condition_number": _summary(values["mixed_regularized_condition"]),
            "ridge": _summary(values["ridge"]), "shrinkage": _summary(values["shrinkage"]),
        })
    validation = []
    for dimension in config.dimensions:
        selected = [row for row in rows if row["dimension"] == dimension]
        predicted = np.array([row["alignment_adjusted_prediction"]["mean"] for row in selected])
        observed = np.array([row["observed_total_deflation"]["mean"] for row in selected])
        validation.append({
            "dimension": dimension,
            "cells": len(selected),
            "pearson_total_vs_alignment_adjusted_prediction": _correlation(predicted, observed),
            "pearson_along_vs_theory_upper": _correlation(
                np.array([row["theory_upper_deflation"]["mean"] for row in selected]),
                np.array([row["observed_along_deflation"]["mean"] for row in selected]),
            ),
            "mean_absolute_error": float(np.mean(np.abs(observed - predicted))),
        })
    return {
        "protocol": {
            "outcome": "held-out Mahalanobis-squared relative deflation",
            "fixed_evaluation": "question-grouped held-out class means; evaluation prevalence never varies",
            "covariance_fit": "label-1 reference rows contaminated by label-0 rows only",
            "covariance_centering": "each clean or mixed fitting pool is centered on its own pooled empirical mean",
            "basis": "PCA fit only on label-1 training rows, fixed before each contamination draw",
            "regularization": "regularizer fit on each clean covariance draw and held fixed across pi",
            "prediction": "along: c/(1+c), c=pi(1-pi)m_delta; total: rho2*c/(1+c)",
            "resolution": "along Delta direction, clean-Mahalanobis orthogonal complement, and cross term",
        },
        "config": {
            "contamination": list(config.contamination), "dimensions": list(config.dimensions),
            "folds": config.folds, "repeats": config.repeats,
            "ridge_fraction": config.ridge_fraction, "shrinkage": config.shrinkage, "seed": config.seed,
            "pool_size": pool_size,
            "blocks": list(block_indices),
        },
        "fold_counts": per_fold_counts,
        "warnings": ([
            (
                "The requested pi range ends at 0.5, so it tests only the increasing arm of pi(1-pi). "
                "Add values above 0.5 (with a smaller common pool if necessary) to test the predicted turnover."
            )
        ] if max(config.contamination) <= 0.5 else []),
        "theory_validation": validation,
        "cells": rows,
    }


def merge_contamination_outputs(paths: list[str | Path]) -> dict:
    """Merge non-overlapping layer chunks from one contamination configuration."""

    if not paths:
        raise ValueError("at least one chunk is required")
    reports = [json.loads(Path(path).read_text()) for path in paths]
    first = reports[0]
    invariant_config = {key: value for key, value in first["config"].items() if key != "blocks"}
    invariant = {key: first.get(key) for key in ("bank", "position", "bank_metadata", "protocol")}
    cells: dict[tuple[int, int, float], dict] = {}
    for report in reports:
        current_config = {key: value for key, value in report["config"].items() if key != "blocks"}
        current = {key: report.get(key) for key in ("bank", "position", "bank_metadata", "protocol")}
        if current_config != invariant_config or current != invariant:
            raise ValueError("cannot merge chunks from different contamination configurations")
        for cell in report["cells"]:
            key = (cell["block_index"], cell["dimension"], cell["requested_pi"])
            if key in cells:
                raise ValueError(f"duplicate contamination cell {key}")
            cells[key] = cell
    merged = {key: value for key, value in first.items()
              if key not in {"cells", "theory_validation", "warnings", "config"}}
    merged["config"] = {**invariant_config, "blocks": sorted({key[0] for key in cells})}
    merged["cells"] = [cells[key] for key in sorted(cells)]
    merged["warnings"] = sorted({warning for report in reports for warning in report["warnings"]})
    validation = []
    for dimension in invariant_config["dimensions"]:
        selected = [cell for cell in merged["cells"] if cell["dimension"] == dimension]
        predicted = np.array([cell["alignment_adjusted_prediction"]["mean"] for cell in selected])
        observed = np.array([cell["observed_total_deflation"]["mean"] for cell in selected])
        validation.append({
            "dimension": dimension,
            "cells": len(selected),
            "pearson_total_vs_alignment_adjusted_prediction": _correlation(predicted, observed),
            "pearson_along_vs_theory_upper": _correlation(
                np.array([cell["theory_upper_deflation"]["mean"] for cell in selected]),
                np.array([cell["observed_along_deflation"]["mean"] for cell in selected]),
            ),
            "mean_absolute_error": float(np.mean(np.abs(observed - predicted))),
        })
    merged["theory_validation"] = validation
    return merged
