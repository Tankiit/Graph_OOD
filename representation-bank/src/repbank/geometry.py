from __future__ import annotations

import numpy as np


def covariance(values: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    centered = values - values.mean(0, keepdims=True)
    result = centered.T @ centered / max(1, len(values) - 1)
    if ridge:
        result = result + ridge * np.eye(result.shape[0], dtype=result.dtype)
    return result


def reduced_basis(values: np.ndarray, max_dim: int = 128) -> tuple[np.ndarray, np.ndarray]:
    center = values.mean(0)
    _, _, right = np.linalg.svd(values - center, full_matrices=False)
    dimension = min(max_dim, len(values) - 2, right.shape[0])
    if dimension < 1:
        raise ValueError("not enough rows for a reduced basis")
    return center, right[:dimension].T


def solve_direction(correct: np.ndarray, wrong: np.ndarray, ridge_fraction: float = 0.1):
    difference = correct.mean(0) - wrong.mean(0)
    pooled = (covariance(correct) + covariance(wrong)) / 2
    ridge = ridge_fraction * np.trace(pooled) / pooled.shape[0]
    direction = np.linalg.solve(pooled + ridge * np.eye(pooled.shape[0]), difference)
    return direction / np.linalg.norm(direction), difference / np.linalg.norm(difference)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))


def whitened_eigenvalues(correct: np.ndarray, wrong: np.ndarray, ridge_fraction: float = 0.1):
    left = covariance(correct)
    right = covariance(wrong)
    ridge = ridge_fraction * np.trace(left) / left.shape[0]
    values, vectors = np.linalg.eigh(left + ridge * np.eye(left.shape[0]))
    inverse_root = (vectors * (1 / np.sqrt(values))) @ vectors.T
    transformed = inverse_root @ right @ inverse_root
    return np.linalg.eigvalsh(transformed)


def covariance_mismatch(correct: np.ndarray, wrong: np.ndarray) -> float:
    left = covariance(correct)
    right = covariance(wrong)
    return float(np.linalg.norm(right - left, ord="fro") / np.linalg.norm(left, ord="fro"))
