"""Straight-line rejection paths in unconstrained Euclidean space."""

from dataclasses import dataclass

import numpy as np


def random_dirs(x, m, rng):
    """Return m isotropic unit vectors; points x+alpha*v are never projected."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("x must be a finite nonempty vector")
    if not isinstance(m, (int, np.integer)) or m < 0:
        raise ValueError("m must be a nonnegative integer")
    vectors = rng.normal(size=(m, len(x)))
    norms = np.linalg.norm(vectors, axis=1)
    while np.any(norms == 0):
        vectors[norms == 0] = rng.normal(size=(np.sum(norms == 0), len(x)))
        norms = np.linalg.norm(vectors, axis=1)
    return vectors / norms[:, None]


@dataclass
class PathTrace:
    alphas: np.ndarray
    scores: np.ndarray
    rejected: np.ndarray
    alpha_star: object
    censored: bool
    status: str
    n_recross: int
    first_bracket: object
    return_brackets: tuple


def trace_path(score_fn, t, x, v, alphas, *, refine_steps=30, grid_scores=None):
    """Measure a full path and its first observed rejection, score(x+alpha*v)>t.

    alphas must start at zero and strictly increase. A detected first crossing
    is bisected; alpha_star is its boundary (zero if already rejected). None
    denotes right censoring at alphas[-1]. n_recross counts all subsequent
    state changes; return_brackets records only rejected-to-accepted changes.
    A grid can miss excursions between samples, particularly for nonmonotone
    spectral/kNN scores. first_bracket is the original grid bracket; bisection
    does not prove there were no earlier excursions. grid_scores may cache
    evaluations of the same fitted scorer at precisely these path points.
    """
    x, v = np.asarray(x, dtype=float), np.asarray(v, dtype=float)
    alphas = np.asarray(alphas, dtype=float)
    if (
        x.ndim != 1
        or x.size == 0
        or v.shape != x.shape
        or not np.isfinite([x, v]).all()
    ):
        raise ValueError("x and v must be finite vectors of the same shape")
    if (
        alphas.ndim != 1
        or not len(alphas)
        or alphas[0] != 0
        or not np.isfinite(alphas).all()
        or np.any(np.diff(alphas) <= 0)
    ):
        raise ValueError("alphas must be finite, start at zero and strictly increase")
    if np.isnan(t) or t == -np.inf or not isinstance(refine_steps, int) or refine_steps < 0:
        raise ValueError(
            "require finite or +inf threshold and nonnegative integer refine_steps"
        )
    scores = np.asarray(
        score_fn(x[None, :] + alphas[:, None] * v) if grid_scores is None else grid_scores,
        dtype=float,
    )
    if scores.shape != alphas.shape or not np.isfinite(scores).all():
        raise ValueError("score_fn must return one finite score per point")
    rejected = scores > t
    returns = tuple(
        (float(alphas[i]), float(alphas[i + 1]))
        for i in np.flatnonzero(rejected[:-1] & ~rejected[1:])
    )
    indices = np.flatnonzero(rejected)
    if not len(indices):
        return PathTrace(alphas, scores, rejected, None, True,
                         "uninformative_threshold" if t == np.inf else "no_crossing_within_cap",
                         0, None, returns)
    first = int(indices[0])
    n_recross = int(np.count_nonzero(np.diff(rejected[first:].astype(int))))
    if first == 0:
        return PathTrace(alphas, scores, rejected, 0.0, False, "already_rejected",
                         n_recross, (0.0, 0.0), returns)
    lo, hi = alphas[first - 1], alphas[first]
    bracket = (float(lo), float(hi))
    for _ in range(refine_steps):
        mid = (lo + hi) / 2
        value = float(np.asarray(score_fn((x + mid * v)[None, :])).item())
        if not np.isfinite(value):
            raise ValueError("score_fn returned a nonfinite score during refinement")
        if value > t:
            hi = mid
        else:
            lo = mid
    return PathTrace(alphas, scores, rejected, float(hi), False, "crossed",
                     n_recross, bracket, returns)


def first_rejection(score_fn, t, x, v, alphas, *, refine_steps=30):
    """Backward-compatible (alpha_star, censored, n_recross) path summary."""
    trace = trace_path(score_fn, t, x, v, alphas, refine_steps=refine_steps)
    return trace.alpha_star, trace.censored, trace.n_recross
