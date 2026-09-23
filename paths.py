"""Straight-line rejection paths in unconstrained Euclidean space."""

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


def first_rejection(score_fn, t, x, v, alphas, *, refine_steps=30):
    """Return (alpha_star, censored, n_recross) for score(x+alpha*v)>t.

    alphas must start at zero and strictly increase. A detected first crossing
    is bisected; alpha_star is its boundary (zero if already rejected). None
    denotes right censoring at alphas[-1]. n_recross counts subsequent changes
    of acceptance state on the supplied grid. A grid can miss excursions
    between samples, particularly for nonmonotone spectral/kNN scores.
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
    if not np.isfinite(t) or not isinstance(refine_steps, int) or refine_steps < 0:
        raise ValueError(
            "require finite threshold and nonnegative integer refine_steps"
        )
    scores = np.asarray(score_fn(x[None, :] + alphas[:, None] * v), dtype=float)
    if scores.shape != alphas.shape or not np.isfinite(scores).all():
        raise ValueError("score_fn must return one finite score per point")
    rejected = scores > t
    indices = np.flatnonzero(rejected)
    if not len(indices):
        return None, True, 0
    first = int(indices[0])
    n_recross = int(np.count_nonzero(np.diff(rejected[first:].astype(int))))
    if first == 0:
        return 0.0, False, n_recross
    lo, hi = alphas[first - 1], alphas[first]
    for _ in range(refine_steps):
        mid = (lo + hi) / 2
        value = float(np.asarray(score_fn((x + mid * v)[None, :])).item())
        if not np.isfinite(value):
            raise ValueError("score_fn returned a nonfinite score during refinement")
        if value > t:
            hi = mid
        else:
            lo = mid
    return float(hi), False, n_recross
