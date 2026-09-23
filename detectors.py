"""Anomaly detectors. Larger scores always mean stronger rejection."""

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist, pdist


def _matrix(X, minimum=1):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) < minimum or X.shape[1] == 0 or not np.isfinite(X).all():
        raise ValueError(f"expected a finite 2-D array with at least {minimum} rows")
    return X


class Maha:
    """Squared Mahalanobis distance using the sample covariance (ddof=1).

    ridge is an absolute diagonal covariance regularizer. Set it to zero for
    the unregularized estimator; the default keeps bootstrap fits invertible.
    """

    def __init__(self, ridge=1e-8):
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be nonnegative")
        self.ridge = ridge

    def fit(self, D):
        D = _matrix(D, 2)
        self.mu = D.mean(axis=0)
        centered = D - self.mu
        self.Sigma = centered.T @ centered / (len(D) - 1)
        self.Sigma += self.ridge * np.eye(D.shape[1])
        np.linalg.cholesky(self.Sigma)
        self.precision = np.linalg.solve(self.Sigma, np.eye(D.shape[1]))
        return self

    def score(self, X):
        centered = _matrix(X) - self.mu
        return np.einsum("ni,ij,nj->n", centered, self.precision, centered)


class KNN:
    """Euclidean distance to the kth neighbour in the fitted pool.

    Pool calibration includes each point itself, as a literal in-pool score.
    k>=2 avoids the identically-zero k=1 pool threshold for distinct points.
    """

    def __init__(self, k=5):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k

    def fit(self, D):
        self.D = _matrix(D, self.k).copy()
        return self

    def score(self, X):
        distances = cdist(_matrix(X), self.D)
        return np.partition(distances, self.k - 1, axis=1)[:, self.k - 1]


class Spectral:
    """Exact insertion change in lambda_2 of the unnormalized RBF Laplacian.

    W_ij=exp(-||x_i-x_j||^2/(2 bandwidth^2)), W_ii=0, L=diag(W1)-W.
    The median nonzero pairwise distance sets bandwidth at fit time; it is
    held fixed for all candidate insertions. No approximate eigensolver is
    used. ``score_signed`` returns lambda_2(D union {x})-lambda_2(D),
    ``score`` its absolute value. Each candidate is inserted independently.
    """

    def __init__(self, bandwidth=None):
        if bandwidth is not None and (not np.isfinite(bandwidth) or bandwidth <= 0):
            raise ValueError("bandwidth must be positive")
        self.bandwidth = bandwidth

    def fit(self, D):
        self.D = _matrix(D, 2).copy()
        distances = pdist(self.D)
        positive = distances[distances > 0]
        self.bandwidth_ = (
            (float(np.median(positive)) if len(positive) else 1.0)
            if self.bandwidth is None
            else self.bandwidth
        )
        weights = np.exp(
            -cdist(self.D, self.D, "sqeuclidean") / (2 * self.bandwidth_**2)
        )
        np.fill_diagonal(weights, 0)
        self.laplacian_ = np.diag(weights.sum(axis=1)) - weights
        values, vectors = eigh(self.laplacian_)
        self.lambda2_ = float(values[1])
        self.fiedler_ = vectors[:, 1]
        self.fiedler_gap_ = float(values[2] - values[1]) if len(values) > 2 else np.nan
        return self

    def score_signed(self, X):
        X = _matrix(X)
        output = np.empty(len(X))
        for i, x in enumerate(X):
            weights = np.exp(
                -np.sum((self.D - x) ** 2, axis=1) / (2 * self.bandwidth_**2)
            )
            augmented = np.empty((len(self.D) + 1, len(self.D) + 1))
            augmented[:-1, :-1] = self.laplacian_ + np.diag(weights)
            augmented[-1, :-1] = augmented[:-1, -1] = -weights
            augmented[-1, -1] = weights.sum()
            output[i] = (
                eigh(augmented, eigvals_only=True, subset_by_index=[1, 1])[0]
                - self.lambda2_
            )
        return output

    def score(self, X):
        return np.abs(self.score_signed(X))

    def score_components(self, X):
        signed = self.score_signed(X)
        return {"dlambda2": signed, "abs_dlambda2": np.abs(signed)}


class Linear(Maha):
    """Signed linear score w^T(x-mu_hat), w=Sigma_hat^-1 Delta_hat.

    The independent pure-Q reference is required explicitly, never inferred
    from analysis labels. mu_hat and Sigma_hat come from the fitted pool.
    """

    def __init__(self, contaminant=None, ridge=1e-8):
        super().__init__(ridge)
        self.contaminant = contaminant

    def fit(self, D, contaminant=None):
        reference = self.contaminant if contaminant is None else contaminant
        if reference is None:
            raise ValueError("Linear requires the independent contaminant split")
        reference = _matrix(reference)
        super().fit(D)
        if reference.shape[1] != len(self.mu):
            raise ValueError("contaminant and fitted pool dimensions differ")
        self.Delta = reference.mean(axis=0) - self.mu
        self.w = np.linalg.solve(self.Sigma, self.Delta)
        return self

    def score(self, X):
        return (_matrix(X) - self.mu) @ self.w


def calibrate(det, D, calib, quantile=0.95):
    """Protocol A: fit on D, threshold on independent clean calibration."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie in (0, 1)")
    fitted = det.fit(D)
    return fitted, float(np.quantile(fitted.score(calib), quantile))


def calibrate_on_pool(det, D, quantile=0.95):
    """Protocol B: empirical quantile of the fitted pool's own scores."""
    return calibrate(det, D, D, quantile)
