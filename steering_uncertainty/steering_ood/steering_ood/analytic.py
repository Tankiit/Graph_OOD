"""Exact crossing control, extracted from the user steering checkout."""
import numpy as np

def alpha_star_maha(x, v, mu, Sigma, t):
    """First rejection boundary for squared Mahalanobis score along x+alpha*v.

    Return the smaller nonnegative *exit* root, or None if there is no exit.
    An initially rejected point returns zero (even if it later enters). This
    convention agrees with first_rejection, including zero-length directions.
    """
    x, v, mu, Sigma = (np.asarray(z, dtype=float) for z in (x, v, mu, Sigma))
    if (
        x.ndim != 1
        or x.size == 0
        or v.shape != x.shape
        or mu.shape != x.shape
        or Sigma.shape != (x.size, x.size)
        or not np.isfinite(t)
        or not all(np.isfinite(z).all() for z in (x, v, mu, Sigma))
    ):
        raise ValueError("require compatible finite vectors, covariance and threshold")
    if not np.allclose(Sigma, Sigma.T):
        raise ValueError("Sigma must be symmetric positive definite")
    np.linalg.cholesky(Sigma)
    y = x - mu
    inv_y, inv_v = np.linalg.solve(Sigma, np.column_stack((y, v))).T
    a, b, c = float(v @ inv_v), float(2 * v @ inv_y), float(y @ inv_y - t)
    if c > 0:
        return 0.0
    if a == 0:
        return None
    discriminant = max(0.0, b * b - 4 * a * c)
    root = np.sqrt(discriminant)
    # Stable evaluation of the positive exit root when b is large and positive.
    return float(-2 * c / (b + root) if b > 0 else (-b + root) / (2 * a))
