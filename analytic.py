"""Population Gaussian-mixture Mahalanobis predictions, without simulation."""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import chi2, norm


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


def contaminated_moments(world, pi):
    if not np.isfinite(pi) or not 0 <= pi <= 1:
        raise ValueError("pi must lie in [0, 1]")
    mu, Sigma, Delta = world
    return mu + pi * Delta, Sigma + pi * (1 - pi) * np.outer(Delta, Delta)


def _component_cdf(t, d, inflation, offset):
    # Whitened score: chi2_(d-1) + (Z+offset)^2/inflation.
    if t <= 0:
        return 0.0
    bound = np.sqrt(inflation * t)
    if d == 1:
        return float(norm.cdf(bound - offset) - norm.cdf(-bound - offset))
    value, _ = quad(
        lambda z: (
            norm.pdf(z - offset) * chi2.cdf(max(0.0, t - z * z / inflation), d - 1)
        ),
        -bound,
        bound,
        epsabs=1e-9,
        epsrel=1e-8,
        limit=150,
    )
    return float(value)


@lru_cache(maxsize=512)
def _threshold(d, sep, pi, protocol, quantile):
    if pi == 0 or sep == 0:
        return float(chi2.ppf(quantile, d))
    inflation = 1 + pi * (1 - pi) * sep**2

    def cdf(t):
        clean = _component_cdf(t, d, inflation, -pi * sep)
        if protocol == "A":
            return clean
        other = _component_cdf(t, d, inflation, (1 - pi) * sep)
        return (1 - pi) * clean + pi * other

    upper = max(1.0, float(chi2.ppf(quantile, d)))
    while cdf(upper) < quantile:
        upper *= 2
    return float(brentq(lambda t: cdf(t) - quantile, 0, upper, xtol=1e-9))


def contaminated_threshold(world, pi, protocol, quantile=0.95):
    """Population quantile including covariance inflation, offset, and mixing.

    A integrates only P scores. B integrates (1-pi)P + pi Q scores. These
    are population predictions, not exact finite-sample calibration formulas.
    """
    if protocol not in ("A", "B") or not 0 < quantile < 1:
        raise ValueError("protocol must be A/B and quantile must lie in (0,1)")
    contaminated_moments(world, pi)
    mu, Sigma, Delta = world
    sep = float(np.sqrt(max(0.0, Delta @ np.linalg.solve(Sigma, Delta))))
    return _threshold(len(mu), sep, float(pi), protocol, float(quantile))


def alpha_star_maha_contaminated(x, v, world, pi, protocol, *, quantile=0.95):
    """Population rejection boundary with lambda=1+pi(1-pi)sep^2."""
    mu, covariance = contaminated_moments(world, pi)
    threshold = contaminated_threshold(world, pi, protocol, quantile)
    return alpha_star_maha(x, v, mu, covariance, threshold)
