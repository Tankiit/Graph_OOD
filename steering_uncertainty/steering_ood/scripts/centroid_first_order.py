"""First-order (delta-method) check of centroid-induced variance in alpha*.

Centroid detector: reject z when ||z - mu_hat|| > t. Along z = x + a v (unit v),
alpha* solves ||x + a v - mu_hat|| = t. Differentiating that identity gives

    delta alpha* = (u*^T delta mu + delta t) / kappa,
    u* = (z* - mu_hat) / ||z* - mu_hat||,   kappa = u*^T v   (at the crossing z*).

With mu_hat the mean of n draws from N(mu, Sigma) and t fixed,
    Var(alpha*) ~= u*^T Sigma u* / (n kappa^2)   (= sigma^2 / (n kappa^2) if isotropic).

Usage: python scripts/centroid_first_order.py
"""
import numpy as np


class IsotropicWorld:
    def __init__(self, d, sigma=1.0):
        self.d, self.sigma, self.mu = d, sigma, np.zeros(d)

    def sample_reference(self, n, seed):
        return self.mu + self.sigma * np.random.default_rng(seed).normal(size=(n, self.d))


def centroid_alpha_star(x, v, mu_hat, t):
    """Smallest a >= 0 with ||x + a v - mu_hat|| = t (unit v); 0 if already outside, inf if never."""
    w = x - mu_hat
    b, c = w @ v, w @ w - t * t          # a^2 + 2 b a + c = 0
    if c >= 0:
        return 0.0                        # already rejected, matching the tracer's convention
    return -b + np.sqrt(b * b - c)        # c < 0 so the discriminant is positive and the root > 0


def crossing_geometry(x, v, mu, t):
    a = centroid_alpha_star(x, v, mu, t)
    u = (x + a * v - mu) / t
    return a, u, u @ v


def probe_path(world, r0, psi):
    """Probe at distance r0 from the centroid; v at angle psi from the outward radial direction."""
    e1, e2 = np.eye(world.d)[0], np.eye(world.d)[1]
    return world.mu + r0 * e1, np.cos(psi) * e1 + np.sin(psi) * e2


def check_first_order(world, x, v, t, n, B=2000):
    mus = [world.sample_reference(n, seed=b).mean(0) for b in range(B)]
    a_emp = np.array([centroid_alpha_star(x, v, m, t) for m in mus])
    _, u, kappa = crossing_geometry(x, v, world.mu, t)
    predicted = world.sigma ** 2 * (u @ u) / (n * kappa ** 2)
    return a_emp.var(), predicted, kappa, float(np.mean(a_emp == 0))


def main():
    n, t, B = 100, 3.0, 4000

    print('Check 1: radial path, isotropic world  (expect n*Var/sigma^2 = 1, kappa = 1)')
    for d in (2, 16, 256):
        for sigma in (0.5, 1.0):
            world = IsotropicWorld(d, sigma)
            x, v = probe_path(world, 1.0, 0.0)
            emp, pred, kappa, _ = check_first_order(world, x, v, t, n, B)
            print(f'  d={d:4d} sigma={sigma}: kappa={kappa:.3f}  n*Var_emp/s^2={n*emp/sigma**2:.3f}'
                  f'  n*Var_pred/s^2={n*pred/sigma**2:.3f}')
    print('  1/n scaling (d=16, sigma=1):')
    world = IsotropicWorld(16)
    x, v = probe_path(world, 1.0, 0.0)
    for nn in (25, 100, 400):
        emp, pred, _, _ = check_first_order(world, x, v, t, nn, B)
        print(f'    n={nn:4d}  Var_emp={emp:.5f}  pred={pred:.5f}  ratio={emp/pred:.3f}')

    print('\nCheck 2a: rotate v away from radial, probe well inside (r0=2, t=3)')
    for d in (2, 64):
        world = IsotropicWorld(d)
        for deg in (0, 30, 60, 75, 90):
            x, v = probe_path(world, 2.0, np.radians(deg))
            emp, pred, kappa, clip = check_first_order(world, x, v, t, n, B)
            print(f'  d={d:3d} psi={deg:3d}  kappa={kappa:.3f}  Var_emp={emp:.5f}  Var_pred={pred:.5f}'
                  f'  emp/pred={emp/pred:.3f}  clipped={clip:.3f}')

    print('\nCheck 2b: tangential path (psi=90), r0 -> t drives kappa -> 0')
    crit = np.sqrt(1.0 / (2 * t * np.sqrt(n)))
    print(f'  curvature breakdown kappa_c = sqrt(sigma/(2 t sqrt n)) = {crit:.3f}')
    for d in (2, 64):
        world = IsotropicWorld(d)
        print(f'  d={d}: dimension term sqrt(2(d-1)) sigma/(2 sqrt(n) t) = {np.sqrt(2*(d-1))/(2*np.sqrt(n)*t):.3f}')
        for kappa_target in (0.6, 0.4, 0.25, 0.18, 0.13, 0.09, 0.06):
            r0 = t * np.sqrt(1 - kappa_target ** 2)
            x, v = probe_path(world, r0, np.pi / 2)
            emp, pred, kappa, clip = check_first_order(world, x, v, t, n, B)
            print(f'    kappa={kappa:.3f} (kappa/kappa_c={kappa/crit:.2f})  emp/pred={emp/pred:.3f}  clipped={clip:.3f}')

if __name__ == '__main__':
    main()
