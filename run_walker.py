"""First run: mean shift, equal covariance, Mahalanobis, straight path along Delta.

Three figures, written to results/walker/:
  1. walked vs analytic alpha* on the *fitted* detector   (tests the walker code)
  2. walked vs analytic alpha* on the *population* P      (finite-reference error)
  3. cosine(v_hat, v_true) over 50 redraws of the dev pool

Fixed choices: PI_DEV is the dev-pool contamination, set once and not revisited.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from analytic import alpha_star_maha
from detectors import Maha, calibrate
from paths import first_rejection

# Load src/data/synth.py by path: importing the src package pulls in the
# steering stack, whose src/data/__init__.py currently expects names that
# synth.py no longer defines.
_spec = importlib.util.spec_from_file_location(
    "synth", Path(__file__).parent / "src" / "data" / "synth.py")
_synth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_synth)
draw, make_world, splits = _synth.draw, _synth.make_world, _synth.splits

PI_DEV = 0.2          # dev pool = (1-PI_DEV) P + PI_DEV Q; estimates the direction
QUANTILE = 0.95       # tau = 0.05
N_GRID = 40
N_REDRAW = 50


def unit(z):
    return z / np.linalg.norm(z)


def estimate_direction(dev_X, ref_X):
    """v_hat from unlabeled data only: mean(dev pool) - mean(clean ref)."""
    return unit(dev_X.mean(0) - ref_X.mean(0))


def check_sep_invariance(d, seed, t):
    """alpha* from x=mu along Delta/|Delta| is sqrt(t / v^T Sigma^-1 v).

    v^T Sigma^-1 v = sep^2/|Delta|^2 and |Delta| = sep |u| / sqrt(u^T Sigma^-1 u),
    so alpha* = sqrt(t) |u| / sqrt(u^T Sigma^-1 u): sep cancels, only the
    direction of u and the P-calibrated threshold t remain.
    """
    out = {}
    for sep in (0.5, 1.0, 3.0, 10.0):
        w = make_world(d=d, sep=sep, seed=seed)
        v = unit(w["Delta"])
        closed = np.sqrt(t / (v @ w["Sinv"] @ v))
        root = alpha_star_maha(w["mu"], v, w["mu"], w["Sigma"], t)
        assert np.isclose(closed, root), (closed, root)
        out[sep] = closed
    assert np.allclose(list(out.values()), out[3.0])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=20)
    p.add_argument("--sep", type=float, default=3.0)
    p.add_argument("--grid", type=int, default=N_GRID, help="grid size K")
    p.add_argument("--n-ref", type=int, default=500, help="reference size n")
    p.add_argument("--pi-dev", type=float, default=PI_DEV)
    p.add_argument("--seed", type=int, default=1, help="data seed (world seed is 0)")
    p.add_argument("--output", default="results/walker")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    world = make_world(d=args.d, sep=args.sep)
    rng = np.random.default_rng(args.seed)
    S = splits(world, rng, n_ref=args.n_ref, pi_dev=args.pi_dev)
    mu, Sigma = world["mu"], world["Sigma"]
    v_true = unit(world["Delta"])

    dev_X, _ = S["dev"]            # labels never used
    v_hat = estimate_direction(dev_X, S["ref"])
    cos_hat = float(v_hat @ v_true)

    det, t = calibrate(Maha(), S["ref"], S["calib"], quantile=QUANTILE)
    t_pop = float(chi2.ppf(QUANTILE, args.d))

    # --- answer key ---------------------------------------------------------
    fitted = np.array([alpha_star_maha(x, v_true, det.mu, det.Sigma, t)
                       for x in S["probe"]])
    population = np.array([alpha_star_maha(x, v_true, mu, Sigma, t_pop)
                           for x in S["probe"]])
    # A quadratic with a > 0 and c <= 0 always has a nonnegative exit root.
    assert not any(a is None for a in fitted) and not any(a is None for a in population)

    alpha_max = 1.25 * max(fitted.max(), population.max())
    alphas = np.linspace(0, alpha_max, args.grid)
    step = alphas[1]

    # --- the walker: raw grid (refine_steps=0) and bisected ------------------
    walked_raw, walked_bis, recross, censored = [], [], [], []
    for x in S["probe"]:
        a0, c0, r0 = first_rejection(det.score, t, x, v_true, alphas, refine_steps=0)
        a1, _, _ = first_rejection(det.score, t, x, v_true, alphas)
        walked_raw.append(np.nan if c0 else a0)
        walked_bis.append(np.nan if c0 else a1)
        recross.append(r0)
        censored.append(c0)
    walked_raw, walked_bis = np.array(walked_raw), np.array(walked_bis)

    err_raw = walked_raw - fitted          # in [0, step) by construction
    err_bis = walked_bis - fitted
    gap = walked_bis - population          # estimation error, not walker error
    starts_outside = fitted == 0

    # --- plot 3 data: redraw only the dev pool -------------------------------
    rng_dev = np.random.default_rng(args.seed + 1)
    cosines = np.array([
        estimate_direction(draw(world, len(dev_X), args.pi_dev, rng_dev)[0], S["ref"]) @ v_true
        for _ in range(N_REDRAW)
    ])
    # Context only (not plotted): how the cosine depends on the choice of PI_DEV.
    pi_sweep = {}
    for pi in (0.05, 0.1, 0.2, 0.3, 0.5):
        r = np.random.default_rng(args.seed + 2)
        pi_sweep[pi] = float(np.mean([
            estimate_direction(draw(world, len(dev_X), pi, r)[0], S["ref"]) @ v_true
            for _ in range(N_REDRAW)
        ]))

    sep_check = check_sep_invariance(args.d, 0, t_pop)

    summary = dict(
        d=args.d, sep=args.sep, pi_dev=args.pi_dev, n_ref=args.n_ref,
        grid=args.grid, seed=args.seed, quantile=QUANTILE,
        t_fitted=t, t_population=t_pop,
        alpha_max=float(alpha_max), grid_step=float(step),
        n_probe=len(S["probe"]), n_censored=int(np.sum(censored)),
        n_start_outside=int(starts_outside.sum()),
        walked_start_outside_is_zero=bool(np.all(walked_raw[starts_outside] == 0)),
        n_recross_total=int(np.sum(recross)),
        max_err_raw=float(np.nanmax(np.abs(err_raw))),
        frac_raw_err_within_one_step=float(np.mean((err_raw >= 0) & (err_raw < step))),
        all_raw_err_within_one_step=bool(np.all((err_raw >= 0) & (err_raw < step))),
        max_err_bisected=float(np.nanmax(np.abs(err_bis))),
        pop_gap_mean=float(np.nanmean(gap)), pop_gap_sd=float(np.nanstd(gap)),
        pop_gap_rmse=float(np.sqrt(np.nanmean(gap**2))),
        cos_vhat_vtrue=cos_hat,
        cos_redraw_mean=float(cosines.mean()), cos_redraw_sd=float(cosines.std()),
        cos_mean_by_pi=pi_sweep,
        alpha_star_center_by_sep=sep_check,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    plot(out, fitted, walked_raw, population, walked_bis, step, cosines, cos_hat,
         args.pi_dev)


def plot(out, fitted, walked_raw, population, walked_bis, step, cosines, cos_hat,
         pi_dev):
    import matplotlib.pyplot as plt

    ink, accent, muted = "#1f2933", "#2a6fdb", "#9aa5b1"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "font.size": 10})

    def diagonal(ax, a, b, title, xlabel):
        lim = [0, 1.05 * np.nanmax([a, b])]
        ax.plot(lim, lim, color=muted, lw=1, zorder=0, label="y = x")
        ax.scatter(a, b, s=14, color=accent, alpha=0.8, linewidths=0)
        ax.set(xlim=lim, ylim=lim, xlabel=xlabel, ylabel="walked α* (fitted detector)",
               title=title, aspect="equal")
        ax.legend(frameon=False, loc="upper left")

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    diagonal(ax, fitted, walked_raw, "1 · walker vs analytic, fitted μ̂, Σ̂, t̂",
             "analytic α* (fitted μ̂, Σ̂, t̂)")
    ax.fill_between([0, ax.get_xlim()[1]], [0, ax.get_xlim()[1]],
                    [step, ax.get_xlim()[1] + step], color=accent, alpha=0.08,
                    lw=0, label="one grid step")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "walker_vs_fitted.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    diagonal(ax, population, walked_bis, "2 · walker vs population boundary",
             "analytic α* (population μ, Σ, χ²_d quantile)")
    fig.tight_layout(); fig.savefig(out / "walker_vs_population.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.hist(cosines, bins=15, color=accent, alpha=0.85, edgecolor="white", lw=1)
    ax.axvline(cos_hat, color=ink, lw=1.5, ls="--", label=f"run's v̂: {cos_hat:.3f}")
    ax.set(xlabel="cos(v̂, v_true)", ylabel="dev-pool redraws",
           title=f"3 · direction estimate over {len(cosines)} dev pools (π = {pi_dev})")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "direction_cosine.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
