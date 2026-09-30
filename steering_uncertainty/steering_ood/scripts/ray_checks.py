"""Check linear-head energy/MSP ray predictions and Mahalanobis censoring on saved runs.

Needs runs/modal/main (crossed runs + heads) and runs/modal/caches/{model}[_pca64].npz.
Usage: python scripts/ray_checks.py [runs/modal]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.special import logsumexp, softmax

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steering_ood.core import threshold

MODELS = ['mpnet', 'minilm', 'bge_base', 'bge_large', 'resnet18', 'resnet50', 'vit_b16', 'dinov2_s']
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/modal')


def load(model, rep):
    cache = np.load(ROOT / 'caches' / f"{model}{'' if rep == 'full' else '_' + rep}.npz")
    state = torch.load(ROOT / 'main' / model / rep / 'head' / 'head.pt')
    return cache, state['weight'].double().numpy(), state['bias'].double().numpy()


def cone_mass(W, draws=200000, seed=0):
    """Fraction of isotropic random unit directions whose every logit decreases (energy eventually rejects)."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(draws // 20000):
        v = rng.normal(size=(20000, W.shape[1]))
        hits += int(np.sum((v @ W.T).max(1) < 0))
    return hits / draws


def gordan_cone(W):
    """Largest s with W v <= -s for some v in [-1,1]^D. s > 0 iff some ray lowers every logit."""
    K, D = W.shape
    res = linprog(np.r_[np.zeros(D), -1.], A_ub=np.c_[W, np.ones(K)], b_ub=np.zeros(K),
                  bounds=[(-1, 1)] * D + [(None, 1)], method='highs')
    return -res.fun


def rejected_intervals(rej):
    """Number of maximal rejected runs along the alpha grid, per path."""
    r = rej.astype(int)
    return r[..., 0] + np.sum(np.diff(r, axis=-1) == 1, axis=-1)


def energy_msp(model, rep, kind, cache, W, b):
    run = ROOT / 'main' / model / rep / f'crossed_{kind}'
    design = np.load(run / 'design.npz')
    V = design['signs'][:, None, None] * design['vectors'][None]          # S, C, D
    alphas = design['alphas']
    x = cache['probe_x'][design['probe_indices']].astype(float)           # P, D
    Wv = np.einsum('scd,kd->sck', V, W)                                    # S, C, K
    m = Wv.max(-1)                                                         # eventual slope of logsumexp
    p0 = softmax(x @ W.T + b, axis=1)                                      # P, K
    slope0 = -np.einsum('pk,sck->scp', p0, Wv)                             # dE/dalpha at alpha=0
    e = np.load(run / 'energy_T1_traces.npz')
    rej_e = e['rejected'][0]                                               # C, S, P, G (energy has no D dependence)
    s_ = e['scores'][0]
    out = dict(model=model, rep=rep, kind=kind,
               frac_rays_eventually_reject=float(np.mean(m < 0)),
               observed_rejected_at_horizon=float(rej_e[..., -1].mean()),
               frac_initial_slope_negative=float(np.mean(slope0 < 0)),
               frac_E_H_below_E_0=float(np.mean(s_[..., -1] < s_[..., 0])),
               max_rejected_intervals_energy=int(rejected_intervals(rej_e).max()))
    # MSP: count argmax switches of the logits along each path and relate to re-acceptance.
    ms = np.load(run / 'msp_T1_traces.npz')
    rej_m = ms['rejected'][0]
    logits = (x @ W.T + b)[None, None, :, None, :] + alphas[None, None, None, :, None] * \
             np.swapaxes(Wv, 0, 1)[:, :, None, None, :]                    # C, S, P, G, K
    switches = np.sum(np.diff(logits.argmax(-1), axis=-1) != 0, axis=-1)  # C, S, P
    first = rej_m.argmax(-1)
    ever = rej_m.any(-1)
    returned = ever & ~rej_m[..., -1] | (np.diff(rej_m.astype(int), axis=-1) == -1).any(-1)
    out.update(msp_return_frac=float(returned[ever].mean()) if ever.any() else None,
               msp_argmax_switch_given_return=float((switches[returned] > 0).mean()) if returned.any() else None,
               msp_argmax_switch_given_no_return=float((switches[ever & ~returned] > 0).mean()) if (ever & ~returned).any() else None,
               msp_max_rejected_intervals=int(rejected_intervals(rej_m).max()),
               msp_top_prob_at_horizon=float(softmax(logits[..., -1, :], -1).max(-1).mean()))
    return out


def maha_exact(model, rep, kind, cache):
    """Exact float64 first crossing for pytorch-ood class-conditional Mahalanobis (no horizon)."""
    run = ROOT / 'main' / model / rep / f'crossed_{kind}'
    design = np.load(run / 'design.npz')
    tr = np.load(run / 'mahalanobis_traces.npz')
    chk = json.loads((run / 'steering_checks.json').read_text())
    radius, H = chk['id_reference_radius'], chk['horizon']
    x = cache['probe_x'][design['probe_indices']].astype(float)
    alpha_exact, err, t32, t64 = [], [], [], []
    for bi, idx in enumerate(design['reference_indices']):
        z, y = cache['reference_x'][idx].astype(float), cache['reference_y'][idx]
        classes = np.unique(y)
        mu = np.array([z[y == c].mean(0) for c in classes])
        S = sum((z[y == c] - mu[i]).T @ (z[y == c] - mu[i]) for i, c in enumerate(classes))
        S += 1e-6 * np.eye(S.shape[0])
        ev, Q = np.linalg.eigh(S)
        L = Q / np.sqrt(ev)                                                # precision = L L^T, exact in float64
        cal = cache['calibration_x'].astype(float)
        Ac = (cal[:, None, :] - mu[None]) @ L
        t = threshold(.5 * (Ac ** 2).sum(-1).min(1), .05)                # float64 recalibration
        t32.append(tr['thresholds'][bi]); t64.append(t)
        A = (x[:, None, :] - mu[None]) @ L                                 # P, K, D whitened offsets
        for ci, v in enumerate(design['vectors']):
            for si, sign in enumerate(design['signs']):
                g = (sign * v) @ L                                         # D
                # pytorch-ood score = 0.5 * min_c d_c^2; q_c(alpha) = a0 + a1 alpha + a2 alpha^2
                a0, a1, a2 = .5 * (A ** 2).sum(-1), A @ g, .5 * g @ g
                if bi == 0:
                    grid = design['alphas']
                    q = (a0[..., None] + a1[..., None] * grid + a2 * grid ** 2).min(1)
                    err.append(np.max(np.abs(q - tr['scores'][0, ci, si]) / np.abs(tr['scores'][0, ci, si]).max()))
                disc = a1 ** 2 - 4 * a2 * (a0 - t)                         # accept interval per class
                r1 = np.where(disc > 0, (-a1 - np.sqrt(np.maximum(disc, 0))) / (2 * a2), np.inf)
                r2 = np.where(disc > 0, (-a1 + np.sqrt(np.maximum(disc, 0))) / (2 * a2), -np.inf)
                for p in range(len(x)):
                    cur = 0.0
                    while True:                                            # walk the union of accept intervals
                        cover = (r1[p] <= cur) & (r2[p] > cur)
                        if not cover.any():
                            break
                        cur = r2[p][cover].max()
                    alpha_exact.append(cur)
    a = np.array(alpha_exact)
    obs = tr['event'].reshape(-1)
    return dict(model=model, rep=rep, kind=kind, horizon_radii=H / radius,
                max_rel_err_float64_vs_traced=float(max(err)),
                observed_crossing_within_horizon=float(obs.mean()),
                exact_crossing_within_horizon=float(np.mean(a <= H)),
                exact_all_finite=bool(np.isfinite(a).all()),
                threshold_ratio_32_over_64=float(np.median(np.array(t32) / np.array(t64))),
                exact_initially_rejected=float(np.mean(a == 0)),
                exact_median_alpha_radii=float(np.median(a) / radius),
                exact_p90_alpha_radii=float(np.quantile(a, .9) / radius))


def conditioning(model, cache):
    """Within-class scatter spectrum, float32 vs float64 precision for random directions."""
    z, y = cache['reference_x'].astype(float), cache['reference_y']
    S = sum((z[y == c] - z[y == c].mean(0)).T @ (z[y == c] - z[y == c].mean(0)) for c in np.unique(y))
    S += 1e-6 * np.eye(len(S))
    ev = np.linalg.eigvalsh(S)
    P64 = np.linalg.inv(S)
    P32 = torch.linalg.inv(torch.tensor(S, dtype=torch.float32)).double().numpy()
    rng = np.random.default_rng(0)
    R = rng.normal(size=(200, len(S))); R /= np.linalg.norm(R, axis=1, keepdims=True)
    q64, q32 = np.einsum('nd,de,ne->n', R, P64, R), np.einsum('nd,de,ne->n', R, P32, R)
    top = np.linalg.eigh(S)[1][:, -1]
    return dict(model=model, dim=len(S), n_ref=len(z), cond=float(ev[-1] / ev[0]),
                smallest_eig_per_sample=float(ev[0] / len(z)), ridge_per_sample=1e-6 / len(z),
                random_vs_top_pc_quadratic=float(np.median(q64) / (top @ P64 @ top)),
                float32_rel_err_random=float(np.median(np.abs(q32 - q64) / q64)))


def norms(model, cache, radius, H):
    n = lambda a: np.linalg.norm(a.astype(float), axis=1)
    out = dict(model=model, id_norm=float(np.median(n(cache['id_test_x']))),
               radius_over_id_norm=radius / float(np.median(n(cache['id_test_x']))),
               horizon_over_id_norm=H / float(np.median(n(cache['id_test_x']))))
    groups = cache['ood_test_groups'] if 'ood_test_groups' in cache.files else np.full(len(cache['ood_test_x']), 'oos')
    for g in np.unique(groups):
        out[f'ood_norm_ratio_{g}'] = float(np.median(n(cache['ood_test_x'][groups == g])) / out['id_norm'])
    return out


def show(title, rows):
    print(f'\n## {title}')
    keys = list(rows[0])
    print(' | '.join(keys))
    for r in rows:
        print(' | '.join(f'{v:.3g}' if isinstance(v, float) else str(v) for v in r.values()))


def main():
    heads, rays, maha, cond, nrm = [], [], [], [], []
    for model in MODELS:
        for rep in ('pca64', 'full'):
            cache, W, b = load(model, rep)
            col = np.linalg.norm(W.sum(0)) / np.linalg.norm(W, axis=1).mean()
            heads.append(dict(model=model, rep=rep, classes=W.shape[0],
                              sum_w_over_mean_w=float(col), iid_init_expectation=float(np.sqrt(W.shape[0])),
                              gordan_s=float(gordan_cone(W)), random_cone_mass=cone_mass(W)))
            for kind in ('pca', 'random'):
                rays.append(energy_msp(model, rep, kind, cache, W, b))
                m = maha_exact(model, rep, kind, cache)
                if m['observed_crossing_within_horizon'] < .999 or rep == 'full':
                    maha.append(m)
            if rep == 'full':
                cond.append(conditioning(model, cache))
                chk = json.loads((ROOT / 'main' / model / rep / 'crossed_pca' / 'steering_checks.json').read_text())
                nrm.append(norms(model, cache, chk['id_reference_radius'], chk['horizon']))
    show('Head structure (softmax gauge): ||sum_c w_c|| / mean ||w_c||; Gordan s>0 => some ray lowers all logits', heads)
    show('Energy / MSP along steered rays', rays)
    show('Mahalanobis: exact float64 crossing without horizon', maha)
    show('Mahalanobis conditioning (full dim)', cond)
    show('Feature norms: ID vs real OOD; steering scale', nrm)


if __name__ == '__main__':
    main()
