"""Closed-form radial predictions checked against traced runs (full-dimensional, float64).

Notation: a = W z (bias-free logits), E(s) = -logsumexp(s a + b) along z -> s z.
  (E1) E is concave in s; dE/ds = -E_p(s)[a] and d/ds E_p(s)[a] = Var_p(s)(a) >= 0.
  (E2) s -> 0: E -> -logsumexp(b).  s -> inf: E -> -inf iff max_c a_c > 0.
  (E3) Growth from an accepted probe: if dE/ds(1) <= 0, E never exceeds E(1) (never rejected);
       otherwise the maximum is at the unique s0 with E_p(s0)[a] = 0.
  (K1) kNN score(0) = k-th smallest reference norm (with bootstrap multiplicity).
       kNN is 1-Lipschitz, so on radial growth alpha* <= t + max_i ||r_i|| - ||x||.
  (M1) Mahalanobis(0) = min_c mu_c^T P mu_c.
Usage: python scripts/radial_predictions.py [runs/modal]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import logsumexp, softmax
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steering_ood.detectors import ShrinkageMahalanobis  # noqa: E402

MODELS = ['mpnet', 'minilm', 'bge_base', 'bge_large', 'resnet18', 'resnet50', 'vit_b16', 'dinov2_s']
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/modal')
K = 5


def run_dir(model, kind):
    return ROOT / 'v2' / model / 'full' / f'crossed_{kind}'


def energy(z, W, b):
    return -logsumexp(z @ W.T + b, axis=-1)


def growth_prediction(x, W, b, t, s_max):
    """Per probe: predicted 'ever rejected on s in [1, s_max]' from (E1)-(E3), no path evaluation."""
    a = x @ W.T
    pred, slope_nonpos = [], []
    for ai, s_end in zip(a, s_max):
        mean_a = lambda s: softmax(s * ai + b) @ ai                  # increasing in s by (E1)
        e = lambda s: -logsumexp(s * ai + b)
        slope_nonpos.append(mean_a(1.) >= 0)
        if mean_a(1.) >= 0:
            s_star = 1.
        elif mean_a(s_end) <= 0:
            s_star = s_end
        else:
            s_star = brentq(mean_a, 1., s_end)
        pred.append(e(s_star) > t)
    return np.array(pred), np.array(slope_nonpos), float(np.mean(a.max(1) > 0))


def near0_agreement(pred, rejected_end, s_end):
    """Fraction of reference fits whose origin prediction matches the majority decision of near-origin probes."""
    near = s_end < .1
    if not near.any():
        return None
    obs = rejected_end.reshape(len(pred), -1, len(s_end))[..., near].mean(axis=(1, 2)) > .5
    return float(np.mean(np.array(pred) == obs))


def knn_decidable(margin, rejected_end, end_norm):
    m = np.array(margin)[:, None]
    obs = rejected_end.reshape(len(margin), -1, len(end_norm)).mean(axis=1) > .5          # fit x probe
    dec = np.abs(m) > end_norm[None]
    agree = (obs == (m > 0))[dec]
    return dict(knn_decidable_pairs=int(dec.sum()),
                knn_decidable_agreement=float(agree.mean()) if dec.any() else None)


def show(title, rows):
    print(f'\n## {title}')
    print(' | '.join(rows[0]))
    for r in rows:
        print(' | '.join(f'{v:.3g}' if isinstance(v, float) else str(v) for v in r.values()))


def main():
    origin_rows, growth_rows, link = [], [], []
    for model in MODELS:
        c = np.load(ROOT / 'caches' / f'{model}.npz')
        state = torch.load(ROOT / 'main' / model / 'full' / 'head' / 'head.pt')
        W, b = state['weight'].double().numpy(), state['bias'].double().numpy()
        run = run_dir(model, 'origin')
        design = np.load(run / 'design.npz')
        x = c['probe_x'][design['probe_indices']].astype(float)
        h = design['alphas'][-1]
        s_end = 1 - h / np.linalg.norm(x, axis=1)

        te = np.load(run / 'energy_T1_traces.npz')
        t_e = te['thresholds'][0]
        tk = np.load(run / 'knn_traces.npz')
        tm = np.load(run / 'mahalanobis_shrinkage_traces.npz')
        knn_pred, maha_pred, knn_margin = [], [], []
        for bi, idx in enumerate(design['reference_indices']):
            ref, yref = c['reference_x'][idx].astype(float), c['reference_y'][idx]
            rk = np.sort(np.linalg.norm(ref, axis=1))[K - 1]
            knn_pred.append(rk > tk['thresholds'][bi]); knn_margin.append(rk - tk['thresholds'][bi])
            det = ShrinkageMahalanobis().fit(ref, yref)
            maha_pred.append(det.score(np.zeros((1, ref.shape[1])))[0] > tm['thresholds'][bi])
        rho = float(np.median(np.linalg.norm(c['reference_x'], axis=1)))
        origin_rows.append(dict(
            model=model, median_s_end=float(np.median(s_end)),
            energy_pred_reject_at_0=bool(-logsumexp(b) > t_e),
            energy_obs_reject_end=float(te['rejected'][..., -1].mean()),
            energy_obs_reject_end_near0=float(te['rejected'][..., -1][..., s_end < .1].mean()) if (s_end < .1).any() else None,
            knn_pred_reject_at_0=float(np.mean(knn_pred)),
            knn_obs_reject_end=float(tk['rejected'][..., -1].mean()),
            knn_implied_cosine=float(1 - (np.mean(tk['thresholds']) / rho) ** 2 / 2),
            maha_pred_reject_at_0=float(np.mean(maha_pred)),
            maha_obs_reject_end=float(tm['rejected'][..., -1].mean()),
            # Limit checks use only probes that reach s < 0.1, per reference fit (None if none do).
            knn_agree_near0=near0_agreement(knn_pred, tk['rejected'][..., -1], s_end),
            # kNN is 1-Lipschitz: a (fit, probe) pair decides the origin rule iff |r_(k) - t| > ||z_end||.
            **knn_decidable(knn_margin, tk['rejected'][..., -1], s_end * np.linalg.norm(x, axis=1)),
            maha_agree_near0=near0_agreement(maha_pred, tm['rejected'][..., -1], s_end),
            n_probes_near0=int((s_end < .1).sum())))

        # growth path (radial), if available
        rg = run_dir(model, 'radial')
        if (rg / 'energy_T1_traces.npz').exists():
            dg = np.load(rg / 'design.npz')
            xg = c['probe_x'][dg['probe_indices']].astype(float)
            s_max = 1 + dg['alphas'][-1] / np.linalg.norm(xg, axis=1)
            eg = np.load(rg / 'energy_T1_traces.npz')
            pred, nonpos, pos_top = growth_prediction(xg, W, b, eg['thresholds'][0], s_max)
            obs = eg['rejected'][0, 0, 0].any(-1)
            kg = np.load(rg / 'knn_traces.npz')
            bound = []
            for bi, idx in enumerate(dg['reference_indices']):
                rmax = np.linalg.norm(c['reference_x'][idx], axis=1).max()
                bound.append(kg['thresholds'][bi] + rmax - np.linalg.norm(xg, axis=1))
            obs_a = np.where(kg['event'][:, 0, 0], kg['observed'][:, 0, 0], np.inf)
            growth_rows.append(dict(
                model=model, frac_positive_biasfree_top_logit=pos_top,
                frac_slope_nonpos_at_probe=float(nonpos.mean()),
                energy_pred_ever_reject=float(pred.mean()), energy_obs_ever_reject=float(obs.mean()),
                energy_pred_obs_agreement=float(np.mean(pred == obs)),
                never_reject_rule_violations=int(np.sum(obs & nonpos & ~eg['rejected'][0, 0, 0, :, 0])),
                knn_obs_cross=float(kg['event'].mean()),
                knn_bound_violations=int(np.sum(obs_a > np.array(bound) + 1e-9))))

        # real-OOD norm link: norm ratio vs detector AUROCs (float64 static from v2)
        m = json.loads((ROOT / 'v2' / model / 'full' / 'static' / 'metrics.json').read_text())
        idn = np.median(np.linalg.norm(c['id_test_x'], axis=1))
        groups = c['ood_test_groups'] if 'ood_test_groups' in c.files else np.full(len(c['ood_test_x']), 'oos')
        for g in np.unique(groups):
            by = lambda d: m[d]['ood_by_dataset'][g]['auroc'] if 'ood_by_dataset' in m[d] else m[d]['auroc']
            link.append(dict(model=model, ood=str(g),
                             norm_ratio=float(np.median(np.linalg.norm(c['ood_test_x'][groups == g], axis=1)) / idn),
                             auroc_knn=by('knn'), auroc_maha=by('mahalanobis_shrinkage'),
                             auroc_energy=by('energy_T1'), energy_minus_distance=by('energy_T1') - max(by('knn'), by('mahalanobis_shrinkage'))))
    show('Shrinkage toward the origin: limit predictions (E2, K1, M1) vs observed at path end', origin_rows)
    if growth_rows:
        show('Radial growth: closed-form energy prediction (E3) and kNN bound (K1) vs traces', growth_rows)
    show('Real OOD: norm ratio vs AUROC (float64 static)', link)
    r = spearmanr([l['norm_ratio'] for l in link], [l['energy_minus_distance'] for l in link])
    print(f'\nSpearman(norm ratio, energy AUROC - best distance AUROC) = {r.statistic:.2f} (p={r.pvalue:.2g}, n={len(link)})')
    return origin_rows, growth_rows, link


if __name__ == '__main__':
    main()
