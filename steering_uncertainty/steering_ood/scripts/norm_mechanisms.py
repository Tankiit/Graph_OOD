"""Mechanism checks for norm-dependent detector failures (full-dimensional caches, float64).

Q1  Distance detectors at the origin: kNN score(0) is the k-th smallest reference norm;
    compare with the calibrated threshold and with typical ID-ID distances.
Q2  Energy at the origin: E(0) = -logsumexp(b), independent of W and of the softmax gauge
    (adding r to every row of W shifts logits by r.z, which vanishes at z = 0).
Q3  Principal-axis alignment: do the top ID principal axes lie in span(W^T) or in the
    class-mean subspace, and does rotating a probe onto them raise the top logit?

Usage: python scripts/norm_mechanisms.py [runs/modal]
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import subspace_angles
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steering_ood.core import threshold  # noqa: E402
from steering_ood.detectors import SKKNN, ShrinkageMahalanobis  # noqa: E402

MODELS = ['mpnet', 'minilm', 'bge_base', 'bge_large', 'resnet18', 'resnet50', 'vit_b16', 'dinov2_s']
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/modal')
K, TAU, NPC = 5, .05, 10


def energy(z, W, b):
    return -logsumexp(z @ W.T + b, axis=1)


def show(title, rows):
    print(f'\n## {title}')
    print(' | '.join(rows[0]))
    for r in rows:
        print(' | '.join(f'{v:.3g}' if isinstance(v, float) else str(v) for v in r.values()))


def main():
    q1, q2, q3 = [], [], []
    rng = np.random.default_rng(0)
    for model in MODELS:
        c = np.load(ROOT / 'caches' / f'{model}.npz')
        ref, yref = c['reference_x'].astype(float), c['reference_y']
        cal, probe = c['calibration_x'].astype(float), c['probe_x'].astype(float)
        state = torch.load(ROOT / 'main' / model / 'full' / 'head' / 'head.pt')
        W, b = state['weight'].double().numpy(), state['bias'].double().numpy()
        origin = np.zeros((1, ref.shape[1]))

        # Q1: geometry and distance detectors at the origin
        norms = np.linalg.norm(ref, axis=1)
        i, j = rng.integers(0, len(ref), (2, 4000)); keep = i != j
        pair = np.linalg.norm(ref[i[keep]] - ref[j[keep]], axis=1)
        unit = ref / norms[:, None]
        knn = SKKNN(K).fit(ref)
        t_knn = threshold(knn.score(cal), TAU)
        maha = ShrinkageMahalanobis().fit(ref, yref)
        t_maha = threshold(maha.score(cal), TAU)
        q1.append(dict(model=model, dim=ref.shape[1], rho_median_norm=float(np.median(norms)),
                       mean_cosine_id=float(np.mean(np.sum(unit[i[keep]] * unit[j[keep]], 1))),
                       global_mean_norm_over_rho=float(np.linalg.norm(ref.mean(0)) / np.median(norms)),
                       id_pair_dist_over_rho=float(np.median(pair) / np.median(norms)),
                       knn_score_origin_over_rho=float(knn.score(origin)[0] / np.median(norms)),
                       knn_threshold_over_rho=float(t_knn / np.median(norms)),
                       knn_origin_accepted=bool(knn.score(origin)[0] <= t_knn),
                       maha_origin_over_threshold=float(maha.score(origin)[0] / t_maha),
                       maha_origin_accepted=bool(maha.score(origin)[0] <= t_maha)))

        # Q2: energy at the origin; gauge r = mean row (the Sigma w_c = 0 gauge) as a check
        t_e = threshold(energy(cal, W, b), TAU)
        r = W.mean(0)
        e0 = energy(origin, W, b)[0]
        q2.append(dict(model=model, energy_origin=float(e0), energy_threshold=float(t_e),
                       energy_origin_rejected=bool(e0 > t_e),
                       energy_origin_gauge_centered=float(energy(origin, W - r, b)[0]),
                       bias_spread=float(b.max() - b.min()),
                       frac_probe_energy_increases_halfway=float(np.mean(
                           energy(.5 * probe, W, b) > energy(probe, W, b)))))

        # Q3: principal-axis alignment (direction pool = ID dev split)
        dev = c['direction_x'].astype(float); ydev = c['direction_y']
        mu = dev.mean(0)
        _, sv, vt = np.linalg.svd(dev - mu, full_matrices=False)
        U = vt[:NPC].T
        means = np.array([dev[ydev == k].mean(0) for k in np.unique(ydev)]) - mu
        cos_w = np.cos(subspace_angles(U, W.T))
        cos_m = np.cos(subspace_angles(U, means.T))
        k_sub = min(np.linalg.matrix_rank(W), U.shape[1])
        between = np.array([np.var(means[np.searchsorted(np.unique(ydev), ydev)] @ u) / np.var((dev - mu) @ u)
                            for u in U.T])
        # rotate probes onto +/- u_1 at their own norm (end point of a 90-degree geodesic)
        rp = np.linalg.norm(probe, axis=1, keepdims=True)
        top = lambda z: (z @ W.T + b).max(1)
        rows = {}
        Wc = W - r                                                 # centered gauge: same softmax
        t_ec = threshold(energy(cal, Wc, b), TAU)
        topc = lambda z: (z @ Wc.T + b).max(1)
        for s in (1, -1):
            z = rp * (s * U[:, 0])[None]
            rows[s] = (float(np.mean(top(z) - top(probe))), float(np.mean(energy(z, W, b) > t_e)),
                       float(np.mean(topc(z) - topc(probe))), float(np.mean(energy(z, Wc, b) > t_ec)))
        q3.append(dict(model=model, top10_var_frac=float((sv[:NPC] ** 2).sum() / (sv ** 2).sum()),
                       cos_pc_vs_W_mean=float(cos_w[:k_sub].mean()),
                       cos_pc_vs_classmeans_mean=float(cos_m[:k_sub].mean()),
                       random_subspace_baseline=float(np.sqrt(len(means) / dev.shape[1])),
                       between_class_frac_pc1=float(between[0]),
                       between_class_frac_top10=float(between.mean()),
                       dtoplogit_rot_to_plus_pc1=rows[1][0], dtoplogit_rot_to_minus_pc1=rows[-1][0],
                       energy_rejects_at_plus_pc1=rows[1][1], energy_rejects_at_minus_pc1=rows[-1][1],
                       common_row_share=float(np.linalg.norm(r) / np.linalg.norm(W, axis=1).mean()),
                       centered_dtoplogit_plus=rows[1][2], centered_dtoplogit_minus=rows[-1][2],
                       centered_energy_rejects_plus=rows[1][3], centered_energy_rejects_minus=rows[-1][3]))
    show('Q1 distance detectors at the origin (rho = median reference norm)', q1)
    show('Q2 energy at the origin: E(0) = -logsumexp(b)', q2)
    show('Q3 principal-axis alignment (top 10 PCs of ID dev split)', q3)


if __name__ == '__main__':
    main()
