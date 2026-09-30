"""kNN origin condition in neighbour-cosine form, and an ID-support check for steering paths.

knn_origin_condition: the calibrated kNN threshold is t = ||z_q - r_q||, where z_q is the calibration
point at the order-statistic rank and r_q its k-th neighbour. With c_k = cos(z_q, r_q):
    t^2 = ||z_q||^2 + ||r_q||^2 - 2 ||z_q|| ||r_q|| c_k,   score(0) = r_(k) (k-th smallest reference norm),
    origin accepted  <=>  r_(k) <= t  <=>  c_k <= c* := (||z_q||^2 + ||r_q||^2 - r_(k)^2) / (2 ||z_q|| ||r_q||).
With all norms equal to rho, c* = 1/2.

path_leaves_id: independent of all four detectors and of the direction/reference/calibration splits,
using the held-out ID test split only as a descriptive support reference:
  (i)   projection of each path point on the steering axis u vs the ID-test quantile range of z.u,
  (ii)  norm of the component orthogonal to u vs the ID-test range of that norm,
  (iii) distance to the ID mean vs the ID-test range.
Usage: python scripts/support_checks.py [runs/modal]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steering_ood.experiment import path_points  # noqa: E402

MODELS = ['mpnet', 'minilm', 'bge_base', 'bge_large', 'resnet18', 'resnet50', 'vit_b16', 'dinov2_s']
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/modal')
K, TAU, Q = 5, .05, (.005, .995)


def knn_origin_condition(ref, cal, k=K, tau=TAU):
    from sklearn.neighbors import NearestNeighbors
    dist, ind = NearestNeighbors(n_neighbors=k).fit(ref).kneighbors(cal)
    order, scores = ind[:, k - 1], dist[:, k - 1]
    rank = math.ceil((len(cal) + 1) * (1 - tau))
    q = np.argsort(scores)[rank - 1]
    zq, rq = cal[q], ref[order[q]]
    nz, nr = np.linalg.norm(zq), np.linalg.norm(rq)
    ck = float(zq @ rq / (nz * nr))
    rk = float(np.sort(np.linalg.norm(ref, axis=1))[k - 1])
    cstar = (nz ** 2 + nr ** 2 - rk ** 2) / (2 * nz * nr)
    return dict(c_k=ck, c_star=float(cstar), t=float(scores[q]), r_k=rk,
                general_equals_norm_rule=bool((ck <= cstar) == (rk <= scores[q] + 1e-9)),
                pred_accept_unit_rule=ck < .5, pred_accept_general=ck <= cstar)


def path_leaves_id(pts, u, id_ref, mu):
    """Fraction of path points outside the ID range on each statistic; pts [..., G, D]."""
    lo_hi = lambda v: np.quantile(v, Q)
    p_ref = id_ref @ u
    o_ref = np.linalg.norm(id_ref - p_ref[:, None] * u, axis=1)
    c_ref = np.linalg.norm(id_ref - mu, axis=1)
    p = pts @ u
    o = np.linalg.norm(pts - p[..., None] * u, axis=-1)
    c = np.linalg.norm(pts - mu, axis=-1)
    out = lambda v, r: (v < lo_hi(r)[0]) | (v > lo_hi(r)[1])
    op, oo, oc = out(p, p_ref), out(o, o_ref), out(c, c_ref)
    return op, oo, oc


def main():
    rows_k, rows_s = [], []
    for model in MODELS:
        c = np.load(ROOT / 'caches' / f'{model}.npz')
        cal = c['calibration_x'].astype(float)
        run = ROOT / 'v2' / model / 'full' / 'crossed_origin'
        design = np.load(run / 'design.npz')
        x = c['probe_x'][design['probe_indices']].astype(float)
        near = 1 - design['alphas'][-1] / np.linalg.norm(x, axis=1) < .1    # probes that reach the origin
        rej = np.load(run / 'knn_traces.npz')['rejected'][..., -1]
        obs = rej[..., near].mean(axis=(1, 2, 3)) if near.any() else np.full(len(rej), np.nan)
        res = [knn_origin_condition(c['reference_x'][idx].astype(float), cal) for idx in design['reference_indices']]
        rows_k.append(dict(model=model, c_k=float(np.mean([r['c_k'] for r in res])),
                           c_star=float(np.mean([r['c_star'] for r in res])),
                           unit_rule_correct=None if not near.any() else float(np.mean([r['pred_accept_unit_rule'] == (o <= .5) for r, o in zip(res, obs)])),
                           general_equals_norm_rule=all(r['general_equals_norm_rule'] for r in res),
                           general_rule_correct=None if not near.any() else float(np.mean([r['pred_accept_general'] == (o <= .5) for r, o in zip(res, obs)])),
                           observed_origin_rejected=float(obs.mean())))

        id_ref = c['id_test_x'].astype(float)
        mu = id_ref.mean(0)
        for kind in ('sphere_pca', 'pca', 'sphere_random'):
            rd = ROOT / ('v2' if kind != 'pca' else 'main') / model / 'full' / f'crossed_{kind}'
            dg = np.load(rd / 'design.npz')
            chk = json.loads((rd / 'steering_checks.json').read_text())
            radius = chk['id_reference_radius']
            x = c['probe_x'][dg['probe_indices']].astype(float)
            alphas = dg['alphas']
            match = .81 if model in ('resnet18', 'resnet50', 'vit_b16', 'dinov2_s') else 1.03
            gm = alphas / radius <= match + 1e-9
            fr = {'p': [], 'o': [], 'c': [], 'any': [], 'end': []}
            for v in dg['vectors']:
                for sgn in dg['signs']:
                    pts = path_points(kind, x, sgn * v, alphas)
                    u = v                     # axis the path steers toward (PC or random direction)
                    op, oo, oc = path_leaves_id(pts, u, id_ref, mu)
                    anyo = op | oo | oc
                    fr['p'].append(op[:, gm].mean()); fr['o'].append(oo[:, gm].mean()); fr['c'].append(oc[:, gm].mean())
                    fr['any'].append(anyo[:, gm].mean()); fr['end'].append(anyo[:, gm][:, -1].mean())
            rows_s.append(dict(model=model, path=kind, horizon_radii=match,
                               frac_outside_proj_u=float(np.mean(fr['p'])), frac_outside_orth_norm=float(np.mean(fr['o'])),
                               frac_outside_dist_mean=float(np.mean(fr['c'])), frac_path_outside_any=float(np.mean(fr['any'])),
                               frac_endpoint_outside_any=float(np.mean(fr['end']))))
    for title, rows in (('kNN origin condition: c_k vs 1/2 (unit rule) and vs c* (general)', rows_k),
                        ('ID support along paths up to the matched horizon (ID-test quantiles 0.5%-99.5%)', rows_s)):
        print(f'\n## {title}')
        print(' | '.join(rows[0]))
        for r in rows:
            print(' | '.join(f'{v:.3g}' if isinstance(v, float) else str(v) for v in r.values()))
    return rows_k, rows_s


if __name__ == '__main__':
    main()
