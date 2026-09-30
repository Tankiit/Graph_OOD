"""Multi-seed statistics for the steering results.

Seeds vary the data split, head training and bootstrap/direction draws together. Seed 0 = original split.
Layout: runs/modal/seeds/seed{s}/{model}/full/{head,static,crossed_*}; caches in runs/modal/caches[/seed{s}].

(A) Matched-horizon rejection R per (modality, path, detector): mean over seeds x encoders, SD of the
    encoder-mean across seeds, and a hierarchical bootstrap 95% CI (resample encoders, then seeds within).
(B) Paired claims: per (seed, encoder) cell, paired bootstrap over probes of R_i - R_j (same probes,
    references, directions); one-sided p; count of cells significant at 0.05 and hierarchical CI of the mean.
(C) Static AUROC / FPR@95 mean +- SD over seeds per encoder and detector.
(D) Exact theory checks (radial_predictions, support_checks) re-run per seed.
R at alpha uses the last grid point <= alpha, so rejection stays a per-probe indicator.
Usage: python scripts/seed_stats.py [runs/modal] > runs/modal/seed_stats.md
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/modal')
SEEDS = [0, 1, 2]
MODALITIES = {'text': ['mpnet', 'minilm', 'bge_base', 'bge_large'],
              'vision': ['resnet18', 'resnet50', 'vit_b16', 'dinov2_s']}
DETS = {'knn': 'kNN', 'mahalanobis_shrinkage': 'Maha', 'energy_T1': 'Energy', 'msp_T1': 'MSP'}
PATHS = {'radial': 'radial growth', 'origin': 'toward origin', 'pca': 'straight, PC', 'random': 'straight, random',
         'sphere_pca': 'geodesic, PC', 'sphere_random': 'geodesic, random'}
B = 4000


def run(seed, model, path):
    return ROOT / 'seeds' / f'seed{seed}' / model / 'full' / f'crossed_{path}'


def radius_alphas(d):
    r = json.loads((d / 'steering_checks.json').read_text())['id_reference_radius']
    return np.load(d / 'design.npz')['alphas'] / r


def matched_alpha(mod):
    return min(radius_alphas(run(s, m, p))[-1] for s in SEEDS for m in MODALITIES[mod] for p in PATHS)


def indicators(seed, model, path, det, alpha):
    """Per-probe rejection at alpha, averaged over references/directions/signs -> [P]."""
    d = run(seed, model, path)
    a = radius_alphas(d)
    i = int(np.searchsorted(a, alpha + 1e-9) - 1)
    rej = np.load(d / f'{det}_traces.npz')['rejected'][..., i]              # B, C, S, P
    return rej.reshape(-1, rej.shape[-1]).mean(0)


def hier_boot(v, rng):
    """v[seed, encoder]; resample encoders, then seeds within each chosen encoder."""
    S, E = v.shape
    enc = rng.integers(0, E, (B, E))
    sd = rng.integers(0, S, (B, E, S))
    means = v[sd, enc[..., None]].mean(axis=(1, 2))
    return np.quantile(means, [.025, .975])


def table_a(rng):
    out = ['## (A) Matched-horizon rejection: mean [95% hierarchical CI] (seed SD)', '']
    for mod in MODALITIES:
        a = matched_alpha(mod)
        out += [f'### {mod}: alpha = {a:.2f} ID radii (common to all seeds, encoders, paths)', '',
                '| path | ' + ' | '.join(DETS.values()) + ' |', '|---' * (len(DETS) + 1) + '|']
        for path, name in PATHS.items():
            cells = []
            for det in DETS:
                v = np.array([[indicators(s, m, path, det, a).mean() for m in MODALITIES[mod]] for s in SEEDS])
                lo, hi = hier_boot(v, rng)
                cells.append(f'{v.mean():.2f} [{lo:.2f}, {hi:.2f}] ({v.mean(1).std(ddof=1):.3f})')
            out.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        out.append('')
        out += [f'Per-encoder means over seeds ({mod}, toward origin / radial growth):', '']
        for path in ('origin', 'radial'):
            for det in DETS:
                v = np.array([[indicators(s, m, path, det, a).mean() for m in MODALITIES[mod]] for s in SEEDS])
                out.append(f'- {PATHS[path]}, {DETS[det]}: ' + ', '.join(
                    f'{m} {v[:, j].mean():.2f}±{v[:, j].std(ddof=1):.2f}' for j, m in enumerate(MODALITIES[mod])))
        out.append('')
    return out


CLAIMS = [  # (label, path, better detectors, worse detectors, modalities)
    ('growth: kNN > energy', 'radial', 'knn', 'energy_T1', ('text', 'vision')),
    ('growth: Maha > energy', 'radial', 'mahalanobis_shrinkage', 'energy_T1', ('text', 'vision')),
    ('growth: kNN > MSP', 'radial', 'knn', 'msp_T1', ('text', 'vision')),
    ('origin: energy > kNN', 'origin', 'energy_T1', 'knn', ('text', 'vision')),
    ('origin: energy > Maha', 'origin', 'energy_T1', 'mahalanobis_shrinkage', ('text', 'vision')),
    ('origin: MSP > kNN', 'origin', 'msp_T1', 'knn', ('text', 'vision')),
    ('straight PC: kNN > energy', 'pca', 'knn', 'energy_T1', ('text', 'vision')),
    ('geodesic PC: MSP > energy', 'sphere_pca', 'msp_T1', 'energy_T1', ('text', 'vision')),
]


def table_b(rng):
    out = ['## (B) Paired comparisons at the matched horizon', '',
           'Per (seed, encoder) cell: paired bootstrap over probes, one-sided p for the stated direction.', '',
           '| claim | modality | mean diff [95% hier. CI] | cells p<0.05 / 12 | cells with diff <= 0 |',
           '|---|---|---|---|---|']
    for label, path, i, j, mods in CLAIMS:
        for mod in mods:
            a = matched_alpha(mod)
            diffs, sig, nonpos = np.zeros((len(SEEDS), 4)), 0, 0
            for si, s in enumerate(SEEDS):
                for mi, m in enumerate(MODALITIES[mod]):
                    d = indicators(s, m, path, i, a) - indicators(s, m, path, j, a)
                    diffs[si, mi] = d.mean()
                    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
                    sig += float(np.mean(bs <= 0)) < .05
                    nonpos += d.mean() <= 0
            lo, hi = hier_boot(diffs, rng)
            out.append(f'| {label} | {mod} | {diffs.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {sig} | {nonpos} |')
    return out + ['']


def table_c():
    out = ['## (C) Static detection, mean ± SD over 3 seeds (float64 Mahalanobis)', '',
           '| encoder | ' + ' | '.join(f'{d} AUROC' for d in DETS.values()) + ' | ' +
           ' | '.join(f'{d} FPR95' for d in DETS.values()) + ' |', '|---' * (2 * len(DETS) + 1) + '|']
    for mod, models in MODALITIES.items():
        for m in models:
            ms = [json.loads((ROOT / 'seeds' / f'seed{s}' / m / 'full' / 'static' / 'metrics.json').read_text()) for s in SEEDS]
            au = [np.array([x[d]['auroc'] for x in ms]) for d in DETS]
            fp = [np.array([x[d]['fpr_at_95_tpr'] for x in ms]) for d in DETS]
            out.append(f'| {m} | ' + ' | '.join(f'{v.mean():.3f}±{v.std(ddof=1):.3f}' for v in au) + ' | ' +
                       ' | '.join(f'{v.mean():.3f}±{v.std(ddof=1):.3f}' for v in fp) + ' |')
    return out + ['']


def seed_view(seed):
    """Symlinked layout that the single-seed check scripts expect."""
    view = ROOT / 'seedviews' / f'seed{seed}'
    (view / 'main').mkdir(parents=True, exist_ok=True)
    src = (ROOT / 'seeds' / f'seed{seed}').resolve()
    for name, target in (('v2', src), ('caches', (ROOT / 'caches' if not seed else ROOT / 'caches' / f'seed{seed}').resolve())):
        if not (view / name).exists():
            os.symlink(target, view / name)
    for m in sum(MODALITIES.values(), []):
        if not (view / 'main' / m).exists():
            os.symlink(src / m, view / 'main' / m)
    return view


def table_d():
    import contextlib, io
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import radial_predictions as rp
    import support_checks as sc
    agg = {k: 0 for k in ('energy_origin', 'knn_origin', 'maha_origin', 'growth_energy', 'knn_bound', 'ck_general', 'ck_unit')}
    n, n_energy, n_near, n_dec, pairs = 0, 0, 0, 0, 0
    for s in SEEDS:
        rp.ROOT = sc.ROOT = seed_view(s)
        with contextlib.redirect_stdout(io.StringIO()):
            origin, growth, _ = rp.main()
            rows_k, _ = sc.main()
        for o, g, k in zip(origin, growth, rows_k):
            n += 1
            if o['energy_obs_reject_end_near0'] is not None:          # some probe reaches s < 0.1
                n_energy += 1
                agg['energy_origin'] += o['energy_pred_reject_at_0'] == (o['energy_obs_reject_end_near0'] == 1)
            if o['knn_decidable_pairs']:
                n_dec += 1
                agg['knn_origin'] += o['knn_decidable_agreement'] == 1
                pairs += o['knn_decidable_pairs']
            agg['ck_general'] += bool(k['general_equals_norm_rule'])
            if o['maha_agree_near0'] is not None:
                n_near += 1
                agg['maha_origin'] += o['maha_agree_near0'] == 1
                if k['unit_rule_correct'] is not None:
                    agg['ck_unit'] += k['unit_rule_correct'] == 1
            agg['growth_energy'] += g['energy_pred_obs_agreement'] == 1 and g['never_reject_rule_violations'] == 0
            agg['knn_bound'] += g['knn_bound_violations'] == 0
    out = ['## (D) Exact theory checks, per (seed, encoder) cell', '', f'n = {n} cells (3 seeds x 8 encoders)', '',
           '| check | cells passing |', '|---|---|']
    labels = {'energy_origin': 'energy origin limit: reject iff -lse(b) > t (probes reaching s < 0.1)',
              'knn_origin': 'kNN origin: reject iff r_(k) > t, on (fit, probe) pairs with |r_(k) - t| > ||z_end|| (1-Lipschitz)',
              'maha_origin': 'Mahalanobis origin: reject iff min mu^T P mu > t',
              'growth_energy': 'radial growth: energy closed form = traces, no new rejections',
              'knn_bound': 'radial growth: kNN crossing within 1-Lipschitz bound',
              'ck_general': 'c_k <= c* is algebraically identical to r_(k) <= t (every fit)',
              'ck_unit': 'kNN origin, unit-norm form c_k < 1/2 (expected to fail off the sphere)'}
    denom = {'energy_origin': n_energy, 'knn_origin': n_dec, 'maha_origin': n_near, 'ck_unit': n_near}
    out += [f'| {labels[k]} | {agg[k]}/{denom.get(k, n)} |' for k in labels]
    out += ['', f'kNN: {pairs} decidable (fit, probe) pairs in {n_dec} cells.',
            'Origin checks count only (seed, encoder) cells where some probe reaches s < 0.1; paths stop at',
            '0.99 x the smallest probe norm, so other probes end away from the origin.']
    return out + ['']


def main():
    rng = np.random.default_rng(0)
    doc = ['# Multi-seed statistics (3 seeds x 4 encoders per modality)', '',
           'Seeds vary split, head and bootstrap/direction draws together; seed 0 is the original split.', '']
    doc += table_a(rng) + table_b(rng) + table_c() + table_d()
    print('\n'.join(doc))


if __name__ == '__main__':
    main()
