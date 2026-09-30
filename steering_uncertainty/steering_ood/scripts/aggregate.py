"""Collect static OOD metrics, crossed-steering metrics and steering checks into tables.

Usage: python scripts/aggregate.py runs/modal/<tag>
Writes static.csv, steering.csv, checks.csv and summary.md into that directory.
Crossing distances are also reported in units of the ID reference radius so that
encoders with different feature scales can be compared.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np


def rows_for(run):
    model, rep = run.parent.name, run.name
    static, steering, checks = [], [], []
    m = json.loads((run / 'static' / 'metrics.json').read_text())
    acc = m.pop('id_classification_accuracy', None); m.pop('intent_accuracy', None)
    for det, r in m.items():
        row = dict(model=model, rep=rep, detector=det, id_acc=acc, auroc=r['auroc'],
                   aupr=r['ood_aupr'], fpr95=r['fpr_at_95_tpr'],
                   id_rejection=r['id_rejection'], ood_detection=r['ood_detection'])
        for g, gr in r.get('ood_by_dataset', {}).items():
            row[f'auroc_{g}'], row[f'fpr95_{g}'] = gr['auroc'], gr['fpr_at_95_tpr']
        static.append(row)
    for crossed in sorted(run.glob('crossed_*')):
        if '.failed' in crossed.name or not (crossed / 'steering_checks.json').exists():
            continue
        kind = crossed.name.split('_', 1)[1]
        chk = json.loads((crossed / 'steering_checks.json').read_text())
        radius = chk['id_reference_radius']
        summary = json.loads((crossed / 'summary.json').read_text())
        for det, s in summary.items():
            tr = np.load(crossed / f'{det}_traces.npz')
            obs, ev = tr['observed'], tr['event']
            crossed_only = obs[ev & ~tr['initially_rejected']]
            v = {sign: s['variance'][sign] for sign in s['variance']}
            tot = sum(v[k]['total'] for k in v) or np.nan
            steering.append(dict(
                model=model, rep=rep, direction=kind, detector=det,
                crossing_prob=s['crossing_probability'],
                initial_rejection=s['initial_rejection_probability'],
                median_alpha_star_radii=float(np.median(crossed_only) / radius) if len(crossed_only) else None,
                restricted_mean_radii=s['restricted_crossing_mean'] / radius,
                return_prob=s['return_or_recross_probability'],
                power_at_horizon=float(tr['power'][..., -1].mean()),
                var_reference=sum(v[k]['reference'] for k in v) / tot,
                var_direction=sum(v[k]['direction'] for k in v) / tot,
                var_interaction=sum(v[k]['interaction'] for k in v) / tot,
                var_residual_probe=1 - sum(v[k][c] for k in v for c in ('reference', 'direction', 'interaction')) / tot))
        for det, c in chk['detectors'].items():
            checks.append(dict(model=model, rep=rep, direction=kind, detector=det, **c))
    return static, steering, checks


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)


def fmt(v):
    return f'{v:.3f}' if isinstance(v, float) else ('' if v is None else str(v))


def md_table(rows, keys):
    lines = ['| ' + ' | '.join(keys) + ' |', '|' + '---|' * len(keys)]
    lines += ['| ' + ' | '.join(fmt(r.get(k)) for k in keys) + ' |' for r in rows]
    return '\n'.join(lines)


def link_rows(static, steering):
    """Join static detection quality with steering sensitivity per (model, rep, detector)."""
    st = {(r['model'], r['rep'], r['detector']): r for r in static}
    out = {}
    for r in steering:
        key = (r['model'], r['rep'], r['detector'])
        if key not in st:
            continue
        row = out.setdefault(key, dict(model=key[0], rep=key[1], detector=key[2],
                                       auroc=st[key]['auroc'], fpr95=st[key]['fpr95']))
        row[f"crossing_{r['direction']}"] = r['crossing_prob']
        row[f"alpha_radii_{r['direction']}"] = r['median_alpha_star_radii']
    return list(out.values())


def spearman(rows, x, y):
    from scipy.stats import spearmanr
    pairs = [(r[x], r[y]) for r in rows if r.get(x) is not None and r.get(y) is not None]
    return spearmanr(*zip(*pairs)) if len(pairs) > 3 else None


def main(root):
    root = Path(root)
    static, steering, checks = [], [], []
    for run in sorted(p.parent.parent for p in root.glob('*/*/static/metrics.json')):
        a, b, c = rows_for(run)
        static += a; steering += b; checks += c
    link = link_rows(static, steering)
    for name, rows in (('static', static), ('steering', steering), ('checks', checks), ('link', link)):
        write_csv(root / f'{name}.csv', rows)
    groups = sorted({k[6:] for r in static for k in r if k.startswith('auroc_')})
    doc = ['# Steering / OOD results', '', '## Static held-out OOD detection', '',
           md_table(static, ['model', 'rep', 'detector', 'id_acc', 'auroc', 'fpr95', 'aupr']
                    + [f'auroc_{g}' for g in groups]), '',
           '## Crossed steering (probes pushed along z + alpha*v)', '',
           md_table(steering, ['model', 'rep', 'direction', 'detector', 'crossing_prob',
                               'initial_rejection', 'median_alpha_star_radii', 'power_at_horizon',
                               'return_prob', 'var_reference', 'var_direction', 'var_interaction']), '',
           '## Steering sensitivity vs static detection', '',
           'Spearman correlation across model x representation cells, per detector. '
           'crossing = fraction of steered ID probes rejected within the horizon; '
           'alpha = median crossing distance in ID radii.', '',
           md_table([dict(detector=d, n=len(rs), **{
               f'rho({a},auroc)': (lambda r: None if r is None else f'{r.statistic:.2f} (p={r.pvalue:.2g})')(spearman(rs, a, 'auroc'))
               for a in ('crossing_pca', 'crossing_random', 'alpha_radii_pca', 'alpha_radii_random')})
               for d in sorted({r['detector'] for r in link}) for rs in [[r for r in link if r['detector'] == d]]],
               ['detector', 'n'] + [f'rho({a},auroc)' for a in ('crossing_pca', 'crossing_random', 'alpha_radii_pca', 'alpha_radii_random')]), '',
           '## Steering checks', '',
           md_table(checks, ['model', 'rep', 'direction', 'detector', 'max_abs_recompute_error',
                             'alpha0_matches_unsteered', 'mean_score_change_over_horizon_in_cal_sd',
                             'frac_paths_score_increases', 'probe_rejection_at_alpha0',
                             'probe_rejection_at_horizon'])]
    (root / 'summary.md').write_text('\n'.join(doc) + '\n')
    print(f'{len(static)} static, {len(steering)} steering, {len(checks)} check rows -> {root}')


if __name__ == '__main__':
    main(sys.argv[1])
