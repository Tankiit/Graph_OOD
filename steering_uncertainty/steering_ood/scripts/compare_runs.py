"""Collect learned-direction runs under an output root into one table, to compare hyperparameters.

    python scripts/compare_runs.py runs/text_hparams
    python scripts/compare_runs.py runs/text_hparams --sort ood_gain --top 20
    python scripts/compare_runs.py runs/text_hparams --where layer=blocks.2 lr=0.001 --mean-over-seeds
    python scripts/compare_runs.py runs/text_hparams --csv runs/text_hparams/summary.csv

Reads every completed run (manifest.json, train_log.jsonl, sanity.json); archived *.old-* and
*.failed-* runs are ignored. Works for text and vision runs alike.

Per detector, rates are from the development sanity check at the training radius:
  ID side:  rejection of steering-dev ID queries, unsteered / learned id2ood / random vectors
  OOD side: acceptance of steering-dev OOD queries, unsteered / learned ood2id / random vectors
id_gain and ood_gain are the learned-minus-random rate averaged over detectors: how much more the
learned vectors flip detectors than random vectors of the same size (higher = stronger attack).
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

KEYS = ('model', 'layer', 'sim_layer', 'ood_mode', 'id_augment', 'radius', 'ood_noise_radius', 'lr', 'tau',
        'steps', 'swarm_size', 'bs_id')
SHORT = dict(knn='knn', mahalanobis_shrinkage='maha', energy='energy', msp='msp')


def fmt(v):
    return f'{v:g}' if isinstance(v, float) else str(v)


def collect(root):
    root, rows = Path(root), []
    for manifest_path in sorted(root.rglob('manifest.json')):
        run = manifest_path.parent
        if '.old-' in run.name or '.failed-' in run.name or not (run / 'sanity.json').exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') != 'completed':
            continue
        cfg, sanity = manifest['config'], json.loads((run / 'sanity.json').read_text())
        log = [json.loads(line) for line in (run / 'train_log.jsonl').read_text().splitlines() if line.strip()]
        seed_dir = next((p for p in run.relative_to(root).parts if p.startswith('seed')), 'seed?')
        row = {k: cfg.get(k) for k in KEYS}
        row.update(seed=seed_dir[4:], seconds=round(manifest.get('seconds') or 0),
                   id2ood_dev_loss=log[-1].get('id2ood_dev_loss'), ood2id_dev_loss=log[-1].get('ood2id_dev_loss'))
        gains = defaultdict(list)
        for side, learned in (('id', 'id2ood'), ('ood', 'ood2id')):
            for det in sanity['thresholds']:
                learned_rate = sanity['sides'][side][learned][det]['rate']
                random_rate = sanity['sides'][side]['random'][det]['rate']
                row[f'{side}_clean_{det}'] = sanity.get('clean', {}).get(side, {}).get(det, {}).get('rate')
                row[f'{side}_learned_{det}'] = learned_rate
                row[f'{side}_random_{det}'] = random_rate
                gains[side].append(learned_rate - random_rate)
        row.update(id_gain=sum(gains['id']) / len(gains['id']), ood_gain=sum(gains['ood']) / len(gains['ood']),
                   path=str(run.relative_to(root)))
        rows.append(row)
    return rows


def mean_over_seeds(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in KEYS)].append(row)
    merged = []
    for key, members in groups.items():
        row = dict(zip(KEYS, key), seed=f'n={len(members)}', path=members[0]['path'].split('/', 1)[-1])
        for col in members[0]:
            if col in row or col in KEYS:
                continue
            values = [m[col] for m in members if isinstance(m.get(col), (int, float))]
            row[col] = sum(values) / len(values) if values else None
        merged.append(row)
    return merged


def show(rows, detectors):
    """Fixed-width table; hyperparameters that are the same in every row are printed once above it."""
    constant = [k for k in KEYS if len({fmt(r[k]) for r in rows}) == 1]
    varying = [k for k in KEYS if k not in constant] + ['seed']
    if constant:
        print('fixed: ' + '  '.join(f'{k}={fmt(rows[0][k])}' for k in constant))
    cols = varying + ['id_gain', 'ood_gain', 'id2ood_dev_loss', 'ood2id_dev_loss']
    cols += [f'id_learned_{d}' for d in detectors] + [f'ood_learned_{d}' for d in detectors]
    head = {c: c.replace('_learned_', ':').replace('mahalanobis_shrinkage', 'maha')
               .replace('_dev_loss', ' dev') for c in cols}
    cell = lambda r, c: '' if r.get(c) is None else (f'{r[c]:.3f}' if isinstance(r[c], float) else fmt(r[c]))
    width = {c: max(len(head[c]), *(len(cell(r, c)) for r in rows)) for c in cols}
    print('  '.join(head[c].ljust(width[c]) for c in cols))
    bold = (lambda s: f'\033[1m{s}\033[0m') if sys.stdout.isatty() else (lambda s: s)
    best = {c: max((r[c] for r in rows if isinstance(r.get(c), float)), default=None) for c in ('id_gain', 'ood_gain')}
    for r in rows:
        print('  '.join((bold if c in best and r.get(c) == best[c] else str)(cell(r, c).ljust(width[c])) for c in cols))
    clean = {side: '  '.join(f"{SHORT.get(d, d)} {rows[0][f'{side}_clean_{d}']:.3f}" for d in detectors
                            if rows[0].get(f'{side}_clean_{d}') is not None) for side in ('id', 'ood')}
    print(f"unsteered (first row): ID rejection {clean['id']}  |  OOD acceptance {clean['ood']}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('root', type=Path, nargs='?', default=Path('runs/learned_directions'))
    p.add_argument('--where', nargs='+', default=[], metavar='KEY=VALUE', help='keep runs matching every filter')
    p.add_argument('--sort', default='id_gain', help='column to sort by, descending (e.g. ood_gain, id_learned_knn)')
    p.add_argument('--top', type=int, help='show only the first N rows')
    p.add_argument('--mean-over-seeds', action='store_true', help='average runs that differ only by seed')
    p.add_argument('--csv', type=Path, help='also write every column of every run to this CSV file')
    a = p.parse_args()
    rows = collect(a.root)
    for item in a.where:
        key, _, value = item.partition('=')
        rows = [r for r in rows if fmt(r.get(key)) == value or str(r.get(key)) == value]
    if not rows:
        raise SystemExit(f'No completed runs under {a.root} match')
    if a.mean_over_seeds:
        rows = mean_over_seeds(rows)
    if a.sort not in rows[0]:
        raise SystemExit(f'Unknown sort column {a.sort!r}; columns: {", ".join(rows[0])}')
    rows.sort(key=lambda r: (r.get(a.sort) is None, -(r.get(a.sort) or 0)) if isinstance(r.get(a.sort), (int, float))
              else (False, 0))
    detectors = [c[len('id_learned_'):] for c in rows[0] if c.startswith('id_learned_')]
    show(rows[:a.top] if a.top else rows, detectors)
    print(f'{len(rows)} run(s)' + (' (seed means)' if a.mean_over_seeds else '') + f', sorted by {a.sort}')
    if a.csv:
        columns = list(dict.fromkeys(c for r in rows for c in r))
        with open(a.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f'wrote {a.csv}')


if __name__ == '__main__':
    main()
