"""Phase-1 audit of all saved runs: ranges, non-finite values, calibration, head training, variance.

Usage: python scripts/audit_results.py [runs/modal]
Prints a JSON report; aggregates are over reference bootstraps and encoders (one seed per run).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'runs/modal')
TEXT = {'mpnet', 'minilm', 'bge_base', 'bge_large'}


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))


def main():
    rep = {'nonfinite': [], 'range_violations': [], 'manifests': {}, 'infinite_thresholds': []}
    # 1. manifests: status, seeds, runtime
    secs, status = {}, {}
    for f in glob.glob(str(ROOT / '*/*/*/*/manifest.json')):
        if '.failed' in f:
            continue
        m = json.loads(Path(f).read_text())
        status[m.get('status')] = status.get(m.get('status'), 0) + 1
        secs[f.split('/')[-5]] = secs.get(f.split('/')[-5], 0) + m.get('seconds', 0)
        rep['manifests'].setdefault('seeds', set()).add(m['config'].get('seed'))
    rep['manifests']['status_counts'] = status
    rep['manifests']['seeds'] = sorted(rep['manifests']['seeds'])
    rep['manifests']['cpu_hours_by_tag_16cpu'] = {k: round(v / 3600, 2) for k, v in secs.items()}
    rep['failed_attempt_dirs'] = len(glob.glob(str(ROOT / '*/*/*/*.failed-*')))

    # 2. static metrics: ranges and held-out ID rejection vs tau
    idrej = {'text': [], 'vision': []}
    for f in glob.glob(str(ROOT / '*/*/*/static/metrics.json')):
        m = json.loads(Path(f).read_text())
        model = f.split('/')[-4]
        for det, r in m.items():
            if not isinstance(r, dict):
                if not 0 <= r <= 1:
                    rep['range_violations'].append((f, det, r))
                continue
            for k in ('auroc', 'ood_aupr', 'fpr_at_95_tpr', 'id_rejection', 'ood_detection'):
                if not 0 <= r[k] <= 1:
                    rep['range_violations'].append((f, det, k, r[k]))
            if r['threshold_infinite']:
                rep['infinite_thresholds'].append((f, det))
            idrej['text' if model in TEXT else 'vision'].append(r['id_rejection'])
    rep['heldout_id_rejection_tau_0.05'] = {k: dict(zip(('mean', 'min', 'max'), (float(np.mean(v)), float(np.min(v)), float(np.max(v)))))
                                            for k, v in idrej.items()}

    # 3. traces: non-finite scores, threshold finiteness
    n_tr = 0
    for f in glob.glob(str(ROOT / '*/*/*/crossed_*/*_traces.npz')):
        if '.failed' in f or 'incomplete' in f:
            continue
        t = np.load(f)
        n_tr += 1
        if not np.isfinite(t['scores']).all():
            rep['nonfinite'].append(f)
        if not np.isfinite(t['thresholds']).all():
            rep['infinite_thresholds'].append(f)
    rep['trace_files_checked'] = n_tr

    # 4. heads: loss trajectory and ID accuracy
    heads = {}
    for f in glob.glob(str(ROOT / 'main/*/*/head/history.json')):
        h = json.loads(Path(f).read_text())
        loss = [e['train_loss'] for e in h]
        heads['/'.join(f.split('/')[-4:-2])] = dict(first=round(loss[0], 3), last=round(loss[-1], 3),
                                                     monotone_frac=round(float(np.mean(np.diff(loss) <= 0)), 2))
    rep['heads_loss'] = heads or 'history.json not downloaded'

    # 5. variance: static AUROC across reference bootstraps (crossed summaries) and across encoders
    var = {}
    for f in glob.glob(str(ROOT / 'v2/*/full/crossed_radial/summary.json')):
        model = f.split('/')[-4]
        s = json.loads(Path(f).read_text())
        for det, r in s.items():
            a = [st['auroc'] for st in r['static']]
            var.setdefault(det, {})[model] = (round(float(np.mean(a)), 3), round(float(np.std(a)), 4))
    rep['auroc_mean_sd_over_reference_bootstraps'] = var

    # 6. encoder-level bootstrap CIs for the headline matched-horizon cells (per-encoder means)
    cis = {}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from path_curves import GRID, MODALITIES, collect, reach
    data = collect()
    for mod in MODALITIES:
        a = reach(data, mod, ('pca', 'random', 'sphere_pca', 'sphere_random', 'origin', 'radial'))
        i = int(np.searchsorted(GRID, a - 1e-9))
        for path in ('radial', 'origin'):
            for det in ('knn', 'mahalanobis_shrinkage', 'energy_T1', 'msp_T1'):
                vals = data[mod, det, path][0][:, i]
                m, lo, hi = boot_ci(vals)
                cis[f'{mod}/{path}/{det}'] = dict(mean=round(m, 3), ci95=(round(lo, 3), round(hi, 3)),
                                                   per_encoder=[round(float(v), 3) for v in vals])
    rep['matched_horizon_encoder_bootstrap'] = cis
    print(json.dumps(rep, indent=1, default=str))


if __name__ == '__main__':
    main()
