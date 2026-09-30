"""Local end-to-end pipeline for the paper grid (no cloud dependencies).

Stages (idempotent: completed steps are skipped; partial crossed runs are archived as *.failed-*):
  prepare  CLINC150 + CIFAR10/CIFAR100/SVHN splits for each seed
  encode   frozen encoders -> feature caches (use --device cuda if available)
  grid     per (seed, model): ID head + static metrics; per path kind: crossed run + steering checks

Seed conventions (identical to the reference runs):
  seed s re-partitions CLINC with seed 20260929+s and CIFAR with 7+s, trains the head with 7+s and
  draws references/directions/probes with 7+s. Seed 0 is the original split.
Horizon: 3 x median ||reference - mean(reference)|| (ID development data only).

Layout under --root (default runs/modal, which the analysis scripts read by default):
  data/[seed{s}/]clinc_splits.json, data/[seed{s}/]cifar10_splits.json, data/images/
  caches/[seed{s}/]{model}.npz
  seeds/seed{s}/{model}/full/{head,static,crossed_<kind>}

Examples:
  python scripts/run_pipeline.py prepare --seeds 0 1 2 --download
  python scripts/run_pipeline.py encode --seeds 0 1 2 --device cuda
  python scripts/run_pipeline.py grid --seeds 0 1 2 --jobs 4
  python scripts/run_pipeline.py grid --seeds 0 --models mpnet --kinds radial origin --budget pilot
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
MODELS = {
    'mpnet':     ('text', ['--model', 'sentence-transformers/all-mpnet-base-v2']),
    'minilm':    ('text', ['--model', 'sentence-transformers/all-MiniLM-L6-v2']),
    'bge_base':  ('text', ['--model', 'BAAI/bge-base-en-v1.5']),
    'bge_large': ('text', ['--model', 'BAAI/bge-large-en-v1.5']),
    'resnet18':  ('vision', ['--backend', 'torchvision', '--model', 'resnet18', '--weights', 'IMAGENET1K_V1']),
    'resnet50':  ('vision', ['--backend', 'torchvision', '--model', 'resnet50', '--weights', 'IMAGENET1K_V2']),
    'vit_b16':   ('vision', ['--backend', 'torchvision', '--model', 'vit_b_16', '--weights', 'IMAGENET1K_V1']),
    'dinov2_s':  ('vision', ['--backend', 'timm', '--model', 'vit_small_patch14_dinov2.lvd142m', '--weights', 'DEFAULT']),
}
KINDS = ['pca', 'random', 'sphere_pca', 'sphere_random', 'origin', 'radial']
DETECTORS = ['knn', 'mahalanobis_shrinkage', 'energy', 'msp']
HORIZON_RADII = 3.0
BUDGETS = {  # paper grid uses 'medium' (full-dimensional features)
    'medium': dict(references=6, directions=6, probes=64, steps=101, calibration_draws=3),
    'pilot': dict(references=3, directions=3, probes=32, steps=41, calibration_draws=2),
}
CLINC_URL = 'https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json'


def cli(*args, threads=None):
    env = dict(os.environ, PYTHONPATH=str(PKG))
    if threads:
        env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    cmd = [sys.executable, '-m', 'steering_ood', *map(str, args)]
    print('+', ' '.join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True, cwd=PKG, env=env, stdout=subprocess.DEVNULL)


def splits(root, seed):
    base = root / 'data' / (f'seed{seed}' if seed else '')
    return {'text': base / 'clinc_splits.json', 'vision': base / 'cifar10_splits.json'}


def cache(root, seed, model):
    return root / 'caches' / (f'seed{seed}' if seed else '') / f'{model}.npz'


def cell(root, seed, model):
    return root / 'seeds' / f'seed{seed}' / model / 'full'


def prepare(root, seed, download):
    src = root / 'data' / 'clinc_data_full.json'
    src.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        urllib.request.urlretrieve(CLINC_URL, src)
    sp = splits(root, seed)
    sp['text'].parent.mkdir(parents=True, exist_ok=True)
    if not sp['text'].exists():
        cli('prepare', '--source', src, '--output', sp['text'], '--seed', 20260929 + seed)
    if not sp['vision'].exists():
        cli('prepare-vision', '--root', root / 'data' / 'images', '--id-dataset', 'cifar10',
            '--ood-datasets', 'cifar100', 'svhn', '--counts', 500, 100, 100, 100, 20,
            '--test-limit', 5000, '--output', sp['vision'], '--seed', 7 + seed,
            *(['--download'] if download else []))


def encode(root, seed, model, device, batch_size):
    out = cache(root, seed, model)
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    modality, args = MODELS[model]
    cli('encode' if modality == 'text' else 'encode-vision', '--splits', splits(root, seed)[modality],
        '--output', out, '--device', device, '--batch-size', batch_size, *args)


def id_radius(x):
    import numpy as np
    return float(np.median(np.linalg.norm(x - x.mean(0), axis=1)))


def verify_steering(cache_path, head_dir, run_dir, detectors, k=5, n_check=64, seed=0):
    """Independently refit reference 0 and recompute scores at sampled points on the steered path."""
    import numpy as np
    sys.path.insert(0, str(PKG))
    from steering_ood.core import load_cache
    from steering_ood.detectors import make_detector
    from steering_ood.experiment import path_points
    from steering_ood.head import load_head
    data, head = load_cache(cache_path), load_head(head_dir, cache_path)
    design = np.load(run_dir / 'design.npz')
    ref = design['reference_indices'][0]
    x = data['probe_x'][design['probe_indices']]
    alphas, vectors, signs = design['alphas'], design['vectors'], design['signs']
    kind = json.loads((run_dir / 'manifest.json').read_text())['config']['direction_kind']
    rng = np.random.default_rng(seed)
    out = {}
    for name in detectors:
        label = f'{name}_T1' if name in ('energy', 'msp') else name
        tr = np.load(run_dir / f'{label}_traces.npz')
        scores, thr = tr['scores'], tr['thresholds']
        det = make_detector(name, 'pytorch', k, head).fit(data['reference_x'][ref], data['reference_y'][ref])
        c, s, p, a = (rng.integers(0, n, n_check) for n in (len(vectors), len(signs), len(x), len(alphas)))
        pts = np.array([path_points(kind, x[pi], signs[si] * vectors[ci], alphas[ai:ai + 1])[0]
                        for ci, si, pi, ai in zip(c, s, p, a)])
        cal = det.score(data['calibration_x'])
        delta = scores[..., -1] - scores[..., 0]
        out[label] = dict(
            max_abs_recompute_error=float(np.max(np.abs(det.score(pts) - scores[0, c, s, p, a]))),
            alpha0_matches_unsteered=float(np.max(np.abs(scores[0, :, :, :, 0] - det.score(x)[None, None]))),
            unit_directions=bool(np.allclose(np.linalg.norm(vectors, axis=1), 1)),
            max_rel_norm_change=float(np.max(np.abs(np.linalg.norm(pts, axis=1) / np.linalg.norm(x[p], axis=1) - 1))),
            mean_score_change_over_horizon_in_cal_sd=float(delta.mean() / cal.std()),
            frac_paths_score_increases=float(np.mean(delta > 0)),
            probe_rejection_at_alpha0=float(tr['rejected'][..., 0].mean()),
            probe_rejection_at_horizon=float(tr['rejected'][..., -1].mean()),
            threshold_mean=float(np.mean(thr)))
    return out


def setup(root, seed, model, threads):
    base = cell(root, seed, model)
    base.mkdir(parents=True, exist_ok=True)
    c, head = cache(root, seed, model), base / 'head'
    if not (head / 'head.pt').exists():
        cli('train-head', '--cache', c, '--output', head, '--epochs', 30, '--seed', 7 + seed, threads=threads)
    if not (base / 'static' / 'metrics.json').exists():
        cli('evaluate', '--cache', c, '--head', head, '--detectors', *DETECTORS, '--k', 5,
            '--output', base / 'static', '--temperatures', 1, 1000, threads=threads)


def crossed(root, seed, model, kind, budget, threads):
    import numpy as np
    sys.path.insert(0, str(PKG))
    from steering_ood.core import load_cache, write_json
    base = cell(root, seed, model)
    c, head, out = cache(root, seed, model), base / 'head', base / f'crossed_{kind}'
    b = dict(BUDGETS[budget], **({'directions': 1} if kind in ('origin', 'radial') else {}))
    radius = id_radius(load_cache(c)['reference_x'])
    done = out / 'manifest.json'
    if not done.exists() or json.loads(done.read_text())['status'] != 'completed':
        if out.exists():
            out.rename(out.with_name(f'{out.name}.failed-{os.urandom(3).hex()}'))
        cli('crossed', '--cache', c, '--head', head, '--detectors', *DETECTORS, '--k', 5, '--output', out,
            '--direction-kind', kind, '--references', b['references'], '--directions', b['directions'],
            '--probes', b['probes'], '--steps', b['steps'], '--refine', 15, '--horizon', HORIZON_RADII * radius,
            '--calibration-draws', b['calibration_draws'], '--seed', 7 + seed, threads=threads)
        (out / 'pip_freeze.txt').write_text(subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                                                           capture_output=True, text=True).stdout)
    checks = verify_steering(c, head, out, DETECTORS)
    write_json(out / 'steering_checks.json', dict(
        id_reference_radius=radius, horizon=json.loads(done.read_text())['effective_horizon'],
        requested_horizon=HORIZON_RADII * radius,
        horizon_rule=f'{HORIZON_RADII} x median ||reference - mean(reference)|| (ID dev only)', detectors=checks))
    return f'seed{seed}/{model}/{kind}: max recompute error ' + \
        f"{max(v['max_abs_recompute_error'] for v in checks.values()):.2e}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['prepare', 'encode', 'grid'])
    p.add_argument('--root', type=Path, default=PKG / 'runs' / 'modal')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    p.add_argument('--kinds', nargs='+', default=KINDS, choices=KINDS)
    p.add_argument('--budget', choices=list(BUDGETS), default='medium')
    p.add_argument('--jobs', type=int, default=1, help='parallel crossed runs')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--download', action='store_true', help='download CIFAR/SVHN if missing (slow)')
    a = p.parse_args()
    root = a.root.resolve()
    if a.stage == 'prepare':
        for s in a.seeds:
            prepare(root, s, a.download)
    elif a.stage == 'encode':
        for s in a.seeds:
            for m in a.models:
                encode(root, s, m, a.device, a.batch_size)
    else:
        threads = max(1, (os.cpu_count() or 1) // a.jobs)
        for s in a.seeds:
            for m in a.models:
                setup(root, s, m, threads)
        tasks = [(root, s, m, k, a.budget, threads) for s in a.seeds for m in a.models for k in a.kinds]
        with ProcessPoolExecutor(a.jobs) as pool:
            futures = {pool.submit(crossed, *t): t for t in tasks}
            for f in as_completed(futures):
                try:
                    print(f.result(), flush=True)
                except Exception as exc:  # report and continue; rerun the command to retry
                    print('FAILED', futures[f][1:4], repr(exc), flush=True)


if __name__ == '__main__':
    main()
