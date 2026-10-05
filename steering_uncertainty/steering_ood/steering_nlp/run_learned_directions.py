"""Train learned steering directions (InfoNCE swarms) on CLINC150 sentence encoders and sanity-check them.

    python -m steering_nlp.run_learned_directions --seeds 0 --budget pilot --device cuda
    python -m steering_nlp.run_learned_directions --seeds 0 --budget pilot --device cuda --model mpnet minilm bge_base bge_large
    python -m steering_nlp.run_learned_directions --seeds 0 1 2 --device cuda --sweep ood_mode=clinc,random lr=0.01,0.001

Text counterpart of scripts/run_learned_directions.py, with the same flags and behaviour. Needs the
outputs of scripts/run_pipeline.py for the same seeds and models: the CLINC splits, the model's
feature cache and its trained head, plus CLINC's data_full.json for the out-of-scope train/val
queries. --model takes one or more of mpnet, minilm, bge_base, bge_large (default mpnet); --layer
defaults to the block at the ViTs' relative depth (blocks.9, blocks.4, blocks.9, blocks.19).
Every TextSteerConfig field is a flag; --budget sets the swarm size, steps and batch sizes unless they
are given explicitly. Completed runs are skipped; partial runs are archived as *.failed-*. --force
recomputes a completed run and keeps the old one as *.old-*. --sweep KEY=V1,V2 ... trains every
combination of the listed values; a failed run is reported and the others continue.

Seed s uses the seed-s splits/cache/head and 7+s for the steering split, swarm initialisation,
batches, dropout views and random vectors.

Layout:
  <output-root>/seed{s}/steer_splits_clinc.json
  <output-root>/seed{s}/{model}/{layer}__sim-{sim_layer}/{ood_mode}[-noaug]/r{radius}__{budget}[__{name}][__{key}-{value}...]/
      manifest.json, train_log.jsonl, vectors.npz, sanity.json
"""
import argparse
import itertools
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
BUDGETS = {  # vision budgets, except 500 confirmatory steps: the text pilots' dev loss is flat by then
    'pilot': dict(swarm_size=8, steps=100, bs_id=32, bs_ood=32, log_every=10),
    'confirmatory': dict(swarm_size=64, steps=500, bs_id=64, bs_ood=64, log_every=25),
}
MODEL_BUDGETS = {'confirmatory': {}}  # per-model overrides of a budget, if a model needs its own step count
SPLIT_KEYS = ('seed', 'dev_fraction')
PATH_KEYS = ('layer', 'sim_layer', 'ood_mode', 'id_augment', 'radius')  # already in the run directory


def inputs(root, seed, model):
    sub = f'seed{seed}' if seed else ''
    return dict(splits=root / 'data' / sub / 'clinc_splits.json', cache=root / 'caches' / sub / f'{model}.npz',
                head_dir=root / 'seeds' / f'seed{seed}' / model / 'full' / 'head')


def run(a, seed, overrides, tag=''):
    from steering_nlp.learned_directions import (STEER_SPLITS_NAME, TextSteerConfig, prepare_steering_text,
                                                 train_text_directions)
    budget = {**BUDGETS[a.budget], **MODEL_BUDGETS.get(a.budget, {}).get(overrides['model'], {})}
    cfg = TextSteerConfig(**{**budget, **overrides, 'seed': 7 + seed}).validate()
    root = a.root.resolve()
    paths, raw = inputs(root, seed, cfg.model), root / 'data' / 'clinc_data_full.json'
    missing = [str(p) for p in (paths['splits'], paths['cache'], paths['head_dir'] / 'head.pt', raw) if not p.exists()]
    if missing:
        raise FileNotFoundError(f'Run scripts/run_pipeline.py for seed {seed} first; missing {missing}')
    base = a.output_root.resolve() / f'seed{seed}'
    steer_splits = base / STEER_SPLITS_NAME
    wanted = {k: getattr(cfg, k) for k in SPLIT_KEYS}
    if not steer_splits.exists():
        # Parallel jobs may prepare the same (deterministic) split; readers only ever see a complete file.
        base.mkdir(parents=True, exist_ok=True)
        partial = steer_splits.with_name(f'{steer_splits.name}.{os.getpid()}.partial')
        prepare_steering_text(paths['splits'], raw, partial, **wanted)
        os.replace(partial, steer_splits)
    found = json.loads(steer_splits.read_text())['metadata']
    if {k: found[k] for k in SPLIT_KEYS} != wanted:
        raise ValueError(f'{steer_splits} was prepared with other settings; use another --output-root')
    leaf = f'r{cfg.radius:g}__{a.budget}' + (f'__{a.name}' if a.name else '') + tag
    out = (base / cfg.model / f'{cfg.layer}__sim-{cfg.sim_layer}'
           / (cfg.ood_mode + ('' if cfg.id_augment else '-noaug')) / leaf)
    done = out / 'manifest.json'
    completed = done.exists() and json.loads(done.read_text())['status'] == 'completed'
    if completed and not a.force:
        return print(f'{out}: already completed (use --force to recompute)', flush=True)
    if out.exists():
        out.rename(out.with_name(f"{out.name}.{'old' if completed else 'failed'}-{os.urandom(3).hex()}"))
    print(f'== {out}', flush=True)
    sanity = train_text_directions(cfg, steer_splits=steer_splits, output=out, device=a.device, **paths)
    print('sanity check on steering-dev queries (vectors applied at the training radius):')
    bold = (lambda s: f'\033[1m{s}\033[0m') if sys.stdout.isatty() else (lambda s: s)
    for side, families in sanity['sides'].items():
        for det in cfg.detectors:
            row = {'no steering': sanity['clean'][side][det]['rate'], **{f: e[det]['rate'] for f, e in families.items()}}
            best = max(row.values())  # the highest flip rate is the strongest attack
            print(f"  {sanity['outcome'][side]:20s} {det:22s} "
                  + '  '.join(f'{k} ' + (bold if v == best else str)(f'{v:.3f}') for k, v in row.items()))
    print(f'saved {out}', flush=True)


def main():
    from steering_nlp.learned_directions import TEXT_MODELS, TextSteerConfig
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=PKG / 'runs' / 'modal', help='run_pipeline.py outputs')
    p.add_argument('--output-root', type=Path, default=PKG / 'runs' / 'learned_directions')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--budget', choices=list(BUDGETS), default='confirmatory')
    p.add_argument('--device', default='cpu')
    p.add_argument('--name', default='', help='suffix that keeps runs with other overrides apart')
    p.add_argument('--force', action='store_true', help='recompute completed runs; the old run is kept as *.old-*')
    skip = ('seed', 'model', 'n_ood_train', 'n_ood_dev')  # the OOD pools are CLINC oos_train / oos_val
    names, types = [], {f.name: f.type for f in fields(TextSteerConfig)}
    for f in fields(TextSteerConfig):
        if f.name in skip:
            continue
        names.append(f.name)
        flag = '--' + f.name.replace('_', '-')
        if f.type is bool:
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=None, help=f'default {f.default}')
        elif f.type is tuple:
            p.add_argument(flag, nargs='+', default=None, help=f'default {" ".join(f.default)}')
        else:
            p.add_argument(flag, type=f.type, default=None, help=f'default {f.default} (or the budget)')
    p.add_argument('--model', nargs='+', choices=list(TEXT_MODELS), default=['mpnet'], help='one or more encoders')
    p.add_argument('--sweep', nargs='+', default=[], metavar='KEY=V1,V2',
                   help='grid: every combination of the listed values of any single-valued hyperparameter')
    a = p.parse_args()
    overrides = {k: getattr(a, k) for k in names if getattr(a, k) is not None}
    if 'detectors' in overrides:
        overrides['detectors'] = tuple(overrides['detectors'])
    grid = []
    for item in a.sweep:
        key, sep, raw = item.partition('=')
        key = key.strip().replace('-', '_')
        if not sep or key not in names or types[key] is tuple:
            p.error(f'--sweep expects KEY=V1,V2 with a single-valued hyperparameter, got {item!r}'
                    + (' (use --model for models)' if key == 'model' else ''))
        if key in SPLIT_KEYS:
            p.error(f'{key} defines the steering split; change it with --{key.replace("_", "-")} and another --output-root')
        try:
            cast = (lambda v: {'true': True, 'false': False}[v.lower()]) if types[key] is bool else types[key]
            grid.append([(key, cast(v.strip())) for v in raw.split(',')])
        except (KeyError, ValueError):
            p.error(f'invalid value in --sweep {item!r}')
    combos = list(itertools.product(*grid))
    total, failed = len(a.model) * len(a.seeds) * len(combos), []
    for model in a.model:
        for seed in a.seeds:
            for combo in combos:
                tag = ''.join(f'__{k}-{v:g}' if isinstance(v, float) else f'__{k}-{v}'
                              for k, v in combo if k not in PATH_KEYS)
                try:
                    run(a, seed, {**overrides, 'model': model, **dict(combo)}, tag)
                except Exception as exc:  # report and continue; rerun the command to retry
                    failed.append(dict(model=model, seed=seed, **dict(combo)))
                    print('FAILED', failed[-1], repr(exc), flush=True)
    if total > 1:
        print(f'{total - len(failed)} of {total} runs completed', flush=True)
    if failed:
        raise SystemExit(f'{len(failed)} run(s) failed: {failed}')


if __name__ == '__main__':
    main()
