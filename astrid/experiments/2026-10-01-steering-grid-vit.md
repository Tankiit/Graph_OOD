# E17: infonce steering grid on the CRIL cluster, ViT-S (2026-10-01)

48 runs, all completed without errors. On the ViT, measuring the similarity at the penultimate feature is
the strongest setting almost everywhere. FashionMNIST is fully fooled at radius 0.5–1, and for the first
time a setting gets steered OOD past Mahalanobis (9.9%). On MNIST the ID side always flips but the OOD side
only partly.

## What ran

- **Code:** not yet committed. Working tree after tag `e16-grid` (`b2670ec`) plus: `make_grid.py --arch vit`,
  `collect_grid.py` layer names, `grid_best_settings.py --arch`, `bench_grid.sh` job-list argument; the steering,
  loss, detector and evaluation code is unchanged since `e16-grid`.
- **Grid** (`configs/steer/grid_vit_jobs.txt`, `scripts/make_grid.py --arch vit`): `vit_small_ft_mnist` and
  `vit_small_ft_fmnist` × steered block `blocks.3` / `blocks.6` / `blocks.9` × similarity at the same block / the
  next one (`blocks.6`, `blocks.9`) / the penultimate feature (for `blocks.9`, next = penultimate, run once) ×
  radius 0.25 / 0.5 / 1.0 = 48 runs.
- **Fixed settings:** as E16 (`configs/steer/grid_base.toml`: infonce τ 0.1, K 128, 1000 steps, lr 0.01, batch 64 + 64,
  fp16, OOD 300K + STL-10, five detectors, full evaluation with 64 learned vectors), plus `vec_chunk = 16`.

## Machine and jobs

- Partition `quad_rtx_8000`, one Quadro RTX 8000 (46 GB) per job, 8 CPUs, 64 GB; torch 2.14.0+cu130 in `~/actdist`.
- Benchmark job 13145485 (`bash scripts/bench_grid.sh configs/steer/bench_vit_jobs.txt`, blocks.3, 20 steps):
  3.3 s/step, 29.7 GB peak, evaluation ~35 min, 36 min 53 s in total.
- Grid, 7 jobs (13145486–13145492), verbatim:
  `cluster slurm launch <deploy copy> slurm --remote-dir /home/cril/klipfel/actdist --partition quad_rtx_8000 --gpus 1 --cpus 8 --mem 64G --time 12:00:00 --job-name steer-gridvit-<i> -- env GRID_JOBS=configs/steer/grid_vit_jobs.txt bash scripts/run_grid_shard.sh <i> 7`
- Shard 0 would have hit the 12 h limit during its last run; it was cancelled at 06:53 (cluster clock) after 6 of
  its 7 runs, and that run (`vit_small_ft_mnist`, blocks.9, sim penultimate, r 0.25) was rerun alone in job 13145494
  (`GRID_JOBS=configs/steer/gridvit_rerun_jobs.txt`, 49 min 37 s).
- Run times: blocks.3 ~92 min; similarity at blocks.6 ~124 min (5.4 s/step instead of 3.3, cause not yet
  investigated); blocks.9 ~50–60 min (2.3 s/step: data loading at 224 px on 8 CPUs sets a floor). My initial
  estimate (blocks.9 at ~30 min) was wrong, hence the time-limit problem.
- Log check: no traceback / out-of-memory lines; the only error-like line is Slurm's notice of the cancellation.
  Logs in `outputs/logs/e17_cluster/`.

## Results

Typical vector (median over the 64 evaluated learned vectors), best setting per detector and radius:
`outputs/reports/e17_grid_vit_typical_vectors.pdf`; all tables `outputs/steering/gridvit_tables.txt`; all rows
`outputs/steering/gridvit_summary.csv`.

**Clean ViT detectors are near perfect:** on MNIST all five reach AUROC ≥ 0.998 with 0% of OOD passing; on
FashionMNIST energy 0.984 (6.9% OOD passing), MSP 0.947, max-logit 0.982, kNN 0.995, Mahalanobis 1.000.

**FashionMNIST** (OOD passing / ID not passing, best setting):

| Detector | r 0.25 | r 0.5 | r 1.0 |
| --- | --- | --- | --- |
| Energy | 31% / 100% | 99.2% / 100% | 100% / 100% |
| kNN | 66% / 100% | 99.4% / 100% | 98.9% / 100% |
| Mahalanobis | 0% / 100% | **9.9%** / 100% | 0% / 100% |

**MNIST:**

| Detector | r 0.25 | r 0.5 | r 1.0 |
| --- | --- | --- | --- |
| Energy | 67% / 71% | 49% / 100% | 40% / 100% |
| kNN | 14% / 87% | 0.1% / 100% | 8.9% / 100% |
| Mahalanobis | 0% / 100% | 0% / 100% | 0% / 100% |

- **Similarity layer:** the penultimate feature is the best choice in 29 of the 30 (model, radius, detector) cells (the exception: Mahalanobis on FashionMNIST at r 1.0, similarity at the next block);
  same-layer similarity, the setting of the first six runs, has almost no effect at r 0.25–0.5.
- **AUROC below 0.0001 in most cells comes mainly from steered ID being pushed out (100% rejected)**; the OOD
  columns say how much of the attack actually gets OOD accepted.
- **Mahalanobis:** 9.9% of steered OOD passes on FashionMNIST (blocks.6, sim penultimate, r 0.5); everywhere else 0%.

## Caveats

- Typical-vector medians; best vectors can be stronger (see E14 analysis).
- infonce never sees a detector, so every detector result is a transfer result.
- `paths.npz` files are left on the cluster in `~/actdist/outputs/steering/gridvit__*`.
