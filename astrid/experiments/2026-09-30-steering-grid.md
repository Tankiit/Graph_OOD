# E16: infonce steering grid on the CRIL cluster (2026-09-30)

48 runs, all completed without errors. Steering at `layer3.0` with the similarity measured downstream
(next block or penultimate feature) is the strongest setting for every detector except Mahalanobis,
which still accepts almost no steered OOD image.

## What ran

- **Code:** commit `b2670ec` on `main`, tag `e16-grid`. The tree was deployed before that commit; the
  deployed code differs from the tag only by two additions the grid does not use
  (`eval_steer.py --eval-vectors/--out`, `scripts/vector_report.py`).
- **Grid:** `resnet18_scratch_mnist` and `resnet18_scratch_fmnist` × steered layer `layer2.0` / `layer3.0` /
  `layer4.0` × similarity layer same / next block / penultimate (for `layer4.0`, next = penultimate, run
  once) × radius 0.25 / 0.5 / 1.0 (× the median ID activation norm at the steered layer) = 48 runs.
- **Fixed settings** (`configs/steer/grid_base.toml`): loss `infonce` (τ 0.1), K = 128 vectors per swarm,
  both swarms, 1000 steps, Adam lr 0.01, batch 64 ID + 64 OOD, `amp = fp16`, OOD = 300K Random Images +
  STL-10, detectors energy / MSP / max-logit / kNN (k 50, 10 000 reference images) / Mahalanobis, t_D at
  95% of clean ID accepted, 1000 + 1000 held-out probes, 64 learned vectors evaluated per swarm (median
  reported).
- **Job list:** `configs/steer/grid_jobs.txt` (written by `scripts/make_grid.py`).

## Machine and jobs

- Partition `quad_rtx_8000`, 1 Quadro RTX 8000 (46 GB, driver 595.71.05) per job, 8 CPUs, 64 GB.
- Environment: torch 2.14.0+cu130, deployed with `cluster deploy` to `~/actdist` from a slim copy
  (code, configs, scripts); data staged with rsync (MNIST, FashionMNIST, 300K, STL-10, 2 checkpoints).
- Benchmark job 13145477 (`bash scripts/bench_grid.sh`): layer2.0, 20 steps + full evaluation, 5 min 13 s;
  0.35 s/step, 13 GB peak. A first benchmark (13145476) failed in 1 s because of a relative
  `--remote-dir` (fixed in the `cluster` tool since, commit `6ad60c3`).
- Grid launch, 7 jobs, one per GPU, verbatim:
  `cluster slurm launch <deploy copy> slurm --remote-dir /home/cril/klipfel/actdist --partition quad_rtx_8000 --gpus 1 --cpus 8 --mem 64G --time 03:00:00 --job-name steer-grid-<i> -- bash scripts/run_grid_shard.sh <i> 7`

| Job | Node | Elapsed | Runs |
| --- | --- | --- | --- |
| 13145478 (shard 0) | nodeGcal1 | 1:07:08 | 7 |
| 13145479 (shard 1) | nodeGcal1 | 1:01:47 | 7 |
| 13145480 (shard 2) | nodeGcal1 | 1:02:08 | 7 |
| 13145481 (shard 3) | nodeGcal2 | 1:04:22 | 7 |
| 13145482 (shard 4) | nodeGcal2 | 0:55:36 | 7 |
| 13145483 (shard 5) | nodeGcal2 | 0:57:17 | 7 |
| 13145484 (shard 6) | nodeGcal2 | 0:55:31 | 6 |

Per run: about 13 min at `layer2.0`, 8 min at `layer3.0`, 4–5 min at `layer4.0` (training + evaluation).
Log check: every shard ends with `FINISHED (failed=0)`; a grep for traceback / error / out of memory /
nan / killed finds nothing. Logs are copied to `outputs/logs/e16_cluster/`.

## Results

Full tables: `outputs/steering/grid_tables.txt`; all rows: `outputs/steering/grid_summary.csv`
(`scripts/collect_grid.py`). Values below are medians over the learned vectors. "Both steered" = ID moved
by the `id2ood` swarm and OOD by the `ood2id` swarm, scored together (0.5 = chance, < 0.5 = inverted).

**Clean detection (AUROC):**

| Model | Energy | MSP | Max-logit | kNN | Mahalanobis |
| --- | --- | --- | --- | --- | --- |
| MNIST | 0.777 | 0.945 | 0.780 | 0.996 | 1.000 |
| FashionMNIST | 0.936 | 0.875 | 0.931 | 0.969 | 0.996 |

**Best settings (both-steered AUROC; ID rejected / OOD accepted):**

| Model · setting | kNN | Energy | MSP | Mahalanobis |
| --- | --- | --- | --- | --- |
| MNIST · layer3.0, sim next, r 0.5 | 0.027 (97% / 68%) | 0.076 (55% / 99.7%) | 0.050 | 0.584 (100% / 0%) |
| MNIST · layer3.0, sim penultimate, r 0.5 | 0.029 (97% / 65%) | 0.069 (51% / 99.8%) | 0.047 | 0.598 (100% / 0%) |
| FMNIST · layer3.0, sim penultimate, r 0.5 | 0.010 (94% / 98%) | 0.151 (62% / 96%) | 0.102 | 0.025 (100% / 0%) |
| FMNIST · layer3.0, sim penultimate, r 1 | 0.00004 (100% / 99.3%) | 0.056 (17% / 99.9%) | 0.022 | 0.001 (100% / 0%) |

**Patterns across the grid:**

- **Layer:** `layer3.0` gives the strongest attacks. `layer2.0` needs radius 1 to have much effect;
  `layer4.0` works well only with the similarity at the penultimate feature.
- **Similarity layer:** measuring the similarity at the same layer as the steering is the weakest choice
  in every block; the next block and the penultimate feature are much stronger and similar to each other.
- **Radius:** larger radii strengthen the attack on kNN and Mahalanobis, but against energy the ID side can
  overshoot at radius 1 (MNIST layer3.0, sim next or penultimate: ID rejected falls from 51–55% at r 0.5 to 10–14% at r 1).
- **Mahalanobis** accepts 0% of steered OOD on MNIST in all 24 runs and at most 2.4% on FashionMNIST. Its
  low both-steered AUROC comes only from steered ID being pushed out (100% rejected). On FashionMNIST the
  median Mahalanobis score of steered OOD (1 394–1 691 at radius 0.25 in 7 of the 8 settings) is close to its threshold (973),
  much closer than on MNIST (6 000–29 000 against 1 475).
- **FashionMNIST is easier to fool than MNIST for kNN** (both sides flip at radius 1), even though its clean
  energy detector is much stronger (0.936 against 0.777).

## Caveats

- Medians over 64 of the 128 vectors per swarm: the E14 per-vector analysis showed the best vectors can be
  much stronger than the median, so these numbers understate the best achievable attack.
- The vectors are learned with a loss that never sees a detector (`infonce`); every detector result is a
  transfer result.
- `paths.npz` files (about 200 MB per run) were left on the cluster in `~/actdist/outputs/steering/`.
