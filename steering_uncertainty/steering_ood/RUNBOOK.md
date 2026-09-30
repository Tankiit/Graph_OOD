# Runbook: steering paths as a probe of OOD detectors

Everything needed to reproduce, extend and write up the experiments on a local machine or cluster.
Target: **AISTATS 2027** (full paper due Tue 6 Oct 2026, 23:59 AoE; 8 pages; double-blind;
AI Use Statement required).

## 1. What is here

| path | what |
|---|---|
| `steering_ood/` | Package: splits, encoders, detectors, calibrated path tracing, crossed runner (CLI: `python -m steering_ood`) |
| `steering_ood/experiment.py` | Path types: `pca`, `random` (straight rays), `sphere_pca`, `sphere_random` (geodesics), `origin` (toward origin), `radial` (radial growth) |
| `steering_ood/detectors.py` | Detectors. Use `mahalanobis_shrinkage` (float64 Ledoit-Wolf); pytorch-ood's `mahalanobis` is numerically wrong in full dimension (see NORM_PATHS.md) |
| `tests/` | 24 tests (`python -m pytest -q`) |
| `scripts/run_pipeline.py` | **Local pipeline**: `prepare`, `encode`, `grid` stages with the exact settings of the reported runs |
| `scripts/seed_stats.py` | **Paper statistics**: 3-seed tables, hierarchical CIs, paired tests, exact theory checks |
| `scripts/path_curves.py` | Figure 3 (R(alpha) curves) and matched-horizon tables |
| `scripts/radial_predictions.py`, `support_checks.py`, `norm_mechanisms.py` | Theory-vs-trace checks and mechanism analyses |
| `scripts/audit_results.py` | Result audit: ranges, NaNs, calibration, heads, variance |
| `scripts/ray_checks.py`, `aggregate.py`, `centroid_first_order.py` | Earlier analyses (first grid, PCA-64, delta-method check) |
| `paper/steering_ood.tex` | Draft (venue-neutral skeleton; switch to the AISTATS style file) |
| `paper/make_fig_polar.py` | Figure 1 (schematic) |
| `THEORY.md` | Formal statements, proof sketches and the check behind each clause |
| `NORM_PATHS.md` | Norm-controlled path results and mechanisms |
| `aistats2027.md` | Pipeline audit, draft prose, peer review, deadline notes, statistics fix |
| `requirements-pinned.txt` | Exact versions used for every reported result |
| `README.md`, `VALIDATION.md` | Original package documentation |

## 2. Setup

```bash
cd steering_uncertainty/steering_ood
python -m venv .venv && source .venv/bin/activate      # Python 3.12
python -m pip install -r requirements-pinned.txt
python -m pip install -e . --no-deps
python -m pytest -q                                      # expect 24 passed
```

Checked on macOS (CPU) on 2026-09-30: installs cleanly and all 24 tests pass.
On a CUDA machine, install the matching CUDA build of torch 2.14.0 / torchvision 0.29.0 first.
Every crossed run also saves `pip freeze` in its directory.

## 3. Run the paper grid

```bash
# 1) splits for 3 seeds (CLINC150 is fetched automatically)
python scripts/run_pipeline.py prepare --seeds 0 1 2 --download
# 2) frozen encoders -> feature caches (GPU strongly recommended for the vision models)
python scripts/run_pipeline.py encode --seeds 0 1 2 --device cuda
# 3) heads, static metrics, 6 path types x 4 detectors, with steering checks
python scripts/run_pipeline.py grid --seeds 0 1 2 --jobs 4
```

Notes:
- **Datasets.**
  - CIFAR10/CIFAR100/SVHN (test split) go to `runs/modal/data/images/`. The CIFAR-100 mirror can be very
    slow; you can instead place `cifar-100-python/`, `cifar-10-batches-py/` and `test_32x32.mat` there
    and omit `--download`. torchvision checks the MD5s.
  - Everything under `runs/` is git-ignored.
- **Seed s:**
  - re-partitions CLINC with seed 20260929+s and CIFAR with 7+s;
  - trains the head with 7+s;
  - draws references, directions and probes with 7+s.

  Seed 0 is the original split.
- **Budget.** 6 references x 6 directions x 64 probes x 101 steps (origin/radial: 1 direction, 1 sign).
  Horizon = 3 median ID radii.
- **Cost (measured on the reference runs).**
  - The full grid is 144 crossed runs totalling ~26 hours on 16 cores (~410 core-hours).
  - The slowest single cell (ResNet-50, 2048-d) takes ~1.7 h on 16 cores.
  - Encoding needs a few GPU-minutes per model and seed.
  - Parallelise with `--jobs` (each job gets cores/jobs threads), or split the grid across machines by
    `--seeds` / `--models`.
  - Start with `--budget pilot --models mpnet` (under a minute per path type) to check the setup.
- **Restarts.** Every stage is idempotent: rerunning skips finished steps and archives partial crossed runs
  as `*.failed-*`.

Output layout (the analysis scripts read `runs/modal/` by default):
`runs/modal/seeds/seed{s}/{model}/full/{head,static,crossed_<kind>}`, `runs/modal/caches/[seed{s}/]{model}.npz`.

## 4. Reproduce the paper numbers

```bash
python scripts/seed_stats.py > runs/modal/seed_stats.md           # Table 2, paired tests, 24/24 checks
# seed_stats.py also creates runs/modal/seedviews/seed{s}, the layout the seed-level scripts expect:
python scripts/path_curves.py runs/modal/seedviews/seed0           # Figure 3 + matched-horizon tables
python scripts/radial_predictions.py runs/modal/seedviews/seed0    # Prop. 3 checks, real-OOD norm link
python scripts/support_checks.py runs/modal/seedviews/seed0        # c_k vs c*, ID support of geodesics
python scripts/norm_mechanisms.py runs/modal/seedviews/seed0       # origin geometry, E(0), axis alignment
cd paper && python make_fig_polar.py && cd ..                      # Figure 1
cd paper && pdflatex steering_ood.tex && pdflatex steering_ood.tex # or upload paper/ to Overleaf
```

Copy `runs/modal/seedviews/seed0/figures/path_curves.pdf` to `paper/figures/` after regenerating Figure 3.

| paper element | produced by |
|---|---|
| Table 2 (3-seed matched horizon, CIs) | `seed_stats.py` (A) |
| "Statistical protocol" paired tests | `seed_stats.py` (B) |
| Static AUROC / FPR@95 ± SD | `seed_stats.py` (C) |
| "24/24 (seed, encoder) cells" | `seed_stats.py` (D) |
| Appendix per-encoder table | `seed_stats.py` (A, per-encoder lines) |
| Figure 1 | `paper/make_fig_polar.py` |
| Figure 3 | `path_curves.py` (seed 0) |
| Angular-path ID-support paragraph | `support_checks.py` |

The reference results behind the current draft are not in git; ask the first author for the archive of
`runs/modal/` if you want to compare against them rather than regenerate.

## 5. Adding a detector or path (most likely next steps)

- **Detector:**
  - Add a class with `fit(x, y)` / `score(x)` (higher = more anomalous) to `steering_ood/detectors.py`
    and register it in `make_detector`.
  - Add it to `DETECTORS` in `scripts/run_pipeline.py` and to `DETS` in `scripts/seed_stats.py`.
  - Wanted baselines: **kNN on L2-normalised features** and **ViM**.
- **Path:**
  - Add it to `CURVED_KINDS` and `path_points` in `steering_ood/experiment.py`.
  - Add the CLI choice in `steering_ood/cli.py` and a geometry test in `tests/test_contract.py`.
  - Add it to `KINDS` (run_pipeline) and `PATHS` (seed_stats).
  - The missing cell of the polar design is shrink-and-rotate (a geodesic on a shrinking sphere).

## 6. Open items (see `aistats2027.md` §3 and §4)

1. Baselines: normalised kNN and ViM on all paths (3 seeds).
2. Proofs: crossing stability (T1) and the ANOVA proposition are `\hole`s in the tex. Prove them or cut them.
   Prop. 3, the cone theorem and Lemma 1 have sketches in `THEORY.md`.
3. Gauge sweep (vary the common row shift r of the head; the origin *decision* moves with the threshold).
4. Related work: Hein et al. (CVPR 2019), energy score (Liu et al. 2020), Mahalanobis (Lee et al. 2018),
   deep kNN (Sun et al. 2022), ViM (Wang et al. 2022), ReAct (Sun et al. 2021). Verify all citations.
5. Switch to the AISTATS 2027 style file, anonymise, add the AI Use Statement and reproducibility
   checklist, fit to 8 pages.
6. Figure 3 is seed 0 only (the caption says so). Optionally add seed bands.
