# Norm-organised detector failures along steering paths

Full-dimensional features, 8 frozen encoders (text: MPNet, MiniLM-L6, BGE-base, BGE-large on
CLINC150; vision: ResNet-18, ResNet-50, ViT-B/16, DINOv2-S on CIFAR10 vs CIFAR100/SVHN).
Distances are in ID radii (median ||reference - mean||). Mahalanobis is the float64 Ledoit-Wolf
class-conditional detector (`mahalanobis_shrinkage`); pytorch-ood's float32 version is not used
for these results (see "Numerical note").

Reproduce: `python scripts/run_pipeline.py grid --seeds 0` (see RUNBOOK.md),
then `python scripts/seed_stats.py` and the seed-0 scripts on `runs/modal/seedviews/seed0` (RUNBOOK.md §4).

## Path types

| path | definition | feature norm along path |
|---|---|---|
| straight (PC / random) | z = x + alpha v | grows (up to 3.3x at 3 radii) |
| geodesic (PC / random) | great circle through x on the sphere of radius ‖x‖, tangent = v projected off x, arc length alpha | preserved (max relative change 9e-8) |
| toward origin | z = x - alpha x/‖x‖, stopped at 0.99 min‖x‖ | shrinks |

## Matched-horizon result (Figure: `runs/modal/figures/path_curves.pdf`)

R(alpha) = fraction of steered ID probes rejected at alpha, mean over encoders; tau = 0.05.
Matched horizon = largest alpha every encoder reaches on every path: 1.03 radii (text), 0.81 (vision).
Full tables, including 3-radius results for straight and geodesic paths: `runs/modal/figures/matched_horizon.md`.

- **Norm growth (straight rays):** logit-based scores stay near tau (energy 0.05-0.10, MSP 0.06-0.25);
  distance-based scores reject 0.20-1.00.
- **Norm shrinkage (toward origin):** distance-based scores stay near or below tau (0.00-0.10);
  energy rejects 1.00 (text) / 0.54 (vision), MSP 1.00 / 0.29.
- **Norm preserved (geodesics):** mixed; neither family is uniformly blind.
- **Exception:** vision PC geodesics are hard for *every* detector at the matched horizon
  (kNN 0.10, Mahalanobis 0.01, energy 0.05, MSP 0.23), not only for energy.

## Mechanisms (`runs/modal/norm_mechanisms.txt`)

**Q1: distance detectors at the origin.** kNN score(0) is the k-th smallest reference norm (about rho).
The origin is accepted iff the calibrated kNN threshold exceeds rho. For near-orthogonal ID features,
the typical ID-ID distance is rho * sqrt(2(1 - mean cosine)), which is more than rho when the mean cosine is below 1/2.
With rho = 1 and d = 768 isotropic features: ID-ID distance ~1.41 > 1, so the origin is predicted to be accepted.

| encoder | mean ID cosine | ID-ID dist / rho | kNN threshold / rho | origin accepted (kNN / Maha) |
|---|---|---|---|---|
| MPNet | 0.09 | 1.36 | 1.12 | yes / yes |
| MiniLM | 0.09 | 1.36 | 1.11 | yes / yes |
| BGE-base | 0.48 | 1.03 | 0.83 | **no / no** |
| BGE-large | 0.48 | 1.02 | 0.82 | **no / no** |
| ResNet-18 | 0.68 | 0.81 | 0.70 | yes / yes |
| ResNet-50 | 0.40 | 1.09 | 0.95 | yes / yes |
| ViT-B/16 | 0.16 | 1.27 | 1.10 | yes / yes |
| DINOv2-S | 0.15 | 1.31 | 1.11 | yes / yes |

Superseded: the mean-cosine rule is the wrong statistic. The exact condition uses the k-th-neighbour cosine
at the calibration quantile against c* (THEORY.md, P7(c)), and it is correct for 8/8 encoders. The mean-cosine
version below holds for 6/8. The anisotropic BGE encoders (mean cosine 0.48, global mean at
0.69 rho) have ID points closer together than to the origin, so the origin is rejected. This accounts for
the text origin-path kNN/Mahalanobis rates of about 0.5 at the full horizon (2 of 4 encoders).
For vision, kNN score(0) is below rho (0.67-0.91 rho) because reference norms vary, and Mahalanobis
score(0) is 0.13-0.62 of its threshold.

**Q2: energy at the origin.** E(0) = -logsumexp(b), about -log C (-5.01 for 150 intents, -2.30 for 10 classes;
bias spread <= 2.4). It does not depend on W, and it is gauge-invariant: shifting every row of W by r
changes logits by r.z = 0 at the origin (checked numerically). The origin endpoint therefore rejects iff the
energy threshold is below -logsumexp(b). This holds for 7/8 heads; ResNet-50's threshold (0.90) is above
E(0) = -2.31, so its origin is accepted. Energy at the half-way point exceeds the starting energy for 90-100% of probes
(monotonicity along the whole path was not checked). Intermediate points on the path *do* depend on the gauge, through -(1-s) r.x.

**Q3: vision PC geodesics.** Collapse-style alignment does not explain them:
- The top-10 ID principal axes lie closer to the class-mean subspace for text (cosine 0.997) and DINOv2
  (0.96) than for the supervised ResNets (0.73-0.78).
- The DINOv2 control therefore does not separate the hypothesis. These backbones are ImageNet-trained,
  so collapse onto CIFAR class means is not expected in the first place.

What does differ: rotating a vision probe onto ±PC1 at its own norm *raises* the top logit (ResNet-18 +28/+15,
ResNet-50 +13/+10, DINOv2 +5/+8). Text rotations lower it (-0.4 to -5.7). The rise persists in the centred
gauge (sum_c w_c = 0), so it is not a gauge artefact.

Candidate mechanism, not yet tested: vision features carry a large shared mean component (global mean
0.39-0.82 rho; post-ReLU CNN features are non-negative). A geodesic moves that non-discriminative norm onto a
discriminative axis (PC1 is 79-90% between-class variance), increasing the margin. Test: geodesics on the
sphere centred at the ID mean rather than the origin.

## Proposed abstract clause

> ...and find that distance-based and logit-based detectors fail along different paths: logit-based scores
> are largely insensitive to paths that increase the feature norm, and distance-based scores to paths that decrease it.

"Largely" covers:
- MSP partially rejecting straight PC rays at 3 radii (text 0.47);
- anisotropic BGE, where distance detectors do reject the origin;
- ResNet-50, whose energy accepts the origin.

The vision PC-geodesic result affects all four detectors, so it is not an exception to the clause, but it
needs its own sentence. The gauge sweep (run 3) is still outstanding.

## Numerical note

pytorch-ood 0.3.3 `Mahalanobis` inverts the unnormalised within-class scatter + 1e-6 I in float32. For
ResNet-50, ViT-B/16 and DINOv2 full-dim features (condition number 4e9-4e10, scatter singular), its scores and
thresholds are wrong by up to ~30x. The earlier "ResNet-50 principal-axis blind spot" (15% crossed) was this
artefact (83% in float64), and the static ResNet-50 full-dim Mahalanobis AUROC is 0.771, not 0.570.
