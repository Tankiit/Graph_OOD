# Multi-seed statistics (3 seeds x 4 encoders per modality)

Seeds vary split, head and bootstrap/direction draws together; seed 0 is the original split.

## (A) Matched-horizon rejection: mean [95% hierarchical CI] (seed SD)

### text: alpha = 1.04 ID radii (common to all seeds, encoders, paths)

| path | kNN | Maha | Energy | MSP |
|---|---|---|---|---|
| radial growth | 1.00 [0.99, 1.00] (0.006) | 0.99 [0.98, 1.00] (0.006) | 0.01 [0.00, 0.01] (0.006) | 0.02 [0.01, 0.03] (0.004) |
| toward origin | 0.11 [0.00, 0.22] (0.012) | 0.13 [0.00, 0.30] (0.041) | 1.00 [1.00, 1.00] (0.000) | 1.00 [1.00, 1.00] (0.000) |
| straight, PC | 0.89 [0.88, 0.91] (0.018) | 0.30 [0.23, 0.38] (0.040) | 0.09 [0.07, 0.12] (0.019) | 0.25 [0.22, 0.28] (0.033) |
| straight, random | 0.96 [0.94, 0.98] (0.019) | 1.00 [1.00, 1.00] (0.000) | 0.07 [0.06, 0.07] (0.011) | 0.06 [0.05, 0.08] (0.007) |
| geodesic, PC | 0.39 [0.23, 0.54] (0.024) | 0.03 [0.00, 0.06] (0.002) | 0.28 [0.15, 0.46] (0.054) | 0.64 [0.52, 0.76] (0.030) |
| geodesic, random | 0.74 [0.58, 0.89] (0.020) | 1.00 [1.00, 1.00] (0.000) | 0.51 [0.21, 0.82] (0.044) | 0.23 [0.12, 0.33] (0.022) |

Per-encoder means over seeds (text, toward origin / radial growth):

- toward origin, kNN: mpnet 0.00±0.00, minilm 0.00±0.00, bge_base 0.18±0.02, bge_large 0.25±0.03
- toward origin, Maha: mpnet 0.00±0.00, minilm 0.00±0.00, bge_base 0.14±0.11, bge_large 0.40±0.07
- toward origin, Energy: mpnet 1.00±0.00, minilm 1.00±0.00, bge_base 1.00±0.00, bge_large 1.00±0.00
- toward origin, MSP: mpnet 1.00±0.00, minilm 1.00±0.00, bge_base 1.00±0.00, bge_large 1.00±0.00
- radial growth, kNN: mpnet 0.99±0.01, minilm 1.00±0.01, bge_base 1.00±0.00, bge_large 1.00±0.00
- radial growth, Maha: mpnet 0.98±0.02, minilm 1.00±0.00, bge_base 1.00±0.00, bge_large 1.00±0.00
- radial growth, Energy: mpnet 0.01±0.01, minilm 0.00±0.00, bge_base 0.01±0.01, bge_large 0.01±0.01
- radial growth, MSP: mpnet 0.03±0.00, minilm 0.01±0.01, bge_base 0.01±0.01, bge_large 0.03±0.01

### vision: alpha = 0.75 ID radii (common to all seeds, encoders, paths)

| path | kNN | Maha | Energy | MSP |
|---|---|---|---|---|
| radial growth | 0.95 [0.92, 0.98] (0.008) | 0.94 [0.89, 0.98] (0.002) | 0.03 [0.01, 0.06] (0.012) | 0.02 [0.01, 0.03] (0.005) |
| toward origin | 0.00 [0.00, 0.00] (0.000) | 0.00 [0.00, 0.00] (0.000) | 0.52 [0.09, 0.94] (0.020) | 0.19 [0.13, 0.25] (0.017) |
| straight, PC | 0.44 [0.28, 0.68] (0.028) | 0.22 [0.05, 0.51] (0.034) | 0.06 [0.03, 0.09] (0.023) | 0.19 [0.11, 0.27] (0.011) |
| straight, random | 0.67 [0.57, 0.82] (0.022) | 1.00 [1.00, 1.00] (0.000) | 0.05 [0.03, 0.06] (0.006) | 0.05 [0.04, 0.07] (0.010) |
| geodesic, PC | 0.10 [0.06, 0.16] (0.014) | 0.02 [0.01, 0.03] (0.010) | 0.06 [0.03, 0.09] (0.023) | 0.21 [0.12, 0.32] (0.010) |
| geodesic, random | 0.31 [0.25, 0.37] (0.014) | 0.94 [0.85, 1.00] (0.010) | 0.08 [0.03, 0.13] (0.008) | 0.07 [0.05, 0.09] (0.009) |

Per-encoder means over seeds (vision, toward origin / radial growth):

- toward origin, kNN: resnet18 0.00±0.00, resnet50 0.00±0.00, vit_b16 0.00±0.00, dinov2_s 0.00±0.00
- toward origin, Maha: resnet18 0.00±0.00, resnet50 0.00±0.00, vit_b16 0.00±0.00, dinov2_s 0.00±0.00
- toward origin, Energy: resnet18 0.17±0.03, resnet50 0.02±0.01, vit_b16 0.88±0.08, dinov2_s 0.99±0.01
- toward origin, MSP: resnet18 0.12±0.02, resnet50 0.16±0.05, vit_b16 0.27±0.04, dinov2_s 0.20±0.05
- radial growth, kNN: resnet18 0.92±0.06, resnet50 0.94±0.04, vit_b16 0.93±0.00, dinov2_s 1.00±0.00
- radial growth, Maha: resnet18 0.93±0.03, resnet50 0.86±0.03, vit_b16 0.96±0.01, dinov2_s 1.00±0.00
- radial growth, Energy: resnet18 0.03±0.01, resnet50 0.07±0.03, vit_b16 0.01±0.01, dinov2_s 0.01±0.02
- radial growth, MSP: resnet18 0.02±0.02, resnet50 0.03±0.02, vit_b16 0.02±0.02, dinov2_s 0.02±0.02

## (B) Paired comparisons at the matched horizon

Per (seed, encoder) cell: paired bootstrap over probes, one-sided p for the stated direction.

| claim | modality | mean diff [95% hier. CI] | cells p<0.05 / 12 | cells with diff <= 0 |
|---|---|---|---|---|
| growth: kNN > energy | text | +0.99 [+0.98, +1.00] | 12 | 0 |
| growth: kNN > energy | vision | +0.92 [+0.87, +0.97] | 12 | 0 |
| growth: Maha > energy | text | +0.99 [+0.97, +1.00] | 12 | 0 |
| growth: Maha > energy | vision | +0.91 [+0.84, +0.98] | 12 | 0 |
| growth: kNN > MSP | text | +0.98 [+0.96, +0.99] | 12 | 0 |
| growth: kNN > MSP | vision | +0.93 [+0.90, +0.96] | 12 | 0 |
| origin: energy > kNN | text | +0.89 [+0.78, +1.00] | 12 | 0 |
| origin: energy > kNN | vision | +0.52 [+0.08, +0.94] | 9 | 0 |
| origin: energy > Maha | text | +0.87 [+0.69, +1.00] | 12 | 0 |
| origin: energy > Maha | vision | +0.52 [+0.09, +0.95] | 9 | 0 |
| origin: MSP > kNN | text | +0.89 [+0.78, +1.00] | 12 | 0 |
| origin: MSP > kNN | vision | +0.19 [+0.13, +0.25] | 12 | 0 |
| straight PC: kNN > energy | text | +0.80 [+0.77, +0.82] | 12 | 0 |
| straight PC: kNN > energy | vision | +0.39 [+0.22, +0.64] | 12 | 0 |
| geodesic PC: MSP > energy | text | +0.36 [+0.25, +0.45] | 12 | 0 |
| geodesic PC: MSP > energy | vision | +0.15 [+0.07, +0.26] | 12 | 0 |

## (C) Static detection, mean ± SD over 3 seeds (float64 Mahalanobis)

| encoder | kNN AUROC | Maha AUROC | Energy AUROC | MSP AUROC | kNN FPR95 | Maha FPR95 | Energy FPR95 | MSP FPR95 |
|---|---|---|---|---|---|---|---|---|
| mpnet | 0.948±0.001 | 0.948±0.001 | 0.976±0.000 | 0.963±0.001 | 0.243±0.006 | 0.221±0.007 | 0.092±0.003 | 0.159±0.007 |
| minilm | 0.944±0.001 | 0.946±0.000 | 0.965±0.000 | 0.953±0.001 | 0.248±0.010 | 0.198±0.005 | 0.144±0.002 | 0.200±0.007 |
| bge_base | 0.940±0.001 | 0.950±0.001 | 0.971±0.001 | 0.961±0.001 | 0.251±0.008 | 0.221±0.012 | 0.112±0.004 | 0.166±0.006 |
| bge_large | 0.949±0.001 | 0.953±0.001 | 0.976±0.001 | 0.966±0.001 | 0.215±0.003 | 0.206±0.004 | 0.105±0.004 | 0.145±0.001 |
| resnet18 | 0.414±0.007 | 0.413±0.000 | 0.844±0.014 | 0.806±0.009 | 0.957±0.012 | 0.993±0.000 | 0.491±0.008 | 0.555±0.009 |
| resnet50 | 0.829±0.001 | 0.794±0.004 | 0.849±0.005 | 0.871±0.002 | 0.586±0.004 | 0.638±0.003 | 0.398±0.011 | 0.403±0.010 |
| vit_b16 | 0.889±0.002 | 0.856±0.005 | 0.949±0.003 | 0.905±0.009 | 0.411±0.007 | 0.508±0.012 | 0.205±0.006 | 0.274±0.019 |
| dinov2_s | 0.892±0.002 | 0.957±0.002 | 0.974±0.003 | 0.950±0.004 | 0.397±0.010 | 0.249±0.010 | 0.118±0.002 | 0.160±0.006 |

## (D) Exact theory checks, per (seed, encoder) cell

n = 24 cells (3 seeds x 8 encoders)

| check | cells passing |
|---|---|
| energy origin limit: reject iff -lse(b) > t (probes reaching s < 0.1) | 24/24 |
| kNN origin: reject iff r_(k) > t, on (fit, probe) pairs with |r_(k) - t| > ||z_end|| (1-Lipschitz) | 24/24 |
| Mahalanobis origin: reject iff min mu^T P mu > t | 24/24 |
| radial growth: energy closed form = traces, no new rejections | 24/24 |
| radial growth: kNN crossing within 1-Lipschitz bound | 24/24 |
| c_k <= c* is algebraically identical to r_(k) <= t (every fit) | 24/24 |
| kNN origin, unit-norm form c_k < 1/2 (expected to fail off the sphere) | 18/24 |

kNN: 7454 decidable (fit, probe) pairs in 24 cells.
Origin checks count only (seed, encoder) cells where some probe reaches s < 0.1; paths stop at
0.99 x the smallest probe norm, so other probes end away from the origin.

