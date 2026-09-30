# Formal statements behind P3-P8, with checks against the runs

Notation. Encoder f: X -> R^d, feature z = f(x). A score S: R^d -> R is calibrated at threshold t
(order statistic of calibration scores), and the decision is delta(z) = 1[S(z) > t]. Polar form: z = rho u,
with rho = ||z|| and ||u|| = 1. The linear head has logits Wz + b, with rows w_c and C classes;
a = Wz are the bias-free logits. Energy is E(z) = -lse(Wz + b); MSP is M(z) = -max_c softmax(Wz + b)_c.
kNN score = distance to the k-th nearest reference. Mahalanobis = min_c (z - mu_c)^T P (z - mu_c),
with P positive definite (float64 Ledoit-Wolf).
Checks: `scripts/radial_predictions.py`, `scripts/norm_mechanisms.py`, `scripts/ray_checks.py`,
`scripts/path_curves.py`; outputs are in `runs/modal/*.txt` and `runs/modal/figures/`.

## P3 Why representation space

**Lemma 1 (reduction).** Let x(beta), beta in [0, B], be an input-space shift, and let gamma be a feature
path such that f(x(beta)) = gamma(tau(beta)) for a map tau into gamma's domain. Then
delta(f(x(beta))) = delta(gamma(tau(beta))) for every beta. In particular, if delta o gamma = 0 on [0, H],
every input shift that realises gamma within [0, H] is accepted throughout.

*Proof.* The detector sees x only through z = f(x), so the decision is delta o f, and the two
compositions agree pointwise. The converse fails: input shifts need not realise every feature path.

**Definition (epsilon-realisation).** An input shift epsilon-realises gamma if there is a monotone tau
with sup_beta ||f(x(beta)) - gamma(tau(beta))|| <= epsilon.

**Corollary 1.** If S is L-Lipschitz, the decisions along x(beta) and gamma(tau(beta)) agree at every beta
where |S(gamma(tau(beta))) - t| > L epsilon. Lipschitz constants for our scores:
- kNN: L = 1 (a k-th order statistic of 1-Lipschitz distances).
- Energy: ||grad E|| = ||W^T p|| <= max_c ||w_c||.
- Mahalanobis: only locally Lipschitz (it grows quadratically).

**Answer to "what counts as realisable".** Endpoint proximity is not enough. Decisions along a path are
not monotone:
- energy's rejection set on a ray is a single interval that can close before the endpoint (T-cone below);
- MSP re-accepts 43-45% of the probes it flagged (straight rays, `runs/modal/main`).

Realisation must control the whole path (the epsilon-tube), or at least the segment up to the first
crossing, if only alpha* is used. The tube width needed is set by the score margin |S - t| / L along the path.

**Real OOD realises norm shrinkage on one encoder.** Median real-OOD norm / ID norm:
- text OOS: exactly 1.00 (the encoders normalise);
- CIFAR100: 0.96-1.01;
- SVHN: ResNet-18 0.70, DINOv2 0.93, ResNet-50/ViT 1.04-1.05.

The strongly shrunk case (ResNet-18, SVHN) has the largest energy-over-distance AUROC gap (0.905 vs 0.224).
Across the 12 (encoder, OOD set) pairs there is no monotone relation (Spearman -0.18, p = 0.58): the link
rests on one case.

## P4 Why shared, declared paths

For shared v, the per-detector crossings alpha*_j(v) are paired: the same probe, path and grid, so the
difference alpha*_i(v) - alpha*_j(v) is a within-path comparison. The worst case min_v alpha*_j(v) uses a
different v for each j and is therefore not a paired statistic.

**Polar design.** Along z + alpha v:
rho(alpha)^2 = rho^2 + 2 alpha rho (u.v) + alpha^2,
and u rotates unless v is parallel to u. The implemented families cover:

| | angular: no | angular: yes |
|---|---|---|
| radial + | `radial` (z + alpha u) | straight rays with u.v >= 0 |
| radial 0 | (identity) | `sphere_pca`, `sphere_random` (arc length) |
| radial - | `origin` (z - alpha u) | straight rays with u.v < 0, only until alpha = -rho(u.v) |

The shrink-and-rotate cell is covered only transiently. Straight rays with u.v < 0 shrink first and then
grow, and for random v in high dimension u.v is near 0. A geodesic on a shrinking sphere would fill that cell;
it is not implemented.

## P5 Why calibrated comparison

**Proposition 1 (invariance).** For strictly increasing g, the order-statistic threshold of g o S is g(t),
so g o S and S make identical decisions and give identical alpha*.

**Proposition 2 (gauge; the limit of P1).** Replacing w_c by w_c + r for all c leaves softmax, predictions
and MSP unchanged, but changes energy:
E_r(z) = E(z) - r.z.
Since r.z depends on the input, E_r is not a monotone transform of E, and calibrated energy decisions
can change while every prediction is identical.

At the origin E_r(0) = E(0): the energy *value* there is gauge-invariant (checked numerically,
`norm_mechanisms.txt` Q2). The calibrated *decision* is not. The threshold is recalibrated on
E_r(z_cal) = E(z_cal) - r.z_cal, so the origin condition -lse(b) > t_r can flip with r.
In the schematic (`paper/make_fig_polar.py`), E(0) = -1.10 is rejected under t = -2.40 but accepted under
t_r = 1.75. Our 8/8 origin checks use each trained head's own calibrated threshold, so they are unaffected.
Our heads carry a common row component of 0.10-0.23 of the mean row norm.

## P7 The complementarity result

**Proposition 3 (radial behaviour).** Fix a probe z and let s -> s z.

(a) *Energy.* E(s) = -lse(s a + b) is concave, with
dE/ds = -E_{p(s)}[a] and d/ds E_{p(s)}[a] = Var_{p(s)}(a) >= 0.
Hence:
- E(0) = -lse(b), and E(s) -> -inf iff max_c a_c > 0.
- If dE/ds(1) <= 0 (the softmax-weighted bias-free logit is >= 0), then E(s) <= E(1) for all s >= 1:
  an accepted probe is never rejected on radial growth.
- Otherwise the maximum is at the unique root s0 of E_{p(s)}[a] = 0.
- On shrinkage, an accepted probe is rejected near the origin iff -lse(b) > t. There is then exactly one
  crossing, since E(1) <= t < E(0) and E is concave.
- A sufficient condition for max_c a_c > 0: the top logit exceeds max_c b_c.

(b) *MSP.* M(0) = -max_c softmax(b)_c, about -1/C. As s -> inf, M -> -1 if argmax_c a_c is unique.

(c) *kNN.* S(0) = r_(k), the k-th smallest reference norm. The origin is accepted iff r_(k) <= t.
The score is 1-Lipschitz, so S(s z) >= s||z|| - max_i ||r_i||, and on growth alpha* <= t + max_i ||r_i|| - ||z||.
If references lie on a sphere of radius rho and the calibration-quantile point's k-th neighbour has cosine c_k,
then t = rho sqrt(2(1 - c_k)), and the origin is accepted iff c_k < 1/2.

(d) *Mahalanobis.* Each q_c(s) is a convex quadratic in s with positive leading coefficient z^T P z.
So the score tends to +inf on growth, and at the origin it equals min_c mu_c^T P mu_c.
For anisotropic embeddings the global mean lies along low-variance directions, which makes this large.

*Proof sketch.* (a) lse of an affine function is convex. Its first two derivatives in s are the mean and
variance of a under p(s). The limits follow from lse(s a + b) = s max_c a_c + O(1). (b)-(d) are direct.

**Evidence** (8 full-dimensional encoders; `radial_predictions.txt`, `path_curves.pdf`).
Replicated over 3 independent seeds (split + head + draws): every exact check passes in 24/24 (seed, encoder)
cells (`runs/modal/seed_stats.md`, part D). The kNN origin rule is tested on 7454 (fit, probe) pairs where the
1-Lipschitz margin makes the observation decisive.

Shrinkage limits, predicted vs observed at the end of the origin path:
- Energy: 8/8, including ResNet-50 accepting because its threshold (0.90) exceeds -lse(b) = -2.31.
- MSP: 8/8.
- kNN: 8/8.
- Mahalanobis: 8/8.
- Vision end-of-path rates are partial for three encoders (energy 0.38-1.00, MSP 0.20-1.00; DINOv2 reaches 1.00)
  because the path stops at 0.99 x the smallest probe norm. Every probe that reaches s < 0.1 matches the limit.

Radial growth:
- Energy: the slope condition holds for 89-100% of probes. The closed-form prediction matches the traces
  for 100% of probes, and growth causes zero new rejections in every encoder.
- The only energy "rejections" (3-9%) are probes already rejected at alpha = 0.
- ResNet-50: 11% of probes have max_c a_c < 0, so energy eventually rejects them, beyond the 3-radius horizon.
- kNN and Mahalanobis reject 0.96-0.99 by the matched horizon, and kNN never exceeds its bound.

Exact neighbour-cosine form of (c): t = ||z_q - r_q||, where z_q is the calibration point setting the
threshold and r_q its k-th neighbour, so the origin is accepted iff c_k <= c*, with
c* = (||z_q||^2 + ||r_q||^2 - r_(k)^2) / (2 ||z_q|| ||r_q||).
This is correct on every reference fit for 8/8 encoders (`support_checks.txt`).
- Text, unit norm, c* = 1/2: MPNet/MiniLM c_k = 0.37/0.38 (accepted); BGE c_k = 0.65/0.66 (rejected).
- Vision: ResNet-18 c_k = 0.79 < c* = 0.82 and ResNet-50 c_k = 0.62 < 0.82, both accepted. Their neighbours
  are strongly aligned, but small-norm references raise c*. So "c_k < 1/2" is the unit-norm special case
  only, and the mean pairwise cosine is not the right statistic at all.

Matched-horizon rejection (1.03 radii text, 0.81 vision; kNN / Mahalanobis / energy / MSP):

| path | text | vision |
|---|---|---|
| radial growth | 0.99 / 0.99 / 0.00 / 0.02 | 0.97 / 0.96 / 0.02 / 0.02 |
| toward origin | 0.10 / 0.09 / 1.00 / 1.00 | 0.00 / 0.00 / 0.54 / 0.29 |

**Angular paths.** With the norm fixed, only rotation relative to the rows of W changes the logits.
The collapse account of the vision geodesic exception is not supported:
- Principal axes are closer to the class-mean span in text (0.997) than in supervised ResNets (0.73-0.78).
- Rotating a vision probe onto ±PC1 raises the top logit even in the centred gauge (ResNet-18 +22/+12).

ID-support check (`support_checks.py`; ID-test split, independent of all four detectors):
- At the matched horizon (0.81 radii), vision PC-geodesic endpoints lie a median of 2.2-2.6 ID sd along the
  axis, against an ID-test 99.5% edge of 1.9-2.1 sd, at 0.42-0.73 of the Mahalanobis threshold. They are
  close to ID-preserving there. Text endpoints reach 3.9-4.2 sd.
- Among path points more than 3 sd out along the axis (full horizon), vision flag rates are: energy 0.00,
  Mahalanobis 0.00-0.30, MSP 0.08-0.49, kNN 0.21-0.85.
- So the matched-horizon "exception" is mostly an ID-preserving path. Beyond it, the paths leave ID and are
  missed by energy and mostly by Mahalanobis, but not by kNN.

Conjecture (not checked): a single-axis excursion of kappa sd raises a whitened d^2 by about kappa^2,
while the calibrated slack grows like sqrt(D). Mahalanobis would then need kappa of order D^(1/4)
(about 7-8 sd here). MPNet/MiniLM show the same miss (1% flagged beyond 3 sd); BGE (0.68) does not fit
cleanly. The mean-component mechanism (`NORM_PATHS.md`) is untested.

**Statement for the paper.**

> For a linear-head energy score and Euclidean distance scores, radial growth drives energy toward
> acceptance and distance scores toward rejection. Radial shrinkage drives distance scores toward their
> value at the origin (the k-th smallest reference norm, accepted whenever it is below the calibrated
> threshold) and energy toward the bias-determined constant -lse(b).

Every clause is checked on 8/8 encoders above. The acceptance condition at the origin is not universal:
the anisotropic BGE encoders reject the origin under both distance scores.

## P8 Contributions

```latex
% P8 Contributions.
% (1) Instrument + validity: calibrated first rejection on shared paths;
%     invariance (Prop. \ref{prop:invariance}), crossing stability
%     (Thm. \ref{thm:stability}), error decomposition (Prop. \ref{prop:anova});
%     reduction to input shifts (Lemma \ref{lem:reduction}) and its
%     \epsilon-realisation corollary for Lipschitz scores.
% (2) Polar analysis of paths: radial vs angular components; complementarity
%     of distance- and logit-based detectors (Prop. \ref{prop:radial}),
%     with the origin limits r_{(k)} (kNN), -\operatorname{lse}(b) (energy),
%     -\max_c \operatorname{softmax}(b)_c (MSP), verified on 8/8 encoders.
% (3) Energy-specific structure: single flagged interval, cone condition for
%     eventual rejection (Thm. \ref{thm:cone}), dependence on a
%     prediction-invariant component of the head (Prop. \ref{prop:gauge}).
% (4) Evidence on four text and four vision encoders, including
%     norm-controlled paths (geodesic, radial growth, radial shrinkage) and the
%     real-OOD norm link, which is currently a single case
%     (ResNet-18/SVHN, norm ratio 0.70).
```

Prop. 1 (invariance), T1 (stability) and the ANOVA proposition are referenced from the existing plan;
this file does not re-derive them. Still outstanding: the gauge sweep (run 3) and the shrink-and-rotate cell.
