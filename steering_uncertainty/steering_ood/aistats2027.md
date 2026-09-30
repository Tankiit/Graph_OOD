# AISTATS 2027: pipeline analysis of the steering / OOD results

Generated 2026-09-30 by the academic pipeline: Phase 1 code-validator, Phase 2 academic-writer
(UAI/AISTATS register), Phase 3 paper-reviewer.

## 0. Deadline (act first)

Official CFP: https://virtual.aistats.org/Conferences/2027/CallForPapers
- **Abstract registration: Tue 29 Sep 2026, 23:59 AoE = Wed 30 Sep, 13:59 CEST.** It is mandatory.
- The author list submitted at the abstract deadline is final. Every author needs an up-to-date OpenReview profile.
- Major changes to the title or abstract after this deadline are flagged for desk rejection. Register an
  abstract you can stand behind.
- Full paper, including supplementary material: Tue 6 Oct 2026, 23:59 AoE.
- Main text: 8 pages. References, the AI Use Statement, the reproducibility checklist and appendices do not count.
- The review is double-blind. **A missing AI Use Statement means desk rejection.**

### Title and abstract to register (every claim below is backed by a verified result in §1)

**Title:** Calibrated Steering Paths Reveal Complementary Failure Modes of Out-of-Distribution Detectors

**Abstract:**
Out-of-distribution (OOD) detectors are usually compared with static metrics such as AUROC. These metrics
average over an unspecified mixture of distribution shifts and do not identify which shifts a detector misses.
We instead measure detectors along shared, declared paths in representation space. In-distribution features
are steered along a path, every detector is calibrated to the same in-distribution rejection rate, and we
record the distance at which each first rejects. A detector depends on its input only through the features,
so a path that it never rejects implies blindness to every input shift that realises that path. Decomposing
paths into radial and angular components, we show that for linear-head energy and Euclidean distance scores,
radial growth drives energy toward acceptance and distance scores toward rejection. Radial shrinkage drives
distance scores toward their value at the origin and energy toward a constant set by the classifier's biases
alone. We give exact conditions for each detector's decision at the origin, and show that calibrated energy
decisions depend on a component of the classifier that leaves every prediction unchanged. Across four text
and four vision encoders, every stated origin condition matches the observed decisions, and distance-based and
logit-based detectors fail along different paths.

Claims and their support:
- The reduction ("implies blindness") is Lemma 1: THEORY.md P3.
- The radial statements and the exact origin conditions are Prop. 3: THEORY.md P7, with 8/8 encoders in
  `runs/modal/radial_predictions.txt` and `support_checks.txt`.
- The gauge dependence is Prop. 2.
- "Fail along different paths" comes from the matched-horizon table (§1.4).

Deliberately left out:
- the vision principal-axis geodesics (not settled);
- the real-OOD norm link (one case);
- any numbers.

---

## 1. Phase 1: code and result validation (code-validator)

Script: `scripts/audit_results.py` (output `runs/modal/audit.json`). Also used: `ray_checks.txt`,
`radial_predictions.txt`, `support_checks.txt`, `norm_mechanisms.txt`.

### 1.1 Passing criteria
| check | result |
|---|---|
| Non-finite scores in 288 trace files | none |
| Metric ranges (AUROC, AUPR, FPR@95, rejection rates) | all in [0, 1] |
| Infinite calibration thresholds | none |
| Run status | 110/110 manifests `completed`; 12 preempted attempts archived as `*.failed-*` and excluded |
| Held-out ID rejection at tau = 0.05 | text 0.051 (0.035-0.065), vision 0.052 (0.034-0.071) |
| Steering is applied (independent refit and recompute at sampled path points) | 272/272 detector cells; max error 1.5e-11 (0.0 for the float32 pytorch-ood runs) |
| Path geometry | geodesic norm drift <= 9e-8; radial growth and shrinkage exact |
| Tests | 24 passed on the Modal image (see warning 6 on pinning) |
| Theory-vs-trace agreement | origin limits 8/8 (energy, MSP, kNN, Mahalanobis); radial-growth energy closed form 100% of probes; kNN growth bound 0 violations |

### 1.2 Warnings
1. **One seed per pipeline.** Seed 7 is used for the splits, heads, bootstraps and directions. Variability
   comes from reference bootstraps (6-10 fits), direction bootstraps and 4 encoders per modality, not from
   independent seeds.
   - Reference-bootstrap SD of static AUROC is <= 0.015 for every detector and encoder (0 for energy and MSP,
     as expected with a frozen head).
   - Head-training and split seeds are not varied.
2. **Headline means hide bimodality.** Vision energy on origin paths at the matched horizon is
   0.17 / 0.00 / 0.98 / 1.00 across encoders (mean 0.54, encoder-bootstrap 95% CI 0.09-0.99).
   The spread is fully explained: ResNet-50's threshold exceeds -lse(b), so its origin is accepted
   (predicted), and ResNet-18's path stops at s = 0.45 because of the horizon cap.
   **Report per-encoder values, not means.**
3. **n = 4 encoders per modality.** Bootstrap CIs over encoders are coarse. The deterministic 8/8
   theory checks carry the argument; the means are descriptive.
4. **Superseded numbers exist.** The `main` tag used pytorch-ood's float32 Mahalanobis, which is numerically
   wrong for ResNet-50, ViT-B/16 and DINOv2 in full dimension. The paper must use only `v2`
   (float64 Ledoit-Wolf) Mahalanobis numbers.
5. **Head training is uneven.**
   - ResNet-50, ViT-B/16 and DINOv2 full-dimensional heads reach training loss 0.000-0.004 (nearly separable,
     so logit scale is large).
   - Text heads end at 0.009-0.020.
   - ResNet-18 ends at 0.168 with a loss that decreases on only 72% of epochs. The fixed learning rate (0.01)
     is likely too high for its large-norm features (median norm 27).
   - ID test accuracy is 0.81-0.96 in vision. Weight decay is negligible (AdamW, 1e-4).
   - Energy results depend on the head (Prop. 2), so the head recipe is part of the result. Tune the learning
     rate on a head-validation split, or report the sensitivity.
6. **The Modal image is unpinned** except for skorch and pytorch-ood. It resolved sentence-transformers 6.1.0,
   while `pyproject.toml` says <6. Pin the image (`uv pip compile`) before the final runs.
7. **Horizon is a development choice** (3 ID radii; origin paths capped at 0.99 x the smallest probe norm).
   The matched-horizon comparison (1.03 radii text, 0.81 vision) is the defensible one.

### 1.3 Concrete fixes before 6 Oct
- Re-run the v2 grid with 3 seeds that vary split, head and bootstrap together; report mean ± SD across seeds.
  Cost: 3 x ~4 wall-hours on 16-vCPU containers.
- Pin the image; record `pip freeze` in every manifest.
- Report per-encoder values in the matched-horizon table (appendix), with encoder-bootstrap CIs in the main text.

### 1.4 Validated headline table (matched horizon; mean over 4 encoders; per-encoder in `audit.json`)
| path | text kNN / Maha / Energy / MSP | vision kNN / Maha / Energy / MSP |
|---|---|---|
| radial growth | 0.99 / 0.99 / 0.00 / 0.02 | 0.97 / 0.96 / 0.02 / 0.02 |
| toward origin | 0.10 / 0.09 / 1.00 / 1.00 | 0.00 / 0.00 / 0.54* / 0.29 |
| straight, PC | 0.88 / 0.33 / 0.10 / 0.25 | 0.50 / 0.20 / 0.05 / 0.18 |
| geodesic, PC | 0.38 / 0.03 / 0.29 / 0.68 | 0.10 / 0.01 / 0.05 / 0.23 |

\*Bimodal across encoders; see warning 2.

---

## 2. Phase 2: draft prose (academic-writer, UAI/AISTATS register)

Register: structural premise, then formal problem, then framework; assumptions stated.
Target: `paper/steering_ood.tex` (replace the placeholders).

**Introduction, P1-P2.** An OOD detector is a score on a representation together with a threshold, and a
static AUROC summarises its behaviour on one mixture of shifts. Two detectors with equal AUROC can reject
disjoint sets of shifted inputs, and the metric cannot distinguish them. We formalise the missing object: the
set of feature-space directions along which a calibrated detector never rejects.

**P3 (representation space).** A detector depends on an input only through its features, so decisions along
any input shift equal the decisions along the feature path it traces (Lemma 1). A feature path that is never
rejected therefore bounds from below the blindness to every input shift realising it. For L-Lipschitz scores,
approximate realisation within epsilon preserves every decision whose score margin exceeds L epsilon
(Corollary 1).

**P4 (shared, declared paths).** We fix the paths before measuring and share them across detectors. The
first-rejection distances are then paired statistics on the same probe and path. A worst-case distance would
use a different direction for each detector and answer a different question for each. Writing z = rho u, every
path changes the norm rho, the direction u, or both. We use one family for each combination (Table 1).

**P5 (calibration).** Each detector is thresholded at the same order statistic of held-out ID scores.
First-rejection distances are therefore invariant to strictly increasing transformations of any score
(Prop. 1). The invariance stops at input-dependent reparameterisations. Energy is one: adding a common
vector to every row of the linear head leaves all predictions unchanged but shifts energy by -r^T z (Prop. 2).

**P7 (result).** Radial growth drives linear-head energy toward acceptance and Euclidean distance scores
toward rejection. Radial shrinkage drives distance scores to their value at the origin and energy to
-lse(b), which is fixed by the biases (Prop. 3). Each origin condition is an inequality we check exactly.
The origin is accepted by kNN iff the k-th smallest reference norm does not exceed the threshold,
equivalently c_k <= c*. It is rejected by energy iff -lse(b) > t.

**Experiments paragraph.** On four text encoders (CLINC150) and four vision encoders (CIFAR10 against
CIFAR100 and SVHN), the conditions of Prop. 3 predict the observed decision at the origin for all eight
encoders, for all four detectors (§1.1). This includes the two cases a naive reading would call exceptions:
- the anisotropic BGE encoders, whose distance scores reject the origin;
- ResNet-50, whose energy accepts it.

Radial growth produces no new energy rejections in any encoder, as the closed-form condition predicts for every probe.

**Limitations paragraph.**
- Linear heads only.
- Feature paths are not claimed to be realised by inputs.
- The shrink-and-rotate cell is covered only transiently.
- Principal-axis geodesics in vision stay close to the ID support up to the matched horizon; beyond it they
  leave ID and are missed by energy, but caught partly by kNN (§3.3).
- The real-OOD norm link rests on one encoder-dataset pair.

**AI Use Statement (required; edit to reflect actual use).** An AI assistant (Claude, Anthropic) was used to
write and run experiment code, to derive and check propositions against saved runs, and to draft text. The
authors verified all results, proofs and claims and take responsibility for the content.

---

## 3. Phase 3: peer review (paper-reviewer; AISTATS standard)

Reviewed: the abstract in §0, the prose in §2, `paper/steering_ood.tex` and `THEORY.md`.

### 3.1 Mathematical soundness
- **Strengths.**
  - Prop. 3(a) is correct and sharp. Concavity follows from lse of an affine map, and the derivative identity
    d/ds E_p[a] = Var_p(a) gives uniqueness of s0.
  - Thm. (cone) is a clean application of Gordan's alternative.
  - Lemma 1 is trivial but correctly one-directional.
- **Assumption gaps to fix.**
  1. Lemma 1 requires a deterministic encoder (inference mode, no dropout or augmentation). State it.
  2. Prop. 3(c) counts bootstrap duplicates in the k-th smallest norm; say so. The equivalence c_k <= c* uses
     the calibration point that attains the order statistic, which exists because the threshold is a sample.
  3. The Prop. 3(a) shrinkage clause "rejected near the origin iff -lse(b) > t" is a limit statement.
     For a finite path it needs E(s_end) > t; state both.
  4. Prop. 1 assumes no ties at the order statistic; ties change strict-inequality rejection.
  5. Crossing stability (T1) and the ANOVA proposition are cited as `\hole`. They must be proved or dropped.
     Reviewers will check them first because the instrument's validity rests on them.
  6. Energy temperature is fixed at T = 1 throughout; say so, since Prop. 3 scales with 1/T.
- **Notation.** Define a = Wz once and use it throughout; the draft mixes Wz and a. Make the sign
  convention "higher = more anomalous" explicit for all four scores.

### 3.2 Statistical rigour: main weakness
- One seed. Variability is reported only across reference bootstraps (small) and across 4 encoders (coarse).
  An AISTATS reviewer will ask for mean ± SD over >= 3 independent seeds (split + head + bootstrap).
- The matched-horizon table has no intervals. Add encoder-bootstrap CIs (`audit.json`) and seed SDs.
- No significance tests are needed for the 8/8 deterministic checks. They are needed for any comparative
  claim between detectors on a path, e.g. "MSP > energy on PC geodesics in text" (0.68 vs 0.29). Use a
  paired bootstrap over probes within an encoder, then report across encoders.

### 3.3 Negative results and failure modes (present; must be foregrounded)
- **Static baselines break on SVHN with ImageNet CNN features** (kNN/Mahalanobis AUROC 0.18-0.22 on
  ResNet-18). This is the unnormalised-feature setting and must be reported, not hidden.
- **Vision PC geodesics.** They are near ID-preserving at the matched horizon (endpoints 2.2-2.6 ID sd along
  the axis vs a 1.9-2.1 edge). Beyond 3 sd, energy flags 0.00, Mahalanobis 0.00-0.30 and kNN 0.21-0.85.
  This is an honest negative result for energy and should be in the paper.
- **The real-OOD norm link is n = 1** (ResNet-18/SVHN). Do not generalise; Spearman over 12 pairs is
  -0.18 (p = 0.58).
- **The float32 pytorch-ood Mahalanobis artefact** deserves a sentence and an appendix. It is a useful warning
  to the community and explains any discrepancy with prior numbers.
- **Missing baselines a reviewer will name.** kNN on L2-normalised features (standard for deep kNN OOD),
  ViM (combines a feature residual with logits and is the closest prior idea to "complementarity"), and ReAct
  (activation clipping, a norm intervention). At least normalised kNN and ViM belong on the same paths.
- **Related work to position against.** Overconfidence of ReLU networks far from the data (Hein et al.,
  CVPR 2019): our radial-growth energy result is its calibrated, linear-head feature-space counterpart.
  Also the energy score (Liu et al., 2020) and Mahalanobis (Lee et al., 2018). Verify all citations
  before submission.

### 3.4 Compute footprint (currently undocumented in the draft)
- Crossed and static runs: `main` 21.0 wall-hours, `v2` 4.2 wall-hours, pilot 0.2, all on 16-vCPU Modal
  containers (~400 vCPU-hours in total). 12 preempted attempts are not counted.
- Encoding: 8 frozen encoders on one NVIDIA L4 each (minutes per model; not logged, so log it).
- No training beyond linear heads (30 epochs, CPU). Add this to the reproducibility checklist with encoder
  parameter counts.

### 3.5 Claim-evidence consistency
| claim | evidence | verdict |
|---|---|---|
| "calibrated to the same ID rejection rate" | held-out 0.051/0.052 at tau = 0.05 | supported (marginal guarantee; say so) |
| "never rejected implies blindness to every realising shift" | Lemma 1 | supported |
| "radial growth drives energy toward acceptance" | 0 new rejections, 8/8; closed form 100% | supported |
| "...distance scores toward rejection" | 0.96-0.99 at the matched horizon | supported |
| "exact conditions for each detector at the origin" | 8/8 for kNN, Maha, energy, MSP | supported |
| "fail along different paths" | matched-horizon table | supported; add CIs |
| tex: "energy 0.54 toward origin (vision)" | per-encoder 0.17/0.00/0.98/1.00 | **misleading as a mean: replace with per-encoder values** |
| "largely accepted by logit-based scores" for norm growth | straight rays 0.05-0.25, radial 0.00-0.02 | supported; "largely" covers MSP 0.25 on text PC rays |
| any vision-geodesic claim in the abstract | none made | correct to omit |

### 3.6 Verdict and readiness for 6 Oct
- **Current standing:** a borderline accept if the statistics are fixed. The theory is sharp and each clause
  is exactly checked, which is unusual and a strength for AISTATS. The main risks are single-seed evidence,
  missing normalised-kNN/ViM baselines, the unproved T1/ANOVA holes and positioning against Hein et al.
- **Six-day plan:**
  1. Today: register the §0 abstract.
  2. Days 1-2: 3-seed v2 re-run with a pinned image; add normalised kNN and ViM to all paths.
  3. Days 2-3: write the proofs (Prop. 3, cone, Lemma 1); prove or cut T1 and ANOVA.
  4. Days 3-4: fill the tex (§2 prose), per-encoder appendix tables, the failure-mode section and related work.
  5. Days 5-6: switch to the AISTATS style file, anonymise, add the AI Use Statement and reproducibility
     checklist, and fit to 8 pages.

---

## 4. Statistics fixed (2026-09-30)

Scripts: `scripts/seed_stats.py` (output `runs/modal/seed_stats.md`). Run: tag `seeds`, 144 cells
(3 seeds x 8 encoders x 6 paths), pinned image, non-preemptible, 0 failures.

**What changed**
- **Warning 1 resolved.** Three independent seeds; each re-partitions the data (CLINC seed 20260929+s,
  CIFAR seed 7+s), retrains the head (seed 7+s) and redraws references, directions and probes.
  The seed SD of the encoder mean is <= 0.054 in every matched-horizon cell.
- **Warning 2 resolved.** Per-encoder mean ± SD over seeds is reported (appendix table, tex).
  Vision energy toward the origin: ResNet-18 0.17±0.03, ResNet-50 0.02±0.01, ViT 0.88±0.08,
  DINOv2 0.99±0.01.
- **Warning 3 addressed.** Every cell carries a 95% hierarchical bootstrap CI (resampling encoders, then
  seeds within encoder).
- **Warning 6 resolved.** The image is pinned to the recorded versions, and `pip freeze` is saved in every
  crossed-run directory.
- **Review 3.2 resolved.** Comparative claims use a paired bootstrap over probes per (seed, encoder) cell,
  reporting significant cells out of 12 and a hierarchical CI of the mean difference:
  - Radial growth, distance > logit: 12/12 cells for every comparison and modality. Mean difference
    >= +0.91; lower CI >= +0.84.
  - Toward origin, logit > distance: text 12/12 (+0.87 to +0.89). Vision energy 9/12 (+0.52, CI 0.08-0.95).
    No cell has a difference <= 0. The 3 non-significant cells are ResNet-50 in every seed, exactly the
    case Prop. 3 predicts.
  - Geodesic PC, MSP > energy: text 12/12 (+0.36), vision 12/12 (+0.15).
- **Exact theory checks: 24/24 cells for every clause.**
  - An earlier 23/24 kNN result was a flaw in my check, not in the rule. Seed-2 ResNet-18 sits on the
    boundary (c_k = 0.7954 vs c* = 0.7960), and its origin path ends at norm 0.2-2.1, farther from the
    origin than the rule's margin.
  - The check now uses only (fit, probe) pairs where the 1-Lipschitz bound makes the observation decisive
    (7454 pairs).
  - The unit-norm c_k < 1/2 form fails on 6/24 cells, exactly ResNet-18/50 in every seed, as expected
    off the sphere.

**Headline numbers at the 3-seed matched horizon (1.04 radii text, 0.75 vision)**
| path | text kNN / Maha / Energy / MSP | vision kNN / Maha / Energy / MSP |
|---|---|---|
| radial growth | 1.00 / 0.99 / 0.01 / 0.02 | 0.95 / 0.94 / 0.03 / 0.02 |
| toward origin | 0.11 / 0.13 / 1.00 / 1.00 | 0.00 / 0.00 / 0.52 / 0.19 |

The CIs are in `seed_stats.md` and in Table 2 of the tex.

**Still open (not statistics)**
- Normalised-kNN and ViM baselines.
- Proofs for T1/ANOVA.
- Figure 3 is seed 0 only; the caption says so.
- The single-seed static AUROCs in §1 are superseded by the 3-seed table in `seed_stats.md` (C): AUROC SDs <= 0.014, FPR@95 SDs <= 0.019.
