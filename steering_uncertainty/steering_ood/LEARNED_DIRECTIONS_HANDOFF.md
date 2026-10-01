# Learned activation directions: collaborator handoff

Status: agreed experiment specification; implementation and benchmark runs are pending.
Date: 2026-10-01. Integration branch: `steering` (the existing steering/OOD branch).
Collaborator source: `steering-astrid/astrid`, reviewed commit
`bcf8e0ce9a706dd2ff83db409c48626ca473b58d`.
Source: https://github.com/Tankiit/Graph_OOD/tree/steering-astrid/astrid

## Research question and scope

Do learned, detector-independent activation directions produce failures explained by
our radial/angular analysis, or reveal an additional mechanism? This is a detector
stress test, not a proposal to improve clean OOD detection. Start with CIFAR10 ID,
CIFAR100 and SVHN OOD, and the existing ImageNet-pretrained ResNet18 plus the
CIFAR10 linear head. Use full-dimensional pooled features and exactly the existing
checkpoint preprocessing. Do not substitute ImageNet logits for the CIFAR10 head.

First implement pooled-feature steering to align exactly with the current theory.
Then add `layer3.0` steering, propagate through the remaining encoder, and evaluate
the same frozen CIFAR10 head. Internal steering is an activation intervention;
it does not establish that an input image realises the modified path.
Do not copy MNIST/FashionMNIST thresholds or checkpoints into this experiment.

## Division of work

Proposed ownership (coordinate in the PR; this is not a notification to collaborators):

| Owner | Work | Deliverable |
| --- | --- | --- |
| Astrid | Port pure InfoNCE swarms, CIFAR data adapter, pooled and internal steering, vector export | Direction-training module, resolved configs, vectors, training logs |
| Tanmoy / steering maintainer | Common detector scoring, calibration, shared path tracing, polar diagnostics, statistics | Evaluation adapter, paired tables, mechanism figures |
| Both | Check split provenance, pilot outputs, seed replication, interpretation | Reviewed pilot and final evidence report |

Astrid: begin with the checklist below; propose changes on your branch and open a
PR targeting `steering`. Keep additions under this package, without replacing its
existing runner. Pin the Astrid source commit and preserve attribution in the
manifest. Avoid merging unrelated branch changes.

## Required implementation checklist

- [ ] Add a CIFAR loader using stable dataset/split/index IDs, with disjoint roles.
- [ ] Load the exact frozen encoder and CIFAR10 head; verify unsteered suffix logits
      and features agree with the original forward pass within recorded tolerances.
- [ ] Port `SteeringSwarm`, `steer`, and pure `infonce` from Astrid's modules.
      Train independent ID-to-OOD and OOD-to-ID swarms; export unit directions and
      resolved training radii. InfoNCE must not use detector scores or thresholds.
- [ ] For internal steering, adapt `SplitModel` so the suffix returns pooled
      features plus logits from our CIFAR10 head, rather than the encoder's head.
- [ ] Add normalized kNN alongside raw kNN; keep distinct method names and scores.
- [ ] Use the shared calibration and path contracts below, including first acceptance
      for OOD, full rejection traces, censoring and grid sensitivity.
- [ ] Export polar diagnostics and per-source metrics, not just aggregate AUROC.
- [ ] Run the software checks and capped pilot before the three-seed benchmark.

Suggested new modules: `steering_ood/learned_directions.py`,
`scripts/run_learned_directions.py`, `scripts/analyze_learned_directions.py`,
`tests/test_learned_directions.py`. These are implementation targets, not existing
commands. Use `steering_ood/core.py:threshold`, `paths.py:trace_path`, and the
existing detector adapters wherever compatible. Check `trace_path`'s return schema
before adapting it for first acceptance.

## Data isolation and OOD transfer

Keep the current ID role contract: `head`, `reference`, `direction`, `calibration`,
`probe`, `id_test`. Partition the direction role further into steering-training and
steering-development sets. Calibration must be untouched by head checkpoint
selection, direction fitting, radius/horizon tuning and detector fitting.
If an existing checkpoint used calibration images for selection, construct a fresh
split or retrain before claiming the calibration guarantee. Report realized counts.

Use CIFAR100 official training images for OOD steering-training/development, split
with recorded IDs. Use CIFAR100 test images only for same-source held-out evaluation.
SVHN test images are the unseen-source evaluation: no SVHN samples may be used for
training, development, vector selection or tuning in this first experiment.
Use ID-development data to set scales and numerical path budgets; do not inspect
final OOD test performance to choose a setting. Audit exact content duplicates
across roles in addition to ID overlap. Save removal logs and dataset hashes.

## Fixed pilot and benchmark design

| Parameter | Capped software/development pilot | Confirmatory run |
| --- | --- | --- |
| Encoder | ResNet18, existing pretrained checkpoint | Same |
| Injection | Pooled feature first | Pooled feature and `layer3.0` |
| Similarity | Pooled feature | Pooled feature for both injection sites |
| Seeds | 0 | 0, 1, 2 |
| Loss | Pure multi-positive InfoNCE | Same |
| Swarm size / evaluated vectors | 8 / all 8 | 64 / all 64 |
| Optimization | 100 steps, Adam lr 0.01, batch 32 ID + 32 OOD | 1000 steps, Adam lr 0.01, batch 64 + 64 |
| InfoNCE temperature / own negative | 0.1 / true | Same |
| Relative training radius | 0.5 | 0.25 and 0.5 |
| Probe count | 32 ID, 32 per OOD source | 256 ID, 256 per OOD source |
| Path grid | 41, then 81 points | 101, then 201 points |
| Horizon | 4 times resolved training radius | Same, unless development censoring motivates a preregistered change |

These are proposed budgets, not a power calculation. Record measured runtime and
peak memory before expanding. Record all overrides before final evaluation.
Seeds must vary direction training, sampling, reference draws and head training;
use the corresponding existing seed-specific heads/caches. Architecture and
pretrained encoder checkpoint stay fixed. Do not treat swarm vectors as independent
training seeds. Evaluate every vector, with no test-based best-vector selection.

Directions: learned, isotropic random (matched count), ID PCA (both signs), and
ID-to-OOD development-centroid direction as an explicitly OOD-informed control.
Fit PCA only on ID direction-training data; report its eigengap. Keep the single
PCA axis as a single axis rather than manufacturing independent replicates.
Share each evaluated path across all detectors. Separate the two learned swarms;
they are distinct interventions, not the same direction or its opposite.

Training radius and horizon are in injection-space units. For CNN maps the offset
is broadcast at every position: full-map displacement is sqrt(H*W) times its channel
norm. Record both. Compare direction families within an injection site at identical
budgets. Across sites, report induced pooled-feature displacement as well; do not
claim equal downstream budgets merely because relative injection radii match.

## Detectors and calibration

Evaluate energy (T=1), MSP (T=1), raw Euclidean kNN, L2-normalized kNN, and
`mahalanobis_shrinkage` (float64 pooled Ledoit-Wolf). Use the same ID reference IDs
for both kNN variants and Mahalanobis. Set k=5 for the aligned primary panel; an
optional k=50 panel must change both kNN variants together and be labeled separately.
Do not label Astrid's covariance-pseudoinverse method as the same Mahalanobis method.

Normalize only inside the normalized-kNN scoring adapter; do not normalize the
head input or the other detectors' features. Explicitly record/reject zero-norm
queries and references for this adapter, rather than assigning an arbitrary unit
vector. Positive radial scaling of a nonzero query must leave normalized-kNN scores
unchanged; it therefore cannot inherit our raw-distance radial-growth conclusion.

For each detector fit, use `threshold(scores, tau=0.05)`: rank
ceil((m+1)*0.95), infinite threshold if the rank exceeds m, strict score > threshold
rejection. Fix thresholds for the entire path, including steered endpoints.
Report clean held-out ID rejection, not just calibration acceptance.
Do not replace this with `np.quantile` or recalibrate on steered test ID.

## Measurements and mechanism analysis

For each probe and shared direction record raw scores, rejection at every alpha,
first ID rejection or first OOD acceptance, initial state, crossing brackets,
re-entry/recrossings, event indicator and censoring horizon. Refine detected brackets
by bisection; compare doubled grids because refinement cannot detect missed excursions.
Define attack success additionally on initially correctly handled probes: initially
accepted ID becoming rejected, initially rejected OOD becoming accepted. Report
unconditional rates too, with denominators and baseline error rates.

Report at the fixed clean threshold:
1. ID rejection and OOD acceptance separately, per OOD source.
2. Clean, ID-only-steered, OOD-only-steered and both-steered AUROC panels.
3. First-flip curves, current-state curves, censoring and re-entry rates.
4. Baseline raw-score distributions and all detector thresholds.

Both-steered AUROC pairs independently learned swarms; it is descriptive and must
not imply a single universal direction attacks both sides. A low AUROC can result
from ID displacement alone, even if OOD never becomes accepted. If reporting
conventional FPR95 from evaluation ID scores, label its recomputed threshold and
keep it separate from acceptance at the fixed clean-calibration threshold.

At every step export downstream z, ||z||/||z0||, angle(z,z0), displacement
||z-z0||, and its radial projection and tangent residual relative to z0.
Compute matched diagnostic projections rho(alpha)*u0 (radial-only) and
rho0*u(alpha) (angular-only). Label these as feature-space counterfactual controls,
not realizable inputs or a causal attribution of the nonlinear internal intervention.
For pooled straight paths, compare exact linear-head energy predictions with traces.
For internal paths, score the actual suffix features; do not apply an affine-ray
formula to a generally nonlinear downstream path.

Interpretation: test whether learned paths primarily change norm, angle, or both;
compare each detector's failure with the corresponding controlled traces. Residual
failures after norm control motivate additional mechanisms; they do not alone prove
one. Include unsuccessful attacks and source-transfer failures.

## Uncertainty, artifacts and acceptance criteria

Report seed-level means and SDs and paired bootstrap intervals over probes within
seed/source; preserve the same resampled probe IDs across detectors. Preserve
cluster structure over vectors (do not count direction-by-probe rows as independent
images). Do not pool seeds, alphas or bootstrap draws as independent observations.
A calibration-only bootstrap is a separate sensitivity arm, not total epistemic
uncertainty. If adding crossed reference/direction variability, freeze the trained
swarm per direction-training draw, refit/recalibrate for every reference draw and
retain interaction terms using the current runner's conventions.

Each run writes a manifest (both code commits, checkpoint/head/data hashes,
preprocessing, role IDs, seeds, detector configuration, precision, normalization,
training parameters, resolved radii, grid, versions, runtime/memory), vectors,
training logs, per-probe score/geometry traces, event/censoring arrays and a summary.
Write outputs under ignored `runs/learned_directions/`; commit a small report and
configuration, not datasets/checkpoints. A compact fixture can be committed.

Required meaningful checks before benchmark:
- [ ] Zero-offset split-model equivalence and frozen model/head weights.
- [ ] Disjoint roles; untouched SVHN; fresh calibration provenance.
- [ ] Same features/paths across detectors and separate thresholds per method.
- [ ] Normalized-kNN positive-scale invariance and declared zero-norm behavior.
- [ ] Known crossing, initially flipped probe, no crossing, re-entry and bisection.
- [ ] Doubled-grid agreement reported; censored paths never silently imputed as events.
- [ ] Nonfinite scores/geometry fail explicitly; pooled energy formula agrees numerically.
- [ ] Seed-specific manifest and pilot raw artifacts can reproduce summary values.

Completion means: aligned pooled pilot, internal-steering pilot, three-seed results
for both sites and sources, paired detector tables, polar figures, and an evidence
report documenting limitations. Existing Astrid MNIST/ViT reports motivate this
experiment; their missing raw outputs have not been independently reproduced here.
