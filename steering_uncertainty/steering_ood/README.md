# OOD steering for text and vision

Version 0.2 generalizes the original NLP package. **One feature-cache contract,
one calibration protocol, one steering runner** serve both modalities.
The canonical package/CLI is `steering_ood`. Existing `steering_nlp` imports,
`python -m steering_nlp`, and the `steering-nlp` command remain compatibility aliases.

## Supported inputs and models

| Input | Preparation | Frozen feature encoder |
|---|---|---|
| Text | CLINC150 split command | Sentence Transformers |
| Vision benchmarks | CIFAR10, CIFAR100, SVHN | TorchVision ResNet-family or ViT; timm backbones |
| Your image dataset | ImageFolder train/test/OOD directories | Same encoders, or your own PyTorch module |
| Existing features | Import a split NPZ with IDs/provenance | Any already-extracted pooled representation |

The common interface is `images or text -> z[N,D] -> score(z)`.
Skorch trains a new ID class head for energy/MSP; distance and graph scores use
features directly. ImageNet classification logits are never silently substituted
for features or for a classifier trained on the declared ID task.

## Install

Install matching Torch and TorchVision builds for your platform first. For CPU:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[vision,nlp,test]'
python -m pytest -q
```

Use `.[vision,test]` for image-only work; it does not require Sentence Transformers.
Use `.[nlp,test]` for text work. Frozen vision feature extraction accepts CPU/CUDA;
head training and the current detector adapters run on CPU.

## First vision run

These commands start a capped development pilot. Increase budgets only after
runtime, calibration and numerical checks. `--counts` means per ID class counts
for **head, reference, direction, calibration, probe**, in that order.

```bash
python -m steering_ood prepare-vision \
  --root data/images --id-dataset cifar10 --ood-datasets cifar100 svhn \
  --counts 100 100 100 100 20 --test-limit 1000 --download \
  --output data/vision_splits.json

python -m steering_ood encode-vision \
  --splits data/vision_splits.json --output data/resnet18.npz \
  --backend torchvision --model resnet18 --weights IMAGENET1K_V1 --device cpu

python -m steering_ood project \
  --cache data/resnet18.npz --output data/resnet18_64.npz --components 64

python -m steering_ood train-head \
  --cache data/resnet18_64.npz --output runs/vision_head --epochs 30

python -m steering_ood evaluate \
  --cache data/resnet18_64.npz --head runs/vision_head --output runs/vision_static

python -m steering_ood crossed \
  --cache data/resnet18_64.npz --head runs/vision_head --output runs/vision_e2 \
  --references 3 --directions 3 --probes 32 --steps 41 --horizon 5
```

Skip `project` and use the original cache throughout to study full features.
Projection is fitted ONLY on head-training features and must be disclosed in results.
The horizon is in exported/projected feature units; choose it on development data.
It is not comparable numerically to MPNet distances or another encoder's distances.
Each OOD source has separate detection/AUROC/AUPR metrics under `ood_by_dataset`;
pooled metrics are also provided and their AUPR depends on the evaluation mixture.

`--download` is explicit and applies only to dataset preparation. Default encoder
weights are pretrained and may be downloaded by TorchVision/timm on first use.
`--weights none` is available for software checks and is marked as untrained.
A backbone's pretraining can include related categories: OOD is relative to the
specified ID task, not necessarily novel to pretraining.

## Other visual models

TorchVision ResNet-family models use the native pooled feature before `fc`;
ViT uses the native CLS feature before `heads`. Checkpoint-specific preprocessing
is taken from the weights enum, including resize/crop/normalization.

```bash
python -m steering_ood encode-vision \
  --splits data/vision_splits.json --output data/vit.npz \
  --backend torchvision --model vit_b_16 --weights IMAGENET1K_V1

python -m steering_ood encode-vision \
  --splits data/vision_splits.json --output data/timm_resnet50.npz \
  --backend timm --model resnet50.a1_in1k --weights DEFAULT
```

For timm, use a tagged model name from your installed model registry, including
available DINO-style encoders if appropriate. The adapter uses `num_classes=0`
and the model's resolved evaluation transform. It validates a pooled `[B,D]`
output; spatial maps and token sequences require explicit pooling. State hashes,
versions and preprocessing are recorded rather than relying only on model names.

## Your own images

Use an ImageFolder layout, with one subdirectory per class. ID train and test
must have identical class names; OOD subfolder labels are discarded and replaced
by -1. Distinct OOD folder names become separately reported groups.

```bash
python -m steering_ood prepare-imagefolder \
  --id-train /data/my_id/train --id-test /data/my_id/test \
  --ood-folders near=/data/near_ood far=/data/far_ood \
  --counts 100 100 100 100 20 --output data/custom_splits.json
```

Decoded RGB-pixel hashes detect repeated content across selected splits. All
copies are excluded and logged; report the realized filtered benchmark counts.
The encoder checks those hashes again and fails if files changed after splitting.
These checks catch exact duplicates, not perceptual duplicates or semantic overlap.
Absolute input paths are recorded for provenance; regenerate the manifest after
moving your datasets to another machine.

For a custom pretrained PyTorch encoder, call the public exporter:

```python
from steering_ood.vision import export_image_cache

# encoder(batch) must return [batch, D]. Choose pooling in your own wrapper.
export_image_cache(
    'data/custom_splits.json', 'data/custom_features.npz',
    encoder=my_encoder, preprocess=my_checkpoint_transform,
    encoder_metadata={'model': 'my model', 'checkpoint': 'my checkpoint identifier'},
    batch_size=32, device='cuda',
)
```

This handles encoding and provenance. Every subsequent command is shared with NLP.

## Reuse an existing feature cache

For each role `head`, `reference`, `direction`, `calibration`, `probe`, `id_test`,
`ood_test`, supply these arrays in a non-pickled NPZ:

- `ROLE_x`: finite `[N,D]` features, same D and representation for every role.
- `ROLE_y`: integer ID class labels starting at zero; OOD labels are -1.
- `ROLE_ids`: stable strings, unique within and across all roles.
- Optional `ood_test_groups`: one string per OOD row.

Provide a metadata JSON with `modality`, `encoder`, and `split_provenance`.
For example, identify the checkpoint/layer/pooling and how disjoint IDs were formed.

```bash
python -m steering_ood import-features \
  --source my_arrays.npz --metadata my_metadata.json --output data/imported.npz
```

The importer checks shapes, labels, finite values and IDs. It cannot infer content
leakage from embeddings; truthful source IDs and splitting remain your responsibility.

## Vision validation and limits

```bash
python scripts/vision_smoke.py --output runs/vision-smoke
```

This runs generated image files through a real ResNet18 architecture, cache/PCA,
Skorch, all four PyTorch-OOD baselines and crossed steering. Default random weights
avoid downloads; `--pretrained` exercises pretrained loading. Both are software
integration checks, not benchmark evidence. Tests also exercise a timm ViT and
custom CNN exporters. See `VISION_VALIDATION.md` for what actually ran here.

Straight-line feature steering is shared across modalities. It does **not** produce
images or sentences. Input-supported corruption/edit paths and semantic validation
remain separate experiments; neither vision accuracy improvements nor semantic
OOD monotonicity is implied by this extension.

---

# NLP workflow (retained)

Runnable research extension of Tanmoy's `Graph_OOD_steering` interface.
Start with E0 and crossed E2; use CLINC150 as the first NLP application.
This package measures calibrated directional rejection. It does not establish
that a feature direction points toward semantic OOD or that steering improves
OOD classification accuracy.

## What is implemented

- CLINC150 split preparation, exact-text deduplication audit, cached Sentence Transformer embeddings.
- Optional PCA fitted only on classifier-training features; fixed thereafter.
- A linear intent head trained with Skorch, raw-logit scoring, saved weights/history.
- PyTorch-OOD 0.3.3 adapters: kNN, class-conditional Mahalanobis, energy, MSP.
- Explicit custom controls: single Gaussian Mahalanobis/NLL, pooled shrinkage
  Mahalanobis, exact unnormalized RBF spectral insertion score.
- Independent order-statistic ID calibration, strict `score > threshold` rejection.
- Original straight-line path tracer: initial rejection, censoring, bisection,
  full traces and returns to acceptance.
- Fresh-sample Gaussian E2 and stratified-bootstrap NLP E2; shared paths, both
  direction signs, reference/direction/interaction decomposition per probe.
- Optional separate calibration-only bootstrap arm, actual logit-temperature sweeps.
- Raw scores, IDs, split/cache/code hashes, configs, package versions and failure records.

## Shared installation details

Python 3.10+ (tested with Python 3.12). Use an isolated environment.
For a CPU environment install matching Torch and TorchVision builds first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[nlp,test]'
python -m pytest -q
```

On CUDA, install a matching Torch/TorchVision pair for your platform before the
editable install. A mismatched TorchVision binary can break PyTorch-OOD imports
although this experiment uses text. `requirements-tested.txt` records the
versions exercised here; it is a version record, not a cross-platform lockfile.

Core Gaussian/sklearn experiments need only `pip install -e '.[test]'`.
No backend silently falls back to another implementation.

## First runnable experiment: E0 then E2

```bash
python -m steering_nlp e0 --include-pytorch --output runs/e0.json
python -m steering_nlp synthetic \
  --output runs/e2-gaussian --detectors gaussian knn \
  --references 10 --directions 10 --probes 64 \
  --n-reference 64 --n-direction 64 --steps 101 --horizon 8
```

Repeat using new output names with sample sizes 256 and with
`--direction-kind oracle`. Independent reference and direction samples are drawn
from `N(0, diag(4,1,...))`. The PCA eigengap is recorded. Oracle directions should
remove direction variability. This diagnostic budget is not a power calculation.

For a dependency-light test use `--backend sklearn --detectors gaussian knn`.
`gaussian` is a pooled single Gaussian, unlike class-conditional `mahalanobis`.

## End-to-end integration check, no external model required

```bash
python -m steering_nlp fixture --output runs/fixture.npz
python -m steering_nlp train-head --cache runs/fixture.npz --output runs/head --epochs 5
python -m steering_nlp evaluate --cache runs/fixture.npz --head runs/head --output runs/static
python -m steering_nlp crossed \
  --cache runs/fixture.npz --head runs/head --output runs/crossed \
  --references 2 --directions 2 --probes 4 --steps 31 --refine 8
```

To run all these checks with one command, use `bash scripts/smoke.sh runs/smoke`.
The fixture contains synthetic features and is never an NLP result.
Included `examples/validation/` outputs exercise this workflow. See `VALIDATION.md`.

## CLINC150 workflow

Get `data/data_full.json` from https://github.com/clinc/oos-eval and retain it locally.
The package does not use OOS training/validation examples.

```bash
python -m steering_nlp prepare --source data/data_full.json --output data/splits.json
python -m steering_nlp encode --splits data/splits.json --output data/mpnet.npz \
  --model sentence-transformers/all-mpnet-base-v2 --device cpu
```

Use `--revision COMMIT` for a pinned model snapshot. The cache records the resolved
model commit where available, encoder modules, dataset hash and split hash.
GPU extraction is available with `--device cuda`; detector/head runners use CPU.
Embeddings are whatever the encoder emits: `normalize_embeddings=False` avoids
an extra normalization, but an encoder can itself contain a Normalize module.
No path point is subsequently projected back to a sphere.

The requested per-intent allocation is 60 head / 20 reference / 20 direction
from train; 10 calibration / 10 probes from validation. The official ID/OOS tests
are reserved. All copies of repeated case/whitespace-normalized text in these
used splits are excluded after assignment. Removals and realized role counts
are recorded. This is a **deduplicated CLINC150 variant**, not the unmodified
published protocol. The checked source excluded ten duplicate records; resulting
counts were 8996 head, 2999 reference, 3000 direction, 1498 calibration, 1499 probes,
4498 ID test and 1000 OOS test. A different source revision may give other counts.

For a cheaper initial CPU pilot, explicitly use a fixed 64-dimensional projection:

```bash
python -m steering_nlp project --cache data/mpnet.npz --output data/mpnet64.npz --components 64
python -m steering_nlp train-head --cache data/mpnet64.npz --output runs/clinc-head --epochs 30
python -m steering_nlp evaluate --cache data/mpnet64.npz --head runs/clinc-head \
  --output runs/clinc-static --temperatures 0.5 1 2 10
python -m steering_nlp crossed --cache data/mpnet64.npz --head runs/clinc-head \
  --output runs/clinc-e2-pilot --references 3 --directions 3 \
  --probes 32 --steps 41 --horizon 3 --calibration-draws 5
```

PCA changes the representation being studied; name the projection in every
result. Use `mpnet.npz` throughout to study the full representation. Train a new
head for that cache. The selected horizon is a development choice, not a universal
OOD distance. Check censoring and double the grid before a confirmatory run.
Tune horizon/step budgets on development data; do not use final OOS performance.
Start with kNN/Gaussian controls if runtime is high, then add class-conditional
Mahalanobis and fixed-head scores. Full reference/direction sweeps are deliberate
follow-ups to the pilot, not automatic defaults for a first resource estimate.

## Connect to the existing steering interface

The old interface is retained:

```python
from steering_nlp.detectors import make_detector
from steering_nlp.core import threshold
from steering_nlp.paths import trace_path

# D, C, x, v and alphas are numpy arrays. y_D contains nonnegative intent IDs.
detector = make_detector('knn', backend='pytorch', k=5).fit(D, y_D)
t = threshold(detector.score(C), tau=0.05)
trace = trace_path(detector.score, t, x, v, alphas)
```

Existing unsupervised detectors with `fit(D)` and `score(X)` can still be used
with `threshold` and `trace_path` directly. To register one in the runner, add an
adapter accepting `fit(D, y=None)` to `make_detector`. The exact RBF spectral
adapter is already available as `spectral`; it requires 2..256 references.
Create a declared small-reference cache for a spectral panel and run ALL compared
detectors on those same references. The code refuses an accidental large dense
spectral run. It is not the PDF's normalized cosine-kNN graph.

## Scientific interpretation

- Target clean-ID rejection defaults to 5%. With `m` calibration points the order
  is `ceil((m+1)*(1-tau))`; insufficient calibration gives an infinite threshold.
  This is a marginal exchangeability guarantee, not an exact conditional 5% rate.
- Direction development, reference fitting, head training, calibration and probes
  are separate. Calibration POINTS stay fixed in E2, but thresholds change with D.
- Learned NLP directions use independently bootstrapped ID PCA estimates. Both
  signs are evaluated. A small covariance eigengap makes the PCA direction weakly
  identified. Random directions are an explicit control (`--direction-kind random`).
- NLP bootstraps describe the fixed empirical pools; fresh Gaussian draws support
  the controlled population experiment. These are different inference targets.
- `observed = min(first crossing, horizon)` always travels with an event indicator.
  No crossing is not an observed crossing at the horizon. Initial rejection is 0.
- ANOVA is descriptive under the empirical product distribution over D and V.
  It is computed per probe/sign before averaging, including the interaction.
  Path steps and bootstrap draws are NOT independent sample-size multipliers.
  No confidence intervals or universal epistemic-uncertainty interpretation are claimed.
- Energy/MSP do not depend on D when the head is frozen. Their reference variance
  should be zero and is not a detector-quality advantage. Temperature acts on real
  logits; each score/temperature receives its own ID threshold.
- Finite grids can miss short excursions. Bisection refines a detected bracket,
  not proof that it is the earliest true crossing. Check doubled-grid agreement.
- Full rejection curves differ from cumulative first-crossing curves after re-entry.
- No feature path is claimed to be a valid sentence or monotone semantic-OOD path.
  Input-supported edits, near-tangent stress tests, local-derivative comparisons,
  population confidence intervals and the final NLP benchmark are follow-up work.

## Output schema

`manifest.json`: full config, code hashes, versions, cache/head provenance, status.
`design.npz`: directions, signs, alphas, stable probe IDs and bootstrap indices.
`METHOD_traces.npz`: axes **reference, direction, sign, probe, alpha** for scores
and rejection; crossing summaries omit the alpha axis. Power arrays average probes.
`METHOD_paths.jsonl`: first brackets, every observed return bracket, censoring/status.
`summary.json`: per-sign reference/direction/interaction variance, crossing fraction,
restricted crossing mean and held-out static metrics per reference fit.
`METHOD_calibration_only.npz`: optional separate fixed-D/fixed-v calibration arm.

Static evaluation saves raw scores and ID/OOS IDs. Positive class is OOD.
Fit failures stop the run and write an explicit failure record; there is no silent
removal of failed fits. Nonempty run directories are refused to prevent overwrite.

## Provenance

`steering_ood/paths.py` is copied unchanged from the user's local `Graph_OOD_steering` checkout,
commit `ccd9d4b6e09c87b517ac24d5e07164f9f7039d5d`.
`analytic.py` extracts its `alpha_star_maha` control. Original checkout left unchanged.
The spectral adapter reproduces its documented RBF operator. This private handoff
does not assert a new redistribution license for the user's existing code.

Dependency documentation: https://pytorch-ood.readthedocs.io/,
https://skorch.readthedocs.io/, https://sbert.net/.

## Learned-direction collaboration

See [the Astrid/Tanmoy experiment handoff](LEARNED_DIRECTIONS_HANDOFF.md) for the agreed CIFAR10
learned-direction extension, proposed ownership, implementation checklist, matched
detector protocol, calibration isolation, source-transfer evaluation and required
artifacts. This is a specification; implementation and benchmark results are pending.
