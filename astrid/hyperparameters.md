# Steering swarms: hyperparameters, losses and detectors

Reference for `actdist.train_steer` (training), `actdist.eval_steer` (evaluation), `actdist.actnorm`
and `actdist.compare_steer`. Every hyperparameter below is a field of `SteerConfig` in
`src/actdist/steer_config.py`; that file is the source of truth for defaults.

## Setting hyperparameters

Values are resolved in three layers, later ones winning:

1. the defaults of `SteerConfig`;
2. a TOML file, `--config configs/steer/<name>.toml` (any subset of the keys);
3. command-line overrides, `--set key=value [key=value ...]` (repeatable).

`--sweep key=v1,v2 [key2=w1,w2]` runs one experiment per combination (Cartesian product).
Values are parsed as TOML: numbers, `true`/`false`, `"strings"`, `["lists"]`; a bare word is a string.
Lists can be swept: `--sweep 'swarms=["id2ood"],["id2ood","ood2id"]'`.

```
uv run python -m actdist.train_steer --help-config                                   # every key and its default
uv run python -m actdist.train_steer --config configs/steer/base.toml --set layer=layer3.1 radius=0.25
uv run python -m actdist.train_steer --config configs/steer/base.toml --sweep radius=0.5,1 --print   # show, don't run
```

An unknown key or an invalid value (e.g. an unknown loss) stops with an error listing the allowed values.
Each experiment writes its fully resolved configuration to `outputs/steering/<name>/config.json`.

Config files: `configs/steer/base.toml` (general starting point) and
`configs/steer/scratch_mnist_layer4.toml` (ResNet-18 scratch on MNIST, layer4.0). Note that
`base.toml` sets `radius_mode = "rel"` while the dataclass default is `"abs"`.

## Setting: what is steered, and against what

- **Classifier.** A frozen checkpoint from `outputs/checkpoints/` (`run`), in eval mode (BatchNorm
  uses its running statistics). No classifier weight changes; only the steering vectors are learned.
- **ID** is the dataset the classifier was trained on (MNIST or FashionMNIST), **OOD** the unlabeled
  sources (`ood_sources`). OOD images get exactly the classifier's preprocessing (grayscale,
  resize, 1→3 channels for pretrained models, the classifier's normalisation).
- **Two swarms**, trained jointly in the same loop and on the same batches:
  - `id2ood`: K vectors that, added to the activations of ID images, make the detector **reject** them;
  - `ood2id`: K vectors that, added to the activations of OOD images, make the detector **accept** them.

### Fixed data splits (not hyperparameters)

Disjoint index sets, following the AISTATS 2027 draft:

| Set | ID | OOD (per source, by index `i`) |
|---|---|---|
| development: `v_toward`, `radius_mode = "rel"` | 5 000 images of the classifier's train split | `i % 10 == 1` (5 000 images, equal share per source) |
| steering-train: batches the swarms are trained on | the other 50 000 images of the train split | `i % 10 >= 2` |
| calibration: threshold `t_D` | the classifier's 5 000-image validation split | not used |
| evaluation probes | `n_probe_id` images of the test split | `i % 10 == 0` (`n_probe_ood` images) |

The ID train/validation split is the one `train.py` used (seed 0): the calibration images were not
trained on (they only served to pick the best checkpoint), and the steering-train images are part of
the classifier's own training data.

## The detectors

A detector maps the classifier's output to a score **s(x)**; a **higher score means more OOD**. With
logits f(x) ∈ R^C and penultimate feature z(x) ∈ R^d (the pooled feature just before the head):

| Name | Score s(x) | Uses | Notes |
|---|---|---|---|
| `energy` | E(x) = −T · log Σ_c exp(f_c(x) / T) | logits | free energy (Liu et al., 2020), T = `energy_T`; the detector the energy losses attack |
| `msp` | −max_c softmax(f(x))_c | logits | maximum softmax probability, negated (Hendrycks & Gimpel, 2017) |
| `maxlogit` | −max_c f_c(x) | logits | maximum logit, negated |
| `knn` | ‖ẑ(x) − ẑ_(k)‖, ẑ = z / ‖z‖ | features | distance from the L2-normalised feature to its k-th nearest normalised reference feature, k = `knn_k` (Sun et al., 2022) |
| `mahalanobis` | min_c (z(x) − μ_c)ᵀ Σ⁺ (z(x) − μ_c) | features | μ_c = class means of the reference features, Σ = covariance shared across classes (pseudo-inverse Σ⁺) (Lee et al., 2018) |

**Reference set of the feature detectors.** `knn` and `mahalanobis` are fitted once, on the clean
penultimate features of `ref_size` ID images drawn (with `seed`) from the steering-train split, i.e. the
classifier's own training images; Mahalanobis uses their labels for the class means. The reference
never changes during training or evaluation: steering moves the probes, not the reference.
Their scores are computed on the (steered) penultimate feature, whatever `layer` the vectors are added at.

**Threshold.** t_D is the `tpr`-quantile of the scores of the clean ID calibration split, so a fraction
`tpr` of clean ID is accepted (τ = 1 − `tpr` = 5% rejected by default). The detector **rejects** x when
s(x) > t_D and **accepts** it otherwise. One threshold per detector; the energy losses always use the
energy threshold.

**Metrics** (in `metrics.json` and `compare_steer`):

- **AUROC** = P(s(OOD) > s(ID)) (Mann–Whitney); 1 = perfect, 0.5 = chance, < 0.5 = inverted.
- **FPR95**: fraction of OOD accepted at the threshold that accepts 95% of the ID scores being compared.
- **ID / OOD accepted at t_D**: fraction with s ≤ t_D, using the fixed calibration threshold.
- **flip rate**: for `id2ood`, the fraction of ID probes rejected; for `ood2id`, the fraction of OOD probes accepted.

`detectors` lists which detectors are evaluated and reported. The energy threshold used by the energy
losses and by the training log's flip rates is computed in every case, whether or not `energy` is listed;
the same holds for the kNN threshold when the loss is `knn_hinge`.

## The steering-vector swarm

### Parametrisation: vectors on a hypersphere

Each swarm holds unconstrained parameters u_k ∈ R^D, k = 1..K, and uses

  v_k = r · u_k / ‖u_k‖,

so every vector has norm exactly r at every optimisation step. Adam updates u; r is fixed.

### Where a vector is added

At the output of block `layer` (names as in `models.ActivationRecorder`). The rest of the network then
runs as usual up to the logits.

| Model | `layer` values | Activation h | How v_k is added | D |
|---|---|---|---|---|
| ResNet-18 | `layer1.0` … `layer4.1` | block output map [B, C, H, W] (after the residual addition and ReLU) | the same C-vector at every spatial position | C (64–512) |
| ViT-S/16 | `blocks.0` … `blocks.11` | block output tokens [B, T, 384] | to every token, CLS included | 384 |
| both | `penultimate` | pooled feature just before the classifier head | to the feature | 512 / 384 |

So **r is the displacement per spatial position / per token / per image**. Only `layer = "penultimate"`
matches the draft's setting, where the displaced feature goes straight to the detector.

### Steering hyperparameters

| Key | Default | Meaning |
|---|---|---|
| `layer` | `"penultimate"` | where vectors are added (table above) |
| `radius` | `1.0` | the sphere radius, before `radius_mode` is applied |
| `radius_mode` | `"abs"` | `abs`: r = `radius` (activation units). `rel`: r = `radius` × the median activation norm at `layer`, measured on the 5 000 ID development images (per position / token / image, as vectors are added). With `rel`, one value means the same relative size at every layer; `actdist.actnorm` shows the medians |
| `swarm_size` | `1024` | K, vectors per swarm |
| `swarms` | `["id2ood", "ood2id"]` | which swarms to train (one or both) |
| `init` | `"randn"` | `randn`: u_k ~ N(0, I). `toward`: u_k = s · v_toward + `init_noise` · ε_k / √D with ε_k ~ N(0, I), s = +1 for `id2ood`, −1 for `ood2id` |
| `init_noise` | `0.5` | noise scale for `init = "toward"` |

**v_toward** (the draft's default direction) is the unit vector along
mean(h, OOD development) − mean(h, ID development), where the mean is over images and over
positions / tokens. It points from the ID mean toward the OOD mean at the injection layer.

### Sharing a batch across the swarm

Each step takes `bs_id` ID and `bs_ood` OOD images. Their activations h at `layer` are computed once
(no gradient), and every vector of a swarm is applied to every image of its side: K × B steered
copies. The `id2ood` swarm is applied to the ID batch, `ood2id` to the OOD batch. The part of the
network after `layer` runs on `vec_chunk` vectors at a time, with a backward pass per chunk; vectors
do not interact, so this only bounds memory and the gradient is exact.

## The losses

Notation for one swarm and one step:

- a_{k,i} = penultimate feature of image i steered by vector k (the anchor), L2-normalised;
- P = clean penultimate features of the batch of the **other** side (targets), L2-normalised;
- N = clean penultimate features of the batch of the **same** side, L2-normalised;
- E_{k,i} = energy of the steered image, t = energy threshold t_D (for `knn_hinge`: the kNN score and threshold);
- s = +1 for `id2ood` (push the energy up, above t), s = −1 for `ood2id` (push it down, below t).

Each loss gives a per-vector value L_k (mean over the batch). The optimised objective is

  L = `weight_id2ood` · Σ_k L_k(id2ood) + `weight_ood2id` · Σ_k L_k(ood2id),

summed (not averaged) over the K vectors, so each vector's gradient does not depend on K.

### `infonce`: contrastive (default)

Multi-positive InfoNCE with temperature τ = `tau`:

  L_{k,i} = log Σ_{c ∈ P ∪ N} exp(a_{k,i} · c / τ) − log Σ_{c ∈ P} exp(a_{k,i} · c / τ)

The steered feature must be more similar (cosine) to the clean features of the other side than to
those of its own side. The features compared are the activations at `sim_layer` (default: the penultimate
feature), for steered and clean images alike; the detectors always score the penultimate feature and logits. If `own_negative = false`, the anchor's own clean feature is removed from N, so
the vector is not pushed away from its starting point specifically.

It acts on features only, not on the detector: it succeeds against the detector only if looking like
the other side's features also moves the score across t_D. When the detector already accepts much of
the OOD, pulling ID toward OOD features can lower the energy instead (seen in experiment E5).

### `energy_hinge`: threshold hinge on the detector

  L_{k,i} = max(0, m − s · (E_{k,i} − t)),  m = `energy_margin`

Zero once the steered energy is past t_D by at least m energy units, on the right side.

### `energy_margin`: pairwise margin against the other side

  L_{k,i} = mean_j max(0, m − s · (E_{k,i} − E^clean_j)),  j over the clean images of the other side

Every steered energy must pass every clean energy of the other side by m: steered ID above all clean
OOD energies (`id2ood`), steered OOD below all clean ID energies (`ood2id`). It does not use t_D.

### `knn_hinge`: threshold hinge on the kNN detector

  L_{k,i} = max(0, m − s · (knn(z_{k,i}) − t_knn)),  m = `knn_margin`

where knn(z) is the kNN score of the steered penultimate feature: the distance of ẑ = z / ‖z‖ to its
`knn_k`-th nearest neighbour among the normalised features of the `ref_size` ID reference images. The
gradient goes through that k-th distance (top-k over the reference), so the vector moves the steered
feature relative to the very reference points the detector uses (white-box, like the energy losses).
Scale for resnet18_scratch_mnist: ID median 0.094, t_D 0.25, OOD median 0.565 (`knn_margin` 0.05 is
about a third of the ID spread).

### `infonce+energy`

  L_{k,i} = InfoNCE_{k,i} + λ · EnergyHinge_{k,i},  λ = `lambda_energy`

### Loss hyperparameters

| Key | Default | Used by | Meaning |
|---|---|---|---|
| `loss` | `"infonce"` | all | `infonce`, `infonce+energy`, `energy_hinge`, `energy_margin` or `knn_hinge` |
| `tau` | `0.1` | infonce | InfoNCE temperature; smaller = sharper preference for the nearest target features |
| `sim_layer` | `"penultimate"` | infonce | where the similarity is measured: `layer` itself or any layer downstream of it (e.g. steer at `layer3.1`, compare at `layer4.0`). An intermediate block is reduced to one vector per image as in `ActivationRecorder` (ResNet: mean over positions; ViT: CLS token). An upstream layer is refused |
| `own_negative` | `true` | infonce | count the anchor's own clean feature among the negatives |
| `lambda_energy` | `1.0` | infonce+energy | weight of the energy hinge |
| `energy_margin` | `1.0` | energy losses | margin m, in energy units (clean ID energies of resnet18_scratch_mnist are around −8) |
| `knn_margin` | `0.05` | knn_hinge | margin m, in kNN distance units (see scale above) |
| `weight_id2ood` | `1.0` | all | weight of the `id2ood` swarm's loss |
| `weight_ood2id` | `1.0` | all | weight of the `ood2id` swarm's loss |
| `energy_T` | `1.0` | energy losses, `energy` detector | temperature of the energy score |

There is no term that keeps the vectors of a swarm apart. With `infonce` at the penultimate layer
all 1024 vectors converged to the same direction (E5); the evaluation reports the mean |cosine|
between vectors so this is visible.

## Optimisation

| Key | Default | Meaning |
|---|---|---|
| `steps` | `2000` | optimisation steps (one ID batch + one OOD batch each) |
| `lr` | `1e-2` | Adam learning rate on u |
| `bs_id` | `64` | ID images per step, shared by every vector of `id2ood` |
| `bs_ood` | `64` | OOD images per step, shared by every vector of `ood2id` |
| `vec_chunk` | `256` | vectors per forward/backward pass after `layer`; memory only |
| `amp` | `"bf16"` | `bf16` autocast on GPU, or `fp32` |
| `log_every` | `50` | steps between lines of `train_log.jsonl`: per swarm, mean loss per vector and flip rate on the training batch for the detector the loss attacks (energy for `infonce`, `infonce+energy` and the energy losses, kNN for `knn_hinge`); also elapsed time and peak GPU memory |

Cost grows with K × batch × the part of the network after `layer`. Measured on the RTX 4060 for
`resnet18_scratch_mnist`, K = 1024, batch 64 + 64: penultimate 0.1 s/step, layer4.0 1.8 s,
layer3.1 3.6 s, layer2.1 8 s, layer1.0 19 s. Lower `vec_chunk` if memory runs out (16 is enough at layer1.0).

## Experiment and data

| Key | Default | Meaning |
|---|---|---|
| `run` | `"resnet18_scratch_mnist"` | classifier checkpoint: `{resnet18_scratch, resnet18_ft, vit_small_ft}_{mnist, fmnist}` |
| `ood_sources` | `["300k"]` | OOD sources: `300k` (300K Random Images), `stl10` (STL-10 unlabeled), `coco` (COCO unlabeled2017), `openimages` (1M OpenImages subset) |
| `ood_weights` | `None` | share of each source in an OOD batch (`None` = equal) |
| `seed` | `0` | swarm initialisation, OOD sampling order, probe selection, random directions |
| `name` | `""` | output folder; empty = `<run>__<layer>__r<radius><a/r>__K<K>__<loss>__<hash of the config>` |
| `workers` | `8` | DataLoader worker processes |

## Evaluation

After training, `evaluate` walks held-out probes along three families of unit directions:

- **learned**: `eval_vectors` directions drawn at random from the swarm;
- **toward**: v_toward (sign flipped for `ood2id`);
- **random**: `n_random` random unit directions, the same for both swarms.

Each direction d is applied at every α on the grid `linspace(0, alpha_max, alpha_steps)` ∪ {r}, i.e.
the activation becomes h + α d.

| Key | Default | Meaning |
|---|---|---|
| `alpha_max` | `0.0` | end of the grid; 0 means 4 × r |
| `alpha_steps` | `41` | grid points including α = 0 (r is always added) |
| `eval_vectors` | `64` | learned directions walked (of the K) |
| `n_random` | `64` | random directions on the same sphere, the baseline |
| `n_probe_id` | `1000` | ID test probes |
| `n_probe_ood` | `1000` | OOD evaluation probes (equal share per source) |
| `bootstrap_B` | `50` | calibration resamples for σ_b |
| `tpr` | `0.95` | fraction of clean ID calibration accepted at t_D |
| `detectors` | `["energy", "msp", "maxlogit", "knn", "mahalanobis"]` | detectors evaluated |
| `knn_k` | `50` | k of the `knn` detector |
| `ref_size` | `10000` | ID reference images the `knn` and `mahalanobis` detectors are fitted on |

Reported per swarm, detector and family, in three separate groups as in the draft:

- **standard**, at α = r: flip rate, AUROC and FPR95 against the clean other side;
- **path**: first-flip length α* (first rejection, `id2ood`) or α† (first acceptance, `ood2id`),
  censoring rate (never flips within the grid), fraction currently / ever flipped at each α,
  recrossings, and the ratio of learned to random median first-flip length;
- **variability**: σ_b, the spread of the median first-flip length when t_D is recomputed on
  `bootstrap_B` bootstrap resamples of the calibration split.

Also reported: clean detection (no steering), classification accuracy of steered ID at r, and
diversity of the swarm (mean |cosine| between its vectors, cosine to v_toward).

`uv run python -m actdist.compare_steer --exps outputs/steering/<name>` puts clean and steered
detection side by side: ID steered, OOD steered, and both steered at once. Besides AUROC, FPR95 and the
fractions accepted, it prints the median detector score of each side (`med ID`, `med OOD`) and every
detector's threshold t_D, so e.g. the median Mahalanobis score of steered OOD can be read against t_D.

`uv run python -m actdist.clean_detect --config <config>` reports clean detection (no steering) for every
detector in `detectors`, on the same reference, calibration split and probes as that config's steering
run, overall and per OOD source (saved to `outputs/clean_detection/<run>.json`).

To evaluate a finished experiment with other detectors, without retraining:
`uv run python -m actdist.eval_steer outputs/steering/<name> --detectors energy msp maxlogit knn mahalanobis`.
The override is recorded in `<name>/eval_overrides.json`; `config.json` keeps the training configuration.
`compare_steer` compares every detector that has saved scores unless `--detectors` is given.

Configs written before the feature detectors existed (`base.toml`, `scratch_mnist_layer4.toml`,
`scratch_mnist_layer3_r0.1.toml`) list only the three logit detectors; add `"knn", "mahalanobis"` to
their `detectors` to evaluate those too.
