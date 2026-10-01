# actdist: activations, OOD detection and steering on MNIST / FashionMNIST

Classifiers (ResNet-18 from scratch, ImageNet-pretrained ResNet-18, ViT-S/16) are trained on MNIST and
FashionMNIST. Their per-neuron activation distributions are recorded and plotted, and learned
**steering vectors** added to their activations are used to fool post-hoc OOD detectors: make ID
images rejected and OOD images (unlabeled natural images) accepted.

Environment: `uv` (Python ≥ 3.12); run everything with `uv run` from this folder.

## Documentation

- **[hyperparameters.md](hyperparameters.md)**: every hyperparameter of the steering experiments, the
  detectors (energy, MSP, max-logit, kNN, Mahalanobis; threshold, metrics), the loss functions and the steering-vector swarm.
- [outputs/steering/EXPERIMENTS.md](outputs/steering/EXPERIMENTS.md): log of all steering experiments
  (running, planned, done) with their results.
- [dataset.md](dataset.md): candidate unlabeled datasets.

## Pipelines

**Classifiers and activation distributions** (`scripts/run_all.sh` runs all of it)

```
uv run python -m actdist.train   --model resnet18_scratch --dataset mnist     # -> outputs/checkpoints/
uv run python -m actdist.extract --run resnet18_scratch_mnist                  # -> outputs/activations/
uv run python -m actdist.plot heatmap --run resnet18_scratch_mnist             # -> outputs/figures/
uv run python -m actdist.plot grid    --run resnet18_scratch_mnist --select selective
```

**Unlabeled OOD data** (`src/actdist/unlabeled.py`): a loader that mixes several sources in fixed
proportions. Sources: STL-10, 300K Random Images, COCO unlabeled2017, OpenImages (1M subset,
`scripts/download_openimages.py`). Checks: `uv run python scripts/check_unlabeled.py`.

**Steering swarms** (see [hyperparameters.md](hyperparameters.md))

```
uv run python -m actdist.train_steer --config configs/steer/base.toml --set layer=layer3.1 radius=0.25
uv run python -m actdist.eval_steer outputs/steering/<name>                 # re-run the evaluation
uv run python -m actdist.compare_steer --exps outputs/steering/<name>       # detection with / without steering
uv run python -m actdist.clean_detect --config configs/steer/base.toml      # clean detection, all detectors
uv run python -m actdist.actnorm --config configs/steer/base.toml --layers all   # activation norms per layer
uv run python -m actdist.plot steer --exp outputs/steering/<name>           # figure of one experiment
uv run python -m actdist.plot steer-sweep --exps 'outputs/steering/<pattern>*'
```

## Layout

| Path | Content |
|---|---|
| `src/actdist/` | `data`, `models`, `train`, `extract`, `plot`, `unlabeled`, `steer`, `steer_config`, `losses`, `detectors`, `train_steer`, `eval_steer`, `compare_steer`, `clean_detect`, `actnorm` |
| `configs/steer/` | TOML configs of steering experiments |
| `scripts/` | `run_all.sh`, `steer_sweep.sh`, `download_openimages.py`, `check_unlabeled.py`, `make_pdf.py` |
| `data/` | MNIST, FashionMNIST, STL-10, 300K Random Images, COCO, OpenImages |
| `outputs/` | `checkpoints/`, `activations/`, `figures/`, `steering/`, `act_norms/`, `clean_detection/`, `logs/` |
