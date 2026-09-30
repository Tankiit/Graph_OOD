# Vision/generalization validation (v0.2)

The full suite passed: **21 tests**, including the 11 existing scientific/NLP
checks plus vision manifest, feature-extraction and integration checks.

Executed:

- The old `steering_nlp` imports/CLI and the new `steering_ood` CLI.
- ImageFolder splitting into disjoint ID roles and separately named OOD sources.
- Duplicate-pixel removal, changed-file detection and class-name mismatch checks.
- An actual custom CNN -> cache -> PCA -> Skorch -> PyTorch-OOD -> crossed runner.
- Actual TorchVision ResNet18 and timm ViT-tiny forward passes with frozen,
  randomly initialized weights; output shape/pooling and model-state hashes.
- `scripts/vision_smoke.py`: generated image files through ResNet18,
  fixed PCA, Skorch training, four OOD adapters, static evaluation and crossed E2.
  Included results are in `examples/vision_validation/`.
- Per-OOD-dataset metrics and modality-neutral classification accuracy.
- External-cache import, compatible with existing vision/text embeddings.
- Installed timm registry verification for the documented `resnet50.a1_in1k`
  and available `vit_base_patch14_dinov2.lvd142m` model names.

Not executed:

- Full CIFAR10/CIFAR100/SVHN downloads or benchmark evaluation.
- Pretrained encoder downloads in this validation run.
- DINOv2 forward passes or TorchVision ViT-B/16 forward passes.
- Input-supported image corruption paths or semantic-OOD validation.

The random-weights image runs are software integration evidence only. They do
not support model-performance claims. Checkpoint-specific pretrained preprocessing
is implemented using the official library APIs; the full benchmark is left as an
explicit runnable workflow in README.

The earlier `VALIDATION.md` and `examples/validation/` preserve the v0.1 handoff's
results and provenance. They are historical outputs, not newly rerun benchmarks.
The new package adds vision inputs without changing its calibration/path protocol.
