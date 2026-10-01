# Unlabeled image datasets for self-supervised pre-training

Candidate datasets for self-supervised pretraining (labels ignored or absent), for the models
in this project (ResNet-18 at 28–32px grayscale, ViT-S/16 at 224px).

## Reference: what ViT was trained on

- The checkpoint used here, `vit_small_patch16_224.augreg_in21k_ft_in1k` (timm), was **supervised**
  pretraining on **ImageNet-21k** (~14M images, ~21k classes), then fine-tuned on **ImageNet-1k**
  (1.28M images, 1000 classes).
- The original ViT (Dosovitskiy et al., 2020) also used **JFT-300M**, which is Google-internal and not released.
- Self-supervised ViTs mostly use ImageNet-1k with the labels dropped: DINO, MAE, iBOT.
- DINOv2 used **LVD-142M**, a curated dataset that was never released.

## Datasets

| Dataset | Size | Resolution | Access | Notes |
|---|---|---|---|---|
| **STL-10 "unlabeled" split** | 100k | 96×96 | `torchvision.datasets.STL10(split="unlabeled")` | Designed for unsupervised learning; small enough for this setup |
| **ImageNet-1k**, labels ignored | 1.28M | variable (~400px) | Registration at image-net.org | The standard self-supervised benchmark (DINO, MAE, SimCLR) |
| **Downsampled ImageNet** (Chrabaszcz et al., 2017) | 1.28M | 32×32 or 64×64 | image-net.org (downsampled variants) | Good match for 28–32px inputs |
| **300K Random Images** (Hendrycks et al.) | 300k | 32×32 | Outlier Exposure repo | Replacement for 80M Tiny Images; used in the OOD literature (Outlier Exposure, OpenOOD) |
| **Places365** | 1.8M | 256×256 | `torchvision.datasets.Places365` | Scenes rather than objects |
| **COCO unlabeled2017** | 123k | variable | cocodataset.org | Complex multi-object scenes |
| **OpenImages** | ~9M | variable | storage.googleapis.com/openimages | CC-BY licensed |
| **ImageNet-21k-P** (winter-21, processed) | ~11–13M | variable | Alibaba-MIIL release; needs ImageNet registration | Only if ImageNet-21k scale is needed |
| **DataComp** (CommonPool / DataComp-1B) | 12.8B / 1.4B URLs | variable | datacomp.ai | Web scale, URL lists to download yourself |
| **COYO-700M** | 747M URLs | variable | Hugging Face `kakaobrain/coyo-700m` | Web scale, URL lists to download yourself |
| **Re-LAION-5B** | ~5.5B URLs | variable | laion.ai | Cleaned re-release (2024); do **not** use the original LAION-5B, which was taken down in 2023 |
| **FractalDB** (Kataoka et al., 2020) | 1k–10k categories, generated | any | Generated procedurally | Synthetic; no real images |
| **Procedural noise** ("Learning to See by Looking at Noise", Baradad et al., 2021) | unlimited | any | Generator code released | Synthetic "random" images; useful as a control for what natural-image pretraining adds |

## Do not use

- **80M Tiny Images**: withdrawn in 2020 because of offensive content.
- **Original LAION-5B / LAION-400M**: taken down in 2023; use Re-LAION-5B instead.

## Recommendations

- **Small-scale experiments** (ResNet on 28–32px grayscale): use the **STL-10 unlabeled** split or
  **32/64px downsampled ImageNet**, converted to grayscale and resized. They train fast on the local
  RTX 4060 and match the look of the inputs.
- **Self-supervised pretraining of the ViT**: use **ImageNet-1k without labels**, the DINO/MAE setup.
- **Later OOD work**: keep the pretraining data away from anything close to MNIST or Fashion-MNIST,
  meaning handwritten characters (e.g. EMNIST) or clothing photos. Otherwise the "OOD" set leaks into
  pretraining.

*This list was compiled from memory on 2026-09-24. Check download links and licences before use.*
