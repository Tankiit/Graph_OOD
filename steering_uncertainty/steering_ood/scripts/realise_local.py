"""Encode transformed CIFAR-10 test images for the realisability figure (RQ5), locally.

For each vision encoder, the first N CIFAR-10 test images are transformed at increasing strength
(contrast, brightness, Gaussian blur, Gaussian noise) and encoded with the encoder's own preprocessing.
Writes runs/modal/realise/{model}.npz with features[transform, level, image, dim]; then run
`python paper/make_figures.py runs/modal fig_realise`.
Usage: python scripts/realise_local.py --device cuda [--images runs/modal/data/images] [--models resnet18 ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
MODELS = {
    'resnet18': ('torchvision', 'resnet18', 'IMAGENET1K_V1'),
    'resnet50': ('torchvision', 'resnet50', 'IMAGENET1K_V2'),
    'vit_b16': ('torchvision', 'vit_b_16', 'IMAGENET1K_V1'),
    'dinov2_s': ('timm', 'vit_small_patch14_dinov2.lvd142m', 'DEFAULT'),
}
LEVELS = {  # level 0 = original image
    'contrast': [1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.12, 0.06],     # factor toward mean gray
    'brightness': [1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.12, 0.06],   # factor toward black
    'blur': [0.0, 0.5, 0.8, 1.2, 1.6, 2.2, 3.0, 4.0],           # Gaussian sigma in pixels
    'noise': [0.0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4],       # Gaussian std in [0,1] pixel units
}


def encode(model, images, out_dir, device, n):
    import torch
    import torchvision.transforms.functional as F
    from PIL import Image
    from torchvision import datasets
    from steering_ood.vision import build_encoder
    out = out_dir / f'{model}.npz'
    if out.exists():
        return f'{model}: exists'
    net, preprocess, meta = build_encoder(*MODELS[model], seed=7, device=device)
    test = datasets.CIFAR10(images, train=False, download=False)
    rng = np.random.default_rng(0)

    def transform(img, kind, level):
        if kind == 'contrast':
            return F.adjust_contrast(img, level)
        if kind == 'brightness':
            return F.adjust_brightness(img, level)
        if kind == 'blur':
            return img if level == 0 else F.gaussian_blur(img, kernel_size=2 * int(3 * level) + 1, sigma=level)
        a = np.asarray(img, dtype=np.float32) / 255
        if level:
            a = a + rng.normal(0, level, a.shape)
        return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))

    blocks = []
    for kind, levels in LEVELS.items():
        per_level = []
        for level in levels:
            batch = torch.stack([preprocess(transform(test[i][0].convert('RGB'), kind, level)) for i in range(n)])
            with torch.inference_mode():
                z = torch.cat([net(batch[j:j + 100].to(device)).float().cpu() for j in range(0, n, 100)])
            per_level.append(z.numpy())
        blocks.append(np.stack(per_level))
    feats = np.stack(blocks)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, features=feats, labels=np.array([test[i][1] for i in range(n)]),
                        transforms=np.array(list(LEVELS)), levels=np.array(list(LEVELS.values())),
                        metadata=json.dumps(dict(model=model, n=n, encoder=str(meta.get('model')))))
    return f'{model}: {feats.shape}'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--images', type=Path, default=PKG / 'runs' / 'modal' / 'data' / 'images')
    p.add_argument('--out', type=Path, default=PKG / 'runs' / 'modal' / 'realise')
    p.add_argument('--models', nargs='+', default=list(MODELS), choices=list(MODELS))
    p.add_argument('--device', default='cpu')
    p.add_argument('--n', type=int, default=500)
    a = p.parse_args()
    for m in a.models:
        print(encode(m, a.images, a.out, a.device, a.n), flush=True)


if __name__ == '__main__':
    main()
