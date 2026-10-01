"""Sanity checks for the balanced multi-source unlabeled loader (src/actdist/unlabeled.py).

    uv run python scripts/check_unlabeled.py                 # STL-10 + two synthetic folders
    uv run python scripts/check_unlabeled.py --folder data/coco/unlabeled2017 --folder data/openimages/train \
                                             --npy data/300K_random_images.npy
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from actdist.unlabeled import NORM, BalancedBatchSampler, build_unlabeled_loader, single, two_crop

ROOT = Path(__file__).resolve().parents[1] / "outputs"


def make_fake_folders(base: Path):
    """A large-image RGB folder (ImageNet-like, 1000 imgs) and a tiny grayscale one (50 imgs, 20x20)."""
    rng = np.random.default_rng(0)
    big, tiny = base / "fake_big", base / "fake_tiny"
    if not big.exists():
        big.mkdir(parents=True)
        for i in range(1000):
            h, w = rng.integers(300, 500, size=2)
            img = (rng.random((h, w, 3)) * [255, 128, 64]).astype(np.uint8)
            Image.fromarray(img).save(big / f"{i:04d}.jpg", quality=80)
    if not tiny.exists():
        (tiny / "sub").mkdir(parents=True)
        for i in range(50):
            Image.fromarray(rng.integers(0, 255, (20, 20), dtype=np.uint8), "L").save(tiny / "sub" / f"{i}.png")
    return big, tiny


def source_of(batch, offsets):
    return np.searchsorted(offsets, np.asarray(batch), side="right") - 1


def check_composition(sizes):
    weights, bs, steps = [1, 1, 2], 250, 200
    s = BalancedBatchSampler(sizes, bs, steps, weights)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    quota = bs * np.array(weights) / sum(weights)
    totals = np.zeros(3)
    for b, batch in enumerate(s):
        assert len(batch) == bs
        c = np.bincount(source_of(batch, offsets), minlength=3)
        assert (np.abs(c - quota) < 1).all(), (b, c, quota)
        totals += c
        assert (np.abs(totals - quota * (b + 1)) <= 1).all()
    print(f"[ok] composition: quotas {quota} per batch; totals after {steps} batches {totals}")


def check_cycling():
    sizes, bs, steps = [100_000, 1000], 250, 400
    s = BalancedBatchSampler(sizes, bs, steps)
    counts = Counter(i - sizes[0] for batch in s for i in batch if i >= sizes[0])
    assert len(counts) == 1000 and set(counts.values()) == {50}, Counter(counts.values())
    print("[ok] cycling: each of the 1000 small-source images drawn exactly 50 times")


def check_reproducibility(sizes):
    a = BalancedBatchSampler(sizes, 64, 20, seed=3)
    b = BalancedBatchSampler(sizes, 64, 20, seed=3)
    assert list(a) == list(b)
    b.set_epoch(1)
    assert list(a) != list(b)
    print("[ok] reproducibility: same (seed, epoch) -> same order; new epoch -> new order")


def check_loader(sources, mode, size, views, name, n_batches=100):
    dl = build_unlabeled_loader(sources, mode=mode, views=views, batch_size=128, steps_per_epoch=n_batches)
    shapes, sid_counts = set(), Counter()
    t0 = time.time()
    for (v1, v2), sid in dl:
        shapes.add((tuple(v1.shape), tuple(v2.shape)))
        sid_counts.update(sid.tolist())
    dt = time.time() - t0
    assert len(shapes) == 1, shapes
    print(f"[ok] {name}: shapes {shapes.pop()}; per-source totals "
          f"{ {dl.source_names[k]: v for k, v in sorted(sid_counts.items())} }; "
          f"{n_batches} batches in {dt:.1f}s ({n_batches * 128 / dt:.0f} img/s)")
    return dl


def save_grid(dl, mode, out):
    (v1, _), sid = next(iter(dl))
    mean, std = (np.array(x).reshape(-1, 1, 1) for x in NORM[mode])
    order = np.argsort(sid.numpy(), kind="stable")[:48]
    fig, axes = plt.subplots(6, 8, figsize=(10, 8))
    for ax, i in zip(axes.flat, order):
        img = np.clip(v1[i].numpy() * std + mean, 0, 1)
        ax.imshow(img[0] if mode == "L" else img.transpose(1, 2, 0), cmap="gray" if mode == "L" else None)
        ax.set_title(dl.source_names[sid[i]], fontsize=7)
        ax.axis("off")
    fig.suptitle("One balanced batch (first view), sorted by source", x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", action="append", default=[], help="extra image folder(s) to mix in")
    p.add_argument("--npy", action="append", default=[], help="extra uint8 [N, H, W, C] .npy array(s) to mix in")
    args = p.parse_args()

    big, tiny = make_fake_folders(ROOT / "fake_folders")
    sources = [dict(kind="stl10", canonical_size=96),
               dict(kind="folder", root=big, name="fake_big", canonical_size=128),
               dict(kind="folder", root=tiny, name="fake_tiny", canonical_size=32)]
    sources += [dict(kind="folder", root=f, canonical_size=128) for f in args.folder]
    sources += [dict(kind="npy", path=f, canonical_size=32) for f in args.npy]

    check_composition([100_000, 1000, 50])
    check_cycling()
    check_reproducibility([100_000, 1000, 50])
    dl = check_loader(sources, "L", 32, two_crop(32, "L"), "gray 32px (ResNet)")
    save_grid(dl, "L", ROOT / "figures" / "unlabeled_batch_L32.png")
    dl = check_loader(sources, "RGB", 224, two_crop(224, "RGB"), "RGB 224px (ViT)", n_batches=30)
    save_grid(dl, "RGB", ROOT / "figures" / "unlabeled_batch_RGB224.png")
    single(32, "L")  # preset builds


if __name__ == "__main__":
    main()
