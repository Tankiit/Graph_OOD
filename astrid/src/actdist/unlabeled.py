"""Balanced multi-source loader of unlabeled images for self-supervised training.

Every batch holds each source in a fixed proportion (equal by default), whatever the source sizes.
Sources are cycled independently (reshuffled when exhausted), and an epoch is a fixed number of steps.

    sources = [dict(kind="stl10", canonical_size=96),
               dict(kind="npy", path=DATA_DIR / "300K_random_images.npy", canonical_size=32),
               dict(kind="folder", root=DATA_DIR / "coco/unlabeled2017", canonical_size=128),
               dict(kind="folder", root=DATA_DIR / "openimages/train", canonical_size=128)]
    dl = build_unlabeled_loader(sources, mode="L", views=two_crop(32, "L"),
                                batch_size=256, steps_per_epoch=1000)
    for epoch in range(E):
        dl.batch_sampler.set_epoch(epoch)
        for (v1, v2), source_id in dl: ...

Image sizes are harmonized in two stages: a per-source `pre` transform brings each source to its
canonical resolution (resize short side + center crop), then the shared `views` transform produces
the final `target_size` crops, so every sample in a batch has the same shape.
"""

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler, Subset
from torchvision import datasets, transforms as T

from .data import DATA_DIR, IMAGENET_MEAN, IMAGENET_STD

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
NORM = {"RGB": (IMAGENET_MEAN, IMAGENET_STD), "L": ((0.449,), (0.226,))}


# ---------------------------------------------------------------- sources


class UnlabeledSource(Dataset):
    """Base class: returns a PIL image converted to `mode`, no label."""

    name: str

    def __init__(self, mode: str):
        self.mode = mode

    def load(self, i: int) -> Image.Image:
        raise NotImplementedError

    def __getitem__(self, i: int) -> Image.Image:
        return self.load(i).convert(self.mode)


class STL10Unlabeled(UnlabeledSource):
    """STL-10 'unlabeled' split: 100k RGB images, 96x96."""

    def __init__(self, mode: str, root: str | Path = DATA_DIR):
        super().__init__(mode)
        self.name = "stl10"
        self.data = datasets.STL10(root, split="unlabeled", download=True).data  # uint8 [N, 3, 96, 96]

    def __len__(self):
        return len(self.data)

    def load(self, i):
        return Image.fromarray(np.transpose(self.data[i], (1, 2, 0)))


class ImageFolderUnlabeled(UnlabeledSource):
    """Every image file under `root` (recursive). The sorted file list is cached in DATA_DIR/.filelists.

    The cache is rebuilt when `root` or one of its direct subfolders was modified after it was
    written (files added or removed deeper than that are not detected).
    """

    def __init__(self, mode: str, root: str | Path, name: str | None = None, exts=IMG_EXTS):
        super().__init__(mode)
        root = Path(root).resolve()
        self.name = name or root.name
        cache = DATA_DIR / ".filelists" / f"{hashlib.md5(str(root).encode()).hexdigest()}.txt"
        if cache.exists() and _newest_mtime(root) <= cache.stat().st_mtime:
            files = cache.read_text().splitlines()
        else:
            files = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.suffix.lower() in exts)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text("\n".join(files))
        if not files:
            raise FileNotFoundError(f"no images found under {root}")
        self.root, self.files = root, files

    def __len__(self):
        return len(self.files)

    def load(self, i):
        with Image.open(self.root / self.files[i]) as im:
            im.load()
            return im


def _newest_mtime(root: Path) -> float:
    return max([root.stat().st_mtime] + [d.stat().st_mtime for d in root.iterdir() if d.is_dir()])


class NpyUnlabeled(UnlabeledSource):
    """A uint8 [N, H, W, C] (or [N, H, W]) .npy array, e.g. 300K Random Images (Hendrycks et al.).

    The array is memory-mapped lazily, so each DataLoader worker opens its own map instead of
    receiving a pickled copy of the whole array.
    """

    def __init__(self, mode: str, path: str | Path = DATA_DIR / "300K_random_images.npy", name: str | None = None):
        super().__init__(mode)
        self.path = Path(path)
        self.name = name or self.path.stem
        self._data = None
        self._len = len(np.load(self.path, mmap_mode="r"))

    def __len__(self):
        return self._len

    def __getstate__(self):
        return {**self.__dict__, "_data": None}

    def load(self, i):
        if self._data is None:
            self._data = np.load(self.path, mmap_mode="r")
        return Image.fromarray(np.asarray(self._data[i]))


SOURCES = {"stl10": STL10Unlabeled, "folder": ImageFolderUnlabeled, "npy": NpyUnlabeled}


SPLITS = {"eval": 0, "dev": 1}  # index i goes to the split of i % 10; every other residue is "train"


def split_indices(n: int, split: str | None) -> np.ndarray:
    """Disjoint, deterministic index sets of a source: eval (i % 10 == 0), dev (== 1), train (the rest)."""
    idx = np.arange(n)
    if split is None:
        return idx
    if split == "train":
        return idx[idx % 10 >= len(SPLITS)]
    return idx[idx % 10 == SPLITS[split]]


class SourceView(Dataset):
    """Source + per-source `pre` transform (to canonical size) + shared `views` transform.

    `split` restricts the view to one of the disjoint index sets of `split_indices`.
    """

    def __init__(self, source: UnlabeledSource, source_id: int, canonical_size: int, views,
                 split: str | None = None):
        self.source, self.source_id, self.views = source, source_id, views
        self.pre = T.Compose([T.Resize(canonical_size), T.CenterCrop(canonical_size)])
        self.indices = split_indices(len(source), split)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.views(self.pre(self.source[int(self.indices[i])])), self.source_id


# ---------------------------------------------------------------- views (SSL augmentations)


def _to_tensor(mode: str):
    mean, std = NORM[mode]
    return [T.ToTensor(), T.Normalize(mean, std)]


def single(target_size: int, mode: str):
    """One un-augmented view."""
    return T.Compose([T.Resize(target_size), T.CenterCrop(target_size), *_to_tensor(mode)])


class PILGaussianBlur:
    """Gaussian blur with random sigma; PIL's box approximation is much faster than T.GaussianBlur."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.sigma)))


class TwoCrops:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return self.transform(img), self.transform(img)


def two_crop(target_size: int, mode: str, scale=(0.2, 1.0)):
    """SimCLR-style pair of augmented views."""
    color = ([T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), T.RandomGrayscale(p=0.2)]
             if mode == "RGB" else [T.RandomApply([T.ColorJitter(0.4, 0.4)], p=0.8)])
    aug = T.Compose([
        T.RandomResizedCrop(target_size, scale=scale),
        T.RandomHorizontalFlip(),
        *color,
        T.RandomApply([PILGaussianBlur((0.1, 2.0) if target_size >= 96 else (0.1, 0.5))], p=0.5),
        *_to_tensor(mode),
    ])
    return TwoCrops(aug)


# ---------------------------------------------------------------- balanced sampling


class BalancedBatchSampler(Sampler[list[int]]):
    """Yields batches of global indices into a ConcatDataset with fixed per-source proportions.

    Each source has its own infinite stream of shuffled permutations. Per-batch quotas are
    `batch_size * w_k`, with leftover slots handed out by accumulated deficit, so every batch has
    exactly `batch_size` items, per-batch counts are within 1 of the quota, and running totals stay
    within 1 of the exact ratio.
    """

    def __init__(self, sizes: list[int], batch_size: int, steps_per_epoch: int,
                 weights: list[float] | None = None, seed: int = 0):
        weights = np.ones(len(sizes)) if weights is None else np.asarray(weights, dtype=float)
        if len(weights) != len(sizes) or (weights <= 0).any():
            raise ValueError("need one positive weight per source")
        self.sizes, self.bs, self.steps, self.seed = sizes, batch_size, steps_per_epoch, seed
        self.w = weights / weights.sum()
        self.offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.steps

    def _stream(self, k: int, rng: np.random.Generator):
        while True:
            yield from rng.permutation(self.sizes[k])

    def __iter__(self):
        rngs = [np.random.default_rng([self.seed, self.epoch, k]) for k in range(len(self.sizes))]
        streams = [self._stream(k, r) for k, r in enumerate(rngs)]
        mix_rng = np.random.default_rng([self.seed, self.epoch, len(self.sizes)])
        carry = np.zeros(len(self.sizes))
        for _ in range(self.steps):
            ideal = self.bs * self.w + carry
            counts = np.floor(ideal).astype(int)
            for k in np.argsort(-(ideal - counts), kind="stable")[: self.bs - counts.sum()]:
                counts[k] += 1
            carry = ideal - counts
            batch = np.concatenate([self.offsets[k] + np.fromiter(streams[k], int, counts[k])
                                    for k in range(len(self.sizes))])
            yield mix_rng.permutation(batch).tolist()


# ---------------------------------------------------------------- builder


def _views(sources: list[dict], mode: str, views, split: str | None) -> list[SourceView]:
    parts = []
    for k, cfg in enumerate(sources):
        cfg = dict(cfg)
        kind, canonical = cfg.pop("kind"), cfg.pop("canonical_size")
        parts.append(SourceView(SOURCES[kind](mode=mode, **cfg), k, canonical, views, split))
    return parts


def build_unlabeled_loader(sources: list[dict], mode: str, views, batch_size: int, steps_per_epoch: int,
                           weights: list[float] | None = None, seed: int = 0, num_workers: int = 8,
                           split: str | None = None):
    """`sources`: dicts with `kind` (see SOURCES), `canonical_size`, and the source's own kwargs
    (e.g. `root`, `name`). Batches are `(views, source_id)`. `split`: see `split_indices`."""
    parts = _views(sources, mode, views, split)
    sampler = BalancedBatchSampler([len(p) for p in parts], batch_size, steps_per_epoch, weights, seed)
    loader = DataLoader(ConcatDataset(parts), batch_sampler=sampler, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    loader.source_names = [p.source.name for p in parts]
    return loader


def build_unlabeled_eval_set(sources: list[dict], mode: str, views, n: int, split: str = "eval",
                             seed: int = 0) -> Dataset:
    """A fixed set of `n` images, an equal share drawn (without replacement, seeded) from each source."""
    parts = _views(sources, mode, views, split)
    rng = np.random.default_rng(seed)
    shares = [n // len(parts) + (k < n % len(parts)) for k in range(len(parts))]
    subsets = [Subset(p, np.sort(rng.choice(len(p), min(m, len(p)), replace=False)).tolist())
               for p, m in zip(parts, shares)]
    ds = ConcatDataset(subsets)
    ds.source_names = [p.source.name for p in parts]
    return ds


# ---------------------------------------------------------------- named sources

NAMED_SOURCES = {  # canonical_size: native resolution for the fixed-size sets, a working size for photo folders
    "stl10": dict(kind="stl10", canonical_size=96),
    "300k": dict(kind="npy", path=DATA_DIR / "300K_random_images.npy", name="300k", canonical_size=32),
    "coco": dict(kind="folder", root=DATA_DIR / "coco" / "unlabeled2017", name="coco", canonical_size=128),
    "openimages": dict(kind="folder", root=DATA_DIR / "openimages" / "train", name="openimages", canonical_size=128),
}


def named_sources(names: list[str], min_size: int = 0) -> list[dict]:
    """Source dicts by name; photo folders are brought to at least `min_size` (the model's input size)."""
    out = []
    for n in names:
        cfg = dict(NAMED_SOURCES[n])
        if cfg["kind"] == "folder":
            cfg["canonical_size"] = max(cfg["canonical_size"], min_size)
        out.append(cfg)
    return out
