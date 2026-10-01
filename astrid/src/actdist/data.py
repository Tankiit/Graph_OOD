"""Datasets and transforms for MNIST / Fashion-MNIST."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

DATASETS = {"mnist": datasets.MNIST, "fmnist": datasets.FashionMNIST}
STATS = {"mnist": (0.1307, 0.3081), "fmnist": (0.2860, 0.3530)}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

CLASS_NAMES = {
    "mnist": [str(i) for i in range(10)],
    "fmnist": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
}


def build_transform(model_cfg: dict, train_dataset: str, train: bool) -> T.Compose:
    """Transform depends on the model's input spec and on the dataset it was trained on.

    The normalization always follows the *training* dataset so that images from any
    dataset fed to a given model go through identical preprocessing.
    """
    size, channels, pretrained = model_cfg["size"], model_cfg["channels"], model_cfg["pretrained"]
    ops = []
    if size != 28:
        ops.append(T.Resize(size, interpolation=T.InterpolationMode.BILINEAR))
    if train:
        pad = max(2, size // 14)
        ops.append(T.RandomCrop(size, padding=pad))
    if channels == 3:
        ops.append(T.Grayscale(num_output_channels=3))
    ops.append(T.ToTensor())
    if pretrained:
        ops.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    else:
        mean, std = STATS[train_dataset]
        ops.append(T.Normalize((mean,) * channels, (std,) * channels))
    return T.Compose(ops)


def get_dataset(name: str, split: str, transform):
    return DATASETS[name](DATA_DIR, train=(split != "test"), download=True, transform=transform)


def train_val_indices(n: int, seed: int = 0) -> tuple[list[int], list[int]]:
    """The 55k/5k train/val split of the official train set used to train the classifiers."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[5000:], perm[:5000]


def get_loaders(model_cfg: dict, dataset: str, bs: int, seed: int = 0, limit: int | None = None,
                num_workers: int = 8):
    """Train/val (55k/5k split of the official train set) and test loaders."""
    train_full = get_dataset(dataset, "train", build_transform(model_cfg, dataset, train=True))
    eval_full = get_dataset(dataset, "train", build_transform(model_cfg, dataset, train=False))
    test = get_dataset(dataset, "test", build_transform(model_cfg, dataset, train=False))

    train_idx, val_idx = train_val_indices(len(train_full), seed)
    if limit:
        train_idx, val_idx = train_idx[:limit], val_idx[: max(limit // 10, 100)]
        test = Subset(test, range(min(limit, len(test))))

    kw = dict(batch_size=bs, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    return (
        DataLoader(Subset(train_full, train_idx), shuffle=True, drop_last=True, **kw),
        DataLoader(Subset(eval_full, val_idx), shuffle=False, **kw),
        DataLoader(test, shuffle=False, **kw),
    )
