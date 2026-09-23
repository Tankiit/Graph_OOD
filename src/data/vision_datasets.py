from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset


class SyntheticVisionDataset(Dataset):
    """Synthetic image dataset with customizable image shapes and class clusters."""

    def __init__(
        self,
        num_samples: int = 100,
        num_classes: int = 10,
        image_shape: Tuple[int, int, int] = (3, 32, 32),
        seed: int = 42
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_shape = image_shape

        # Generate clustered synthetic data
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.images = torch.randn(num_samples, *image_shape)
        # Shift means per class for separable features
        for c in range(num_classes):
            mask = (self.labels == c)
            self.images[mask] += (c / num_classes) * 1.5

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "images": self.images[idx],
            "label": self.labels[idx].item(),
            "id": f"syn_vis_{idx}"
        }


def get_torchvision_dataset(
    name: str,
    root: str = "./data",
    train: bool = True,
    transform: Optional[Any] = None,
    download: bool = True
) -> Dataset:
    """Helper to instantiate standard torchvision datasets (CIFAR10, CIFAR100, SVHN, etc.)."""
    import torchvision.datasets as tv_datasets
    name_lower = name.lower()

    if name_lower == "cifar10":
        return tv_datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
    elif name_lower == "cifar100":
        return tv_datasets.CIFAR100(root=root, train=train, transform=transform, download=download)
    elif name_lower == "svhn":
        split = "train" if train else "test"
        return tv_datasets.SVHN(root=root, split=split, transform=transform, download=download)
    elif name_lower == "mnist":
        return tv_datasets.MNIST(root=root, train=train, transform=transform, download=download)
    else:
        raise ValueError(f"Unsupported torchvision dataset: {name}")
