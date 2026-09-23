import pytest
import torch
from src.data.contrastive_dataset import ContrastiveDataset
from src.data.text_datasets import TextClassificationDataset
from src.data.vision_datasets import SyntheticVisionDataset
from src.data.loader import create_dataloader, contrastive_collate_fn, get_train_val_split


def test_contrastive_dataset():
    ds = ContrastiveDataset.create_synthetic_ood_pairs(num_samples=10)
    assert len(ds) == 10
    sample = ds[0]
    assert sample.positive_text is not None
    assert sample.negative_text is not None

    loader = create_dataloader(ds, batch_size=4, shuffle=False, collate_fn=contrastive_collate_fn)
    batch = next(iter(loader))
    assert len(batch["positive_texts"]) == 4
    assert len(batch["negative_texts"]) == 4


def test_text_classification_dataset():
    ds = TextClassificationDataset.create_synthetic_classification(num_samples_per_class=10, num_classes=2)
    assert len(ds) == 20
    assert len(ds.texts) == 20
    assert len(ds.labels) == 20

    train_ds, val_ds = get_train_val_split(ds, val_ratio=0.2, seed=42)
    assert len(train_ds) == 16
    assert len(val_ds) == 4


def test_synthetic_vision_dataset():
    vds = SyntheticVisionDataset(num_samples=20, num_classes=5, image_shape=(3, 16, 16))
    assert len(vds) == 20
    item = vds[0]
    assert item["images"].shape == (3, 16, 16)
    assert 0 <= item["label"] < 5
