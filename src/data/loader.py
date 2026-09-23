from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .base import ContrastivePair, TextSample


def text_collate_fn(tokenizer: Any, max_length: int = 512) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """Create collation function that tokenizes raw text batches on the fly."""
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch if item["label"] is not None]
        ids = [item.get("id", str(i)) for i, item in enumerate(batch)]

        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        result: Dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "texts": texts,
            "ids": ids
        }
        if "token_type_ids" in encoded:
            result["token_type_ids"] = encoded["token_type_ids"]

        if len(labels) == len(batch):
            result["labels"] = torch.tensor(labels, dtype=torch.long)

        return result
    return collate


def contrastive_collate_fn(batch: List[ContrastivePair]) -> Dict[str, Any]:
    """Collate contrastive pairs into positive and negative text lists."""
    pos_texts = [p.positive_text for p in batch]
    neg_texts = [p.negative_text for p in batch]
    concepts = [p.concept for p in batch]
    return {
        "positive_texts": pos_texts,
        "negative_texts": neg_texts,
        "concepts": concepts,
        "pairs": batch
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False
) -> DataLoader:
    """Create standard PyTorch DataLoader with safe defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def get_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """Deterministically split a dataset into train and validation subsets."""
    val_len = int(len(dataset) * val_ratio)
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len], generator=generator)
