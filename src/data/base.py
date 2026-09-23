from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset


@dataclass
class TextSample:
    """Dataclass representing a text classification/generation sample."""
    text: str
    label: Optional[Union[int, str]] = None
    id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ContrastivePair:
    """Dataclass representing a positive/negative contrastive pair for steering vector extraction."""
    positive_text: str
    negative_text: str
    concept: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseDataset(Dataset):
    """Base dataset interface for representation and steering experiments."""

    def __init__(self, samples: List[Any]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Any:
        return self.samples[idx]
