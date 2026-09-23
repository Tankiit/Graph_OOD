import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

from .base import TextSample


class TextClassificationDataset(Dataset):
    """General text dataset for classification, probing, and OOD evaluation."""

    def __init__(self, samples: List[TextSample], label_to_id: Optional[Dict[str, int]] = None):
        self.samples = samples
        self.label_to_id = label_to_id or self._build_label_map()

    def _build_label_map(self) -> Dict[str, int]:
        unique_labels = sorted(list({str(s.label) for s in self.samples if s.label is not None}))
        return {lbl: i for i, lbl in enumerate(unique_labels)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        target = self.label_to_id[str(sample.label)] if sample.label is not None and str(sample.label) in self.label_to_id else sample.label
        return {
            "text": sample.text,
            "label": target,
            "id": sample.id or str(idx),
            "metadata": sample.metadata or {}
        }

    @property
    def texts(self) -> List[str]:
        return [s.text for s in self.samples]

    @property
    def labels(self) -> List[Optional[Union[int, str]]]:
        return [s.label for s in self.samples]

    @classmethod
    def from_csv(
        cls,
        file_path: Union[str, Path],
        text_col: str = "text",
        label_col: Optional[str] = "label",
        delimiter: str = ","
    ) -> "TextClassificationDataset":
        import pandas as pd
        df = pd.read_csv(file_path, delimiter=delimiter)
        samples = []
        for idx, row in df.iterrows():
            lbl = row[label_col] if (label_col and label_col in row) else None
            samples.append(TextSample(text=str(row[text_col]), label=lbl, id=str(idx)))
        return cls(samples)

    @classmethod
    def from_jsonl(
        cls,
        file_path: Union[str, Path],
        text_key: str = "text",
        label_key: Optional[str] = "label"
    ) -> "TextClassificationDataset":
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)
                txt = obj.get(text_key, "")
                lbl = obj.get(label_key, None) if label_key else None
                samples.append(TextSample(text=txt, label=lbl, id=str(idx), metadata=obj))
        return cls(samples)

    @classmethod
    def create_synthetic_classification(
        cls,
        num_samples_per_class: int = 50,
        num_classes: int = 2
    ) -> "TextClassificationDataset":
        """Create synthetic text classification dataset for unit tests and local pipeline checks."""
        class_templates = {
            0: [
                "This movie was an absolute masterpiece with wonderful acting.",
                "I truly enjoyed the presentation and the clarity of the explanations.",
                "A brilliant achievement in modern engineering and scientific discovery.",
                "The customer support was incredibly helpful, polite, and rapid.",
                "Outstanding performance and beautiful design that exceeded expectations.",
            ],
            1: [
                "This was a terrible experience and a complete waste of time.",
                "The product broke on the first day and customer support was rude.",
                "Deeply disappointed by the poor quality and inaccurate descriptions.",
                "The movie had awful pacing, bad dialogue, and uninspired direction.",
                "Horrible service with frustrating delays and zero communication.",
            ],
            2: [
                "The quarterly financial report indicated steady baseline growth.",
                "Meeting scheduled for Tuesday afternoon at the central headquarters.",
                "Standard operating procedure requires verification of all credentials.",
                "The weather forecast predicts moderate temperatures with light cloud cover.",
                "The system update will be deployed according to standard protocol.",
            ]
        }
        samples = []
        for c in range(min(num_classes, len(class_templates))):
            templates = class_templates[c]
            for i in range(num_samples_per_class):
                text = templates[i % len(templates)] + f" (Variant {i})"
                samples.append(TextSample(text=text, label=c, id=f"class_{c}_sample_{i}"))
        return cls(samples)
