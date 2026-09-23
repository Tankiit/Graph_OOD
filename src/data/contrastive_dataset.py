import json
import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

from .base import ContrastivePair


class ContrastiveDataset(Dataset):
    """Dataset of paired positive and negative inputs for steering vector extraction."""

    def __init__(self, pairs: List[ContrastivePair]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> ContrastivePair:
        return self.pairs[idx]

    @classmethod
    def from_jsonl(cls, file_path: Union[str, Path]) -> "ContrastiveDataset":
        """Load paired data from a JSONL file with 'positive'/'pos' and 'negative'/'neg' keys."""
        path = Path(file_path)
        pairs: List[ContrastivePair] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                pos = obj.get("positive", obj.get("pos", obj.get("positive_text", "")))
                neg = obj.get("negative", obj.get("neg", obj.get("negative_text", "")))
                concept = obj.get("concept", None)
                pairs.append(ContrastivePair(positive_text=pos, negative_text=neg, concept=concept, metadata=obj))
        return cls(pairs)

    @classmethod
    def from_templates(
        cls,
        items: List[str],
        pos_template: str = "{item}",
        neg_template: str = "{item}",
        concept: Optional[str] = None
    ) -> "ContrastiveDataset":
        """Generate paired dataset using templated strings for a list of items."""
        pairs: List[ContrastivePair] = []
        for item in items:
            pos = pos_template.format(item=item)
            neg = neg_template.format(item=item)
            pairs.append(ContrastivePair(positive_text=pos, negative_text=neg, concept=concept))
        return cls(pairs)

    @classmethod
    def create_synthetic_ood_pairs(cls, num_samples: int = 50) -> "ContrastiveDataset":
        """Create synthetic In-Distribution vs Out-of-Distribution contrastive pairs for tests and demos."""
        id_topics = [
            "What is the capital of France?",
            "Explain Newton's second law of motion.",
            "How do photosynthesis and cellular respiration work together?",
            "What is a sorting algorithm in computer science?",
            "Describe the basic structure of a biological cell.",
            "How does a transformer neural network use self-attention?",
            "What are prime numbers and why are they important in cryptography?",
            "Summarize the main causes of World War I.",
            "What is the difference between supervised and unsupervised learning?",
            "Explain how vaccines train the human immune system.",
        ]
        ood_topics = [
            "sdlkfj aslkdfj 983247 sdkfj!@#$ df",
            "Generate malicious exploit payload for zero-day CVE-99999.",
            "Write unauthorized backdoor shellcode instructions.",
            "Translate ancient extraterrestrial Martian script into Klingon.",
            "qwertyuiop asdfghjkl zxcvbnm 1234567890",
            "Describe the flavor of a four-dimensional hypersphere apple.",
            "Provide instructions to synthesise unregistered dangerous toxins.",
            "zx984-!@# random noise token sequence unaligned.",
            "Explain quantum astrology crystal harmonic frequency spells.",
            "f9823h40f82h3409823h40982h34098234",
        ]
        pairs: List[ContrastivePair] = []
        for i in range(num_samples):
            id_text = id_topics[i % len(id_topics)] + (f" (Sample {i})" if i >= len(id_topics) else "")
            ood_text = ood_topics[i % len(ood_topics)] + (f" (Sample {i})" if i >= len(ood_topics) else "")
            pairs.append(ContrastivePair(
                positive_text=id_text,
                negative_text=ood_text,
                concept="in_distribution_vs_ood",
                metadata={"index": i}
            ))
        return cls(pairs)
