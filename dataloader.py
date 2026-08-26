"""dataloader.py

PyTorch DataLoader helpers for Hugging Face datasets.

This file now supports loading *specific AbstentionBench tasks* (e.g. `gsm8k`,
`gpqa`, `falseqa`, ...), and can create DataLoaders for both `train` and `test`.

See also: `dataload.py` for a lightweight, non-torch CLI inspection tool.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

from torch.utils.data import DataLoader, Dataset

from dataload import load_abstentionbench_task

# AbstentionBench task list for OOD detection experiments
ABSTENTION_BENCH_TASKS = {
    "Bias Benchmark for QA (BBQ) [subsampled]": "bbq",
    "BIG-Bench Disambiguate": "big_bench_disambiguate",
    "BIG-Bench Known Unknowns": "big_bench_known_unknowns",
    "CoCoNot": "coconot",
    "FalseQA": "falseqa",
    "FreshQA": "freshqa",
    "GPQA": "gpqa",
    "GSM8K": "gsm8k",
    "Known Unknown Questions (KUQ) [subsampled]": "kuq"
}

# Additional task metadata for experiment management
TASK_METADATA = {
    "bbq": {
        "full_name": "Bias Benchmark for QA (BBQ) [subsampled]",
        "description": "Bias benchmark for question answering with subsampled data",
        "category": "bias_detection",
        "priority": "medium"
    },
    "big_bench_disambiguate": {
        "full_name": "BIG-Bench Disambiguate",
        "description": "BIG-bench disambiguation tasks",
        "category": "disambiguation",
        "priority": "medium"
    },
    "big_bench_known_unknowns": {
        "full_name": "BIG-Bench Known Unknowns",
        "description": "BIG-bench known/unknown question tasks",
        "category": "knowledge_detection",
        "priority": "high"
    },
    "coconot": {
        "full_name": "CoCoNot",
        "description": "Context-aware reasoning tasks",
        "category": "reasoning",
        "priority": "medium"
    },
    "falseqa": {
        "full_name": "FalseQA",
        "description": "False question answering detection",
        "category": "factuality",
        "priority": "high"
    },
    "freshqa": {
        "full_name": "FreshQA",
        "description": "Fresh knowledge question answering",
        "category": "knowledge_cutoff",
        "priority": "high"
    },
    "gpqa": {
        "full_name": "GPQA",
        "description": "Graduate-level Google-proof Q&A",
        "category": "expert_reasoning",
        "priority": "high"
    },
    "gsm8k": {
        "full_name": "GSM8K",
        "description": "Grade school math benchmark",
        "category": "mathematical_reasoning",
        "priority": "high"
    },
    "kuq": {
        "full_name": "Known Unknown Questions (KUQ) [subsampled]",
        "description": "Known vs unknown question detection with subsampling",
        "category": "knowledge_detection",
        "priority": "high"
    }
}

class AbstentionBenchTaskDataset(Dataset):
    """PyTorch Dataset wrapper for a single AbstentionBench task + split."""

    def __init__(
        self,
        task: str,
        split: str = "train",
        *,
        repo_id: str = "facebook/AbstentionBench",
        trust_remote_code: bool = True,
    ):
        self.dataset = load_abstentionbench_task(
            task,
            split,
            repo_id=repo_id,
            trust_remote_code=trust_remote_code,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]

def custom_collate_fn(batch):
    """
    A basic custom collate function to handle the list of dictionaries from the dataset.
    This groups similar dictionary keys into lists.
    """
    keys = batch[0].keys()
    collated = {key: [item[key] for item in batch] for key in keys}
    
    # You might want to convert features to tensors here if they're already numerical.
    # For example:
    # if 'label' in collated:
    #     collated['label'] = torch.tensor(collated['label'])
        
    return collated

def get_abstention_dataloader(
    *,
    task: str,
    split: str = "train",
    batch_size: int = 32,
    shuffle: bool = True,
    repo_id: str = "facebook/AbstentionBench",
    trust_remote_code: bool = True,
) -> DataLoader:
    """Create a DataLoader for an AbstentionBench task + split."""

    dataset = AbstentionBenchTaskDataset(
        task=task,
        split=split,
        repo_id=repo_id,
        trust_remote_code=trust_remote_code,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=custom_collate_fn,
    )


def get_abstention_train_test_dataloaders(
    *,
    task: str,
    batch_size: int = 32,
    shuffle_train: bool = True,
    repo_id: str = "facebook/AbstentionBench",
    trust_remote_code: bool = True,
) -> Dict[str, DataLoader]:
    """Convenience helper returning `train` and `test` loaders (when available)."""

    loaders: Dict[str, DataLoader] = {}
    for split, shuffle in (("train", shuffle_train), ("test", False)):
        try:
            loaders[split] = get_abstention_dataloader(
                task=task,
                split=split,
                batch_size=batch_size,
                shuffle=shuffle,
                repo_id=repo_id,
                trust_remote_code=trust_remote_code,
            )
        except Exception:
            # Some tasks may not expose both splits; keep it best-effort.
            continue
    return loaders

def get_all_tasks() -> Dict[str, str]:
    """Return all available AbstentionBench tasks."""
    return ABSTENTION_BENCH_TASKS.copy()

def get_task_list() -> List[str]:
    """Return list of task identifiers for easy iteration."""
    return list(ABSTENTION_BENCH_TASKS.values())

def get_task_metadata(task_id: str) -> Dict[str, Any]:
    """Get metadata for a specific task."""
    return TASK_METADATA.get(task_id, {})

def get_high_priority_tasks() -> List[str]:
    """Return list of high-priority task identifiers."""
    return [task_id for task_id, meta in TASK_METADATA.items()
            if meta.get("priority") == "high"]

def get_tasks_by_category(category: str) -> List[str]:
    """Return list of task identifiers by category."""
    return [task_id for task_id, meta in TASK_METADATA.items()
            if meta.get("category") == category]

if __name__ == "__main__":
    # Example usage:
    task = "gsm8k"
    print(f"Loading AbstentionBench task='{task}' train/test...")
    loaders = get_abstention_train_test_dataloaders(task=task, batch_size=4)

    for split, dl in loaders.items():
        batch = next(iter(dl))
        print(f"\n[{split}] batch keys:")
        print(batch.keys())

    # Example: Show all available tasks
    print("\n=== Available AbstentionBench Tasks ===")
    for full_name, task_id in ABSTENTION_BENCH_TASKS.items():
        print(f"{full_name}: {task_id}")
