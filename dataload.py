"""dataload.py

Small, practical dataset-loading examples for a few AbstentionBench tasks.

The tasks requested here are commonly available via Hugging Face Datasets as
either:
  1) separate configs under `facebook/AbstentionBench`, or
  2) a single table that includes a `dataset`/`task` column.

This script tries (1) first, then falls back to (2) by filtering.

Datasets (task ids):
  - BIG-Bench Disambiguate: `big_bench_disambiguate`
  - BIG-Bench Known Unknowns: `big_bench_known_unknowns`
  - CoCoNot: `coconot`
  - FalseQA: `falseqa`
  - FreshQA: `freshqa`
  - GPQA: `gpqa`
  - GSM8K: `gsm8k`
  - Known Unknown Questions (KUQ) [subsampled]: `kuq`

Examples:
  python dataload.py --task gsm8k --split test --n 3
  python dataload.py --task gpqa --split train --show-keys
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TASKS: Tuple[str, ...] = (
    "big_bench_disambiguate",
    "big_bench_known_unknowns",
    "coconot",
    "falseqa",
    "freshqa",
    "gpqa",
    "gsm8k",
    "kuq",
)


def _guess_task_column(keys: Sequence[str]) -> Optional[str]:
    for candidate in ("dataset", "task", "subset", "source"):
        if candidate in keys:
            return candidate
    return None


def _extract_text_fields(example: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort extraction of prompt/answer-ish fields.

    Different tasks expose different schemas. This attempts to pick the most
    human-readable fields for quick inspection.
    """

    prompt_keys = (
        "question",
        "prompt",
        "input",
        "query",
        "instruction",
        "problem",
    )
    answer_keys = (
        "answer",
        "target",
        "output",
        "label",
        "completion",
        "response",
    )
    choices_keys = ("choices", "options", "candidates", "answers")

    prompt = next((example[k] for k in prompt_keys if k in example), None)
    answer = next((example[k] for k in answer_keys if k in example), None)
    choices = next((example[k] for k in choices_keys if k in example), None)

    extracted: Dict[str, Any] = {}
    if prompt is not None:
        extracted["prompt"] = prompt
    if choices is not None:
        extracted["choices"] = choices
    if answer is not None:
        extracted["answer"] = answer

    # If nothing matched, just return the first few key/values.
    if not extracted:
        for k in list(example.keys())[:6]:
            extracted[k] = example[k]
    return extracted


def load_abstentionbench_task(
    task: str,
    split: str,
    *,
    repo_id: str = "facebook/AbstentionBench",
    trust_remote_code: bool = True,
):
    """Load a single AbstentionBench task split.

    Returns a `datasets.Dataset` (or similar) object.
    """

    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Expected one of: {', '.join(TASKS)}")

    try:
        import datasets  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: `datasets`. Install via `pip install datasets`."
        ) from e

    # Attempt: each task is a named config.
    try:
        return datasets.load_dataset(
            repo_id,
            task,
            split=split,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        pass

    # Fallback: a single split containing multiple tasks.
    ds = datasets.load_dataset(
        repo_id,
        split=split,
        trust_remote_code=trust_remote_code,
    )
    task_col = _guess_task_column(getattr(ds, "column_names", []))
    if task_col is None:
        raise RuntimeError(
            "Could not find a task column to filter by. "
            "Expected something like 'dataset' or 'task'."
        )

    filtered = ds.filter(lambda ex: ex.get(task_col) == task)
    if len(filtered) == 0:
        raise RuntimeError(
            f"Loaded `facebook/AbstentionBench` split='{split}', but couldn't find "
            f"any rows where {task_col} == '{task}'."
        )
    return filtered


def make_torch_dataloader(
    ds,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
):
    """Optional: wrap a HF dataset as a PyTorch DataLoader."""

    try:
        from torch.utils.data import DataLoader  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: `torch`. Install via `pip install torch`."
        ) from e

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        keys = batch[0].keys()
        return {k: [row.get(k) for row in batch] for k in keys}

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def iter_n(ds: Iterable[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, ex in enumerate(ds):
        if i >= n:
            break
        out.append(ex)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a few AbstentionBench tasks.")
    parser.add_argument(
        "--repo",
        default="facebook/AbstentionBench",
        help="HF dataset repo id (default: facebook/AbstentionBench)",
    )
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=3, help="Number of examples to print")
    parser.add_argument("--torch", action="store_true", help="Iterate via PyTorch DataLoader")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--show-keys",
        action="store_true",
        help="Print dataset column names before examples",
    )
    args = parser.parse_args()

    ds = load_abstentionbench_task(args.task, args.split, repo_id=args.repo)

    if args.torch:
        dl = make_torch_dataloader(ds, batch_size=args.batch_size)
        batch = next(iter(dl))
        print(f"torch batch keys: {list(batch.keys())}")
        # Still print a few individual examples (more readable than a batch).

    if args.show_keys:
        colnames = getattr(ds, "column_names", None)
        if colnames is not None:
            print("columns:", colnames)

    for i, ex in enumerate(iter_n(ds, args.n)):
        extracted = _extract_text_fields(ex)
        print(f"\n[{args.task} | {args.split}] example {i}")
        for k, v in extracted.items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
