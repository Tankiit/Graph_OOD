#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=4)
    parser.add_argument("--min-calibration-labels", type=int, default=2)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.generations.read_text().splitlines() if line.strip()]
    labeled = [row for row in rows if row["label"] is not None]
    calibration: dict[str, list[int]] = defaultdict(list)
    evaluation_indices: dict[str, list[int]] = defaultdict(list)
    for frozen_index, row in enumerate(labeled):
        if row["sample_id"] < args.calibration_samples:
            calibration[row["pair_id"]].append(int(row["label"]))
        else:
            evaluation_indices[row["pair_id"]].append(frozen_index)

    questions = {}
    for question, labels in calibration.items():
        if len(labels) < args.min_calibration_labels or not evaluation_indices[question]:
            continue
        correct_rate = sum(labels) / len(labels)
        difficulty = 1 - correct_rate
        bin_name = "easy" if difficulty <= 0.25 else "medium" if difficulty <= 0.75 else "hard"
        questions[question] = {
            "difficulty": difficulty,
            "difficulty_bin": bin_name,
            "calibration_labels": len(labels),
            "evaluation_row_indices": evaluation_indices[question],
        }
    payload = {
        "protocol": "sample_ids 0-3 grade difficulty; 4-7 evaluate; unknown labels omitted",
        "questions": questions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "questions": len(questions),
        "evaluation_rows": sum(len(item["evaluation_row_indices"]) for item in questions.values()),
        "bins": {name: sum(item["difficulty_bin"] == name for item in questions.values())
                 for name in ("easy", "medium", "hard")},
    }, indent=2))


if __name__ == "__main__":
    main()
