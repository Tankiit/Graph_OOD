"""
fix_truthfulqa_pt.py
────────────────────
Fixes the y_expert labels in a TruthfulQA hidden_states.pt file that was
extracted without the category join (y_expert=1 for all questions).

Loads the .pt file, re-downloads TruthfulQA generation split (tiny, <1MB),
joins on question text to get real categories, recomputes y_expert from
Lin et al. (2022) human accuracy table, and saves a patched .pt file.

Usage:
    python fix_truthfulqa_pt.py --input ./truthfulqa.pt
    python fix_truthfulqa_pt.py --input ./truthfulqa.pt --output ./truthfulqa_fixed.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path

HUMAN_ACC_BY_CATEGORY = {
    "Misconceptions":        0.65,
    "Conspiracies":          0.52,
    "Myths and Fairytales":  0.67,
    "Paranormal":            0.60,
    "Superstitions":         0.63,
    "Fiction":               0.75,
    "Advertising":           0.72,
    "Psychology":            0.78,
    "Sociology":             0.76,
    "Economics":             0.79,
    "History":               0.88,
    "Politics":              0.81,
    "Law":                   0.84,
    "Health":                0.83,
    "Science":               0.92,
    "Nutrition":             0.80,
    "Statistics":            0.85,
    "Weather":               0.90,
    "Geography":             0.91,
    "Religion":              0.77,
    "Language":              0.82,
    "Logical Falsehoods":    0.88,
    "Distraction":           0.85,
    "__default__":           0.80,
}

EXPERT_THRESHOLD = 0.80


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",     default="./truthfulqa.pt")
    p.add_argument("--output",    default=None,
                   help="Output path (default: input_fixed.pt)")
    p.add_argument("--threshold", type=float, default=EXPERT_THRESHOLD)
    return p.parse_args()


def main():
    args   = parse_args()
    in_path  = Path(args.input)
    out_path = Path(args.output) if args.output \
               else in_path.with_stem(in_path.stem + "_fixed")

    # ── Load .pt file ─────────────────────────────────────────────────────────
    print(f"Loading {in_path}...")
    data = torch.load(in_path, map_location="cpu", weights_only=False)
    N    = data["n_questions"]
    print(f"  {N} questions | model: {data.get('model', 'unknown')}")

    # Check current state
    y_exp_c = data["y_expert_correct"].numpy()
    y_exp_w = data["y_expert_wrong"].numpy()
    print(f"  Current y_expert_correct: {int(y_exp_c.sum())}/{N} are 1")
    print(f"  Current categories sample: {data['categories'][:3]}")

    if y_exp_c.sum() < N and len(set(data["categories"])) > 1:
        print("  y_expert looks fine — nothing to fix.")
        return

    # ── Download TruthfulQA generation split (has category field) ─────────────
    print("\nDownloading TruthfulQA generation split for category labels...")
    from datasets import load_dataset
    gen_ds = load_dataset("truthful_qa", "generation", split="validation")

    # Build question → (category, human_acc) map
    q_to_info = {}
    for item in gen_ds:
        cat       = item["category"]
        human_acc = HUMAN_ACC_BY_CATEGORY.get(cat,
                    HUMAN_ACC_BY_CATEGORY["__default__"])
        q_to_info[item["question"]] = (cat, human_acc)

    print(f"  Built lookup for {len(q_to_info)} questions")

    # ── Reconstruct per-question y_expert ──────────────────────────────────────
    questions    = data["questions"]
    new_cats     = []
    new_haccs    = []
    new_y_exp_c  = []

    n_matched   = 0
    n_default   = 0

    for q in questions:
        if q in q_to_info:
            cat, hacc = q_to_info[q]
            n_matched += 1
        else:
            cat, hacc = "__default__", HUMAN_ACC_BY_CATEGORY["__default__"]
            n_default += 1

        new_cats.append(cat)
        new_haccs.append(hacc)
        new_y_exp_c.append(int(hacc >= args.threshold))

    new_y_exp_c = np.array(new_y_exp_c, dtype=int)
    new_y_exp_w = new_y_exp_c.copy()   # expert reliability is question-level

    print(f"\n  Matched {n_matched}/{N} questions to generation split")
    print(f"  Unmatched (using __default__): {n_default}")
    print(f"  Categories found: {len(set(new_cats))}")
    print(f"  Expert reliable (human_acc >= {args.threshold}): "
          f"{int(new_y_exp_c.sum())}/{N} ({new_y_exp_c.mean()*100:.0f}%)")
    print(f"  Expert UNreliable: "
          f"{int((new_y_exp_c == 0).sum())}/{N} "
          f"← Conspiracies, Myths, Superstitions etc.")

    # ── Patch and save ─────────────────────────────────────────────────────────
    data["y_expert_correct"] = torch.tensor(new_y_exp_c, dtype=torch.long)
    data["y_expert_wrong"]   = torch.tensor(new_y_exp_w, dtype=torch.long)
    data["categories"]       = new_cats
    data["human_accs"]       = new_haccs

    torch.save(data, out_path)
    print(f"\n✓  Saved fixed file → {out_path}")

    # Quick sanity
    n_classes = len(set(new_y_exp_c.tolist()))
    print(f"  y_expert classes: {n_classes}  ({'✓ OK' if n_classes == 2 else '✗ still broken'})")
    print(f"\nNow run:")
    print(f"  python train_probe.py --input {out_path} "
          f"--output outputs/results_tqa_fixed.json")

    # Show category breakdown
    from collections import Counter
    cat_counts = Counter(new_cats)
    reliable_cats   = [c for c in cat_counts if HUMAN_ACC_BY_CATEGORY.get(c, 0.80) >= args.threshold]
    unreliable_cats = [c for c in cat_counts if HUMAN_ACC_BY_CATEGORY.get(c, 0.80) <  args.threshold]
    print(f"\n  Unreliable categories (y_expert=0, {len(unreliable_cats)} cats):")
    for c in unreliable_cats:
        acc = HUMAN_ACC_BY_CATEGORY.get(c, 0.80)
        print(f"    {c:<30}  human_acc={acc:.2f}  n={cat_counts[c]}")


if __name__ == "__main__":
    main()