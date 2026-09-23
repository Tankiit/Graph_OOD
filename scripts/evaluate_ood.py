#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.logging import setup_logging, get_logger
from src.models.transformer_wrapper import TransformerWrapper
from src.data.text_datasets import TextClassificationDataset
from src.data.loader import create_dataloader, text_collate_fn
from src.steering.base import SteeringVector
from src.evaluation.ood_evaluator import OODEvaluator

logger = get_logger("EvaluateOOD")


def main():
    parser = argparse.ArgumentParser(description="Evaluate OOD Detection benchmarks with steering.")
    parser.add_argument("--model", type=str, default="gpt2", help="Hugging Face model name or path.")
    parser.add_argument("--scoring", type=str, default="all", choices=["all", "msp", "energy", "projection"])
    parser.add_argument("--steering-vector", type=str, default=None, help="Path to steering vector file (.pt).")
    parser.add_argument("--target-layer", type=str, default=None, help="Target layer for projection scoring.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    device = get_device(args.device)

    logger.info(f"Loading model: {args.model} on {device}")
    model = TransformerWrapper(args.model, device=device)

    logger.info("Setting up In-Distribution (ID) and Out-of-Distribution (OOD) benchmark datasets...")
    id_dataset = TextClassificationDataset.create_synthetic_classification(num_samples_per_class=30, num_classes=2)
    ood_dataset = TextClassificationDataset([
        TextClassificationDataset.create_synthetic_classification(num_samples_per_class=10, num_classes=3).samples[i]
        for i in range(20, 30)
    ])

    id_loader = create_dataloader(
        id_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=text_collate_fn(model.tokenizer)
    )
    ood_loader = create_dataloader(
        ood_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=text_collate_fn(model.tokenizer)
    )

    evaluator = OODEvaluator(model, device=device)
    methods_to_test = ["msp", "energy"] if args.scoring == "all" else [args.scoring]

    sv = None
    if args.steering_vector and Path(args.steering_vector).exists():
        sv = SteeringVector.load(args.steering_vector)
        target_layer = args.target_layer or sv.layer_names[len(sv.layer_names) // 2]
        if args.scoring in ("all", "projection"):
            methods_to_test.append("projection")

    print("\n" + "=" * 60)
    print("           OOD DETECTION BENCHMARK RESULTS           ")
    print("=" * 60)
    print(f"{'Scoring Method':<20} | {'AUROC (%)':<10} | {'FPR95 (%)':<10} | {'AUPR-In (%)':<10}")
    print("-" * 60)

    for method in methods_to_test:
        if method == "projection" and sv is not None:
            res = evaluator.evaluate_ood(
                id_loader,
                ood_loader,
                scoring_method="projection",
                steering_vector=sv,
                target_layer=target_layer
            )
        else:
            res = evaluator.evaluate_ood(id_loader, ood_loader, scoring_method=method)

        m = res["metrics"]
        print(f"{method.upper():<20} | {m['auroc']:<10.2f} | {m['fpr95']:<10.2f} | {m['aupr_in']:<10.2f}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
