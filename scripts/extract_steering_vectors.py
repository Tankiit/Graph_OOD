#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.logging import setup_logging, get_logger
from src.utils.config import load_config
from src.models.transformer_wrapper import TransformerWrapper
from src.data.contrastive_dataset import ContrastiveDataset
from src.steering.extractor import extract_steering_vectors

logger = get_logger("ExtractSteering")


def main():
    parser = argparse.ArgumentParser(description="Extract steering vectors from contrastive datasets.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--model", type=str, default="gpt2", help="Hugging Face model name or local path.")
    parser.add_argument("--method", type=str, default="mean_difference", choices=["mean_difference", "pca", "mass_mean", "linear_probe"])
    parser.add_argument("--data-file", type=str, default=None, help="Path to JSONL contrastive pairs file.")
    parser.add_argument("--output-path", type=str, default="./outputs/steering_vectors.pt", help="Path to save output .pt file.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for forward passes.")
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "mean", "cls", "first"])
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, mps, cpu, auto).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    device = get_device(args.device)

    logger.info(f"Loading model: {args.model} on {device}")
    model = TransformerWrapper(args.model, device=device)

    if args.data_file:
        logger.info(f"Loading contrastive dataset from: {args.data_file}")
        dataset = ContrastiveDataset.from_jsonl(args.data_file)
    else:
        logger.info("Using synthetic ID vs OOD contrastive dataset.")
        dataset = ContrastiveDataset.create_synthetic_ood_pairs(num_samples=40)

    logger.info(f"Extracting steering vectors using method: '{args.method}' across {len(model.get_layer_names())} layers...")
    steering_vector = extract_steering_vectors(
        model=model,
        dataset=dataset,
        method=args.method,
        batch_size=args.batch_size,
        pooling=args.pooling,
        normalize=True,
        show_progress=True
    )

    out_path = Path(args.output_path)
    steering_vector.save(out_path)
    logger.info(f"Steering vectors successfully saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
