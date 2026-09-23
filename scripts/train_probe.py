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
from src.training.probe_trainer import ProbeTrainer

logger = get_logger("TrainProbe")


def main():
    parser = argparse.ArgumentParser(description="Train linear/MLP probes across model layers.")
    parser.add_argument("--model", type=str, default="gpt2", help="Hugging Face model name or path.")
    parser.add_argument("--probe-type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--epochs", type=int, default=30, help="Number of probe training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for feature extraction.")
    parser.add_argument("--output-path", type=str, default="./outputs/probe_steering_vectors.pt")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)
    device = get_device(args.device)

    logger.info(f"Initializing model: {args.model} on {device}")
    model = TransformerWrapper(args.model, device=device)

    logger.info("Generating dataset for probe training...")
    dataset = TextClassificationDataset.create_synthetic_classification(num_samples_per_class=40, num_classes=2)
    texts = dataset.texts
    labels = torch.tensor([s["label"] for s in dataset], dtype=torch.long)

    target_layers = model.get_layer_names()
    logger.info(f"Extracting representations across {len(target_layers)} layers...")
    reps = model.extract_representations(texts, layer_names=target_layers, pooling="last")

    trainer = ProbeTrainer(
        probe_type=args.probe_type,
        lr=args.lr,
        epochs=args.epochs,
        device=device
    )

    logger.info("Training layer probes...")
    results = trainer.train_multi_layer_probes(
        train_reps=reps,
        train_labels=labels,
        num_classes=2
    )

    sv = results["steering_vector"]
    if sv:
        out_path = Path(args.output_path)
        sv.save(out_path)
        logger.info(f"Probe directional vectors saved to: {out_path.resolve()}")

    logger.info("Probe training complete!")


if __name__ == "__main__":
    main()
