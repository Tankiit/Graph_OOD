#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.seed import set_seed
from src.utils.device import get_device
from src.utils.logging import setup_logging, get_logger
from src.models.base import BaseModelWrapper
from src.models.hook_manager import HookManager
from src.models.transformer_wrapper import TransformerWrapper
from src.data.contrastive_dataset import ContrastiveDataset
from src.data.text_datasets import TextClassificationDataset
from src.data.loader import create_dataloader, text_collate_fn
from src.steering.extractor import extract_steering_vectors
from src.training.probe_trainer import ProbeTrainer
from src.evaluation.ood_evaluator import OODEvaluator
from src.evaluation.steering_evaluator import SteeringEvaluator

logger = get_logger("Pipeline")


class TinyMockModel(BaseModelWrapper):
    """Lightweight mock transformer-like model for instant CI/CD pipeline verification."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 4, num_classes: int = 2):
        super(BaseModelWrapper, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Layer blocks
        self.embedding = nn.Linear(32, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.hook_manager = HookManager(self)
        self.tokenizer = None

    def get_layer_names(self):
        return [f"layers.{i}" for i in range(len(self.layers))]

    def get_layer_module(self, layer_name: str):
        return self.hook_manager.get_submodule(layer_name)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        # input_ids: [B, S]
        # create mock float tokens
        b, s = input_ids.shape
        x = torch.zeros(b, s, 32, device=input_ids.device)
        for i in range(min(32, s)):
            x[:, i, i] = input_ids[:, i].float() + 1.0

        h = self.embedding(x)  # [B, S, D]
        for layer in self.layers:
            h = layer(h)
        logits = self.classifier(h.mean(dim=1))
        return logits

    def extract_representations(self, batch, layer_names=None, pooling="last", **kwargs):
        target_layers = layer_names or self.get_layer_names()
        if isinstance(batch, (list, str)):
            # convert list of texts to mock input_ids
            texts = [batch] if isinstance(batch, str) else batch
            ids = [[hash(w) % 100 for w in t.split()[:8]] for t in texts]
            max_l = max(len(row) for row in ids)
            padded = [row + [0] * (max_l - len(row)) for row in ids]
            input_tensor = torch.tensor(padded, dtype=torch.long)
        elif isinstance(batch, dict) and "input_ids" in batch:
            input_tensor = batch["input_ids"]
        else:
            input_tensor = batch

        with torch.no_grad():
            with self.hook_manager.capture_activations(target_layers) as storage:
                _ = self.forward(input_tensor)

        reps = {}
        for name in target_layers:
            acts = storage[name][0]
            reps[name] = acts[:, -1, :] if pooling == "last" else acts.mean(dim=1)
        return reps


def run_pipeline(mock_run: bool = False, model_name: str = "gpt2", device_name: str = "auto"):
    setup_logging()
    set_seed(42)
    device = get_device(device_name)
    logger.info(f"Starting pipeline on device: {device} (Mock run: {mock_run})")

    output_dir = Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Model
    if mock_run:
        logger.info("Initializing TinyMockModel for rapid verification...")
        model = TinyMockModel()
    else:
        logger.info(f"Loading transformer model: {model_name}...")
        model = TransformerWrapper(model_name, device=device)

    layer_names = model.get_layer_names()
    logger.info(f"Model initialized with {len(layer_names)} layers: {layer_names}")

    # 2. Extract Steering Vectors
    logger.info("\n--- STEP 1: Extracting Steering Vectors ---")
    contrastive_data = ContrastiveDataset.create_synthetic_ood_pairs(num_samples=20)
    sv = extract_steering_vectors(
        model=model,
        dataset=contrastive_data,
        method="mean_difference",
        batch_size=8,
        pooling="last",
        normalize=True,
        show_progress=False
    )
    sv_path = output_dir / "pipeline_steering_vector.pt"
    sv.save(sv_path)
    logger.info(f"Extracted and saved steering vector to {sv_path}")

    # 3. Train Probes
    logger.info("\n--- STEP 2: Training Representation Probes ---")
    cls_dataset = TextClassificationDataset.create_synthetic_classification(num_samples_per_class=20, num_classes=2)
    texts = cls_dataset.texts
    labels = torch.tensor([s["label"] for s in cls_dataset], dtype=torch.long)
    reps = model.extract_representations(texts, layer_names=layer_names, pooling="last")

    probe_trainer = ProbeTrainer(probe_type="linear", epochs=15, device=device)
    probe_results = probe_trainer.train_multi_layer_probes(
        train_reps=reps,
        train_labels=labels,
        num_classes=2
    )
    logger.info(f"Probe Accuracies by layer: {probe_results['accuracies']}")

    # 4. Evaluate OOD Detection
    logger.info("\n--- STEP 3: Evaluating OOD Detection Benchmarks ---")
    id_dataset = TextClassificationDataset.create_synthetic_classification(num_samples_per_class=15, num_classes=2)
    ood_dataset = TextClassificationDataset.create_synthetic_classification(num_samples_per_class=15, num_classes=3)

    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        collate_fn = text_collate_fn(model.tokenizer)
    else:
        # Mock model collation
        def mock_collate(batch):
            texts = [b["text"] for b in batch]
            ids = [[(hash(w) % 100) + 1 for w in t.split()[:8]] for t in texts]
            max_l = max(len(row) for row in ids)
            padded = [row + [0] * (max_l - len(row)) for row in ids]
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "texts": texts,
                "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long)
            }
        collate_fn = mock_collate

    id_loader = create_dataloader(id_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    ood_loader = create_dataloader(ood_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    evaluator = OODEvaluator(model, device=device)
    target_layer = layer_names[len(layer_names) // 2]
    proj_results = evaluator.evaluate_ood(
        id_loader,
        ood_loader,
        scoring_method="projection",
        steering_vector=sv,
        target_layer=target_layer
    )
    logger.info(f"OOD Projection Detection Results: AUROC={proj_results['metrics']['auroc']}% | FPR95={proj_results['metrics']['fpr95']}%")

    logger.info("\n==============================================")
    logger.info("Pipeline execution completed successfully!")
    logger.info("==============================================\n")


def main():
    parser = argparse.ArgumentParser(description="End-to-end Steering & OOD Pipeline Runner.")
    parser.add_argument("--mock-run", action="store_true", help="Run fast verification pipeline with mock model.")
    parser.add_argument("--model", type=str, default="gpt2", help="Hugging Face model name.")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, mps, cpu).")
    args = parser.parse_args()

    run_pipeline(mock_run=args.mock_run, model_name=args.model, device_name=args.device)


if __name__ == "__main__":
    main()
