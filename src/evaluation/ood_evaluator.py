from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .metrics import compute_ood_metrics
from ..models.base import BaseModelWrapper
from ..models.probes import MahalanobisDetector
from ..steering.base import SteeringVector
from ..steering.interveners import AdditiveIntervener
from ..utils.device import get_device, to_device
from ..utils.logging import get_logger

logger = get_logger("OODEvaluator")


class OODEvaluator:
    """Benchmark evaluator for Out-of-Distribution (OOD) detection methods."""

    def __init__(
        self,
        model: BaseModelWrapper,
        device: Optional[Union[torch.device, str]] = None
    ):
        self.device = get_device(device) if isinstance(device, str) or device is None else device
        self.model = model

    def compute_scores_msp(self, dataloader: DataLoader) -> torch.Tensor:
        """Compute Maximum Softmax Probability (MSP) confidence scores."""
        self.model.eval()
        scores = []
        with torch.no_grad():
            for batch in dataloader:
                batch = to_device(batch, self.device)
                if isinstance(batch, dict):
                    inputs = {k: v for k, v in batch.items() if k not in ("label", "labels", "id", "ids", "texts", "metadata")}
                    outputs = self.model(**inputs) if inputs else self.model(batch["input_ids"])
                else:
                    inputs, _ = batch
                    outputs = self.model(inputs)

                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                probs = F.softmax(logits, dim=-1)
                max_probs, _ = torch.max(probs, dim=-1)
                scores.append(max_probs.cpu())
        return torch.cat(scores, dim=0)

    def compute_scores_energy(self, dataloader: DataLoader, temperature: float = 1.0) -> torch.Tensor:
        """Compute Helmholtz free energy scores: T * LogSumExp(logits / T)."""
        self.model.eval()
        scores = []
        with torch.no_grad():
            for batch in dataloader:
                batch = to_device(batch, self.device)
                if isinstance(batch, dict):
                    inputs = {k: v for k, v in batch.items() if k not in ("label", "labels", "id", "ids", "texts", "metadata")}
                    outputs = self.model(**inputs) if inputs else self.model(batch["input_ids"])
                else:
                    inputs, _ = batch
                    outputs = self.model(inputs)

                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                energy = temperature * torch.logsumexp(logits / temperature, dim=-1)
                scores.append(energy.cpu())
        return torch.cat(scores, dim=0)

    def compute_scores_projection(
        self,
        dataloader: DataLoader,
        steering_vector: SteeringVector,
        target_layer: str,
        pooling: str = "last"
    ) -> torch.Tensor:
        """Compute projection score along steering direction: h . v_norm (higher = more ID)."""
        vec = steering_vector[target_layer].to(self.device).float()
        norm = torch.norm(vec, p=2)
        if norm > 1e-9:
            vec = vec / norm

        scores = []
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict) and "texts" in batch:
                    inputs = batch["texts"]
                else:
                    inputs = batch

                reps = self.model.extract_representations(inputs, layer_names=[target_layer], pooling=pooling)
                h = reps[target_layer].to(self.device).float()  # [B, D]
                proj = torch.mv(h, vec)
                scores.append(proj.cpu())
        return torch.cat(scores, dim=0)

    def evaluate_ood(
        self,
        id_loader: DataLoader,
        ood_loader: DataLoader,
        scoring_method: str = "msp",
        steering_vector: Optional[SteeringVector] = None,
        target_layer: Optional[str] = None,
        pooling: str = "last",
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """Evaluate OOD detection performance distinguishing ID from OOD loader samples.

        Args:
            id_loader: In-Distribution DataLoader.
            ood_loader: Out-of-Distribution DataLoader.
            scoring_method: 'msp', 'energy', or 'projection'.
            steering_vector: Optional steering vector for projection scoring.
            target_layer: Target layer name for representations.
            pooling: Token pooling method for sequences.
            temperature: Softmax/Energy temperature.

        Returns:
            Dict containing metric results and score arrays.
        """
        logger.info(f"Evaluating OOD detection using scoring method: {scoring_method}")

        if scoring_method == "msp":
            id_scores = self.compute_scores_msp(id_loader)
            ood_scores = self.compute_scores_msp(ood_loader)
        elif scoring_method == "energy":
            id_scores = self.compute_scores_energy(id_loader, temperature=temperature)
            ood_scores = self.compute_scores_energy(ood_loader, temperature=temperature)
        elif scoring_method == "projection":
            if steering_vector is None or target_layer is None:
                raise ValueError("steering_vector and target_layer are required for projection scoring.")
            id_scores = self.compute_scores_projection(id_loader, steering_vector, target_layer, pooling=pooling)
            ood_scores = self.compute_scores_projection(ood_loader, steering_vector, target_layer, pooling=pooling)
        else:
            raise ValueError(f"Unsupported scoring method: {scoring_method}")

        results = compute_ood_metrics(id_scores, ood_scores)
        results["scoring_method"] = scoring_method
        logger.info(f"Results [{scoring_method}]: AUROC: {results['auroc']}% | FPR95: {results['fpr95']}% | AUPR-In: {results['aupr_in']}%")

        return {
            "metrics": results,
            "id_scores": id_scores,
            "ood_scores": ood_scores
        }
