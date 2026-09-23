from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..models.base import BaseModelWrapper
from ..steering.base import SteeringVector
from ..steering.interveners import AdditiveIntervener
from ..utils.device import get_device, to_device
from ..utils.logging import get_logger

logger = get_logger("SteeringEvaluator")


class SteeringEvaluator:
    """Evaluates the behavioral impact of steering vectors across intervention strengths."""

    def __init__(
        self,
        model: BaseModelWrapper,
        device: Optional[Union[torch.device, str]] = None
    ):
        self.device = get_device(device) if isinstance(device, str) or device is None else device
        self.model = model

    def evaluate_coefficient_sweep(
        self,
        dataloader: DataLoader,
        steering_vector: SteeringVector,
        target_layer: str,
        coefficients: List[float],
        target_class_idx: int = 1
    ) -> Dict[str, Any]:
        """Sweep steering strength coefficients and measure target class probability shifts.

        Args:
            dataloader: Test evaluation DataLoader.
            steering_vector: SteeringVector instance.
            target_layer: Layer to inject vector into.
            coefficients: List of multiplier coefficients (e.g. [-2.0, -1.0, 0.0, 1.0, 2.0]).
            target_class_idx: Class index whose probability is tracked.

        Returns:
            Dict mapping coefficient -> mean target class probability and accuracy.
        """
        self.model.eval()
        vec = steering_vector[target_layer].to(self.device)

        results = {
            "coefficients": coefficients,
            "mean_target_probs": [],
            "accuracies": []
        }

        for coeff in coefficients:
            intervener = AdditiveIntervener(vector=vec, coefficient=coeff)
            interventions = {target_layer: intervener}

            target_probs = []
            correct = 0
            total = 0

            with torch.no_grad():
                with self.model.hook_manager.apply_steering(interventions):
                    for batch in dataloader:
                        batch = to_device(batch, self.device)
                        if isinstance(batch, dict):
                            inputs = {k: v for k, v in batch.items() if k not in ("label", "labels", "id", "ids", "texts", "metadata")}
                            labels = batch.get("label", batch.get("labels", None))
                            outputs = self.model(**inputs) if inputs else self.model(batch["input_ids"])
                        else:
                            inputs, labels = batch
                            outputs = self.model(inputs)

                        logits = outputs.logits if hasattr(outputs, "logits") else outputs
                        probs = F.softmax(logits, dim=-1)

                        if logits.shape[-1] > target_class_idx:
                            target_probs.append(probs[:, target_class_idx].cpu())

                        if labels is not None:
                            preds = torch.argmax(logits, dim=-1)
                            correct += (preds == labels).sum().item()
                            total += len(labels)

            mean_p = torch.cat(target_probs, dim=0).mean().item() if target_probs else 0.0
            acc = (correct / total) if total > 0 else 0.0

            results["mean_target_probs"].append(round(mean_p, 4))
            results["accuracies"].append(round(acc * 100, 2))

            logger.info(f"Coeff {coeff:+.2f} -> Target Class P: {mean_p:.4f} | Acc: {acc * 100:.2f}%")

        return results
