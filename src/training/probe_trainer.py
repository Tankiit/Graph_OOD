from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from ..models.probes import LinearProbe, MLPProbe
from ..steering.base import SteeringVector
from ..utils.device import get_device, to_device
from ..utils.logging import get_logger

logger = get_logger("ProbeTrainer")


class ProbeTrainer:
    """Trains and evaluates classification probes across multiple model layers."""

    def __init__(
        self,
        probe_type: str = "linear",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 64,
        device: Optional[Union[torch.device, str]] = None
    ):
        self.probe_type = probe_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = get_device(device) if isinstance(device, str) or device is None else device

    def train_layer_probe(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None
    ) -> Tuple[nn.Module, float]:
        """Train a single probe on feature representations [N, D].

        Returns:
            Tuple of (trained_probe, best_val_accuracy).
        """
        train_features = train_features.float().to(self.device)
        train_labels = train_labels.long().to(self.device)
        input_dim = train_features.shape[1]
        classes = num_classes or len(torch.unique(train_labels))

        if self.probe_type == "linear":
            probe = LinearProbe(input_dim=input_dim, num_classes=classes).to(self.device)
        else:
            probe = MLPProbe(input_dim=input_dim, num_classes=classes).to(self.device)

        optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(train_features, train_labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            probe.train()
            for x_b, y_b in loader:
                optimizer.zero_grad()
                logits = probe(x_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()

        # Evaluate accuracy
        probe.eval()
        with torch.no_grad():
            if val_features is not None and val_labels is not None:
                vf = val_features.float().to(self.device)
                vl = val_labels.long().to(self.device)
                val_logits = probe(vf)
                preds = torch.argmax(val_logits, dim=-1)
                accuracy = (preds == vl).float().mean().item()
            else:
                train_logits = probe(train_features)
                preds = torch.argmax(train_logits, dim=-1)
                accuracy = (preds == train_labels).float().mean().item()

        return probe, accuracy

    def train_multi_layer_probes(
        self,
        train_reps: Dict[str, torch.Tensor],
        train_labels: torch.Tensor,
        val_reps: Optional[Dict[str, torch.Tensor]] = None,
        val_labels: Optional[torch.Tensor] = None,
        num_classes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train probes for all layers in parallel/sequence.

        Returns:
            Dict with 'probes', 'accuracies', and 'steering_vector'.
        """
        probes: Dict[str, nn.Module] = {}
        accuracies: Dict[str, float] = {}
        steering_weights: Dict[str, torch.Tensor] = {}

        for layer_name, feats in train_reps.items():
            vf = val_reps[layer_name] if val_reps and layer_name in val_reps else None
            probe, acc = self.train_layer_probe(
                train_features=feats,
                train_labels=train_labels,
                val_features=vf,
                val_labels=val_labels,
                num_classes=num_classes
            )
            probes[layer_name] = probe
            accuracies[layer_name] = acc

            if isinstance(probe, LinearProbe):
                steering_weights[layer_name] = probe.get_direction(class_idx=1, normalize=True)

            logger.info(f"Layer '{layer_name}' Probe Accuracy: {acc * 100:.2f}%")

        sv = None
        if steering_weights:
            sv = SteeringVector(
                vectors=steering_weights,
                concept="probe_classification_direction",
                method="linear_probe",
                metadata={"accuracies": accuracies}
            )

        return {
            "probes": probes,
            "accuracies": accuracies,
            "steering_vector": sv
        }
