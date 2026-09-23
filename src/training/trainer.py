from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.device import get_device, to_device
from ..utils.logging import get_logger

logger = get_logger("Trainer")


class Trainer:
    """General PyTorch training loop engine with validation and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        scheduler: Optional[Any] = None,
        device: Optional[Union[torch.device, str]] = None,
        grad_clip: Optional[float] = 1.0,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ):
        self.device = get_device(device) if isinstance(device, str) or device is None else device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": []
        }

    def train_epoch(self, dataloader: DataLoader, show_progress: bool = True) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        iterator = tqdm(dataloader, desc="Training Epoch", leave=False) if show_progress else dataloader

        for batch in iterator:
            batch = to_device(batch, self.device)
            self.optimizer.zero_grad()

            if isinstance(batch, dict):
                inputs = {k: v for k, v in batch.items() if k not in ("label", "labels", "id", "ids", "texts", "metadata")}
                labels = batch.get("label", batch.get("labels", None))
                outputs = self.model(**inputs) if inputs else self.model(batch["input_ids"])
            else:
                inputs, labels = batch
                outputs = self.model(inputs)

            # Handle model outputs (logits tensor or HuggingFace ModelOutput)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = self.criterion(logits, labels)

            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if self.scheduler:
            self.scheduler.step()

        return total_loss / max(1, num_batches)

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on a validation dataloader, returning (avg_loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        with torch.no_grad():
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
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        accuracy = (correct / total) if total > 0 else 0.0
        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        early_stopping_patience: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, List[float]]:
        """Train for multiple epochs with optional validation and early stopping."""
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, show_progress=show_progress)
            self.history["train_loss"].append(train_loss)

            log_msg = f"Epoch {epoch:03d}/{epochs:03d} - Train Loss: {train_loss:.4f}"

            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                log_msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if self.checkpoint_dir:
                        torch.save(self.model.state_dict(), self.checkpoint_dir / "best_model.pt")
                else:
                    patience_counter += 1
                    if early_stopping_patience and patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break

            logger.info(log_msg)

        return self.history
