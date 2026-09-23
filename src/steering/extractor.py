from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm

from .base import SteeringVector
from ..models.base import BaseModelWrapper
from ..data.contrastive_dataset import ContrastiveDataset
from ..data.loader import create_dataloader, contrastive_collate_fn
from ..models.probes import LinearProbe


class SteeringExtractor:
    """Base class for computing steering vectors from paired contrastive activations."""

    def __init__(self, method_name: str):
        self.method_name = method_name

    def extract_from_activations(
        self,
        pos_acts: torch.Tensor,
        neg_acts: torch.Tensor
    ) -> torch.Tensor:
        """Compute steering direction from positive and negative activation tensors [N, D]."""
        raise NotImplementedError


class MeanDiffExtractor(SteeringExtractor):
    """Contrastive Activation Addition (CAA) / Mean Difference: v = Mean(H+) - Mean(H-)."""

    def __init__(self):
        super().__init__("mean_difference")

    def extract_from_activations(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
        pos_mean = pos_acts.mean(dim=0)
        neg_mean = neg_acts.mean(dim=0)
        diff = pos_mean - neg_mean
        return diff


class PCAExtractor(SteeringExtractor):
    """PCA / SVD Direction: 1st Principal Component of pairwise difference vectors (H+ - H-)."""

    def __init__(self, center: bool = True):
        super().__init__("pca")
        self.center = center

    def extract_from_activations(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
        diffs = (pos_acts - neg_acts).float()  # [N, D]
        if self.center:
            diffs = diffs - diffs.mean(dim=0, keepdim=True)

        # SVD: diffs = U @ S @ V.T -> top direction is V[:, 0]
        # or torch.pca_lowrank
        _, _, v = torch.pca_lowrank(diffs, q=1, center=False)
        top_vec = v[:, 0]

        # Ensure consistent sign (aligned with mean difference)
        mean_diff = (pos_acts.mean(dim=0) - neg_acts.mean(dim=0)).float()
        if torch.dot(top_vec, mean_diff) < 0:
            top_vec = -top_vec

        return top_vec.to(pos_acts.dtype)


class MassMeanExtractor(SteeringExtractor):
    """Fisher Linear Discriminant (LDA) Direction: inv(S_W + lambda*I) * (Mean(H+) - Mean(H-))."""

    def __init__(self, ridge: float = 1e-3):
        super().__init__("mass_mean_lda")
        self.ridge = ridge

    def extract_from_activations(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
        pos_acts = pos_acts.float()
        neg_acts = neg_acts.float()
        mu_pos = pos_acts.mean(dim=0)
        mu_neg = neg_acts.mean(dim=0)
        diff = mu_pos - mu_neg

        # Within-class scatter
        centered_pos = pos_acts - mu_pos
        centered_neg = neg_acts - mu_neg
        cov_pos = torch.mm(centered_pos.t(), centered_pos)
        cov_neg = torch.mm(centered_neg.t(), centered_neg)
        s_w = (cov_pos + cov_neg) / max(1, (len(pos_acts) + len(neg_acts) - 2))

        dim = s_w.shape[0]
        reg_cov = s_w + self.ridge * torch.eye(dim, device=s_w.device)
        inv_cov = torch.linalg.pinv(reg_cov)

        direction = torch.mv(inv_cov, diff)
        return direction.to(pos_acts.dtype)


class ProbeExtractor(SteeringExtractor):
    """Linear Probe Direction: Normalized weight vector of a logistic regression probe."""

    def __init__(self, epochs: int = 50, lr: float = 1e-2):
        super().__init__("linear_probe")
        self.epochs = epochs
        self.lr = lr

    def extract_from_activations(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
        x = torch.cat([neg_acts, pos_acts], dim=0).float()
        y = torch.cat([
            torch.zeros(len(neg_acts), dtype=torch.long, device=x.device),
            torch.ones(len(pos_acts), dtype=torch.long, device=x.device)
        ])

        dim = x.shape[1]
        probe = LinearProbe(input_dim=dim, num_classes=2).to(x.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = probe(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        return probe.get_direction(class_idx=1, normalize=False).to(pos_acts.dtype)


EXTRACTOR_REGISTRY = {
    "mean_diff": MeanDiffExtractor,
    "mean_difference": MeanDiffExtractor,
    "caa": MeanDiffExtractor,
    "pca": PCAExtractor,
    "lda": MassMeanExtractor,
    "mass_mean": MassMeanExtractor,
    "probe": ProbeExtractor,
    "linear_probe": ProbeExtractor,
}


def extract_steering_vectors(
    model: BaseModelWrapper,
    dataset: ContrastiveDataset,
    method: str = "mean_difference",
    layer_names: Optional[List[str]] = None,
    batch_size: int = 16,
    pooling: str = "last",
    normalize: bool = True,
    show_progress: bool = True,
    **kwargs
) -> SteeringVector:
    """Extract steering vectors across model layers using contrastive pairs.

    Args:
        model: Wrapped model instance.
        dataset: Paired ContrastiveDataset.
        method: Extraction algorithm ('mean_difference', 'pca', 'mass_mean', 'probe').
        layer_names: Subset of layers to hook (defaults to all discovered layers).
        batch_size: Batch size for extraction forward passes.
        pooling: Token pooling method for sequence models ('last', 'mean', etc.).
        normalize: Whether to unit-normalize output steering vectors.
        show_progress: Display tqdm progress bar.

    Returns:
        SteeringVector instance containing vectors per layer.
    """
    if method not in EXTRACTOR_REGISTRY:
        raise ValueError(f"Unknown extraction method: {method}. Available: {list(EXTRACTOR_REGISTRY.keys())}")

    extractor = EXTRACTOR_REGISTRY[method](**kwargs)
    target_layers = layer_names or model.get_layer_names()

    pos_storage: Dict[str, List[torch.Tensor]] = {l: [] for l in target_layers}
    neg_storage: Dict[str, List[torch.Tensor]] = {l: [] for l in target_layers}

    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=contrastive_collate_fn
    )

    iterator = tqdm(dataloader, desc="Extracting Activations") if show_progress else dataloader

    for batch in iterator:
        pos_texts = batch["positive_texts"]
        neg_texts = batch["negative_texts"]

        pos_reps = model.extract_representations(pos_texts, layer_names=target_layers, pooling=pooling)
        neg_reps = model.extract_representations(neg_texts, layer_names=target_layers, pooling=pooling)

        for l in target_layers:
            pos_storage[l].append(pos_reps[l].cpu())
            neg_storage[l].append(neg_reps[l].cpu())

    # Compute direction per layer
    computed_vectors: Dict[str, torch.Tensor] = {}
    for l in target_layers:
        all_pos = torch.cat(pos_storage[l], dim=0)
        all_neg = torch.cat(neg_storage[l], dim=0)
        vec = extractor.extract_from_activations(all_pos, all_neg)
        computed_vectors[l] = vec

    sv = SteeringVector(
        vectors=computed_vectors,
        concept=getattr(dataset.pairs[0], "concept", "general") if len(dataset) > 0 else "general",
        method=method,
        metadata={"num_samples": len(dataset), "pooling": pooling}
    )

    if normalize:
        sv = sv.normalize()

    return sv
