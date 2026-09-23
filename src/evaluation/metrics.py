from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
from sklearn import metrics


def _to_numpy(x: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
    """Convert tensor or list to 1D flat numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy().flatten()
    elif isinstance(x, list):
        return np.array(x, dtype=np.float32).flatten()
    elif isinstance(x, np.ndarray):
        return x.flatten().astype(np.float32)
    raise ValueError(f"Unsupported data type for metric computation: {type(x)}")


def compute_auroc(id_scores: Union[torch.Tensor, np.ndarray, list], ood_scores: Union[torch.Tensor, np.ndarray, list]) -> float:
    """Compute Area Under ROC Curve (AUROC) distinguishing ID from OOD.

    Convention: Higher score = In-Distribution (label 1), Lower score = Out-of-Distribution (label 0).
    """
    id_arr = _to_numpy(id_scores)
    ood_arr = _to_numpy(ood_scores)

    y_true = np.concatenate([np.ones_like(id_arr), np.zeros_like(ood_arr)])
    y_scores = np.concatenate([id_arr, ood_arr])

    # Check for NaN / Inf
    valid_mask = np.isfinite(y_scores)
    if not np.all(valid_mask):
        y_true = y_true[valid_mask]
        y_scores = y_scores[valid_mask]

    if len(np.unique(y_true)) < 2:
        return 0.5

    return float(metrics.roc_auc_score(y_true, y_scores))


def compute_fpr95(id_scores: Union[torch.Tensor, np.ndarray, list], ood_scores: Union[torch.Tensor, np.ndarray, list]) -> float:
    """Compute False Positive Rate at 95% True Positive Rate (FPR@95).

    Returns:
        FPR value in [0.0, 1.0] (lower is better).
    """
    id_arr = _to_numpy(id_scores)
    ood_arr = _to_numpy(ood_scores)

    y_true = np.concatenate([np.ones_like(id_arr), np.zeros_like(ood_arr)])
    y_scores = np.concatenate([id_arr, ood_arr])

    valid_mask = np.isfinite(y_scores)
    if not np.all(valid_mask):
        y_true = y_true[valid_mask]
        y_scores = y_scores[valid_mask]

    if len(np.unique(y_true)) < 2:
        return 1.0

    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
    # Find smallest FPR where TPR >= 0.95
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0
    return float(fpr[idx[0]])


def compute_aupr(
    id_scores: Union[torch.Tensor, np.ndarray, list],
    ood_scores: Union[torch.Tensor, np.ndarray, list]
) -> Tuple[float, float]:
    """Compute Area Under Precision-Recall Curve: (AUPR-In, AUPR-Out)."""
    id_arr = _to_numpy(id_scores)
    ood_arr = _to_numpy(ood_scores)

    y_true_in = np.concatenate([np.ones_like(id_arr), np.zeros_like(ood_arr)])
    y_scores = np.concatenate([id_arr, ood_arr])

    valid_mask = np.isfinite(y_scores)
    if not np.all(valid_mask):
        y_true_in = y_true_in[valid_mask]
        y_scores = y_scores[valid_mask]

    # AUPR In
    precision_in, recall_in, _ = metrics.precision_recall_curve(y_true_in, y_scores)
    aupr_in = float(metrics.auc(recall_in, precision_in))

    # AUPR Out (invert scores so higher score = OOD)
    y_true_out = np.concatenate([np.zeros_like(id_arr), np.ones_like(ood_arr)])
    precision_out, recall_out, _ = metrics.precision_recall_curve(y_true_out, -y_scores)
    aupr_out = float(metrics.auc(recall_out, precision_out))

    return aupr_in, aupr_out


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) for classification probabilities."""
    confidences, predictions = torch.max(probs, dim=-1)
    accuracies = predictions.eq(labels)

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_ood_metrics(
    id_scores: Union[torch.Tensor, np.ndarray, list],
    ood_scores: Union[torch.Tensor, np.ndarray, list]
) -> Dict[str, float]:
    """Calculate all standard OOD detection metrics at once.

    Returns:
        Dict with 'auroc', 'fpr95', 'aupr_in', 'aupr_out'.
    """
    auroc = compute_auroc(id_scores, ood_scores)
    fpr95 = compute_fpr95(id_scores, ood_scores)
    aupr_in, aupr_out = compute_aupr(id_scores, ood_scores)

    return {
        "auroc": round(auroc * 100, 2),
        "fpr95": round(fpr95 * 100, 2),
        "aupr_in": round(aupr_in * 100, 2),
        "aupr_out": round(aupr_out * 100, 2),
    }
