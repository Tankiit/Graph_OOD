import pytest
import numpy as np
import torch
from src.evaluation.metrics import (
    compute_auroc,
    compute_fpr95,
    compute_aupr,
    compute_ece,
    compute_ood_metrics,
)


def test_metrics_perfect_separation():
    id_scores = torch.tensor([10.0, 9.0, 8.5, 9.5, 11.0])
    ood_scores = torch.tensor([1.0, 2.0, 0.5, -1.0, 0.0])

    auroc = compute_auroc(id_scores, ood_scores)
    assert auroc == 1.0

    fpr95 = compute_fpr95(id_scores, ood_scores)
    assert fpr95 == 0.0

    metrics_dict = compute_ood_metrics(id_scores, ood_scores)
    assert metrics_dict["auroc"] == 100.0
    assert metrics_dict["fpr95"] == 0.0


def test_ece_computation():
    probs = torch.tensor([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.3, 0.7],
        [0.2, 0.8]
    ])
    labels = torch.tensor([0, 0, 1, 1])
    ece = compute_ece(probs, labels)
    assert 0.0 <= ece <= 1.0
