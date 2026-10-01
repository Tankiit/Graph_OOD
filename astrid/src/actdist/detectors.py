"""Post-hoc OOD detectors on a classifier's logits and penultimate features. Higher score = more OOD.

    dets = build_detectors(["energy", "knn", "mahalanobis"], cfg, ref_feat, ref_labels, device)
    s = dets["knn"](logits, feat)          # any leading shape: [..., C] logits, [..., d] features

Logit detectors (energy, msp, maxlogit) need no fitting. Feature detectors are fitted once on
the penultimate features of an ID reference set (clean training images of the classifier):
    knn          distance of the L2-normalised feature to its k-th nearest reference feature (Sun et al., 2022)
    mahalanobis  min over classes of the Mahalanobis distance to the class mean, shared covariance (Lee et al., 2018)

A detector rejects x when score(x) > t, with t the `tpr`-quantile of the scores of a clean
ID calibration split, so a fraction `tpr` of ID points is accepted.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu


def energy(logits: torch.Tensor, feat=None, T: float = 1.0) -> torch.Tensor:
    """Negative free energy, -T logsumexp(logits / T) (Liu et al., 2020); differentiable."""
    return -T * torch.logsumexp(logits.float() / T, dim=-1)


def msp(logits: torch.Tensor, feat=None, T: float = 1.0) -> torch.Tensor:
    return -torch.softmax(logits.float(), dim=-1).amax(-1)


def maxlogit(logits: torch.Tensor, feat=None, T: float = 1.0) -> torch.Tensor:
    return -logits.float().amax(-1)


LOGIT_SCORES = {"energy": energy, "msp": msp, "maxlogit": maxlogit}


class _FeatureDetector:
    """Scores features in row chunks (so K x B steered batches never build a huge distance matrix)."""

    chunk = 4096

    def __call__(self, logits, feat, T: float = 1.0) -> torch.Tensor:
        lead = feat.shape[:-1]
        f = feat.reshape(-1, feat.shape[-1]).float().to(self.device)
        out = torch.cat([self._score(f[i:i + self.chunk]) for i in range(0, len(f), self.chunk)])
        return out.reshape(lead)


class KNN(_FeatureDetector):
    def __init__(self, ref_feat: torch.Tensor, k: int, device):
        self.device, self.k = device, k
        self.ref = F.normalize(ref_feat.float(), dim=1).to(device)       # [n_ref, d]

    def _score(self, f):
        d = torch.cdist(F.normalize(f, dim=1), self.ref)                  # [n, n_ref]
        return d.topk(self.k, dim=1, largest=False).values[:, -1]


class Mahalanobis(_FeatureDetector):
    def __init__(self, ref_feat: torch.Tensor, ref_labels: torch.Tensor, device):
        self.device = device
        x, y = ref_feat.double().to(device), ref_labels.to(device)
        classes = torch.unique(y)
        self.mu = torch.stack([x[y == c].mean(0) for c in classes])       # [C, d]
        centered = x - self.mu[torch.searchsorted(classes, y)]
        cov = centered.T @ centered / len(x)
        self.prec = torch.linalg.pinv(cov, hermitian=True)                # [d, d]
        self.mu_term = ((self.mu @ self.prec) * self.mu).sum(1)          # mu_c^T P mu_c

    def _score(self, f):
        f = f.double()
        fp = f @ self.prec
        d2 = (fp * f).sum(1, keepdim=True) - 2 * fp @ self.mu.T + self.mu_term  # [n, C]
        return d2.min(1).values.float()


FEATURE_DETECTORS = ("knn", "mahalanobis")
DETECTORS = tuple(LOGIT_SCORES) + FEATURE_DETECTORS


def build_detectors(names, cfg, ref_feat=None, ref_labels=None, device="cpu") -> dict:
    """name -> callable(logits, feat) -> scores (higher = more OOD), fitted on the ID reference."""
    dets = {}
    for n in names:
        if n in LOGIT_SCORES:
            fn = LOGIT_SCORES[n]
            dets[n] = lambda logits, feat, fn=fn: fn(logits, T=cfg.energy_T)
        elif n == "knn":
            dets[n] = KNN(ref_feat, cfg.knn_k, device)
        elif n == "mahalanobis":
            dets[n] = Mahalanobis(ref_feat, ref_labels, device)
        else:
            raise ValueError(f"unknown detector {n!r}; choose from {DETECTORS}")
    return dets


def calibrate(id_scores: np.ndarray, tpr: float) -> float:
    return float(np.quantile(id_scores, tpr))


def auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """P(score_ood > score_id); 0.5 = inseparable, < 0.5 = inverted."""
    u = mannwhitneyu(ood_scores, id_scores, alternative="two-sided").statistic
    return float(u / (len(ood_scores) * len(id_scores)))


def fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, tpr: float = 0.95) -> float:
    """Fraction of OOD accepted at the threshold that accepts `tpr` of these ID scores."""
    return float((ood_scores <= np.quantile(id_scores, tpr)).mean())
