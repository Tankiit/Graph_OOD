"""Losses for steering swarms. Selected by name with the `loss` hyperparameter.

Every loss has the signature

    f(feat, logits, same, other, dets, thr, sign, cfg, sim) -> [K] per-vector loss (mean over the batch)

feat [K, B, d], logits [K, B, C]: the steered batch, one row per vector (penultimate features, logits).
sim [K, B, d']: the steered activation at `sim_layer`, where infonce measures similarity.
same / other: `Clean` features, logits and sim-layer activations of the steered side and of the target side.
dets / thr: detectors (name -> callable(logits, feat), higher = more OOD) and their thresholds t_D;
a loss uses the detectors listed for it in LOSS_TARGETS.
sign = +1 for ID->OOD (push the score above t), -1 for OOD->ID (push it below).
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F

from .detectors import energy


class Clean(NamedTuple):
    feat: torch.Tensor    # [B, d] penultimate features
    logits: torch.Tensor  # [B, C]
    sim: torch.Tensor     # [B, d'] activation at `sim_layer` (= feat when sim_layer is penultimate)


def infonce(feat, logits, same: Clean, other: Clean, dets, thr, sign, cfg, sim):
    """Multi-positive InfoNCE on unit-normalised activations at `sim_layer`: the steered activation must
    be closer to the clean activations of the other side (positives) than to those of its own side (negatives)."""
    a = F.normalize(sim.float(), dim=-1)                                     # [K, B, d']
    pos = a @ F.normalize(other.sim.float(), dim=-1).T / cfg.tau             # [K, B, Bo]
    neg = a @ F.normalize(same.sim.float(), dim=-1).T / cfg.tau              # [K, B, B]
    if not cfg.own_negative:
        neg = neg.masked_fill(torch.eye(neg.shape[-1], dtype=torch.bool, device=neg.device), float("-inf"))
    loss = torch.logsumexp(torch.cat([pos, neg], -1), -1) - torch.logsumexp(pos, -1)
    return loss.mean(1)


def energy_hinge(feat, logits, same: Clean, other: Clean, dets, thr, sign, cfg, sim=None):
    """Push the steered energy past the energy threshold by `energy_margin`."""
    e = energy(logits, T=cfg.energy_T)                                       # [K, B]
    return F.relu(cfg.energy_margin - sign * (e - thr["energy"])).mean(1)


def energy_margin(feat, logits, same: Clean, other: Clean, dets, thr, sign, cfg, sim=None):
    """Pairwise: every steered energy must pass every clean energy of the other side by the margin."""
    e = energy(logits, T=cfg.energy_T)                                       # [K, B]
    e_other = energy(other.logits, T=cfg.energy_T)                           # [Bo]
    return F.relu(cfg.energy_margin - sign * (e[..., None] - e_other)).mean((1, 2))


def knn_hinge(feat, logits, same: Clean, other: Clean, dets, thr, sign, cfg, sim=None):
    """Push the steered kNN score (distance of the normalised penultimate feature to its k-th nearest
    ID reference feature) past the kNN threshold by `knn_margin`."""
    s = dets["knn"](logits, feat)                                            # [K, B], differentiable
    return F.relu(cfg.knn_margin - sign * (s - thr["knn"])).mean(1)


def infonce_energy(feat, logits, same, other, dets, thr, sign, cfg, sim):
    return infonce(feat, logits, same, other, dets, thr, sign, cfg, sim) + cfg.lambda_energy * energy_hinge(
        feat, logits, same, other, dets, thr, sign, cfg)


LOSSES = {"infonce": infonce, "infonce+energy": infonce_energy,
          "energy_hinge": energy_hinge, "energy_margin": energy_margin, "knn_hinge": knn_hinge}

# detector each loss attacks (also the one whose flip rate the training log reports)
LOSS_TARGETS = {"infonce": "energy", "infonce+energy": "energy", "energy_hinge": "energy",
                "energy_margin": "energy", "knn_hinge": "knn"}
