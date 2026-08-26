#!/usr/bin/env python3
"""
diagnose_constant_field.py -- artefact or finding?

A near-zero gradient dispersion has two explanations that look identical from
the outside:

  (A) ARTEFACT. LogRatioHead was fed raw hidden states (norm ~140 for
      LLaMA-3.1-8B). Both Tanh layers saturate, the local derivative (1 - t^2)
      collapses, and the head degenerates to an affine function of the few
      unsaturated units. It can still reach 0.93 pair accuracy while emitting a
      gradient whose direction never moves.

  (B) FINDING. log r really is affine, the optimal steering direction is
      constant, and it equals the Fisher direction Sigma^{-1}(mu_1 - mu_0).

Four checks. (A) and (B) make opposite predictions on every one.

Deps: numpy, torch, scikit-learn.
"""

import argparse
import json

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #

def saturation_report(head, H):
    """
    CHECK 1 -- how much of the head is saturated, and how big are the gradients?

    (A) predicts: most units |tanh| > 0.99, gradient norms ~1e-6 or smaller.
    (B) predicts: units spread across the tanh range, gradient norms O(1).
    """
    h = torch.tensor(H, dtype=torch.float32, requires_grad=True)
    acts, x = [], h
    for layer in head.f:
        x = layer(x)
        if isinstance(layer, torch.nn.Tanh):
            acts.append(x.detach().abs())
    g, = torch.autograd.grad(head(h).sum(), h)
    return dict(
        frac_saturated=[float((a > 0.99).float().mean()) for a in acts],
        median_abs_act=[float(a.median()) for a in acts],
        median_grad_norm=float(g.norm(dim=1).median()),
    )


def refit_standardised(Hc, Hw, n_pca=50, seed=0):
    """
    CHECK 2 -- standardise + PCA, then refit. This is what step0 already does.

    (A) predicts: dispersion jumps to tens of degrees once saturation is removed.
    (B) predicts: dispersion stays ~1 degree; scale was never the issue.
    """
    from jacobian_probe import fit_paired_head, grad_log_r, dispersion

    X = np.vstack([Hc, Hw])
    sc = StandardScaler().fit(X)
    pca = PCA(n_components=n_pca, random_state=seed).fit(sc.transform(X))
    tc, tw = pca.transform(sc.transform(Hc)), pca.transform(sc.transform(Hw))
    head, acc = fit_paired_head(tc, tw, seed=seed)
    U = grad_log_r(head, np.vstack([tc, tw]))
    return dict(pair_acc=acc, **dispersion(U))


def linear_head_accuracy(Hc, Hw, seed=0):
    """
    CHECK 3 -- a strictly linear paired head. Same conditional objective.

    If linear matches the MLP's 0.93, the MLP was never using its nonlinearity.
    Under (B) that is the correct answer. Under (A) it is because it could not
    reach one. Read alongside checks 1 and 2, not alone.
    """
    torch.manual_seed(seed)
    a = torch.tensor(Hc, dtype=torch.float32)
    b = torch.tensor(Hw, dtype=torch.float32)
    n = len(a); ntr = int(0.8 * n)
    w = torch.nn.Linear(a.shape[1], 1)
    opt = torch.optim.AdamW(w.parameters(), lr=1e-3, weight_decay=1e-2)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(400):
        opt.zero_grad()
        d = (w(b[:ntr]) - w(a[:ntr])).squeeze(-1)
        bce(d, torch.ones_like(d)).backward()
        opt.step()
    with torch.no_grad():
        acc = float(((w(b[ntr:]) - w(a[ntr:])).squeeze(-1) > 0).float().mean())
    return acc


def covariance_check(Hc, Hw):
    """
    CHECK 4 -- the corroborating evidence for (B).

    Under Gaussianity, log r is affine iff the class covariances agree. If the
    whitened eigenvalues cluster near 1, affine is the EXPECTED answer and the
    constant field is a finding rather than a bug. This check is independent of
    the head entirely, which is why it matters most.
    """
    S0, S1 = np.cov(Hc, rowvar=False), np.cov(Hw, rowvar=False)
    w_, V = np.linalg.eigh(S0)
    W = V @ np.diag(1.0 / np.sqrt(np.maximum(w_, 1e-8))) @ V.T
    ev = np.linalg.eigvalsh(W @ S1 @ W)
    return dict(eig_median=float(np.median(ev)), eig_min=float(ev.min()),
                eig_max=float(ev.max()),
                frac_within_10pct=float((np.abs(ev - 1) < 0.1).mean()))


def fisher_direction(Hc, Hw, shrink=1e-3):
    """
    If (B) holds, THIS is the optimal steering vector: Sigma^{-1}(mu_1 - mu_0).

    Compare against the raw difference-in-means that CAA/ITI actually use. A
    large angle between them is the paper: everyone is in the right family but
    using the wrong member, and the correction is derivable and testable.
    """
    mu0, mu1 = Hc.mean(0), Hw.mean(0)
    S = 0.5 * (np.cov(Hc, rowvar=False) + np.cov(Hw, rowvar=False))
    S += shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
    v_fisher = np.linalg.solve(S, mu1 - mu0)
    v_dim = mu1 - mu0
    cos = float(v_fisher @ v_dim /
                (np.linalg.norm(v_fisher) * np.linalg.norm(v_dim)))
    return dict(angle_fisher_vs_diffmeans_deg=float(np.degrees(np.arccos(
        np.clip(cos, -1, 1)))))


def main(a):
    d = np.load(a.cache)
    Hc = d["h_correct"][:, a.layer, :].astype(np.float32)
    Hw = d["h_wrong"][:, a.layer, :].astype(np.float32)

    out = {}
    if a.head:
        head = torch.load(a.head, weights_only=False)
        out["saturation"] = saturation_report(head, np.vstack([Hc, Hw]))
    out["standardised_refit"] = refit_standardised(Hc, Hw, a.n_pca, a.seed)
    out["linear_pair_acc"] = linear_head_accuracy(Hc, Hw, a.seed)
    out["covariance"] = covariance_check(Hc, Hw)
    out["fisher"] = fisher_direction(Hc, Hw)
    print(json.dumps(out, indent=2))

    print("\n--- VERDICT ---")
    print("ARTEFACT if: high frac_saturated, tiny median_grad_norm, and the "
          "standardised refit dispersion jumps to tens of degrees.")
    print("FINDING  if: saturation low, standardised dispersion stays ~1 deg, "
          "linear head matches the MLP, and whitened eigenvalues cluster at 1.")
    print("Either way, report the Fisher-vs-difference-in-means angle: under "
          "FINDING it is the correction the steering literature is missing.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache/cache.npz")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--head", default=None, help="pickled LogRatioHead, if saved")
    ap.add_argument("--n-pca", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
