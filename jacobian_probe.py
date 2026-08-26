#!/usr/bin/env python3
"""
jacobian_probe.py -- VJP field, JVP mechanism check, and the curvature split.

THREE GRADIENTS, KEPT SEPARATE
    grad log r      statistics of the two populations. No architecture in it.
    J = dh_L/dh_l   the network map. No data distribution in it.
    v*(h) = J^T u   the METHOD: steepest direction at the intervention layer l
                    for moving log r at the readout layer L.  u = grad log r(h_L).

    v* is a VECTOR-Jacobian product -- reverse mode, one backward pass through
    the frozen model.  The JVP (forward) is the DIAGNOSTIC, not the method.

WHY A CONSTANT STEERING VECTOR IS TWO APPROXIMATIONS, NOT ONE
    v*(h) = J(h)^T grad log r(h_L) is constant iff BOTH factors are constant:
      (a) grad log r constant  <=> log r affine  <=> homoscedastic Gaussian classes
      (b) J(h)^T u constant    <=> the network acts position-independently
    Measuring them separately tells you what the paper is about.  If (a) fails,
    the contribution is the density-ratio machinery.  If (b) fails, the paper is
    about network geometry and the ratio is nearly incidental.  Find out BEFORE
    writing the introduction.

MEMORY NOTE
    Freeze all parameters: gradients are needed w.r.t. ACTIVATIONS only, so no
    parameter-gradient buffers are allocated for the 8B weights.  hidden_states
    are already in the autograd graph, so autograd.grad(h_L, h_l, ...) works
    directly -- no hand-rolled layer subset, no rotary/mask reconstruction.

Deps: torch, transformers, numpy.
"""

import argparse
import json

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Readout direction: u = grad log r at layer L
# --------------------------------------------------------------------------- #

class LogRatioHead(torch.nn.Module):
    """
    Estimates log r up to an additive constant from PAIRED data.

    Conditional likelihood of which member of a pair is the wrong one:
        P(wrong = b | {a,b}) = sigmoid( s(h_b) - s(h_a) )
    Every pair-level nuisance (topic, length, difficulty) cancels.  The additive
    constant is unidentified -- and irrelevant, because only grad s is used.
    """

    def __init__(self, d, hidden=64):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden), torch.nn.Tanh(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, h):
        return self.f(h).squeeze(-1)


def fit_paired_head(H_correct, H_wrong, epochs=400, lr=1e-3, wd=1e-2, seed=0):
    """Conditional (Bradley-Terry) fit. Returns the head and held-out pair accuracy."""
    torch.manual_seed(seed)
    a = torch.tensor(H_correct, dtype=torch.float32)
    b = torch.tensor(H_wrong, dtype=torch.float32)
    n = len(a); ntr = int(0.8 * n)
    head = LogRatioHead(a.shape[1])
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    bce = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        d = head(b[:ntr]) - head(a[:ntr])          # should be > 0
        bce(d, torch.ones_like(d)).backward()
        opt.step()
    with torch.no_grad():
        acc = float(((head(b[ntr:]) - head(a[ntr:])) > 0).float().mean())
    return head, acc


def grad_log_r(head, H):
    """u(h) = grad_h log r(h). The statistical factor."""
    h = torch.tensor(H, dtype=torch.float32, requires_grad=True)
    g, = torch.autograd.grad(head(h).sum(), h)
    return g.detach().numpy()


# --------------------------------------------------------------------------- #
# VJP: the method
# --------------------------------------------------------------------------- #

def vjp_field(model, tok, prompts, u_fn, layer_l, layer_L):
    """
    v*(h) = J(h)^T u(h_L) for each prompt.  One backward pass per prompt.

    u_fn maps the layer-L state to the readout direction (i.e. grad log r there).

    TODO(batching): this is one prompt at a time for clarity. Batch it before
    running at scale -- the backward pass dominates and batching is most of the
    available speedup.
    """
    for p in model.parameters():
        p.requires_grad_(False)                    # activations only

    out = []
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").to(model.device)
        res = model(**ids, output_hidden_states=True)
        h_l = res.hidden_states[layer_l][:, -1]
        h_L = res.hidden_states[layer_L][:, -1]
        u = torch.tensor(u_fn(h_L.detach().float().cpu().numpy()),
                         device=h_L.device, dtype=h_L.dtype)
        v, = torch.autograd.grad(h_L, h_l, grad_outputs=u, retain_graph=False)
        out.append(v[0].float().cpu().numpy())
    return np.stack(out)


# --------------------------------------------------------------------------- #
# JVP: the mechanism diagnostic
# --------------------------------------------------------------------------- #

def jvp_finite_diff(model, tok, prompt, v, layer_l, layer_L, eps=1e-2):
    """
    (h_L(h_l + eps*v) - h_L(h_l)) / eps  -- how an identical, class-agnostic
    perturbation propagates.

    THE MECHANISM CLAIM: v is the same vector for truthful and hallucinated
    states, and covariance is translation-invariant, so at layer l the two
    populations are shifted identically and separation is unchanged.  If the
    JVPs differ systematically between populations, that difference IS the
    mechanism by which a translation yields class-dependent separation.  If they
    do not differ, additive steering should not separate at all and the observed
    effect originates somewhere you have not identified.

    Finite differences first: easier to get right than torch.func.jvp, and if
    the population difference is absent here it will be absent in exact mode.
    TODO: choose eps by checking the difference is linear in eps over a decade.
    """
    handle_state = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if handle_state.get("shift") is not None:
            h[:, -1] = h[:, -1] + handle_state["shift"]
        return (h,) + out[1:] if isinstance(out, tuple) else h

    h_handle = model.model.layers[layer_l].register_forward_hook(hook)
    try:
        ids = tok(prompt, return_tensors="pt").to(model.device)
        vt = torch.tensor(v, device=model.device).to(next(model.parameters()).dtype)
        with torch.no_grad():
            handle_state["shift"] = None
            base = model(**ids, output_hidden_states=True).hidden_states[layer_L][0, -1]
            handle_state["shift"] = eps * vt
            pert = model(**ids, output_hidden_states=True).hidden_states[layer_L][0, -1]
    finally:
        h_handle.remove()
    return ((pert - base) / eps).float().cpu().numpy()


# --------------------------------------------------------------------------- #
# The decomposition
# --------------------------------------------------------------------------- #

def dispersion(V):
    """Angles to the mean direction; resultant length 1 = perfectly constant field."""
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    m = Vn.mean(0); m /= max(np.linalg.norm(m), 1e-12)
    ang = np.degrees(np.arccos(np.clip(Vn @ m, -1, 1)))
    return dict(median_deg=float(np.median(ang)),
                p90_deg=float(np.percentile(ang, 90)),
                resultant=float(np.linalg.norm(Vn.mean(0))))


def split_curvature(U, V_fixed_u, V_star):
    """
    U          : grad log r(h_L) across h   -> STATISTICAL factor
    V_fixed_u  : J(h)^T u_bar (u held FIXED at its mean) -> NETWORK factor
    V_star     : J(h)^T u(h)                -> total field

    Reading the result:
      statistical >> network -> the DRE machinery is the contribution
      network >> statistical -> the paper is about network geometry
      both small              -> the constant vector is already the field; stop
    """
    return dict(statistical=dispersion(U),
                network=dispersion(V_fixed_u),
                total=dispersion(V_star))


def main(a):
    d = np.load(a.cache)
    # TODO: use view_paired() from cache_schema.py rather than assuming order
    Hc, Hw = d["h_correct"], d["h_wrong"]
    head, acc = fit_paired_head(Hc[:, a.layer_L, :], Hw[:, a.layer_L, :])
    print(f"paired head held-out pair accuracy: {acc:.3f}")
    # Sanity: near 0.5 means log r was not learned and every field below is noise.

    U = grad_log_r(head, np.vstack([Hc, Hw])[:, a.layer_L, :])
    print(json.dumps(dispersion(U), indent=2))
    # TODO: load the model, compute V_star and V_fixed_u via vjp_field, then
    # split_curvature(U, V_fixed_u, V_star). Run the statistical half first --
    # it needs no model at all.


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./cache/cache.npz")
    ap.add_argument("--layer-l", type=int, default=7)
    ap.add_argument("--layer-L", type=int, default=15)
    ap.add_argument("--model", default=None)
    main(ap.parse_args())
