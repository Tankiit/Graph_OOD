#!/usr/bin/env python3
"""
build_probe_cache.py -- adapt the ova-arr-extract-full volume layout to the
cache.npz that jacobian_probe.py reads.

SOURCE (Modal volume `ova-arr-extract-full`, full tree)
    {model}/{dataset}/contrastive_h.pt     paired hidden states, all layers
    {model}/{dataset}/contrastive_meta.pt  records: question / correct / wrong

    The `compact/` tree in the same volume is NOT usable here: it keeps a single
    pooled layer (layer_rel=-4, response_mean), and the whole point of the probe
    is to relate an intervention layer l to a readout layer L.

TARGET
    cache.npz with
        h_correct  [n, n_layers, d]   float32
        h_wrong    [n, n_layers, d]   float32
        prompts    [n]                for the VJP/JVP half (needs the model)
        layers     [n_layers]         absolute layer indices kept
"""

import argparse
import os

import numpy as np
import torch


def _find_pair(obj):
    """Return (h_pos, h_neg) from whatever contrastive_h.pt happens to hold."""
    if isinstance(obj, dict):
        for kp, kn in (("h_pos", "h_neg"), ("pos", "neg"),
                       ("h_correct", "h_wrong"), ("correct", "wrong")):
            if kp in obj and kn in obj:
                return obj[kp], obj[kn]
        # single tensor stored under one key
        tens = [v for v in obj.values() if torch.is_tensor(v)]
        if len(tens) == 1:
            obj = tens[0]
        else:
            raise KeyError(f"cannot find a pos/neg pair in keys {list(obj)}")
    if torch.is_tensor(obj):
        # a pair axis of exactly 2 somewhere near the front
        for ax in (1, 2):
            if obj.ndim > ax and obj.shape[ax] == 2:
                return obj.select(ax, 0), obj.select(ax, 1)
        if obj.shape[0] % 2 == 0:                      # stacked [2n, ...]
            n = obj.shape[0] // 2
            return obj[:n], obj[n:]
    raise TypeError(f"unrecognised contrastive_h payload: {type(obj)}")


def _pick_pool(h, n_pools, pool_idx):
    """Drop the pooling axis if the file carries one (pool_names in the meta)."""
    if n_pools and h.ndim >= 3:
        for ax in range(1, h.ndim):
            if h.shape[ax] == n_pools and ax != h.ndim - 1:
                return h.select(ax, pool_idx)
    return h


def build(src, model, dataset, out, pool="response_mean", layers=None):
    base = os.path.join(src, model, dataset)
    if not os.path.isfile(os.path.join(base, "contrastive_meta.pt")):
        alt = os.path.join(base, dataset)       # `modal volume get` nests the leaf
        if os.path.isfile(os.path.join(alt, "contrastive_meta.pt")):
            base = alt

    meta = torch.load(os.path.join(base, "contrastive_meta.pt"),
                      map_location="cpu", weights_only=False)
    m = meta.get("meta", {})
    pool_names = m.get("pool_names", [])
    pool_idx = pool_names.index(pool) if pool in pool_names else 0

    raw = torch.load(os.path.join(base, "contrastive_h.pt"),
                     map_location="cpu", weights_only=False)
    hc, hw = _find_pair(raw)
    hc = _pick_pool(hc, m.get("n_pools"), pool_idx).float()
    hw = _pick_pool(hw, m.get("n_pools"), pool_idx).float()

    n_layers = hc.shape[1]
    keep = np.arange(n_layers) if layers is None else np.array(layers)
    hc, hw = hc[:, keep], hw[:, keep]

    recs = meta.get("records", [])
    prompts = np.array([r.get("question", "") for r in recs], dtype=object)

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    np.savez_compressed(
        out,
        h_correct=hc.numpy(), h_wrong=hw.numpy(),
        prompts=prompts, layers=keep,
        model=model, dataset=dataset, pool=pool,
    )
    print(f"{model}/{dataset}: h_correct{tuple(hc.shape)} h_wrong{tuple(hw.shape)}"
          f"  pool={pool}  -> {out}")
    return hc.shape


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/ova_arr_full/full")
    ap.add_argument("--model", default="llama3_8b")
    ap.add_argument("--dataset", default="truthfulqa")
    ap.add_argument("--pool", default="response_mean")
    ap.add_argument("--out", default="./cache/cache.npz")
    build(**vars(ap.parse_args()))
