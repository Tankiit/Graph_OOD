"""Clean OOD detection (no steering) for every detector, on the probes a steering run evaluates on.

    uv run python -m actdist.clean_detect --config configs/steer/scratch_mnist_layer3_r0.1.toml
    uv run python -m actdist.clean_detect --set run=resnet18_ft_fmnist 'ood_sources=["300k","stl10"]'

Uses the same configuration as train_steer (run, ood_sources, detectors, knn_k, ref_size, tpr,
n_probe_id, n_probe_ood, seed): same ID reference, calibration split and held-out probes, so the
numbers are the "no steering" baseline of that steering run. Reports AUROC, FPR95 and the fractions
accepted at t_D, over all OOD probes and per OOD source.
Output: printed table and outputs/clean_detection/<run>.json.
"""

import json

import numpy as np

from .detectors import auroc, fpr_at_tpr
from .steer_config import parse_configs
from .train_steer import ROOT, Context


def run(cfg):
    ctx = Context(cfg)
    probes = {"id": ctx.encode(ctx.id_probe, keep_h=False), "ood": ctx.encode(ctx.ood_probe, keep_h=False)}
    source = probes["ood"]["label"].numpy()                   # OOD "labels" are source indices
    names = ctx.ood_probe.source_names
    res = {}
    for d in cfg.detectors:
        s_id = ctx.score(d, probes["id"]["logits"], probes["id"]["feat"])
        s_ood = ctx.score(d, probes["ood"]["logits"], probes["ood"]["feat"])
        t = ctx.thresholds[d]
        row = {"all": _metrics(s_id, s_ood, t, cfg.tpr)}
        for k, n in enumerate(names):
            row[n] = _metrics(s_id, s_ood[source == k], t, cfg.tpr)
        row["threshold"] = t
        res[d] = row

    print(f"\n{cfg.run} · clean detection (no steering) · ID test {len(probes['id']['logits'])} probes, "
          f"OOD {len(source)} probes ({', '.join(names)}) · t_D at {cfg.tpr:.0%} ID accepted · "
          f"knn k={cfg.knn_k}, reference {cfg.ref_size} ID train images")
    head = f"{'detector':12s} {'OOD set':12s} {'AUROC':>6s} {'FPR95':>6s} {'ID acc.':>8s} {'OOD acc.':>8s}"
    print(head)
    print("-" * len(head))
    for d, row in res.items():
        for n in ["all"] + names:
            m = row[n]
            print(f"{d:12s} {n:12s} {m['auroc']:6.3f} {m['fpr95']:6.3f} {m['id_acc']:8.3f} {m['ood_acc']:8.3f}")
    out = ROOT / "clean_detection" / f"{cfg.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": {k: getattr(cfg, k) for k in
                                          ("run", "ood_sources", "detectors", "knn_k", "ref_size", "tpr",
                                           "n_probe_id", "n_probe_ood", "seed")}, "results": res}, indent=2))
    print(f"saved {out}")


def _metrics(s_id, s_ood, t, tpr):
    return {"auroc": auroc(s_id, s_ood), "fpr95": fpr_at_tpr(s_id, s_ood, tpr),
            "id_acc": float((s_id <= t).mean()), "ood_acc": float((s_ood <= t).mean()), "n_ood": int(len(s_ood))}


def main():
    cfgs, _ = parse_configs(description=__doc__)
    for cfg in cfgs:
        run(cfg)


if __name__ == "__main__":
    main()
