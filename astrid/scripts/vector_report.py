"""Compare the vectors of a steering swarm with each other.

    uv run python scripts/vector_report.py outputs/steering/<name> --sub all_vectors

Needs an evaluation of every vector (eval_steer --eval-vectors K --out all_vectors). For each swarm
and detector, at the trained radius, every learned vector is scored on its own side only:
    id2ood  ID rejected (fraction of ID probes over t_D) and AUROC(steered ID, clean OOD)
    ood2id  OOD accepted (fraction of OOD probes under t_D) and AUROC(clean ID, steered OOD)
(lower AUROC = stronger attack). Reported: the spread across vectors, the best and worst vector,
whether the same vectors are best for every detector (Spearman across vectors), the swarm's
geometry (pairwise cosines, clusters of near-identical vectors, cosine to the mean direction) and
the both-steered AUROC of the best id2ood + best ood2id pair against the median pair.
Writes <exp>/<sub>/vectors.csv (one row per swarm x vector).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actdist.detectors import auroc  # noqa: E402

CLUSTER_COS = 0.99  # vectors with cosine above this are "the same direction"


def clusters(cos: np.ndarray, thr: float) -> np.ndarray:
    """Connected components of the graph cos > thr: a cluster id per vector, largest cluster = 0."""
    n = len(cos)
    label = -np.ones(n, int)
    c = 0
    for i in range(n):
        if label[i] >= 0:
            continue
        stack, label[i] = [i], c
        while stack:
            j = stack.pop()
            for k in np.where((cos[j] > thr) & (label < 0))[0]:
                label[k] = c
                stack.append(k)
        c += 1
    sizes = np.bincount(label)
    order = np.argsort(-sizes, kind="stable")
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[label]


def summ(x) -> str:
    x = np.asarray(x, float)
    q = np.quantile(x, [0, 0.25, 0.5, 0.75, 1])
    return " ".join(f"{v:7.3f}" for v in q)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp", type=Path)
    p.add_argument("--sub", default="all_vectors")
    args = p.parse_args()
    ev = args.exp / args.sub
    m = json.loads((ev / "metrics.json").read_text())
    z = np.load(ev / "paths.npz")
    dirs = torch.load(args.exp / "swarms.pt", map_location="cpu")["directions"]
    i_r, dets = m["radius_index"], list(m["thresholds"])
    dets = [d for d in dets if f"id2ood__{d}__scores" in z.files]

    S = {(s, d): z[f"{s}__{d}__scores"].astype(np.float32) for s in ("id2ood", "ood2id") for d in dets}
    id0 = {d: S["id2ood", d][0, 0] for d in dets}      # alpha = 0: clean ID probes
    ood0 = {d: S["ood2id", d][0, 0] for d in dets}     # clean OOD probes

    per, rows = {}, []
    print(f"{args.exp.name}  ·  radius {m['radius']:.3g}  ·  every vector scored alone, at the trained radius")
    for s in ("id2ood", "ood2id"):
        b = z[f"{s}__family_bounds"]
        idx = np.array(m["swarms"][s]["eval_vector_indices"])
        U = torch.nn.functional.normalize(dirs[s].float(), dim=1).numpy()
        K = len(U)
        if len(idx) != K:
            raise SystemExit(f"{s}: only {len(idx)} of {K} vectors were evaluated; rerun with --eval-vectors {K}")
        cos = U @ U.T
        mean_dir = U.mean(0) / np.linalg.norm(U.mean(0))
        cos_mean = U @ mean_dir
        lab = clusters(cos, CLUSTER_COS)
        off = cos[~np.eye(K, dtype=bool)]
        print(f"\n=== {s}: {K} vectors · pairwise cosine min {off.min():.3f} median {np.median(off):.3f} max {off.max():.3f}"
              f" · {lab.max() + 1} clusters at cos > {CLUSTER_COS} (sizes {np.bincount(lab)[:8].tolist()}"
              f"{'…' if lab.max() >= 8 else ''})")
        print(f"    {'detector':12s} {'metric':13s}   min     q25  median     q75     max   best vec  worst vec")
        per[s] = {}
        for d in dets:
            X = S[s, d][b[0]:b[1], i_r]                      # [K, N] steered scores at the radius
            t = m["thresholds"][d]
            if s == "id2ood":
                flip = (X > t).mean(1)
                au = np.array([auroc(x, ood0[d]) for x in X])
                fname = "ID rejected"
            else:
                flip = (X <= t).mean(1)
                au = np.array([auroc(id0[d], x) for x in X])
                fname = "OOD accepted"
            med = np.median(X, axis=1)
            per[s][d] = dict(flip=flip, auroc=au, median_score=med)
            best, worst = int(idx[np.argmin(au)]), int(idx[np.argmax(au)])
            print(f"    {d:12s} {fname:13s} {summ(flip)}")
            print(f"    {'':12s} {'AUROC':13s} {summ(au)}   #{best:<7d} #{worst}")
            if d == "mahalanobis":
                print(f"    {'':12s} {'median score':13s} {summ(med)}   (t_D {t:.4g})")
        # same vectors best for every detector?
        print(f"    rank agreement across vectors (Spearman of per-vector AUROC):")
        for i, d1 in enumerate(dets):
            line = " ".join(f"{spearmanr(per[s][d1]['auroc'], per[s][d2]['auroc']).statistic:6.2f}" for d2 in dets)
            print(f"      {d1:12s} {line}")
        print(f"      {'':12s} " + " ".join(f"{d[:6]:>6s}" for d in dets))
        # geometry vs performance
        for d in dets:
            rho = spearmanr(cos_mean, per[s][d]["auroc"]).statistic
            print(f"    cos to mean direction vs AUROC ({d}): Spearman {rho:+.2f}", end="")
            big = lab == 0
            print(f"   · largest cluster ({big.sum()} vectors) median AUROC {np.median(per[s][d]['auroc'][big]):.3f}"
                  f", others {np.median(per[s][d]['auroc'][~big]) if (~big).any() else float('nan'):.3f}")
        for k in range(K):
            r = {"swarm": s, "vector": int(idx[k]), "cluster": int(lab[k]), "cos_to_mean": float(cos_mean[k])}
            for d in dets:
                r[f"{d}_flip"] = float(per[s][d]["flip"][k])
                r[f"{d}_auroc"] = float(per[s][d]["auroc"][k])
                r[f"{d}_median_score"] = float(per[s][d]["median_score"][k])
            rows.append(r)

    # best pair vs median pair, both steered
    print("\n=== both steered: best id2ood vector + best ood2id vector (chosen per detector) vs the median over pairs")
    b1, b2 = z["id2ood__family_bounds"], z["ood2id__family_bounds"]
    for d in dets:
        A = S["id2ood", d][b1[0]:b1[1], i_r]
        B = S["ood2id", d][b2[0]:b2[1], i_r]
        pair_med = np.median([auroc(A[k], B[k]) for k in range(min(len(A), len(B)))])
        best = auroc(A[np.argmin(per["id2ood"][d]["auroc"])], B[np.argmin(per["ood2id"][d]["auroc"])])
        print(f"    {d:12s} median pair {pair_med:.3f}   best pair {best:.3f}   (clean {m['clean'][d]['auroc']:.3f})")

    out = ev / "vectors.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
