"""Best learned vector per swarm, for every run of a grid: its ID / OOD rates and the best-pair AUROC.

    uv run python scripts/best_vectors.py --pattern 'grid__resnet18_scratch_fmnist__*'

For each run and detector, at the trained radius, among the evaluated learned vectors:
    best id2ood vector   lowest AUROC(steered ID, clean OOD)   -> its ID rejected ("ID not passing")
    best ood2id vector   lowest AUROC(clean ID, steered OOD)   -> its OOD accepted ("OOD passing")
    best pair            both steered with those two vectors   -> AUROC
Then, per radius and detector, the setting with the lowest best-pair AUROC.
The vectors are picked on the evaluation probes themselves, so these numbers are optimistic.
Writes outputs/steering/best_vectors_<tag>.csv (one row per run x detector).
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from actdist.detectors import auroc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "outputs" / "steering"


def sim_label(layer, sim):
    return "same" if sim == layer else ("penultimate" if sim == "penultimate" else "next")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", required=True)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    rows = []
    for d in sorted(glob.glob(str(ROOT / args.pattern))):
        d = Path(d)
        if not (d / "paths.npz").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())
        m = json.loads((d / "metrics.json").read_text())
        z = np.load(d / "paths.npz")
        i_r = m["radius_index"]
        b_id, b_ood = z["id2ood__family_bounds"], z["ood2id__family_bounds"]
        for det in cfg["detectors"]:
            t = m["thresholds"][det]
            Sa = z[f"id2ood__{det}__scores"].astype(np.float32)
            Sb = z[f"ood2id__{det}__scores"].astype(np.float32)
            id0, ood0 = Sa[0, 0], Sb[0, 0]
            A = Sa[b_id[0]:b_id[1], i_r]                  # [M, N] steered ID, learned vectors
            B = Sb[b_ood[0]:b_ood[1], i_r]                # [M, N] steered OOD
            au_a = np.array([auroc(x, ood0) for x in A])
            au_b = np.array([auroc(id0, x) for x in B])
            ka, kb = int(np.argmin(au_a)), int(np.argmin(au_b))
            rows.append({
                "run": cfg["run"], "layer": cfg["layer"], "sim": sim_label(cfg["layer"], cfg["sim_layer"]),
                "radius": cfg["radius"], "detector": det,
                "best_id2ood_vector": m["swarms"]["id2ood"]["eval_vector_indices"][ka],
                "best_ood2id_vector": m["swarms"]["ood2id"]["eval_vector_indices"][kb],
                "id_not_passing_best": float((A[ka] > t).mean()),
                "ood_passing_best": float((B[kb] <= t).mean()),
                "auroc_best_pair": auroc(A[ka], B[kb]),
                "id_not_passing_median": float(np.median((A > t).mean(1))),
                "ood_passing_median": float(np.median((B <= t).mean(1))),
                "clean_id_not_passing": float((id0 > t).mean()), "clean_ood_passing": float((ood0 <= t).mean()),
                "clean_auroc": auroc(id0, ood0),
            })
    out = ROOT / f"best_vectors{('_' + args.tag) if args.tag else ''}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len({(r['run'], r['layer'], r['sim'], r['radius']) for r in rows})} runs -> {out}")

    for run in sorted({r["run"] for r in rows}):
        print(f"\n=== {run}")
        for det in dict.fromkeys(r["detector"] for r in rows):
            c = next(r for r in rows if r["run"] == run and r["detector"] == det)
            print(f"  clean {det:12s} OOD passing {c['clean_ood_passing']:.1%}  ID not passing {c['clean_id_not_passing']:.1%}"
                  f"  AUROC {c['clean_auroc']:.3f}")
        for rad in sorted({r["radius"] for r in rows}):
            print(f"  r = {rad:g}")
            for det in dict.fromkeys(r["detector"] for r in rows):
                cand = [r for r in rows if r["run"] == run and r["detector"] == det and r["radius"] == rad]
                b = min(cand, key=lambda r: r["auroc_best_pair"])
                print(f"    {det:12s} {b['layer']} · sim {b['sim']:12s} OOD passing {b['ood_passing_best']:7.1%}"
                      f"  ID not passing {b['id_not_passing_best']:7.1%}  AUROC {b['auroc_best_pair']:.5f}"
                      f"  (vectors #{b['best_id2ood_vector']} / #{b['best_ood2id_vector']};"
                      f" median vectors: {b['ood_passing_median']:.1%} / {b['id_not_passing_median']:.1%})")


if __name__ == "__main__":
    main()
