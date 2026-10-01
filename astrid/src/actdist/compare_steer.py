"""Detection with and without steering, from the saved scores of finished experiments.

    uv run python -m actdist.compare_steer --exps 'outputs/steering/resnet18_scratch_mnist__penultimate__*'
    uv run python -m actdist.compare_steer --exps outputs/steering/<name> --detectors energy msp

Same held-out probes, same threshold t_D (set on clean ID calibration at `tpr`), vectors at the
trained radius. Rows:
    clean            no steering
    ID steered       ID probes moved by the id2ood swarm, OOD clean
    OOD steered      OOD probes moved by the ood2id swarm, ID clean
    both steered     both at once (the k-th direction of each swarm paired)
for each direction family: learned swarm, v_toward, random directions on the same sphere.
"median" is over directions; "best" is the learned direction with the lowest AUROC.
"""

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

from .detectors import auroc

FAMILIES = ("learned", "toward", "random")


def detection(id_s: np.ndarray, ood_s: np.ndarray, t: float, tpr: float) -> dict:
    return {"auroc": auroc(id_s, ood_s),
            "fpr95": float((ood_s <= np.quantile(id_s, tpr)).mean()),
            "id_acc": float((id_s <= t).mean()),
            "ood_acc": float((ood_s <= t).mean()),
            "med_id": float(np.median(id_s)),                # median detector score of the ID side
            "med_ood": float(np.median(ood_s))}              # ... and of the OOD side (compare with t_D)


def aggregate(rows: list[dict]) -> tuple[dict, dict]:
    """Median over directions, and the single direction with the lowest AUROC."""
    med = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    return med, min(rows, key=lambda r: r["auroc"])


def compare(exp: Path, detectors: list[str] | None) -> list[tuple]:
    m = json.loads((exp / "metrics.json").read_text())
    cfg = json.loads((exp / "config.json").read_text())
    z = np.load(exp / "paths.npz")
    if set(m["swarms"]) != {"id2ood", "ood2id"}:
        raise SystemExit(f"{exp.name}: needs both swarms")
    i_r, tpr = m["radius_index"], cfg["tpr"]
    saved = sorted({k.split("__")[1] for k in z.files if k.endswith("__scores")},
                   key=lambda d: cfg["detectors"].index(d) if d in cfg["detectors"] else len(cfg["detectors"]))
    out = []
    for d in detectors or saved:
        t = m["thresholds"][d]
        S = {s: z[f"{s}__{d}__scores"].astype(np.float32) for s in ("id2ood", "ood2id")}
        B = {s: z[f"{s}__family_bounds"] for s in S}
        id0, ood0 = S["id2ood"][0, 0], S["ood2id"][0, 0]           # alpha = 0: clean probes
        out.append((d, "clean", "", detection(id0, ood0, t, tpr), None))
        for f_i, f in enumerate(FAMILIES):
            fam = {s: S[s][B[s][f_i]:B[s][f_i + 1], i_r] for s in S}  # [M, N] steered at the radius
            n = min(len(fam["id2ood"]), len(fam["ood2id"]))
            settings = {
                "ID steered": [detection(fam["id2ood"][k], ood0, t, tpr) for k in range(len(fam["id2ood"]))],
                "OOD steered": [detection(id0, fam["ood2id"][k], t, tpr) for k in range(len(fam["ood2id"]))],
                "both steered": [detection(fam["id2ood"][k], fam["ood2id"][k], t, tpr) for k in range(n)],
            }
            for setting, rows in settings.items():
                med, best = aggregate(rows)
                out.append((d, setting, f, med, best if f == "learned" else None))
    return out


CSV_CONFIG_KEYS = ("run", "layer", "sim_layer", "radius", "radius_mode", "loss", "swarm_size", "steps", "seed")


def csv_row(exp: Path, cfg: dict, m: dict, d, setting, fam, med, best) -> dict:
    """One CSV row: the run's settings, then the median-over-directions metrics (and the best learned direction)."""
    row = {"experiment": exp.name, **{k: cfg.get(k) for k in CSV_CONFIG_KEYS}, "radius_abs": m["radius"],
           "detector": d, "threshold": m["thresholds"][d], "setting": setting, "directions": fam or "none",
           "auroc": med["auroc"], "fpr95": med["fpr95"],
           "id_rejected": 1 - med["id_acc"], "ood_accepted": med["ood_acc"],
           "median_score_id": med["med_id"], "median_score_ood": med["med_ood"]}
    row["best_auroc"] = best["auroc"] if best else None
    return row


def fmt(r: dict) -> str:
    return (f"{r['auroc']:6.3f} {r['fpr95']:6.3f} {r['id_acc']:8.3f} {r['ood_acc']:8.3f} "
            f"{r['med_id']:10.4g} {r['med_ood']:10.4g}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exps", nargs="+", required=True, help="experiment folders or glob patterns")
    p.add_argument("--detectors", nargs="+", help="default: every detector with saved scores")
    p.add_argument("--csv", type=Path, help="also write every row (all experiments) to this CSV file")
    args = p.parse_args()
    exps = sorted({Path(x) for pat in args.exps for x in glob.glob(pat) if (Path(x) / "paths.npz").exists()})
    if not exps:
        raise SystemExit("no finished experiments matched")
    csv_rows = []
    for exp in exps:
        m = json.loads((exp / "metrics.json").read_text())
        print(f"\n{exp.name}   (radius {m['radius']:.3g})")
        print("thresholds t_D: " + ", ".join(f"{d} {t:.4g}" for d, t in m["thresholds"].items()))
        head = f"{'detector':9s} {'setting':13s} {'directions':10s} {'AUROC':>6s} {'FPR95':>6s} {'ID acc.':>8s} " \
               f"{'OOD acc.':>8s} {'med ID':>10s} {'med OOD':>10s}   {'best learned direction':>54s}"
        print(head)
        print("-" * len(head))
        cfg = json.loads((exp / "config.json").read_text())
        for d, setting, fam, med, best in compare(exp, args.detectors):
            tail = f"   {fmt(best)}" if best else ""
            print(f"{d:9s} {setting:13s} {fam:10s} {fmt(med)}{tail}")
            csv_rows.append(csv_row(exp, cfg, m, d, setting, fam, med, best))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"wrote {len(csv_rows)} rows to {args.csv}")
    print("\nAUROC: 1 = perfect detection, 0.5 = chance, < 0.5 = inverted. FPR95: OOD accepted when 95% of ID is. "
          "ID/OOD acc.: fraction accepted as ID at t_D. Directions: median over the family; "
          "'best' = learned direction with the lowest AUROC. med ID / med OOD: median detector score of each "
          "side (higher = more OOD; accepted when <= t_D).")


if __name__ == "__main__":
    main()
