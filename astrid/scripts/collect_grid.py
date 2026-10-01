"""Collect the E16 grid: merge every outputs/steering/grid__*/summary.csv and print the result tables.

    uv run python scripts/collect_grid.py                     # -> outputs/steering/grid_summary.csv + tables
    uv run python scripts/collect_grid.py --pattern 'grid__*' --out outputs/steering/grid_summary.csv

Per model and detector, rows = (steered layer, similarity layer), columns = radius, for the learned
swarm (median over its directions):
    ID rejected   fraction of ID probes rejected when steered by id2ood
    OOD accepted  fraction of OOD probes accepted when steered by ood2id
    AUROC         both steered (ID by id2ood, OOD by ood2id); clean AUROC in the header
For mahalanobis, also the median score of steered ID and steered OOD (threshold t_D in the header).
"""

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs" / "steering"
SIM_ORDER = ["same", "next", "penultimate"]


def sim_label(layer: str, sim: str) -> str:
    if sim == layer:
        return "same"
    return "penultimate" if sim == "penultimate" else "next"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pattern", default="grid__*")
    p.add_argument("--out", type=Path, default=ROOT / "grid_summary.csv")
    args = p.parse_args()

    files = sorted(glob.glob(str(ROOT / args.pattern / "summary.csv")))
    rows = [r for f in files for r in csv.DictReader(open(f))]
    if not rows:
        raise SystemExit("no summary.csv found")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(files)} runs, {len(rows)} rows -> {args.out}")

    # index: (run, detector, setting, directions, layer, sim_label, radius) -> row
    idx, clean, thr = {}, {}, {}
    radii = sorted({float(r["radius"]) for r in rows})
    for r in rows:
        key = (r["run"], r["detector"], r["setting"], r["directions"], r["layer"],
               sim_label(r["layer"], r["sim_layer"]), float(r["radius"]))
        idx[key] = r
        if r["setting"] == "clean":
            clean[(r["run"], r["detector"])] = float(r["auroc"])
            thr[(r["run"], r["detector"])] = float(r["threshold"])
    missing = defaultdict(int)

    def cell(run, det, setting, layer, sim, rad, field, transform=float):
        r = idx.get((run, det, setting, "learned", layer, sim, rad))
        if r is None:
            missing[(run, layer, sim, rad)] += 1
            return "      -"
        return transform(r[field])

    runs = sorted({r["run"] for r in rows})
    layer_order = sorted({r["layer"] for r in rows}, key=lambda l: (l.split(".")[0], int(l.split(".")[1])))
    dets = list(dict.fromkeys(r["detector"] for r in rows))
    for run in runs:
        for det in dets:
            print(f"\n=== {run} · {det} · clean AUROC {clean.get((run, det), float('nan')):.3f} · t_D {thr.get((run, det), float('nan')):.4g}")
            blocks = [("ID rejected", "ID steered", "id_rejected"), ("OOD accepted", "OOD steered", "ood_accepted"),
                      ("AUROC, both steered", "both steered", "auroc")]
            if det == "mahalanobis":
                blocks += [("median score, steered ID", "ID steered", "median_score_id"),
                           ("median score, steered OOD", "OOD steered", "median_score_ood")]
            for title, setting, field in blocks:
                print(f"  {title}")
                print(f"    {'steer':9s} {'sim':12s} " + " ".join(f"r={rad:<7g}" for rad in radii))
                for layer in layer_order:
                    for sim in SIM_ORDER:
                        if not any((run, det, setting, "learned", layer, sim, rad) in idx for rad in radii):
                            continue
                        vals = [cell(run, det, setting, layer, sim, rad, field) for rad in radii]
                        print(f"    {layer:9s} {sim:12s} " + " ".join(
                            f"{v:9.4g}" if isinstance(v, float) else f"{v:>9s}" for v in vals))
    print(f"\nruns found: {len(files)} (48 per grid)")


if __name__ == "__main__":
    main()
