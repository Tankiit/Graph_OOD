"""Markdown report of the E16 grid: per model and radius, the best setting for each detector
(lowest both-steered AUROC of the typical vector) with OOD passing / ID not passing / AUROC.

    uv run python scripts/grid_best_settings.py > outputs/reports/e16_grid_typical_vectors.md
    uv run python scripts/grid_best_settings.py --arch vit > outputs/reports/e17_grid_vit_typical_vectors.md

"Typical vector" = the median over the evaluated learned vectors of a swarm, as stored in
outputs/steering/grid_summary.csv (scripts/collect_grid.py).
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DETS = [("energy", "Energy"), ("msp", "MSP"), ("maxlogit", "Max-logit"), ("knn", "kNN"), ("mahalanobis", "Mahalanobis")]
ARCHS = {  # csv, models, title, network description, date
    "resnet": ("grid_summary.csv", [("resnet18_scratch_fmnist", "FashionMNIST"), ("resnet18_scratch_mnist", "MNIST")],
               "E16 steering grid", "the scratch ResNet-18", "30 September 2026"),
    "vit": ("gridvit_summary.csv", [("vit_small_ft_fmnist", "FashionMNIST"), ("vit_small_ft_mnist", "MNIST")],
            "E17 steering grid, ViT-S", "the fine-tuned ViT-S/16", "1 October 2026"),
}
RADII = ["0.25", "0.5", "1.0"]


def pct(x: float) -> str:
    """0% / 100% only when exact; two decimals just below 100% so 99.95% never prints as 100%."""
    x *= 100
    if x in (0, 100):
        return f"{x:.0f}%"
    one = f"{x:.1f}"
    return f"{x:.2f}%" if one in ("100.0", "0.0") else f"{one}%"


def auroc(x: float) -> str:
    return "< 0.0001" if x < 1e-4 else (f"{x:.4f}" if x < 0.01 else f"{x:.3f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHS), default="resnet")
    csv_name, MODELS, title, network, date = ARCHS[ap.parse_args().arch]
    rows = list(csv.DictReader(open(ROOT / "outputs" / "steering" / csv_name)))
    get = {(r["experiment"], r["detector"], r["setting"], r["directions"]): r for r in rows}
    out = ["---", f'title: "{title}: best setting per detector (typical vector)"',
           f'date: "{date}"', 'geometry: "margin=2.2cm"', "fontsize: 10pt", "header-includes:",
           "  - \\usepackage{booktabs}", "  - \\usepackage{needspace}", "---", "",
           f"Steering vectors trained with `infonce` (K = 128 per swarm, 1000 steps) on {network}, "
           "OOD = 300K Random Images + STL-10. For each radius and detector, the table shows the setting "
           "(steered layer, layer where the similarity is measured) with the lowest AUROC when both sides are "
           "steered.", "",
           "- **OOD passing**: share of steered OOD images the detector accepts as ID.",
           "- **ID not passing**: share of steered ID images the detector rejects.",
           "- **AUROC**: ID moved by the ID→OOD swarm and OOD by the OOD→ID swarm, scored together "
           "(0.5 = chance, below 0.5 = inverted).",
           "- All values are for the **typical vector**: the median over the 64 evaluated learned vectors of "
           "each swarm, on 1000 ID and 1000 OOD held-out images. Thresholds accept 95% of clean ID.", ""]
    for model, mname in MODELS:
        mrows = [r for r in rows if r["run"] == model]
        exps = sorted({r["experiment"] for r in mrows})
        out += [f"# {mname}", "", "**Without steering**", "",
                "| Detector | OOD passing | ID not passing | AUROC |", "|---|---|---|---|"]
        for d, dn in DETS:
            c = next(r for r in mrows if r["detector"] == d and r["setting"] == "clean")
            out.append(f"| {dn} | {pct(float(c['ood_accepted']))} | {pct(float(c['id_rejected']))} | {auroc(float(c['auroc']))} |")
        for rad in RADII:
            out += ["", "\\needspace{12\\baselineskip}", "", f"**r = {rad}**", "",
                    "| Detector | Best setting | OOD passing | ID not passing | AUROC |",
                    "|-----------|------------------------------|----------|-----------|--------|"]
            for d, dn in DETS:
                cand = [e for e in exps if get[(e, d, "both steered", "learned")]["radius"] == rad]
                e = min(cand, key=lambda e: float(get[(e, d, "both steered", "learned")]["auroc"]))
                b = get[(e, d, "both steered", "learned")]
                sim = "same" if b["sim_layer"] == b["layer"] else ("penultimate" if b["sim_layer"] == "penultimate" else "next")
                idr = float(get[(e, d, "ID steered", "learned")]["id_rejected"])
                ooa = float(get[(e, d, "OOD steered", "learned")]["ood_accepted"])
                out.append(f"| {dn} | {b['layer']} · sim {sim} | {pct(ooa)} | {pct(idr)} | {auroc(float(b['auroc']))} |")
        out += ["", "\\newpage", ""]
    print("\n".join(out[:-2]))


if __name__ == "__main__":
    main()
