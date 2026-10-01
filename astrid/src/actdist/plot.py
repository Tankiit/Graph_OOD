"""Display 1-D per-neuron activation distributions.

Two views, both read from outputs/activations/{run}__{eval}.npz:

  grid     one panel per neuron; inside each panel a ridgeline with one row per class
           (correctly classified images, grouped by class) and a gray row for misclassified images.
  heatmap  every neuron of a layer at once: one row per neuron, color = density over
           per-neuron standardized activation.

    uv run python -m actdist.plot grid    --run resnet18_scratch_mnist --layer layer4.1 --select selective -k 16
    uv run python -m actdist.plot grid    --run vit_small_ft_fmnist --layer blocks.11 --neurons 3,17,42
    uv run python -m actdist.plot heatmap --run vit_small_ft_fmnist --layer all
    uv run python -m actdist.plot steer   --exp outputs/steering/<name>
    uv run python -m actdist.plot steer-sweep --exps 'outputs/steering/resnet18_scratch_mnist__*'
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .data import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[2] / "outputs"

# chart chrome (light mode)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES, SERIES_EDGE = "#2a78d6", "#1c5cab"
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 8, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 9, "axes.titlecolor": INK,
})


def load(run: str, eval_ds: str | None):
    train_ds = run.rsplit("_", 1)[1]
    eval_ds = eval_ds or train_ds
    z = np.load(ROOT / "activations" / f"{run}__{eval_ds}.npz")
    return z, eval_ds


def class_selectivity(a: np.ndarray, label: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Between-class share of the variance (eta^2, in [0, 1]) per neuron, on correct images only."""
    a, label = a[mask].astype(np.float32), label[mask]
    mu = a.mean(0)
    between = sum((label == c).sum() * (a[label == c].mean(0) - mu) ** 2
                  for c in np.unique(label)) / len(a)
    return between / (a.var(0) + 1e-8)


def select_neurons(a, label, correct, how: str, k: int, seed: int = 0) -> np.ndarray:
    d = a.shape[1]
    if how == "first":
        return np.arange(min(k, d))
    if how == "random":
        return np.sort(np.random.default_rng(seed).choice(d, min(k, d), replace=False))
    if how == "selective":
        return np.argsort(-class_selectivity(a, label, correct))[:k]
    if how == "variance":
        return np.argsort(-a.astype(np.float32).var(0))[:k]
    raise ValueError(how)


def ridgeline(ax, x: np.ndarray, groups: list[tuple[str, np.ndarray, str]], bins: int, show_labels: bool):
    """One density row per group on shared bins; each row is scaled to its own peak."""
    lo, hi = np.quantile(x, [0.005, 0.995])
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    dens = [np.histogram(np.clip(v, lo, hi), edges, density=True)[0] if len(v) else np.zeros(bins)
            for _, v, _ in groups]
    n = len(groups)
    for i, ((name, v, color), d) in enumerate(zip(groups, dens)):
        base = n - 1 - i
        ax.fill_between(centers, base, base + 0.9 * d / (d.max() or 1.0), color=color, alpha=0.35, lw=0, step="mid")
        ax.step(centers, base + 0.9 * d / (d.max() or 1.0), where="mid", color=color, lw=1.0)
        ax.axhline(base, color=GRID, lw=0.5, zorder=0)
        if show_labels:
            ax.text(lo, base + 0.15, f"{name} ", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.1, n + 0.8)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=6.5, length=2)


def cmd_grid(args):
    z, eval_ds = load(args.run, args.eval)
    layers = list(z["layers"])
    layer = args.layer if args.layer != "last" else layers[-1]
    a, label, correct = z[f"act__{layer}"].astype(np.float32), z["label"], z["correct"]
    names = CLASS_NAMES[eval_ds]

    idx = (np.array([int(i) for i in args.neurons.split(",")]) if args.neurons
           else select_neurons(a, label, correct, args.select, args.k, args.seed))
    ncol = min(args.cols, len(idx))
    nrow = int(np.ceil(len(idx) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol + 0.6, 2.6 * nrow + 0.6), squeeze=False)
    sel = class_selectivity(a, label, correct)
    for j, (ax, n) in enumerate(zip(axes.flat, idx)):
        x = a[:, n]
        groups = [(names[c], x[correct & (label == c)], SERIES) for c in range(10)]
        groups.append((f"wrong ({(~correct).sum()})", x[~correct], MUTED))
        ridgeline(ax, x, groups, args.bins, show_labels=(j % ncol == 0))
        ax.set_title(f"neuron {n}  ·  sel {sel[n]:.2f}", loc="left")
    for ax in axes.flat[len(idx):]:
        ax.set_visible(False)
    fig.suptitle(f"{args.run} on {eval_ds} test  ·  {layer}\nrows = correctly classified images per class; "
                 "gray = misclassified; each row scaled to its peak",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.supxlabel("activation", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0.03, 0, 1, 0.95))
    out = args.out or ROOT / "figures" / f"{args.run}__{eval_ds}__{layer}__grid_{args.select if not args.neurons else 'custom'}.png"
    save(fig, out)


def cmd_heatmap(args):
    z, eval_ds = load(args.run, args.eval)
    layers = list(z["layers"]) if args.layer == "all" else [args.layer]
    correct = z["correct"]
    ncol = min(4, len(layers))
    nrow = int(np.ceil(len(layers) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.4 * nrow), squeeze=False)
    edges = np.linspace(-4, 4, args.bins + 1)
    for ax, layer in zip(axes.flat, layers):
        a = z[f"act__{layer}"].astype(np.float32)[correct]
        # standardize each neuron so all rows share one x-axis (z-score)
        zs = (a - a.mean(0)) / (a.std(0) + 1e-8)
        H = np.stack([np.histogram(np.clip(zs[:, n], -4, 4), edges, density=True)[0] for n in range(a.shape[1])])
        order = np.argsort(-((zs ** 3).mean(0)))  # sort neurons by skewness
        ax.imshow(H[order], aspect="auto", cmap=SEQ, extent=(-4, 4, a.shape[1], 0),
                  vmin=0, vmax=np.quantile(H, 0.99), interpolation="nearest")
        ax.set_title(f"{layer}  ·  {a.shape[1]} neurons", loc="left")
        ax.set_xlabel("z-scored activation")
        ax.set_ylabel("neuron (sorted by skew)")
        for s in ax.spines.values():
            s.set_visible(False)
    for ax in axes.flat[len(layers):]:
        ax.set_visible(False)
    fig.suptitle(f"{args.run} on {eval_ds} test  ·  per-neuron density, correctly classified images",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = args.out or ROOT / "figures" / f"{args.run}__{eval_ds}__heatmap_{args.layer}.png"
    save(fig, out)


def save(fig, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"saved {out}")


# steering direction families: categorical slots 1-3 (validated all-pairs, light mode)
FAMILY_COLORS = {"learned": "#2a78d6", "toward": "#eb6834", "random": "#1baf7a"}
FAMILY_LABELS = {"learned": "learned swarm", "toward": "v_toward", "random": "random"}
SWARM_TITLES = {"id2ood": "ID → OOD: fraction of ID rejected", "ood2id": "OOD → ID: fraction of OOD accepted"}


def _flips(z, m, swarm: str, det: str) -> np.ndarray:
    S = z[f"{swarm}__{det}__scores"].astype(np.float32)
    t = m["thresholds"][det]
    return S > t if swarm == "id2ood" else S <= t


def _label_ends(ax, x, ends: list[tuple[float, str, str]], min_gap: float = 0.08):
    """Direct labels at the line ends, spread vertically so they never overlap."""
    ends = sorted(ends, reverse=True)
    ys = []
    for y, _, _ in ends:  # top-down: a label only moves if it would collide with the one above
        ys.append(min(y, 1.0) if not ys else min(y, ys[-1] - min_gap))
    for (y, text, color), ly in zip(ends, ys):
        ax.plot([x], [y], "o", ms=4, color=color, mec=SURFACE, mew=1, clip_on=False)
        ax.annotate(text, (x, y), xytext=(x + 0.02 * x, ly), textcoords="data", va="center",
                    fontsize=7, color=INK2, annotation_clip=False)


def cmd_steer(args):
    """One experiment: flipped fraction along the path (per family) and per-vector flip rate at the radius."""
    exp = Path(args.exp)
    z, m = np.load(exp / "paths.npz"), json.loads((exp / "metrics.json").read_text())
    alphas, r = z["alphas"], m["radius"]
    swarms = list(m["swarms"])
    fig, axes = plt.subplots(len(swarms), 2, figsize=(9.5, 3.3 * len(swarms)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1.6, 1]))
    for row, swarm in zip(axes, swarms):
        flip = _flips(z, m, swarm, args.detector)
        b = z[f"{swarm}__family_bounds"]
        ax, ax2 = row
        ends = []
        for f, (lo, hi) in zip(("learned", "toward", "random"), zip(b[:-1], b[1:])):
            cur = flip[lo:hi].mean(2)                                    # [M, A] per direction
            ever = np.maximum.accumulate(flip[lo:hi], axis=1).mean(2)
            c = FAMILY_COLORS[f]
            if hi - lo > 1:
                ax.fill_between(alphas, *np.quantile(cur, [0.25, 0.75], axis=0), color=c, alpha=0.15, lw=0)
            ax.plot(alphas, np.median(cur, 0), color=c, lw=2, label=FAMILY_LABELS[f])
            ax.plot(alphas, np.median(ever, 0), color=c, lw=1, ls=(0, (3, 2)))
            ends.append((float(np.median(cur, 0)[-1]), FAMILY_LABELS[f], c))
            rates = cur[:, m["radius_index"]]
            if hi - lo > 1:
                ax2.hist(rates, bins=np.linspace(0, 1, 26), histtype="step", lw=2, color=c, label=FAMILY_LABELS[f])
            else:
                ax2.axvline(rates[0], color=c, lw=2, label=FAMILY_LABELS[f])
        _label_ends(ax, alphas[-1], ends)
        ax.axvline(r, color=MUTED, lw=1, zorder=0)
        ax.annotate("trained radius", (r, 1.0), xytext=(3, -2), textcoords="offset points", fontsize=7,
                    color=INK2, va="top")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(0, alphas[-1] * 1.18)
        ax.set_title(SWARM_TITLES[swarm], loc="left")
        ax.set_xlabel("path length α (activation units)")
        ax.set_ylabel("fraction flipped  (solid: currently, dashed: ever)")
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax2.set_title("per-direction flip rate at the trained radius", loc="left")
        ax2.set_xlabel("flip rate")
        ax2.set_ylabel("directions")
        ax2.set_xlim(0, 1)
        for a in row:
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
    cfg = json.loads((exp / "config.json").read_text())
    clean = m["clean"][args.detector]
    fig.suptitle(f"{cfg['run']} · {cfg['layer']} · r={r:.3g} · K={cfg['swarm_size']} · loss {cfg['loss']} · "
                 f"OOD {'+'.join(cfg['ood_sources'])}\n{args.detector}: clean AUROC {clean['auroc']:.3f}, "
                 f"t_D at {cfg['tpr']:.0%} ID accepted; bands = IQR over directions",
                 x=0.01, ha="left", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, args.out or exp / f"steer_{args.detector}.png")


LINESTYLES = ["-", (0, (4, 2)), (0, (1, 1.5)), (0, (5, 1.5, 1, 1.5))]
SWEEP_AXES = ("radius", "layer", "name", "seed", "workers", "log_every")  # never used to split lines


def cmd_steer_sweep(args):
    """Many experiments: median flip rate at the radius against the radius, one panel per (swarm, layer).

    Experiments that differ in any other hyperparameter (e.g. loss) become separate lines, told apart
    by line style; --where key=value keeps only matching experiments.
    """
    where = [w.split("=", 1) for w in args.where]
    exps = []
    for e in sorted({p for pat in args.exps for p in glob.glob(pat)}):
        e = Path(e)
        if not (e / "metrics.json").exists():
            continue
        cfg = json.loads((e / "config.json").read_text())
        if all(str(cfg.get(k)) == v or json.dumps(cfg.get(k)) == v for k, v in where):
            exps.append((cfg, json.loads((e / "metrics.json").read_text())))
    if not exps:
        raise SystemExit("no finished experiments matched")
    varying = sorted(k for k in exps[0][0] if k not in SWEEP_AXES
                     and len({json.dumps(c.get(k)) for c, _ in exps}) > 1)
    combos = sorted({tuple(json.dumps(c.get(k)) for k in varying) for c, _ in exps})
    if len(combos) > len(LINESTYLES):
        raise SystemExit(f"{len(combos)} settings of {varying} to compare; narrow them with --where key=value")

    rows = []  # (layer, swarm, family, combo, radius, radius_mode, value)
    for cfg, m in exps:
        combo = tuple(json.dumps(cfg.get(k)) for k in varying)
        for swarm, s in m["swarms"].items():
            for f in ("learned", "random"):
                rows.append((cfg["layer"], swarm, f, combo, cfg["radius"], cfg["radius_mode"],
                             s[args.detector][f]["standard_at_radius"]["flip_rate"]["median"]))
    layers = sorted({r[0] for r in rows}, key=lambda l: (l == "penultimate", l))
    swarms = sorted({r[1] for r in rows})
    fig, axes = plt.subplots(len(swarms), len(layers), figsize=(2.6 * len(layers) + 1.6, 2.6 * len(swarms) + 0.8),
                             squeeze=False, sharey=True)
    for i, swarm in enumerate(swarms):
        for j, layer in enumerate(layers):
            ax = axes[i, j]
            for f in ("learned", "random"):
                for combo, ls in zip(combos, LINESTYLES):
                    pts = sorted((r[4], r[6]) for r in rows if r[:4] == (layer, swarm, f, combo))
                    if pts:
                        x, y = zip(*pts)
                        tag = ", ".join(f"{k}={json.loads(v)}" for k, v in zip(varying, combo))
                        ax.plot(x, y, marker="o", ls=ls, color=FAMILY_COLORS[f], lw=2, ms=4, mec=SURFACE,
                                label=FAMILY_LABELS[f] + (f" · {tag}" if tag else ""))
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{layer}" if i == 0 else "", loc="left")
            if j == 0:
                ax.set_ylabel(f"{swarm}\nmedian flip rate at r")
            ax.grid(axis="y", color=GRID, lw=0.5)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
    modes = {r[5] for r in rows}
    fig.supxlabel("radius" + (" (× median activation norm)" if modes == {"rel"} else ""), fontsize=8, color=INK2)
    handles, labels = {}, []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(list(handles.values()), list(handles), loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=7)
    fig.suptitle(f"Steering sweep · {args.detector} · learned swarm vs random directions on the same sphere",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.94))
    out = Path(args.out or ROOT / "figures" / f"steer_sweep_{args.detector}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("steer", help="one steering experiment")
    st.add_argument("--exp", required=True, help="outputs/steering/<name>")
    st.add_argument("--detector", default="energy")
    st.add_argument("--out")
    sw = sub.add_parser("steer-sweep", help="flip rate against radius and layer over many experiments")
    sw.add_argument("--exps", nargs="+", required=True, help="experiment folders or glob patterns")
    sw.add_argument("--detector", default="energy")
    sw.add_argument("--where", nargs="+", default=[], metavar="KEY=VALUE", help="keep matching experiments only")
    sw.add_argument("--out")
    for name in ("grid", "heatmap"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--eval", help="evaluation dataset (default: training dataset)")
        s.add_argument("--layer", default="last" if name == "grid" else "all")
        s.add_argument("--bins", type=int, default=50 if name == "grid" else 80)
        s.add_argument("--out")
    g = sub.choices["grid"]
    g.add_argument("--neurons", help="comma-separated neuron indices")
    g.add_argument("--select", default="selective", choices=["selective", "variance", "random", "first"])
    g.add_argument("-k", type=int, default=16)
    g.add_argument("--cols", type=int, default=4)
    g.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    {"grid": cmd_grid, "heatmap": cmd_heatmap, "steer": cmd_steer, "steer-sweep": cmd_steer_sweep}[args.cmd](args)


if __name__ == "__main__":
    main()
