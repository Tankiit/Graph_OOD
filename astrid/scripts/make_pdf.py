"""Bundle all activation-distribution figures into one PDF: outputs/activation_distributions.pdf."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "outputs"
MODELS = ["resnet18_scratch", "resnet18_ft", "vit_small_ft"]
DATASETS = ["mnist", "fmnist"]
LABELS = {"resnet18_scratch": "ResNet-18 (from scratch)", "resnet18_ft": "ResNet-18 (ImageNet fine-tuned)",
          "vit_small_ft": "ViT-S/16 (IN-21k fine-tuned)", "mnist": "MNIST", "fmnist": "Fashion-MNIST"}
VIEWS = [("heatmap_all", "All layers: per-neuron density (z-scored), correctly classified images"),
         ("penultimate__grid_selective", "Penultimate layer: 16 most class-selective neurons"),
         ("penultimate__grid_variance", "Penultimate layer: 16 highest-variance neurons"),
         ("penultimate__grid_random", "Penultimate layer: 16 random neurons")]


def text_page(pdf, title, lines):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.92, title, fontsize=20, weight="bold")
    fig.text(0.08, 0.88, "\n".join(lines), fontsize=11, va="top", family="monospace", linespacing=1.6)
    pdf.savefig(fig)
    plt.close(fig)


def image_page(pdf, path, caption):
    img = Image.open(path)
    w, h = img.size
    fig = plt.figure(figsize=(11.69, 11.69 * h / w + 0.5))
    ax = fig.add_axes((0, 0, 1, (11.69 * h / w) / (11.69 * h / w + 0.5)))
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.01, 0.995, caption, fontsize=12, va="top", weight="bold")
    pdf.savefig(fig, dpi=160)
    plt.close(fig)


def main():
    out = ROOT / "activation_distributions.pdf"
    with PdfPages(out) as pdf:
        rows = [f"{'model':34s} {'MNIST':>8s} {'FMNIST':>8s}"]
        for m in MODELS:
            accs = [json.loads((ROOT / "checkpoints" / f"{m}_{d}.json").read_text())["test_acc"] for d in DATASETS]
            rows.append(f"{LABELS[m]:34s} " + " ".join(f"{a * 100:7.2f}%" for a in accs))
        text_page(pdf, "Per-neuron activation distributions", [
            "Models trained on MNIST / Fashion-MNIST, evaluated on the",
            "test set of the training dataset (10k images).", "",
            "Neuron = one channel of a ResNet block output (global-avg",
            "pooled) or one hidden dim of the ViT CLS token.", "",
            "Grid pages: one panel per neuron; one row per class",
            "(correctly classified images), gray row = misclassified.",
            "Each row is scaled to its own peak.",
            "sel = share of variance explained by class (eta^2).", "",
            "Test accuracy", *rows, "",
            "Pages: for each dataset x model:",
            *[f"  - {v}" for _, v in VIEWS],
        ])
        for d in DATASETS:
            for m in MODELS:
                for key, desc in VIEWS:
                    path = ROOT / "figures" / f"{m}_{d}__{d}__{key}.png"
                    image_page(pdf, path, f"{LABELS[m]} · trained on {LABELS[d]} — {desc}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
