"""Dump per-neuron activations of a trained model on a test set.

    uv run python -m actdist.extract --run resnet18_scratch_mnist
    uv run python -m actdist.extract --run resnet18_scratch_mnist --eval-dataset fmnist

Output: outputs/activations/{run}__{eval_dataset}.npz with
    label, pred, correct, logits, and act__<layer> arrays of shape [N, D] (float16).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data import build_transform, get_dataset
from .models import MODELS, ActivationRecorder, build_model

ROOT = Path(__file__).resolve().parents[2] / "outputs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="e.g. vit_small_ft_fmnist")
    p.add_argument("--eval-dataset", choices=["mnist", "fmnist"], help="defaults to the training dataset")
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    ckpt = torch.load(ROOT / "checkpoints" / f"{args.run}.pt", map_location="cpu")
    model_name, train_ds = ckpt["model"], ckpt["dataset"]
    eval_ds = args.eval_dataset or train_ds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_name, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    rec = ActivationRecorder(model, model_name)

    # preprocessing always follows the training dataset of the model
    ds = get_dataset(eval_ds, "test", build_transform(MODELS[model_name], train_ds, train=False))
    if args.limit:
        ds = Subset(ds, range(args.limit))
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=8, pin_memory=True)

    labels, logits, acts = [], [], {}
    with torch.no_grad():
        for x, y in tqdm(dl, desc=f"{args.run} on {eval_ds}"):
            out = model(x.to(device, non_blocking=True)).float().cpu()
            labels.append(y)
            logits.append(out)
            for k, v in rec.pop().items():
                acts.setdefault(k, []).append(v)

    label = torch.cat(labels).numpy()
    logit = torch.cat(logits).numpy()
    pred = logit.argmax(1)
    arrays = {"label": label, "pred": pred, "correct": pred == label, "logits": logit.astype(np.float32)}
    arrays |= {f"act__{k}": torch.cat(v).numpy().astype(np.float16) for k, v in acts.items()}
    arrays["layers"] = np.array(list(acts))  # keeps layer order

    out_dir = ROOT / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.run}__{eval_ds}.npz"
    np.savez_compressed(path, **arrays)
    print(f"saved {path}  acc={arrays['correct'].mean():.4f}  layers: "
          + ", ".join(f"{k}[{v[0].shape[1]}]" for k, v in acts.items()))


if __name__ == "__main__":
    main()
