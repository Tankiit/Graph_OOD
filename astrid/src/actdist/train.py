"""Train one (model, dataset) run.

    uv run python -m actdist.train --model resnet18_scratch --dataset mnist
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from .data import get_loaders
from .models import MODELS, build_model

OUT = Path(__file__).resolve().parents[2] / "outputs" / "checkpoints"

DEFAULTS = {  # epochs, lr, batch size, optimizer
    "resnet18_scratch": dict(epochs=20, lr=0.1, bs=128, opt="sgd"),
    "resnet18_ft": dict(epochs=5, lr=1e-3, bs=128, opt="adamw"),
    "vit_small_ft": dict(epochs=5, lr=1e-4, bs=64, opt="adamw"),
}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--dataset", required=True, choices=["mnist", "fmnist"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--bs", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, help="subsample for smoke tests")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    cfg = {**DEFAULTS[args.model], **{k: v for k, v in vars(args).items() if v is not None and k in ("epochs", "lr", "bs")}}

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model).to(device)
    train_dl, val_dl, test_dl = get_loaders(MODELS[args.model], args.dataset, cfg["bs"], args.seed,
                                            args.limit, args.workers)

    if cfg["opt"] == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=5e-4, nesterov=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.05)
    warmup = len(train_dl)  # one epoch of linear warmup
    total_steps = cfg["epochs"] * len(train_dl)

    def lr_at(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.0)

    run = f"{args.model}_{args.dataset}"
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT / f"{run}.pt"
    best_val, history = -1.0, []
    for epoch in range(cfg["epochs"]):
        model.train()
        t0, run_loss = time.time(), 0.0
        for x, y in tqdm(train_dl, desc=f"{run} ep{epoch}", leave=False):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item()
        val_acc = evaluate(model, val_dl, device)
        history.append(dict(epoch=epoch, loss=run_loss / len(train_dl), val_acc=val_acc, time=time.time() - t0))
        print(f"[{run}] epoch {epoch} loss {history[-1]['loss']:.4f} val {val_acc:.4f} ({history[-1]['time']:.0f}s)")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": args.model, "dataset": args.dataset, "state_dict": model.state_dict()}, ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    test_acc = evaluate(model, test_dl, device)
    print(f"[{run}] best val {best_val:.4f} test {test_acc:.4f}")
    metrics = dict(run=run, cfg=cfg, best_val=best_val, test_acc=test_acc, history=history, limit=args.limit)
    (OUT / f"{run}.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
