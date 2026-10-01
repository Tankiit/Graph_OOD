"""Measure activation norms at the steering layer(s), per dataset, with the steering hyperparameters.

    uv run python -m actdist.actnorm --config configs/steer/base.toml
    uv run python -m actdist.actnorm --config configs/steer/base.toml --layers all
    uv run python -m actdist.actnorm --set run=vit_small_ft_fmnist 'ood_sources=["300k","stl10"]' --layers blocks.3,blocks.6
    uv run python -m actdist.actnorm --config configs/steer/base.toml --sweep run=resnet18_scratch_mnist,resnet18_ft_mnist

The model (`run`), `layer`, `ood_sources`, `radius`, `radius_mode`, `amp` and `seed` come from the same
config as train_steer (--config / --set / --sweep); --layers overrides `layer` with a list or "all".

Norms are taken where a steering vector is added: per spatial position over channels (ResNet
maps), per token over width (ViT), per sample at "penultimate". Datasets:
    id_dev    the 5000 ID development images radius_mode=rel is measured on (always in full)
    id_test   --n ID test images
    <source>  --n images of each OOD source (--ood-split, default dev), measured separately

For every (layer, dataset): quantiles of the per-position norm, the per-sample norm of the whole
activation, and the norm of the mean activation vector; plus the radius the config resolves to
and its ratio to each dataset's median norm.
Output: printed table and outputs/act_norms/<run>.json (merged with earlier measurements).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import build_transform, get_dataset, train_val_indices
from .models import MODELS, build_model
from .steer import SplitModel
from .steer_config import SteerConfig, parse_configs
from .train_steer import AMP_DTYPES, CLASSIFIER_SPLIT_SEED, N_DEV, ROOT
from .unlabeled import build_unlabeled_eval_set, named_sources

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def layer_modules(model, model_name: str) -> dict:
    """Block modules by ActivationRecorder name; "penultimate" is the classifier head (pre-hooked)."""
    if model_name.startswith("resnet"):
        mods = {f"{l}.{i}": b for l in ["layer1", "layer2", "layer3", "layer4"] for i, b in enumerate(getattr(model, l))}
        mods["penultimate"] = model.fc
    else:
        mods = {f"blocks.{i}": b for i, b in enumerate(model.blocks)}
        mods["penultimate"] = model.head
    return mods


class NormStats:
    """Accumulates per-position norms, per-sample norms and the activation sum for one layer."""

    def __init__(self):
        self.pos, self.sample, self.sum, self.count = [], [], 0.0, 0

    def add(self, h: torch.Tensor):
        h = h.float()
        self.pos.append(SplitModel.position_norms(h).cpu())
        self.sample.append(h.flatten(1).norm(dim=1).cpu())
        self.sum = self.sum + SplitModel.position_mean(h) * len(h)
        self.count += len(h)

    def summary(self, radius: float) -> dict:
        pos, sample = torch.cat(self.pos).numpy(), torch.cat(self.sample).numpy()
        q = np.quantile(pos, QUANTILES)
        return {"n_images": self.count, "n_positions": int(pos.size), "dim": int((self.sum / self.count).numel()),
                "position_norm": {"mean": float(pos.mean()), "std": float(pos.std()),
                                  **{f"q{int(p * 100):02d}": float(v) for p, v in zip(QUANTILES, q)}},
                "sample_norm_median": float(np.median(sample)),
                "mean_vector_norm": float((self.sum / self.count).norm()),
                "radius_over_median": radius / float(q[2]) if q[2] > 0 else None}


@torch.no_grad()
def measure(model, mods: dict, layers: list[str], ds, device, amp_dtype, workers: int, bs: int = 256) -> dict:
    """One forward pass over `ds` with hooks on every requested layer -> {layer: NormStats}."""
    stats = {l: NormStats() for l in layers}
    handles = []
    for l in layers:
        if l == "penultimate":
            handles.append(mods[l].register_forward_pre_hook(lambda m, inp, l=l: stats[l].add(inp[0])))
        else:
            handles.append(mods[l].register_forward_hook(lambda m, inp, out, l=l: stats[l].add(out)))
    try:
        for x, _ in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True):
            with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
                model(x.to(device, non_blocking=True))
    finally:
        for h in handles:
            h.remove()
    return stats


def run(cfg: SteerConfig, layers_arg: str | None, n: int, ood_split: str, bs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = AMP_DTYPES.get(cfg.amp) if device.type == "cuda" else None
    ckpt = torch.load(ROOT / "checkpoints" / f"{cfg.run}.pt", map_location="cpu")
    model_name, id_name = ckpt["model"], ckpt["dataset"]
    model = build_model(model_name, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval().requires_grad_(False)
    mods = layer_modules(model, model_name)
    all_layers = list(mods)
    layers = all_layers if layers_arg == "all" else (layers_arg.split(",") if layers_arg else [cfg.layer])
    unknown = [l for l in layers if l not in mods]
    layers = [l for l in layers if l in mods]
    if unknown:  # a sweep over architectures may name layers only some models have
        print(f"{cfg.run}: skipping layer(s) {unknown} (not in {model_name}; choose from {all_layers} or 'all')")
    if not layers:
        print(f"{cfg.run}: no layer to measure")
        return

    spec = MODELS[model_name]
    tf = build_transform(spec, id_name, train=False)
    id_train, id_test = get_dataset(id_name, "train", tf), get_dataset(id_name, "test", tf)
    train_idx, _ = train_val_indices(len(id_train), CLASSIFIER_SPLIT_SEED)
    rng = np.random.default_rng(cfg.seed)
    datasets = {"id_dev": Subset(id_train, train_idx[:N_DEV]),
                "id_test": Subset(id_test, np.sort(rng.choice(len(id_test), min(n, len(id_test)), replace=False)).tolist())}
    for name, src in zip(cfg.ood_sources, named_sources(cfg.ood_sources, spec["size"])):
        datasets[name] = build_unlabeled_eval_set([src], "L", tf, n, split=ood_split, seed=cfg.seed)

    t0 = time.time()
    raw = {name: measure(model, mods, layers, ds, device, amp, cfg.workers, bs) for name, ds in datasets.items()}
    result = {}
    for l in layers:
        ref_median = float(np.median(torch.cat(raw["id_dev"][l].pos).numpy()))
        radius = cfg.radius * (ref_median if cfg.radius_mode == "rel" else 1.0)
        result[l] = {"radius": radius, "radius_setting": f"{cfg.radius:g} ({cfg.radius_mode})",
                     "datasets": {name: raw[name][l].summary(radius) for name in datasets}}
    print_table(cfg, model_name, id_name, result, time.time() - t0)
    save(cfg, result, ood_split)


def print_table(cfg, model_name, id_name, result, dt):
    print(f"\n{cfg.run} ({model_name} on {id_name}) · amp {cfg.amp} · radius {cfg.radius:g} ({cfg.radius_mode}) · {dt:.0f}s")
    head = f"{'layer':12s} {'dataset':11s} {'dim':>5s} {'images':>7s} {'p05':>8s} {'p25':>8s} {'median':>8s} " \
           f"{'p75':>8s} {'p95':>8s} {'|mean|':>8s} {'sample':>9s} {'r':>8s} {'r/med':>7s}"
    print(head)
    print("-" * len(head))
    for l, lr in result.items():
        for name, s in lr["datasets"].items():
            p = s["position_norm"]
            ratio = s["radius_over_median"]
            print(f"{l:12s} {name:11s} {s['dim']:5d} {s['n_images']:7d} {p['q05']:8.3g} {p['q25']:8.3g} "
                  f"{p['q50']:8.3g} {p['q75']:8.3g} {p['q95']:8.3g} {s['mean_vector_norm']:8.3g} "
                  f"{s['sample_norm_median']:9.4g} {lr['radius']:8.3g} {'-' if ratio is None else f'{ratio:7.3f}'}")
    print("norms per position (ResNet) / token (ViT) / sample (penultimate); |mean| = norm of the mean "
          "activation vector; sample = median norm of the whole activation of an image; "
          "r = resolved radius, r/med = r / median of that dataset")


def save(cfg, result, ood_split):
    out = ROOT / "act_norms" / f"{cfg.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(out.read_text()) if out.exists() else {}
    for l, lr in result.items():
        entry = data.setdefault(l, {"datasets": {}})
        entry["datasets"].update(lr["datasets"])
        entry["amp"], entry["ood_split"], entry["seed"] = cfg.amp, ood_split, cfg.seed
        entry.setdefault("radii", {})[lr["radius_setting"]] = lr["radius"]
    out.write_text(json.dumps(data, indent=2))
    print(f"saved {out}")


def main():
    def extra(p):
        p.add_argument("--layers", help="comma-separated layer names or 'all' (default: the config's layer)")
        p.add_argument("-n", type=int, default=2000, help="images per dataset (id_dev is always its full 5000)")
        p.add_argument("--ood-split", default="dev", choices=["train", "dev", "eval"])
        p.add_argument("--bs", type=int, default=256)

    cfgs, args = parse_configs(extra=extra, description=__doc__)
    for cfg in cfgs:
        run(cfg, args.layers, args.n, args.ood_split, args.bs)


if __name__ == "__main__":
    main()
