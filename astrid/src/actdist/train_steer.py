"""Train two steering swarms against the OOD detector of a frozen classifier, then evaluate them.

    uv run python -m actdist.train_steer --config configs/steer/base.toml
    uv run python -m actdist.train_steer --config configs/steer/base.toml --set layer=layer3.1 radius=2 swarm_size=128
    uv run python -m actdist.train_steer --config configs/steer/base.toml --sweep radius=0.5,1,2,4
    uv run python -m actdist.train_steer --help-config      # every hyperparameter with its default

ID = the classifier's own dataset, OOD = the unlabeled sources. Swarm id2ood learns K vectors
that make ID images rejected, swarm ood2id K vectors that make OOD images accepted; both share
every batch and are trained in the same loop. See steer_config.py for all hyperparameters.

Disjoint index sets (as in the AISTATS draft):
    ID   train split (classifier's 55k) -> 5k development (v_toward) + 50k steering-train
         val split (5k)                 -> calibration of the threshold t_D
         test split                     -> evaluation probes
    OOD  i % 10 == 1 development, == 0 evaluation probes, others steering-train (unlabeled.split_indices)

Output: outputs/steering/<name>/{config.json, train_log.jsonl, swarms.pt, metrics.json, paths.npz}
"""

import json
import sys
import time
from dataclasses import asdict
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from .data import build_transform, get_dataset, train_val_indices
from .detectors import build_detectors, calibrate
from .losses import LOSS_TARGETS, LOSSES, Clean
from .models import MODELS, build_model
from .steer import SplitModel, SteeringSwarm, steer
from .steer_config import SteerConfig, load_configs
from .unlabeled import build_unlabeled_eval_set, build_unlabeled_loader, named_sources

ROOT = Path(__file__).resolve().parents[2] / "outputs"
CLASSIFIER_SPLIT_SEED = 0  # seed train.py used for its train/val split
N_DEV = 5000
AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}  # fp32 = no autocast
SIGN = {"id2ood": +1.0, "ood2id": -1.0}


class Context:
    """Frozen model cut at the layer, data splits, thresholds and the quantities derived from them."""

    def __init__(self, cfg: SteerConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = AMP_DTYPES.get(cfg.amp) if self.device.type == "cuda" else None

        ckpt = torch.load(ROOT / "checkpoints" / f"{cfg.run}.pt", map_location="cpu")
        self.model_name, self.id_name = ckpt["model"], ckpt["dataset"]
        model = build_model(self.model_name, pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(self.device).eval().requires_grad_(False)
        self.split = SplitModel(model, self.model_name, cfg.layer)
        self.split.check_sim_layer(cfg.sim_layer)

        spec = MODELS[self.model_name]
        tf = build_transform(spec, self.id_name, train=False)  # OOD images get exactly the ID preprocessing
        id_train = get_dataset(self.id_name, "train", tf)
        train_idx, val_idx = train_val_indices(len(id_train), CLASSIFIER_SPLIT_SEED)
        rng = np.random.default_rng(cfg.seed)
        id_test = get_dataset(self.id_name, "test", tf)
        self.id_dev = Subset(id_train, train_idx[:N_DEV])
        self.id_steer = Subset(id_train, train_idx[N_DEV:])
        self.id_cal = Subset(id_train, val_idx)
        self.id_probe = Subset(id_test, np.sort(rng.choice(len(id_test), cfg.n_probe_id, replace=False)).tolist())

        self.sources = named_sources(cfg.ood_sources, spec["size"])
        self.tf = tf
        self.ood_dev = build_unlabeled_eval_set(self.sources, "L", tf, N_DEV, split="dev", seed=cfg.seed)
        self.ood_probe = build_unlabeled_eval_set(self.sources, "L", tf, cfg.n_probe_ood, split="eval", seed=cfg.seed)

        # detectors: feature detectors are fitted on clean penultimate features of an ID reference set
        # drawn from the steering-train split (the classifier's own training images)
        ref_idx = np.sort(rng.choice(len(self.id_steer), min(cfg.ref_size, len(self.id_steer)), replace=False))
        ref = self.encode(Subset(self.id_steer, ref_idx.tolist()), keep_h=False)
        # the evaluated detectors, plus energy and the detector the loss attacks (its flip rate is logged)
        self.target = LOSS_TARGETS[cfg.loss]
        names = list(dict.fromkeys([*cfg.detectors, "energy", self.target]))
        self.detectors = build_detectors(names, cfg, ref["feat"], ref["label"], self.device)

        # thresholds t_D on the clean calibration split
        cal = self.encode(self.id_cal, keep_h=False)
        self.cal_scores = {d: self.score(d, cal["logits"], cal["feat"]) for d in self.detectors}
        self.thresholds = {d: calibrate(s, cfg.tpr) for d, s in self.cal_scores.items()}

        # development pools: v_toward (the draft's default direction) and the activation scale
        dev_id = self.encode(self.id_dev, keep_h=False, stats=True)
        dev_ood = self.encode(self.ood_dev, keep_h=False, stats=True)
        delta = dev_ood["mean_h"] - dev_id["mean_h"]
        self.v_toward = (delta / delta.norm()).to(self.device)
        self.dim = self.v_toward.numel()
        self.act_norm_median = float(dev_id["median_norm"])
        self.radius = cfg.radius * (self.act_norm_median if cfg.radius_mode == "rel" else 1.0)

    def score(self, detector: str, logits, feat) -> np.ndarray:
        with torch.no_grad():
            return self.detectors[detector](logits, feat).float().cpu().numpy()

    def autocast(self):
        return torch.autocast(self.device.type, dtype=self.amp_dtype or torch.float32,
                              enabled=self.amp_dtype is not None)

    @torch.no_grad()
    def encode(self, ds, keep_h: bool = True, stats: bool = False, bs: int = 256) -> dict:
        """Clean pass: h at the layer (optional), penultimate features, logits, labels; or h statistics."""
        out = {k: [] for k in ("h", "feat", "logits", "label")}
        sum_h, count, norms = 0.0, 0, []
        dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=self.cfg.workers, pin_memory=True)
        for x, y in dl:
            with self.autocast():
                h = self.split.prefix(x.to(self.device, non_blocking=True))
                feat, logits = self.split.suffix(h)
            if stats:
                hf = h.float()
                sum_h = sum_h + SplitModel.position_mean(hf) * len(hf)
                count += len(hf)
                norms.append(SplitModel.position_norms(hf).cpu())
            if keep_h:
                out["h"].append(h)
            out["feat"].append(feat.float().cpu())
            out["logits"].append(logits.float().cpu())
            out["label"].append(torch.as_tensor(y))
        res = {k: torch.cat(v) for k, v in out.items() if v}
        if stats:
            res["mean_h"] = (sum_h / count).cpu()
            res["median_norm"] = torch.cat(norms).median()
        return res


def init_u(ctx: Context, name: str, gen: torch.Generator) -> torch.Tensor:
    K, D = ctx.cfg.swarm_size, ctx.dim
    noise = torch.randn(K, D, generator=gen)
    if ctx.cfg.init == "randn":
        return noise
    toward = ctx.v_toward.cpu() * SIGN[name]
    return toward + ctx.cfg.init_noise * noise / D ** 0.5


def train(ctx: Context, out_dir: Path) -> nn.ModuleDict:
    cfg, dev, split = ctx.cfg, ctx.device, ctx.split
    gen = torch.Generator().manual_seed(cfg.seed)
    swarms = nn.ModuleDict({n: SteeringSwarm(cfg.swarm_size, ctx.dim, ctx.radius, init_u(ctx, n, gen))
                            for n in cfg.swarms}).to(dev)
    opt = torch.optim.Adam(swarms.parameters(), lr=cfg.lr)
    # fp16 needs loss scaling so small gradients do not underflow; a no-op for bf16 / fp32
    scaler = torch.amp.GradScaler(dev.type, enabled=ctx.amp_dtype == torch.float16)
    loss_fn, weight = LOSSES[cfg.loss], {"id2ood": cfg.weight_id2ood, "ood2id": cfg.weight_ood2id}

    id_loader = DataLoader(ctx.id_steer, batch_size=cfg.bs_id, shuffle=True, drop_last=True,
                           num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0,
                           generator=torch.Generator().manual_seed(cfg.seed))
    ood_loader = build_unlabeled_loader(ctx.sources, "L", ctx.tf, cfg.bs_ood, cfg.steps, cfg.ood_weights,
                                        cfg.seed, cfg.workers, split="train")

    log = open(out_dir / "train_log.jsonl", "w")
    acc = {n: dict(loss=0.0, flip=0.0) for n in cfg.swarms}
    t0 = time.time()
    for step, ((x_ood, _), (x_id, _)) in enumerate(zip(ood_loader, cycle(id_loader))):
        with torch.no_grad(), ctx.autocast():
            h = {"id": split.prefix(x_id.to(dev, non_blocking=True)),
                 "ood": split.prefix(x_ood.to(dev, non_blocking=True))}
            clean = {}
            for k, v in h.items():
                sim, feat, logits = split.suffix_sim(v, cfg.sim_layer)
                clean[k] = Clean(feat, logits, sim)
        opt.zero_grad(set_to_none=True)
        for name in cfg.swarms:
            src, tgt = ("id", "ood") if name == "id2ood" else ("ood", "id")
            hs, B, sign = h[src], len(h[src]), SIGN[name]
            for c in range(0, cfg.swarm_size, cfg.vec_chunk):
                v = swarms[name]()[c:c + cfg.vec_chunk]
                with ctx.autocast():
                    sim, feat, logits = split.suffix_sim(steer(hs, v), cfg.sim_layer)
                feat, logits = feat.view(len(v), B, -1), logits.view(len(v), B, -1)
                sim = sim.view(len(v), B, -1)
                per_vec = loss_fn(feat, logits, clean[src], clean[tgt], ctx.detectors, ctx.thresholds, sign, cfg, sim)
                scaler.scale(weight[name] * per_vec.sum()).backward()
                with torch.no_grad():
                    e, t = ctx.detectors[ctx.target](logits, feat), ctx.thresholds[ctx.target]
                    acc[name]["loss"] += per_vec.sum().item()
                    acc[name]["flip"] += ((e > t) if sign > 0 else (e <= t)).float().sum(0).mean().item()
        scaler.step(opt)
        scaler.update()

        if (step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps:
            n = cfg.log_every if (step + 1) % cfg.log_every == 0 else (step + 1) % cfg.log_every
            rec = {"step": step + 1, "time": round(time.time() - t0, 1)}
            if dev.type == "cuda":
                rec["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            for name in cfg.swarms:
                rec[f"{name}_loss"] = acc[name]["loss"] / (n * cfg.swarm_size)
                rec[f"{name}_flip"] = acc[name]["flip"] / (n * cfg.swarm_size)
                acc[name] = dict(loss=0.0, flip=0.0)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print("  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
    log.close()
    return swarms


def run_experiment(cfg: SteerConfig):
    from .eval_steer import evaluate

    out_dir = ROOT / "steering" / cfg.exp_name()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"== {out_dir.name}", flush=True)

    ctx = Context(cfg)
    print(f"layer {cfg.layer}: dim {ctx.dim}, median activation norm {ctx.act_norm_median:.3g}, "
          f"radius {ctx.radius:.3g}; loss {cfg.loss} attacks {ctx.target}, t_D {ctx.thresholds[ctx.target]:.4g}"
          f" (flip rates in the log are for {ctx.target})", flush=True)
    swarms = train(ctx, out_dir)
    torch.save({"directions": {n: s.directions().cpu() for n, s in swarms.items()},
                "radius": ctx.radius, "thresholds": ctx.thresholds, "target_detector": ctx.target,
                "v_toward": ctx.v_toward.cpu(), "act_norm_median": ctx.act_norm_median,
                "config": asdict(cfg)}, out_dir / "swarms.pt")
    evaluate(ctx, {n: s.directions() for n, s in swarms.items()}, out_dir)


def main():
    if "--help-config" in sys.argv:
        for k, v in asdict(SteerConfig()).items():
            print(f"{k:16s} {v!r}")
        return
    for cfg in load_configs():
        run_experiment(cfg)


if __name__ == "__main__":
    main()
