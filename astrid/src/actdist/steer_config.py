"""All hyperparameters of a steering-swarm experiment, in one place.

Resolution order: dataclass defaults -> TOML file (--config) -> --set key=value overrides.
--sweep key=v1,v2 key2=w1,w2 expands to the Cartesian product of experiments.

    uv run python -m actdist.train_steer --config configs/steer/base.toml --set layer=layer3.1 radius=2
    uv run python -m actdist.train_steer --config configs/steer/base.toml --sweep radius=0.5,1,2 layer=layer3.1,penultimate

Values are parsed as TOML (numbers, booleans, ["lists"], "strings"); a bare word is a string.
"""

import argparse
import hashlib
import itertools
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class SteerConfig:
    # ---- experiment
    run: str = "resnet18_scratch_mnist"      # checkpoint in outputs/checkpoints/ (its dataset is the ID set)
    ood_sources: list[str] = field(default_factory=lambda: ["300k"])  # stl10 | 300k | coco | openimages
    ood_weights: list[float] | None = None   # per-source share of each OOD batch (None = equal)
    seed: int = 0                            # swarm init, OOD sampling, probe and random-direction draws
    name: str = ""                           # output folder under outputs/steering/ ("" = derived from config)

    # ---- steering
    layer: str = "penultimate"               # layer name as in models.ActivationRecorder (layer3.1, blocks.6, ...)
    radius: float = 1.0                      # norm of every vector (per spatial position / per token)
    radius_mode: str = "abs"                 # abs: radius as is | rel: radius x median activation norm at the layer
    swarm_size: int = 1024                   # K vectors per swarm
    swarms: list[str] = field(default_factory=lambda: ["id2ood", "ood2id"])  # trained jointly
    init: str = "randn"                      # randn | toward (v_toward of the paper + noise)
    init_noise: float = 0.5                  # noise scale for init=toward (relative to the unit direction)

    # ---- loss
    loss: str = "infonce"                    # infonce | infonce+energy | energy_hinge | energy_margin | knn_hinge
    tau: float = 0.1                         # InfoNCE temperature
    sim_layer: str = "penultimate"           # where InfoNCE measures similarity: `layer` or any layer downstream of it
    own_negative: bool = True                # the anchor's own clean feature counts as a negative
    lambda_energy: float = 1.0               # weight of the energy hinge in infonce+energy
    energy_margin: float = 1.0               # margin (energy units) in energy_hinge / energy_margin
    knn_margin: float = 0.05                 # margin (kNN distance units) in knn_hinge
    weight_id2ood: float = 1.0               # weight of the ID->OOD swarm loss
    weight_ood2id: float = 1.0               # weight of the OOD->ID swarm loss

    # ---- optimisation
    steps: int = 2000
    lr: float = 1e-2                         # Adam on the unconstrained parameters u (v = r u/|u|)
    bs_id: int = 64                          # ID images per step, shared by every vector
    bs_ood: int = 64                         # OOD images per step, shared by every vector
    vec_chunk: int = 256                     # vectors per suffix forward (memory only, gradient is exact)
    amp: str = "bf16"                        # bf16 | fp16 (with a gradient scaler; use on Turing GPUs such as the RTX 8000) | fp32
    log_every: int = 50

    # ---- detector
    detectors: list[str] = field(default_factory=lambda: ["energy", "msp", "maxlogit", "knn", "mahalanobis"])  # evaluated; energy losses target energy
    energy_T: float = 1.0
    knn_k: int = 50                          # k of the kNN detector (distance to the k-th nearest reference feature)
    ref_size: int = 10000                    # ID reference images the feature detectors (knn, mahalanobis) are fitted on
    tpr: float = 0.95                        # t_D = tpr-quantile of ID calibration scores (tau = 1 - tpr)

    # ---- evaluation
    alpha_max: float = 0.0                   # end of the path grid (0 = 4 x the resolved radius)
    alpha_steps: int = 41                    # grid points, including alpha = 0
    eval_vectors: int = 64                   # learned vectors walked along the path (random subset of the swarm)
    n_random: int = 64                       # matched random unit directions
    n_probe_id: int = 1000                   # ID test probes
    n_probe_ood: int = 1000                  # OOD eval probes
    bootstrap_B: int = 50                    # calibration resamples for sigma_b

    # ---- data
    workers: int = 8

    def validate(self):
        from .losses import LOSSES
        from .unlabeled import NAMED_SOURCES
        from .detectors import DETECTORS
        checks = [
            (self.loss in LOSSES, f"loss must be one of {list(LOSSES)}"),
            (set(self.ood_sources) <= set(NAMED_SOURCES), f"ood_sources must be among {list(NAMED_SOURCES)}"),
            (self.ood_weights is None or len(self.ood_weights) == len(self.ood_sources), "one weight per OOD source"),
            (set(self.swarms) <= {"id2ood", "ood2id"} and self.swarms, "swarms must be a non-empty subset of id2ood, ood2id"),
            (set(self.detectors) <= set(DETECTORS) and self.detectors, f"detectors must be among {list(DETECTORS)}"),
            (self.radius_mode in ("abs", "rel"), "radius_mode must be abs or rel"),
            (self.init in ("randn", "toward"), "init must be randn or toward"),
            (self.amp in ("bf16", "fp16", "fp32"), "amp must be bf16, fp16 or fp32"),
            (self.radius >= 0 and 0 < self.tpr < 1, "radius >= 0 and 0 < tpr < 1"),
            (0 < self.knn_k <= self.ref_size, "0 < knn_k <= ref_size"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(f"invalid config: {msg}")
        return self

    def exp_name(self) -> str:
        if self.name:
            return self.name
        cfg = {k: v for k, v in asdict(self).items() if k not in ("name", "workers", "log_every")}
        h = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
        return f"{self.run}__{self.layer}__r{self.radius:g}{self.radius_mode[0]}__K{self.swarm_size}__{self.loss}__{h}"


_FIELDS = {f.name: f for f in fields(SteerConfig)}


def _parse_value(key: str, raw: str):
    if key not in _FIELDS:
        raise KeyError(f"unknown hyperparameter {key!r}; known: {', '.join(_FIELDS)}")
    try:
        v = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        v = raw  # bare word -> string
    default = _FIELDS[key].default
    if isinstance(default, float) and isinstance(v, int) and not isinstance(v, bool):
        v = float(v)
    return v


def _split_top(s: str) -> list[str]:
    """Split on commas that are not inside [...] (so list values can be swept)."""
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += ch == "["
        depth -= ch == "]"
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur]


def _kv(items: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"expected key=value, got {it!r}")
        k, v = it.split("=", 1)
        pairs.append((k.strip(), v.strip()))
    return pairs


def load_configs(argv: list[str] | None = None) -> list[SteerConfig]:
    return parse_configs(argv)[0]


def parse_configs(argv: list[str] | None = None, extra=None, description: str = __doc__):
    """Resolved configs plus the parsed arguments; `extra(parser)` may add tool-specific flags."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    if extra:
        extra(p)
    p.add_argument("--config", type=Path, help="TOML file with any subset of the hyperparameters")
    p.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                   help="override hyperparameters (repeatable; later values win)")
    p.add_argument("--sweep", nargs="+", action="extend", default=[], metavar="KEY=V1,V2",
                   help="Cartesian product over values (repeatable)")
    p.add_argument("--print", action="store_true", help="print the resolved configs and exit")
    args = p.parse_args(argv)

    base = {}
    if args.config:
        base = tomllib.loads(args.config.read_text())
        unknown = set(base) - set(_FIELDS)
        if unknown:
            raise KeyError(f"unknown hyperparameters in {args.config}: {sorted(unknown)}")
    for k, raw in _kv(args.set):
        base[k] = _parse_value(k, raw)

    grid = [(k, [_parse_value(k, x) for x in _split_top(raw)]) for k, raw in _kv(args.sweep)]
    combos = itertools.product(*[vals for _, vals in grid]) if grid else [()]
    cfgs = [SteerConfig(**{**base, **dict(zip([k for k, _ in grid], c))}).validate() for c in combos]
    if args.print:
        for c in cfgs:
            print(c.exp_name(), json.dumps(asdict(c)))
        raise SystemExit(0)
    return cfgs, args
