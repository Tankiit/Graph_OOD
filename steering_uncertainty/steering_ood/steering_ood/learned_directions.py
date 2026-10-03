"""Learned steering directions: InfoNCE swarms added at one layer of the frozen encoder.

    split = SplitModel(encoder, head, 'layer3.0')
    h = split.prefix(x)                                   # activation at the steering layer
    swarm = SteeringSwarm(K=64, dim=split.dim(h), radius=r)
    sim, feat, logits = split.suffix_sim(steer(h, swarm()), 'pooled')   # K*B steered passes

SplitModel, steer, SteeringSwarm and the pure multi-positive InfoNCE are ported from
Graph_OOD steering-astrid, commit bcf8e0ce9a706dd2ff83db409c48626ca473b58d
(astrid/src/actdist/{steer,losses,train_steer}.py). Differences: logits come from this package's
CIFAR10 linear head, training uses no detector score or threshold, ID images can be augmented and
the OOD side is either CIFAR100 train images or ID activations plus a large random vector.
A vector is added at every spatial position of a ResNet map or every token of a ViT, so its norm
is the displacement per position; at 'pooled' it is added to the pooled feature, just before the head.
Supported encoders are the vision models of scripts/run_pipeline.py: ResNet-18/50, ViT-B/16 and DINOv2-S.
"""
from dataclasses import asdict, dataclass
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .core import file_hash, load_cache, new_output, source_hashes, threshold, versions, write_json
from .detectors import make_detector
from .experiment import estimate_direction
from .head import load_head
from .vision import build_encoder, state_hash
from .vision_data import _identity, image_hash, open_dataset

SOURCE_COMMIT = 'bcf8e0ce9a706dd2ff83db409c48626ca473b58d'
MODELS = {  # same checkpoints as scripts/run_pipeline.py, so the cached features and heads match
    'resnet18': ('torchvision', 'resnet18', 'IMAGENET1K_V1'),
    'resnet50': ('torchvision', 'resnet50', 'IMAGENET1K_V2'),
    'vit_b16': ('torchvision', 'vit_b_16', 'IMAGENET1K_V1'),
    'dinov2_s': ('timm', 'vit_small_patch14_dinov2.lvd142m', 'DEFAULT'),
}
DEFAULT_LAYERS = dict(resnet18='layer4.0', resnet50='layer4.0', vit_b16='blocks.9', dinov2_s='blocks.9')
STEER_ROLES = ('id_steer_train', 'id_steer_dev', 'ood_steer_train', 'ood_steer_dev')
SWARMS = ('id2ood', 'ood2id')
EQUIVALENCE_TOLERANCE = dict(direct=1e-4, cached=1e-2)  # relative to the largest feature magnitude


class SplitModel(nn.Module):
    """An encoder cut after `layer`: prefix(x) -> h, suffix(h) -> (pooled feature, head logits).

    ResNet blocks are named layer1.0 ... layer4.N and give maps [B, C, H, W]; ViT blocks are named
    blocks.0 ... blocks.N and give tokens [B, T, D] (TorchVision and timm layouts). The pooled
    feature is the encoder's own output (ResNet average pool, ViT CLS token after the final norm).
    """

    def __init__(self, encoder, head, layer):
        super().__init__()
        self.encoder, self.head, self.layer = encoder, head, layer
        if hasattr(encoder, 'layer1'):
            self.kind = 'resnet'
            names, blocks = [], []
            for lname in ('layer1', 'layer2', 'layer3', 'layer4'):
                for i, b in enumerate(getattr(encoder, lname)):
                    names.append(f'{lname}.{i}'); blocks.append(b)
        elif hasattr(encoder, 'class_token'):
            self.kind, blocks = 'torchvision_vit', list(encoder.encoder.layers)
        elif hasattr(encoder, 'blocks') and hasattr(encoder, 'patch_embed'):
            self.kind, blocks = 'timm_vit', list(encoder.blocks)
        else:
            raise ValueError('SplitModel supports ResNet and TorchVision/timm ViT encoders')
        if self.kind != 'resnet':
            names = [f'blocks.{i}' for i in range(len(blocks))]
        self.layers = names + ['pooled']
        if layer not in self.layers:
            raise ValueError(f'unknown layer {layer!r}; choose from {self.layers}')
        self.cut = len(blocks) if layer == 'pooled' else names.index(layer) + 1
        self.pre_blocks, self.post_blocks = nn.Sequential(*blocks[:self.cut]), nn.Sequential(*blocks[self.cut:])
        self.block_names = names

    def _stem(self, x):
        m = self.encoder
        if self.kind == 'resnet':
            return m.maxpool(m.relu(m.bn1(m.conv1(x))))
        if self.kind == 'torchvision_vit':
            x = m._process_input(x)
            x = torch.cat([m.class_token.expand(len(x), -1, -1), x], dim=1)
            return m.encoder.dropout(x + m.encoder.pos_embedding)
        return m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(x))))

    def _pool(self, h):
        m = self.encoder
        if self.kind == 'resnet':
            return torch.flatten(m.avgpool(h), 1)
        if self.kind == 'torchvision_vit':
            return m.encoder.ln(h)[:, 0]
        return m.forward_head(m.norm(h), pre_logits=True)

    def prefix(self, x):
        h = self.pre_blocks(self._stem(x))
        return self._pool(h) if self.layer == 'pooled' else h

    def suffix(self, h):
        feat = h if self.layer == 'pooled' else self._pool(self.post_blocks(h))
        return feat, self.head(feat.float())

    def check_sim_layer(self, sim_layer):
        """`sim_layer` must be `layer` itself or downstream of it."""
        if sim_layer not in self.layers:
            raise ValueError(f'unknown sim_layer {sim_layer!r}; choose from {self.layers}')
        if self.layers.index(sim_layer) < self.layers.index(self.layer):
            raise ValueError(f'sim_layer {sim_layer!r} is upstream of the steered layer {self.layer!r}')

    @staticmethod
    def reduce(h):
        """One vector per image: a ResNet map is averaged over positions, ViT tokens give the CLS token."""
        return h.mean(dim=(2, 3)) if h.dim() == 4 else h[:, 0] if h.dim() == 3 else h

    def suffix_sim(self, h, sim_layer):
        """suffix(h) plus the activation at `sim_layer` (reduced to [B, d]): (sim, feat, logits)."""
        if sim_layer == 'pooled':
            feat, logits = self.suffix(h)
            return feat, feat, logits
        x = h
        sim = self.reduce(x) if sim_layer == self.layer else None
        for name, block in zip(self.block_names[self.cut:], self.post_blocks):
            x = block(x)
            if name == sim_layer:
                sim = self.reduce(x)
        feat = self._pool(x)
        return sim, feat, self.head(feat.float())

    @staticmethod
    def dim(h):
        """Size of a steering vector for activations h: channels (maps), width (tokens, pooled)."""
        return h.shape[-1] if h.dim() == 3 else h.shape[1]

    @staticmethod
    def positions(h):
        """Positions that receive the vector per image: H*W (maps), T (tokens) or 1 (pooled)."""
        return int(np.prod(h.shape[2:])) if h.dim() == 4 else h.shape[1] if h.dim() == 3 else 1

    @staticmethod
    def position_norms(h):
        """Norm of the activation at every position / token / sample, flattened."""
        return h.norm(dim=-1 if h.dim() == 3 else 1).flatten()

    @staticmethod
    def position_mean(h):
        """Mean activation vector over samples and positions: [dim]."""
        if h.dim() == 4:
            return h.mean((0, 2, 3))
        return h.reshape(-1, h.shape[-1]).mean(0)


def steer(h, v):
    """h [B, ...], v [K, dim] -> [K*B, ...]: every vector applied to every sample of the batch."""
    if h.dim() == 4:        # ResNet map [B, C, H, W]: same channel offset at every position
        out = h[None] + v[:, None, :, None, None]
    elif h.dim() == 3:      # ViT tokens [B, T, D]: added to every token
        out = h[None] + v[:, None, None, :]
    else:                   # pooled feature [B, D]
        out = h[None] + v[:, None, :]
    return out.flatten(0, 1)


def offset(h, v):
    """h [B, ...], v [B, dim] -> [B, ...]: one vector per sample (synthetic OOD)."""
    return h + (v[:, :, None, None] if h.dim() == 4 else v[:, None, :] if h.dim() == 3 else v)


class SteeringSwarm(nn.Module):
    """K vectors constrained to the sphere of radius r: v_k = r * u_k / |u_k|, u unconstrained."""

    def __init__(self, K, dim, radius, init=None):
        super().__init__()
        self.u = nn.Parameter(torch.randn(K, dim) if init is None else init.clone())
        self.register_buffer('radius', torch.tensor(float(radius)))

    def forward(self):
        return self.radius * F.normalize(self.u, dim=1)

    def directions(self):
        return F.normalize(self.u.detach(), dim=1)


def infonce(sim, same, other, tau=.1, own_negative=True):
    """Multi-positive InfoNCE on unit-normalised activations -> [K] per-vector loss.

    sim [K, B, d] steered anchors; other [Bo, d] clean activations of the target side (positives);
    same [B, d] clean activations of the anchors' own side (negatives), row i belonging to anchor i.
    """
    a = F.normalize(sim.float(), dim=-1)
    pos = a @ F.normalize(other.float(), dim=-1).T / tau
    neg = a @ F.normalize(same.float(), dim=-1).T / tau
    if not own_negative:
        neg = neg.masked_fill(torch.eye(neg.shape[-1], dtype=torch.bool, device=neg.device), float('-inf'))
    loss = torch.logsumexp(torch.cat([pos, neg], -1), -1) - torch.logsumexp(pos, -1)
    return loss.mean(1)


@dataclass
class SteerConfig:
    seed: int = 7
    model: str = 'resnet18'           # resnet18 | resnet50 | vit_b16 | dinov2_s
    # steering
    layer: str = ''                   # where the vector is added; '' = model default (DEFAULT_LAYERS)
    sim_layer: str = 'pooled'         # where InfoNCE measures similarity: `layer` or downstream of it
    radius: float = .25               # x median ID activation norm at `layer` (per position)
    swarm_size: int = 64
    # training-only data choices
    id_augment: bool = True           # two SimCLR-style views per ID image (anchor, clean reference)
    ood_mode: str = 'cifar100'        # cifar100: real OOD images | random: ID image + large random vector
    ood_noise_radius: float = 2.      # x median ID activation norm at `layer` (ood_mode=random)
    # loss and optimisation
    tau: float = .1                   # InfoNCE temperature
    own_negative: bool = True         # the anchor's own clean reference counts as a negative
    steps: int = 1000
    lr: float = .01                   # Adam on the unconstrained parameters u
    bs_id: int = 64
    bs_ood: int = 64
    vec_chunk: int = 8                # vectors per suffix forward (memory only, gradient is exact)
    amp: str = 'fp32'                 # fp32 | bf16
    log_every: int = 50
    workers: int = 4
    # steering splits
    dev_fraction: float = .2          # share of the ID direction role held out as steering-dev
    n_ood_train: int = 2000
    n_ood_dev: int = 500
    # development sanity check (not the benchmark)
    detectors: tuple = ('knn', 'mahalanobis_shrinkage', 'energy', 'msp')
    k: int = 5
    reject_tau: float = .05
    sanity_vectors: int = 8
    sanity_probes: int = 200

    def __post_init__(self):
        self.layer = self.layer or DEFAULT_LAYERS.get(self.model, '')

    def validate(self):
        if self.model not in MODELS:
            raise ValueError(f'model must be one of {list(MODELS)}')
        if self.ood_mode not in ('cifar100', 'random') or self.amp not in ('fp32', 'bf16'):
            raise ValueError('ood_mode must be cifar100 or random; amp must be fp32 or bf16')
        if min(self.swarm_size, self.steps, self.bs_id, self.bs_ood, self.vec_chunk, self.log_every,
               self.n_ood_train, self.n_ood_dev, self.sanity_vectors, self.sanity_probes, self.k) < 1 or self.workers < 0:
            raise ValueError('Counts must be positive')
        if min(self.radius, self.ood_noise_radius, self.tau, self.lr) <= 0:
            raise ValueError('radius, ood_noise_radius, tau and lr must be positive')
        if not 0 < self.dev_fraction < 1 or not 0 < self.reject_tau < 1 or not self.detectors:
            raise ValueError('Invalid split fraction, rejection rate or detector list')
        return self


def relocate(spec, image_root=None):
    """Open a dataset whose images now live under `image_root` (e.g. on a cluster); IDs keep the original root."""
    return dict(spec, root=str(image_root)) if image_root else spec


def prepare_steering(splits, output, seed=7, dev_fraction=.2, n_ood_train=2000, n_ood_dev=500, image_root=None):
    """Steering-train/dev roles: the ID direction role is partitioned per class; OOD is CIFAR100 train.

    No calibration, reference, probe, head or test row is used, and no SVHN image. CIFAR100 train
    images whose pixels repeat or occur anywhere in the main manifest are skipped and logged.
    """
    if Path(output).exists():
        raise FileExistsError(output)
    manifest = json.loads(Path(splits).read_text())
    if manifest['sources']['train']['name'] != 'cifar10':
        raise ValueError('Learned directions are set up for CIFAR10 as the ID dataset')
    rng, rows = np.random.default_rng(seed), []
    direction = [r for r in manifest['rows'] if r['role'] == 'direction']
    for c in sorted({r['label'] for r in direction}):
        group = [r for r in direction if r['label'] == c]
        n_dev = round(len(group) * dev_fraction)
        if not 0 < n_dev < len(group):
            raise ValueError(f'Class {c} cannot be split into steering train and dev')
        for j, i in enumerate(rng.permutation(len(group))):
            rows.append(dict(group[i], role='id_steer_dev' if j < n_dev else 'id_steer_train'))
    spec = dict(name='cifar100', root=manifest['sources']['train']['root'], split='train')
    ood = open_dataset(relocate(spec, image_root))
    seen, dropped, n = {r['pixel_sha256'] for r in manifest['rows']}, [], 0
    for index in rng.permutation(len(ood)):
        pixels = image_hash(ood[int(index)][0])
        if pixels in seen:
            dropped.append(dict(id=_identity(spec, int(index)), reason='pixels repeat or occur in the main manifest'))
            continue
        seen.add(pixels)
        rows.append(dict(id=_identity(spec, int(index)), source='ood_steer', index=int(index),
                         role='ood_steer_train' if n < n_ood_train else 'ood_steer_dev',
                         label=-1, pixel_sha256=pixels))
        n += 1
        if n == n_ood_train + n_ood_dev:
            break
    else:
        raise ValueError('CIFAR100 train has too few distinct images for the requested OOD pools')
    write_json(output, dict(sources=dict(train=manifest['sources']['train'], ood_steer=spec), rows=rows,
        metadata=dict(seed=seed, dev_fraction=dev_fraction, n_ood_train=n_ood_train, n_ood_dev=n_ood_dev,
                      main_manifest_sha256=file_hash(splits), role_counts=dict(Counter(r['role'] for r in rows)),
                      dropped_duplicates=dropped, id_source='main manifest role direction, stratified per class',
                      ood_source='CIFAR100 official train split; test splits and SVHN untouched')))


class Pool:
    """Images of one steering role; an item is one tensor per transform (views of the same image)."""

    def __init__(self, rows, datasets, transforms):
        self.rows, self.datasets, self.transforms = rows, datasets, transforms

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        image = self.datasets[row['source']][row['index']][0].convert('RGB')
        if image_hash(image) != row['pixel_sha256']:
            raise ValueError(f"Source image changed after split preparation: {row['id']}")
        return tuple(t(image) for t in self.transforms)


def augment(preprocess, size):
    """SimCLR-style view of the raw image, followed by the unchanged checkpoint preprocessing."""
    from torchvision import transforms as T
    return T.Compose([T.RandomResizedCrop(size, scale=(.2, 1.)), T.RandomHorizontalFlip(),
                      T.RandomApply([T.ColorJitter(.4, .4, .4, .1)], p=.8), T.RandomGrayscale(p=.2), preprocess])


def batches(pool, batch_size, seed, workers):
    """Endless shuffled batches, moving on to a new epoch when the pool is exhausted."""
    loader = DataLoader(pool, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers,
                        persistent_workers=workers > 0, generator=torch.Generator().manual_seed(seed))
    while True:
        yield from loader


@torch.no_grad()
def activations(split, pool, device, reduce=False, batch_size=128):
    """Clean activations at the steering layer for a whole pool, on the CPU."""
    out = []
    for (x,) in DataLoader(pool, batch_size=batch_size):
        h = split.prefix(x.to(device)).float()
        out.append((split.reduce(h) if reduce else h).cpu())
    return torch.cat(out)


@torch.no_grad()
def equivalence(split, pool, cache, device, n=64):
    """The split forward must reproduce the encoder and the cached features of the same images."""
    x = torch.stack([pool[i][0] for i in range(min(n, len(pool)))]).to(device)
    feat = split.suffix(split.prefix(x))[0].float().cpu().numpy()
    direct = split.encoder(x).float().cpu().numpy()
    where = {i: j for j, i in enumerate(cache['direction_ids'].tolist())}
    cached = cache['direction_x'][[where[r['id']] for r in pool.rows[:len(x)]]]
    scale = float(np.abs(direct).max())
    result = dict(n=len(x), feature_scale=scale, direct=float(np.abs(feat - direct).max() / scale),
                  cached=float(np.abs(feat - cached).max() / scale), tolerance=EQUIVALENCE_TOLERANCE,
                  units='max abs error relative to the largest feature magnitude')
    for key, tol in EQUIVALENCE_TOLERANCE.items():
        if not result[key] <= tol:
            raise ValueError(f'Split model disagrees with the {key} features: {result}')
    return result


def swarm_losses(split, cfg, swarms, h_id, sim_id, h_ood, sim_ood, autocast, backward=False):
    """Mean InfoNCE per swarm. id2ood steers ID anchors toward clean OOD; ood2id the reverse.

    sim_id are the clean ID references (row i for ID anchor i), sim_ood the clean OOD activations.
    """
    result = {}
    for name, swarm in swarms.items():
        hs, same, other = (h_id, sim_id, sim_ood) if name == 'id2ood' else (h_ood, sim_ood, sim_id)
        total = 0.
        for c in range(0, cfg.swarm_size, cfg.vec_chunk):
            v = swarm()[c:c + cfg.vec_chunk]
            with autocast():
                sim = split.suffix_sim(steer(hs, v), cfg.sim_layer)[0]
            per_vec = infonce(sim.view(len(v), len(hs), -1), same, other, cfg.tau, cfg.own_negative)
            if backward:
                per_vec.sum().backward()
            total += per_vec.sum().item()
        result[name] = total / cfg.swarm_size
    return result


@torch.no_grad()
def sanity_check(cfg, split, data, head, vectors, radius, h_id, h_ood, device):
    """Development-only flip rates at the fixed clean threshold; descriptive, not the benchmark.

    Each vector is applied as trained, at the training radius, with no scale to sweep. `head` must
    be a separate instance from the one inside `split`: the detector adapters move it to the CPU.
    Rates are ID rejection (id2ood) and OOD acceptance (ood2id), next to the unsteered probes and a
    matched random control, for the first `sanity_vectors` vectors on steering-dev probes.
    """
    dets = {}
    for name in cfg.detectors:
        det = make_detector(name, 'pytorch', cfg.k, head).fit(data['reference_x'], data['reference_y'])
        dets[name] = (det, threshold(det.score(data['calibration_x']), cfg.reject_tau))
    result = dict(note='steering-dev probes only (no test data, no SVHN); descriptive development check',
                  radius=radius, thresholds={n: t for n, (_, t) in dets.items()},
                  outcome=dict(id='ID rejection rate', ood='OOD acceptance rate'), clean={}, sides={})
    for side, h, learned in (('id', h_id, 'id2ood'), ('ood', h_ood, 'ood2id')):
        h = h[:cfg.sanity_probes].to(device)
        clean = split.suffix(h)[0].float()
        attacked = lambda scores, t: scores > t if side == 'id' else scores <= t
        # Unsteered baseline: the same probes scored without any vector.
        result['clean'][side], correct = {}, {}
        for name, (det, t) in dets.items():
            scores = det.score(clean.cpu().numpy().astype('float64'))
            correct[name] = ~attacked(scores, t)  # initially accepted ID / initially rejected OOD
            result['clean'][side][name] = dict(scores=scores, mean_score=float(scores.mean()),
                                               rate=float(attacked(scores, t).mean()))
        result['sides'][side] = {}
        for family in (learned, 'random'):
            v = radius * torch.tensor(vectors[family][:cfg.sanity_vectors], dtype=torch.float32, device=device)
            feats = torch.cat([split.suffix(steer(h, v[i:i + 1]))[0].float() for i in range(len(v))]
                              ).view(len(v), len(h), -1)                               # [V, P, D]
            entry = dict(n_vectors=len(v), n_probes=len(h),
                         pooled_displacement=float((feats - clean).norm(dim=-1).mean()),
                         pooled_norm_ratio=float((feats.norm(dim=-1) / clean.norm(dim=-1)).mean()))
            x = feats.cpu().numpy().astype('float64').reshape(-1, feats.shape[-1])
            for name, (det, t) in dets.items():
                scores = det.score(x).reshape(feats.shape[:2])
                flipped = attacked(scores, t)
                entry[name] = dict(rate=float(flipped.mean()), mean_score=float(scores.mean()),
                                   rate_initially_correct=float(flipped[:, correct[name]].mean()) if correct[name].any() else None,
                                   n_initially_correct=int(correct[name].sum()))
            result['sides'][side][family] = entry
    return result


def code_commit():
    if os.environ.get('GIT_SHA'):  # compute nodes have no git; the launcher passes the commit in
        return os.environ['GIT_SHA']
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train_directions(cfg, splits, steer_splits, cache, head_dir, output, device='cpu', image_root=None):
    """Train the id2ood and ood2id swarms, export unit directions and controls, run the sanity check."""
    cfg.validate()
    out, start, device = new_output(output), time.perf_counter(), torch.device(device)
    torch.manual_seed(cfg.seed)
    encoder, preprocess, details = build_encoder(*MODELS[cfg.model], seed=cfg.seed, device=str(device))
    data = load_cache(cache)
    if data['metadata'].get('actual_encoder_state_sha256') != details['state_sha256']:
        raise ValueError('Cache was exported with a different encoder checkpoint')
    split = SplitModel(encoder, load_head(head_dir, cache).to(device).requires_grad_(False), cfg.layer)
    split.check_sim_layer(cfg.sim_layer)
    frozen = (state_hash(encoder), state_hash(split.head))
    autocast = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp == 'bf16')

    steering = json.loads(Path(steer_splits).read_text())
    if steering['metadata']['main_manifest_sha256'] != file_hash(splits):
        raise ValueError('Steering splits were prepared from a different main manifest')
    datasets = {k: open_dataset(relocate(v, image_root)) for k, v in steering['sources'].items()}
    rows = {role: [r for r in steering['rows'] if r['role'] == role] for role in STEER_ROLES}
    view = augment(preprocess, datasets['train'][0][0].size[::-1])
    pool = lambda role, *transforms: Pool(rows[role], datasets, transforms or (preprocess,))

    manifest = dict(config=asdict(cfg), source_commit=SOURCE_COMMIT, code_commit=code_commit(),
        attribution='SplitModel/steer/SteeringSwarm/infonce ported from Graph_OOD steering-astrid',
        encoder=details, head_sha256=file_hash(Path(head_dir)/'head.pt'), cache_sha256=file_hash(cache),
        splits_sha256=file_hash(splits), steer_splits_sha256=file_hash(steer_splits),
        role_counts={k: len(v) for k, v in rows.items()}, image_root=str(image_root) if image_root else None,
        preprocess=repr(preprocess),
        augmentation=repr(view) if cfg.id_augment else None,
        ood_training_data='CIFAR100 train' if cfg.ood_mode == 'cifar100' else 'ID steering-train activation + random vector',
        loss='pure multi-positive InfoNCE; no detector score or threshold enters training',
        versions=versions(), source_hashes=source_hashes(), status='running')
    write_json(out/'manifest.json', manifest)
    try:
        manifest['equivalence'] = equivalence(split, pool('id_steer_dev'), data, device)
        h_id_dev = activations(split, pool('id_steer_dev'), device)
        h_ood_dev = activations(split, pool('ood_steer_dev'), device)
        median = float(SplitModel.position_norms(h_id_dev).median())
        radius, noise = cfg.radius * median, cfg.ood_noise_radius * median
        dim, positions = SplitModel.dim(h_id_dev), SplitModel.positions(h_id_dev)
        manifest['radii'] = dict(activation_norm_median=median, radius=radius, noise_radius=noise,
            positions=positions, full_map_radius=radius * positions**.5, dim=dim,
            units='injection space, per position; resolved on un-augmented ID steering-dev')
        gen = torch.Generator().manual_seed(cfg.seed)
        noise_gen = torch.Generator(device=device).manual_seed(cfg.seed)
        random_unit = lambda n: F.normalize(torch.randn(n, dim, generator=noise_gen, device=device), dim=1)
        swarms = nn.ModuleDict({n: SteeringSwarm(cfg.swarm_size, dim, radius, torch.randn(cfg.swarm_size, dim, generator=gen))
                                for n in SWARMS}).to(device)
        opt = torch.optim.Adam(swarms.parameters(), lr=cfg.lr)

        # Fixed un-augmented development batch that mirrors the training objective.
        n_dev = min(cfg.bs_id, len(h_id_dev) // 2)
        dev_id = h_id_dev[:n_dev].to(device)
        dev_ood = (h_ood_dev[:cfg.bs_ood].to(device) if cfg.ood_mode == 'cifar100'
                   else offset(h_id_dev[n_dev:2 * n_dev].to(device), noise * random_unit(n_dev)))
        with torch.no_grad():
            dev_sims = [split.suffix_sim(h, cfg.sim_layer)[0] for h in (dev_id, dev_ood)]

        id_batches = batches(pool('id_steer_train', view, view) if cfg.id_augment else pool('id_steer_train'),
                             cfg.bs_id, cfg.seed, cfg.workers)
        ood_batches = batches(pool('ood_steer_train' if cfg.ood_mode == 'cifar100' else 'id_steer_train'),
                              cfg.bs_ood, cfg.seed + 1, cfg.workers)
        acc = dict.fromkeys(SWARMS, 0.)
        with open(out/'train_log.jsonl', 'w') as log:
            for step in range(1, cfg.steps + 1):
                views, (x_ood,) = next(id_batches), next(ood_batches)
                with torch.no_grad(), autocast():
                    h_id = split.prefix(views[0].to(device))
                    # Anchor and clean reference are different views of the same ID images.
                    h_ref = split.prefix(views[1].to(device)) if cfg.id_augment else h_id
                    h_ood = split.prefix(x_ood.to(device))
                    if cfg.ood_mode == 'random':
                        h_ood = offset(h_ood, noise * random_unit(len(h_ood)).to(h_ood.dtype))
                    sim_id, sim_ood = (split.suffix_sim(h, cfg.sim_layer)[0] for h in (h_ref, h_ood))
                opt.zero_grad(set_to_none=True)
                for name, value in swarm_losses(split, cfg, swarms, h_id, sim_id, h_ood, sim_ood, autocast, backward=True).items():
                    acc[name] += value
                opt.step()
                if step % cfg.log_every == 0 or step == cfg.steps:
                    n = (step - 1) % cfg.log_every + 1
                    with torch.no_grad():
                        dev = swarm_losses(split, cfg, swarms, dev_id, dev_sims[0], dev_ood, dev_sims[1], autocast)
                    rec = dict(step=step, seconds=round(time.perf_counter() - start, 1))
                    for name in SWARMS:
                        rec[f'{name}_loss'], rec[f'{name}_dev_loss'] = acc[name] / n, dev[name]
                    acc = dict.fromkeys(SWARMS, 0.)
                    log.write(json.dumps(rec) + '\n'); log.flush()
                    print('  '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in rec.items()), flush=True)

        # Controls in the same injection space; PCA sees ID steering-train only, the centroid is OOD-informed.
        reduced = activations(split, pool('id_steer_train'), device, reduce=True).numpy().astype('float64')
        pca, pca_diagnostics = estimate_direction(reduced)
        centroid = SplitModel.position_mean(h_ood_dev) - SplitModel.position_mean(h_id_dev)
        vectors = {n: s.directions().cpu().numpy() for n, s in swarms.items()}
        vectors.update(random=F.normalize(torch.randn(cfg.swarm_size, dim, generator=gen), dim=1).numpy(),
                       pca=pca[None], centroid=F.normalize(centroid, dim=0).numpy()[None])
        np.savez_compressed(out/'vectors.npz', **vectors, radius=radius, noise_radius=noise,
                            activation_norm_median=median)
        manifest['controls'] = dict(pca=pca_diagnostics, random='isotropic, matched count',
                                    centroid='ID->OOD steering-dev mean difference; OOD-informed (CIFAR100 dev)')
        if (state_hash(encoder), state_hash(split.head)) != frozen:
            raise RuntimeError('Encoder or head weights changed during direction training')
        sanity = sanity_check(cfg, split, data, load_head(head_dir, cache), vectors, radius, h_id_dev, h_ood_dev, device)
        write_json(out/'sanity.json', sanity)
        manifest.update(status='completed', seconds=time.perf_counter() - start,
                        peak_gpu_memory_gb=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None)
        write_json(out/'manifest.json', manifest)
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc), seconds=time.perf_counter() - start)
        write_json(out/'manifest.json', manifest)
        raise
    return sanity
