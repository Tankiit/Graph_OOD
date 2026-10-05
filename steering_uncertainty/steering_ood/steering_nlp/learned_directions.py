"""Learned steering directions for text: InfoNCE swarms on frozen CLINC150 sentence encoders.

Text counterpart of steering_ood.learned_directions, whose swarm, loss and run conventions are reused
unchanged. A SentenceTransformer (Transformer -> Pooling -> Normalize) is cut after one transformer
block: prefix(features) -> Tokens(h, mask), suffix(Tokens) -> (unit-norm embedding, CLINC logits).
A vector is added to every token, so its norm is the displacement per token. At 'pooled' it is added
to the final unit-norm embedding without renormalising, like the existing text steering paths.
ID views for training (`id_augment`) are SimCSE-style: the prefix runs twice with dropout on.
OOD training queries are CLINC oos_train (steering-train) and oos_val (steering-dev); oos_test,
used for evaluation, is never read.
"""
from dataclasses import asdict, dataclass
import copy
import inspect
import json
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from steering_ood.core import file_hash, load_cache, new_output, source_hashes, threshold, versions, write_json
from steering_ood.detectors import make_detector
from steering_ood.experiment import estimate_direction
from steering_ood.head import load_head
from steering_ood.learned_directions import (EQUIVALENCE_TOLERANCE, SOURCE_COMMIT, STEER_ROLES, SWARMS,
                                             SteerConfig, SteeringSwarm, code_commit, infonce)
from steering_ood.vision import state_hash

TEXT_MODELS = {  # same checkpoints as scripts/run_pipeline.py, so the cached features and heads match
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'bge_base': 'BAAI/bge-base-en-v1.5',
    'bge_large': 'BAAI/bge-large-en-v1.5',
}
# Same relative depth as the ViT default blocks.9 of 12.
TEXT_DEFAULT_LAYERS = dict(mpnet='blocks.9', minilm='blocks.4', bge_base='blocks.9', bge_large='blocks.19')
STEER_SPLITS_NAME = 'steer_splits_clinc.json'
canonicalize = lambda text: ' '.join(text.lower().split())  # as steering_ood.data.prepare_clinc


@dataclass
class TextSteerConfig(SteerConfig):
    """SteerConfig for text. n_ood_train/n_ood_dev are unused: the OOD pools are CLINC oos_train/oos_val."""
    model: str = 'mpnet'              # mpnet | minilm | bge_base | bge_large
    ood_mode: str = 'clinc'           # clinc: CLINC out-of-scope queries | random: ID query + large random vector
    workers: int = 0                  # tokenisation is cheap; keep it in the main process

    def __post_init__(self):
        self.layer = self.layer or TEXT_DEFAULT_LAYERS.get(self.model, '')

    def validate(self):
        if self.model not in TEXT_MODELS or self.ood_mode not in ('clinc', 'random'):
            raise ValueError(f'model must be one of {list(TEXT_MODELS)}; ood_mode must be clinc or random')
        # The remaining checks are SteerConfig's, run on a copy with names it accepts.
        vision = copy.copy(self)
        vision.model, vision.ood_mode = 'resnet18', 'random'
        SteerConfig.validate(vision)
        return self


@dataclass
class Tokens:
    """Token activations [B, T, D] with their attention mask [B, T] (1 = real token)."""
    h: torch.Tensor
    mask: torch.Tensor

    def __len__(self):
        return len(self.h)

    def __getitem__(self, i):
        return Tokens(self.h[i], self.mask[i])

    def to(self, device):
        return Tokens(self.h.to(device), self.mask.to(device))

    def float(self):
        return Tokens(self.h.float(), self.mask)


def cat_tokens(items):
    """Concatenate Tokens batches of different lengths, padding with masked zeros."""
    T = max(t.h.shape[1] for t in items)
    pad = lambda x, value: F.pad(x, (0, 0, 0, T - x.shape[1]) if x.dim() == 3 else (0, T - x.shape[1]), value=value)
    return Tokens(torch.cat([pad(t.h, 0.) for t in items]), torch.cat([pad(t.mask, 0) for t in items]))


def steer(h, v):
    """h Tokens [B, T, D] or pooled [B, D], v [K, D] -> K*B: every vector applied to every sample (every token)."""
    if isinstance(h, Tokens):
        return Tokens((h.h[None] + v[:, None, None, :]).flatten(0, 1), h.mask.repeat(len(v), 1))
    return (h[None] + v[:, None, :]).flatten(0, 1)


def offset(h, v):
    """One vector per sample (synthetic OOD), added to every token."""
    if isinstance(h, Tokens):
        return Tokens(h.h + v[:, None, :], h.mask)
    return h + v


def dim(h):
    return h.h.shape[-1] if isinstance(h, Tokens) else h.shape[-1]


def position_norms(h):
    """Norm of every real token (or pooled sample), flattened."""
    if isinstance(h, Tokens):
        return h.h.norm(dim=-1)[h.mask.bool()]
    return h.norm(dim=-1)


def position_mean(h):
    """Mean activation over samples and real tokens: [D]."""
    if isinstance(h, Tokens):
        m = h.mask[..., None].to(h.h.dtype)
        return (h.h * m).sum((0, 1)) / m.sum()
    return h.mean(0)


def positions(h):
    """Mean number of real tokens that receive the vector (1 when pooled)."""
    return float(h.mask.sum(1).float().mean()) if isinstance(h, Tokens) else 1.


class TextSplitModel(nn.Module):
    """A SentenceTransformer cut after `layer` (blocks.0 ... blocks.N or pooled), with our CLINC head."""

    def __init__(self, st, head, layer):
        super().__init__()
        kinds = [type(m).__name__ for m in st]
        if kinds != ['Transformer', 'Pooling', 'Normalize']:
            raise ValueError(f'Expected Transformer -> Pooling -> Normalize, got {kinds}')
        self.st, self.head, self.layer = st, head, layer
        self.hf = getattr(st[0], 'auto_model', None) or st[0].model
        self.pooling, self.normalize = st[1], st[2]
        blocks = list(self.hf.encoder.layer)
        self.block_names = [f'blocks.{i}' for i in range(len(blocks))]
        self.layers = self.block_names + ['pooled']
        if layer not in self.layers:
            raise ValueError(f'unknown layer {layer!r}; choose from {self.layers}')
        self.cut = len(blocks) if layer == 'pooled' else self.block_names.index(layer) + 1
        self.pre_encoder = self._sliced(blocks[:self.cut])
        self.post_single = [self._sliced([b]) for b in blocks[self.cut:]]
        self.embed_params = set(inspect.signature(self.hf.embeddings.forward).parameters)

    def _sliced(self, blocks):
        """The model's own encoder (mask handling, MPNet position bias) restricted to some blocks."""
        enc = copy.copy(self.hf.encoder)
        enc._modules = dict(enc._modules)
        enc.layer = nn.ModuleList(blocks)
        return enc

    def _attention(self, h, mask):
        from transformers.masking_utils import create_bidirectional_mask
        return create_bidirectional_mask(config=self.hf.config, inputs_embeds=h, attention_mask=mask)

    @staticmethod
    def _run(encoder, h, attention):
        return encoder(h, attention_mask=attention)[0] if len(encoder.layer) else h

    def _embed(self, features):
        return self.hf.embeddings(**{k: features[k] for k in ('input_ids', 'token_type_ids')
                                     if k in features and k in self.embed_params})

    def _sentence(self, h, mask):
        """The encoder's pooling of a token sequence, without normalisation."""
        return self.pooling({'token_embeddings': h, 'attention_mask': mask})['sentence_embedding']

    def reduce(self, h):
        """One vector per sentence: the model's own pooling (mean or CLS) of the tokens."""
        return self._sentence(h.h, h.mask) if isinstance(h, Tokens) else h

    @contextmanager
    def dropout(self, active=True):
        """Dropout views: train mode switches dropout on (weights stay frozen, no gradient)."""
        self.hf.train(active)
        try:
            yield
        finally:
            self.hf.eval()

    def prefix(self, features):
        mask = features['attention_mask']
        h = self._embed(features)
        h = self._run(self.pre_encoder, h, self._attention(h, mask))
        if self.layer == 'pooled':
            return self.normalize({'sentence_embedding': self._sentence(h, mask)})['sentence_embedding']
        return Tokens(h, mask)

    def suffix_sim(self, h, sim_layer):
        """(activation at sim_layer reduced to [B, d], unit-norm embedding, logits)."""
        if self.layer == 'pooled':  # added after normalisation, as the existing text paths
            return h, h, self.head(h.float())
        x, mask = h.h, h.mask
        attention = self._attention(x, mask)
        sim = self._sentence(x, mask) if sim_layer == self.layer else None
        for name, block in zip(self.block_names[self.cut:], self.post_single):
            x = self._run(block, x, attention)
            if name == sim_layer:
                sim = self._sentence(x, mask)
        feat = self.normalize({'sentence_embedding': self._sentence(x, mask)})['sentence_embedding']
        if sim_layer == 'pooled':
            sim = feat
        return sim, feat, self.head(feat.float())

    def suffix(self, h):
        _, feat, logits = self.suffix_sim(h, 'pooled')
        return feat, logits

    def check_sim_layer(self, sim_layer):
        if sim_layer not in self.layers:
            raise ValueError(f'unknown sim_layer {sim_layer!r}; choose from {self.layers}')
        if self.layers.index(sim_layer) < self.layers.index(self.layer):
            raise ValueError(f'sim_layer {sim_layer!r} is upstream of the steered layer {self.layer!r}')


def load_encoder(model, device='cpu'):
    """Frozen SentenceTransformer in eval mode with provenance."""
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(TEXT_MODELS[model], device=str(device)).eval().requires_grad_(False)
    hf = getattr(st[0], 'auto_model', None) or st[0].model
    details = dict(model=TEXT_MODELS[model], max_seq_length=st.max_seq_length,
                   resolved_revision=getattr(hf.config, '_commit_hash', None),
                   encoder_modules=str(st), state_sha256=state_hash(st))
    return st, details


def prepare_steering_text(splits, raw, output, seed=7, dev_fraction=.2):
    """ID: the CLINC direction role split per intent. OOD: oos_train (train) and oos_val (dev).

    Queries whose normalised text occurs in the main splits, or repeats, are dropped and logged.
    No calibration, reference, probe, head or test row, and no oos_test query, is used.
    """
    if Path(output).exists():
        raise FileExistsError(output)
    manifest = json.loads(Path(splits).read_text())
    data = json.loads(Path(raw).read_text())
    rng, rows = np.random.default_rng(seed), []
    direction = [r for r in manifest['rows'] if r['role'] == 'direction']
    for c in sorted({r['label'] for r in direction}):
        group = [r for r in direction if r['label'] == c]
        n_dev = round(len(group) * dev_fraction)
        if not 0 < n_dev < len(group):
            raise ValueError(f'Intent {c} cannot be split into steering train and dev')
        for j, i in enumerate(rng.permutation(len(group))):
            rows.append(dict(group[i], role='id_steer_dev' if j < n_dev else 'id_steer_train'))
    seen, dropped = {canonicalize(r['text']) for r in manifest['rows']}, []
    for split, role in (('oos_train', 'ood_steer_train'), ('oos_val', 'ood_steer_dev')):
        for i, (text, label) in enumerate(data[split]):
            if label != 'oos':
                raise ValueError(f'{split} contains an in-scope query')
            if canonicalize(text) in seen:
                dropped.append(dict(id=f'clinc:{split}:{i}', reason='normalised text repeats or occurs in the main splits'))
                continue
            seen.add(canonicalize(text))
            rows.append(dict(id=f'clinc:{split}:{i}', text=text, label=-1, role=role))
    write_json(output, dict(rows=rows, metadata=dict(
        seed=seed, dev_fraction=dev_fraction, main_manifest_sha256=file_hash(splits),
        raw_sha256=file_hash(raw), role_counts=dict(Counter(r['role'] for r in rows)),
        dropped_duplicates=dropped, id_source='main splits role direction, stratified per intent',
        ood_source='CLINC oos_train (steering-train) and oos_val (steering-dev); oos_test untouched')))


class TextPool(torch.utils.data.Dataset):
    """Query texts of one steering role."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]['text']


def tokenizer_collate(st):
    def collate(texts):
        features = st.preprocess(list(texts)) if hasattr(st, 'preprocess') else st.tokenize(list(texts))
        return {k: v for k, v in features.items() if isinstance(v, torch.Tensor)}
    return collate


def to_device(features, device):
    return {k: v.to(device) for k, v in features.items()}


def text_batches(pool, st, batch_size, seed, workers):
    """Endless shuffled, tokenised batches."""
    loader = DataLoader(pool, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=workers,
                        collate_fn=tokenizer_collate(st), persistent_workers=workers > 0,
                        generator=torch.Generator().manual_seed(seed))
    while True:
        yield from loader


@torch.no_grad()
def text_activations(split, pool, device, batch_size=128):
    """Clean activations at the steering layer for a whole pool, on the CPU (Tokens padded together)."""
    out = []
    for features in DataLoader(pool, batch_size=batch_size, collate_fn=tokenizer_collate(split.st)):
        h = split.prefix(to_device(features, device))
        out.append(Tokens(h.h.float().cpu(), h.mask.cpu()) if isinstance(h, Tokens) else h.float().cpu())
    return cat_tokens(out) if isinstance(out[0], Tokens) else torch.cat(out)


@torch.no_grad()
def text_equivalence(split, pool, cache, device, n=64):
    """The split forward must reproduce SentenceTransformer and the cached features of the same queries."""
    rows = pool.rows[:min(n, len(pool))]
    features = to_device(tokenizer_collate(split.st)([r['text'] for r in rows]), device)
    feat = split.suffix(split.prefix(features))[0].float().cpu().numpy()
    direct = split.st(dict(features))['sentence_embedding'].float().cpu().numpy()
    where = {i: j for j, i in enumerate(cache['direction_ids'].tolist())}
    cached = cache['direction_x'][[where[r['id']] for r in rows]]
    scale = float(np.abs(direct).max())
    result = dict(n=len(rows), feature_scale=scale, direct=float(np.abs(feat - direct).max() / scale),
                  cached=float(np.abs(feat - cached).max() / scale), tolerance=EQUIVALENCE_TOLERANCE,
                  units='max abs error relative to the largest feature magnitude')
    for key, tol in EQUIVALENCE_TOLERANCE.items():
        if not result[key] <= tol:
            raise ValueError(f'Split model disagrees with the {key} features: {result}')
    return result


def text_swarm_losses(split, cfg, swarms, h_id, sim_id, h_ood, sim_ood, autocast, backward=False):
    """Mean InfoNCE per swarm; as steering_ood.learned_directions.swarm_losses, for Tokens."""
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
def text_sanity_check(cfg, split, data, head, vectors, radius, h_id, h_ood, device):
    """Development-only flip rates at the fixed clean threshold; as the vision sanity check.

    `head` must be a separate instance from the one inside `split` (detector adapters move it to CPU).
    """
    dets = {}
    for name in cfg.detectors:
        det = make_detector(name, 'pytorch', cfg.k, head).fit(data['reference_x'], data['reference_y'])
        dets[name] = (det, threshold(det.score(data['calibration_x']), cfg.reject_tau))
    result = dict(note='steering-dev queries only (no test data, no oos_test); descriptive development check',
                  radius=radius, thresholds={n: t for n, (_, t) in dets.items()},
                  outcome=dict(id='ID rejection rate', ood='OOD acceptance rate'), clean={}, sides={})
    for side, h, learned in (('id', h_id, 'id2ood'), ('ood', h_ood, 'ood2id')):
        h = h[:cfg.sanity_probes].to(device)
        clean = split.suffix(h)[0].float()
        attacked = lambda scores, t: scores > t if side == 'id' else scores <= t
        result['clean'][side], correct = {}, {}
        for name, (det, t) in dets.items():
            scores = det.score(clean.cpu().numpy().astype('float64'))
            correct[name] = ~attacked(scores, t)
            result['clean'][side][name] = dict(scores=scores, mean_score=float(scores.mean()),
                                               rate=float(attacked(scores, t).mean()))
        result['sides'][side] = {}
        for family in (learned, 'random'):
            v = radius * torch.tensor(vectors[family][:cfg.sanity_vectors], dtype=torch.float32, device=device)
            feats = torch.cat([split.suffix(steer(h, v[i:i + 1]))[0].float() for i in range(len(v))]
                              ).view(len(v), len(h), -1)
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


def train_text_directions(cfg, splits, steer_splits, cache, head_dir, output, device='cpu'):
    """Train id2ood and ood2id swarms on a text encoder, export directions and controls, sanity-check."""
    cfg.validate()
    out, start, device = new_output(output), time.perf_counter(), torch.device(device)
    torch.manual_seed(cfg.seed)
    st, details = load_encoder(cfg.model, device)
    data = load_cache(cache)
    if data['metadata'].get('encoder') != TEXT_MODELS[cfg.model]:
        raise ValueError(f"Cache was encoded with {data['metadata'].get('encoder')}, not {TEXT_MODELS[cfg.model]}")
    split = TextSplitModel(st, load_head(head_dir, cache).to(device).requires_grad_(False), cfg.layer)
    split.check_sim_layer(cfg.sim_layer)
    frozen = (state_hash(st), state_hash(split.head))
    autocast = lambda: torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp == 'bf16')

    steering = json.loads(Path(steer_splits).read_text())
    if steering['metadata']['main_manifest_sha256'] != file_hash(splits):
        raise ValueError('Steering splits were prepared from a different main manifest')
    rows = {role: [r for r in steering['rows'] if r['role'] == role] for role in STEER_ROLES}
    pool = lambda role: TextPool(rows[role])

    manifest = dict(config=asdict(cfg), modality='text', source_commit=SOURCE_COMMIT, code_commit=code_commit(),
        attribution='steering_nlp text adaptation of the swarm/InfoNCE port from Graph_OOD steering-astrid',
        encoder=details, head_sha256=file_hash(Path(head_dir)/'head.pt'), cache_sha256=file_hash(cache),
        splits_sha256=file_hash(splits), steer_splits_sha256=file_hash(steer_splits),
        role_counts={k: len(v) for k, v in rows.items()},
        augmentation='SimCSE dropout views (two prefix passes with dropout on)' if cfg.id_augment else None,
        pooled_steering='added to the final unit-norm embedding, not renormalised',
        ood_training_data='CLINC oos_train' if cfg.ood_mode == 'clinc' else 'ID steering-train activation + random vector',
        loss='pure multi-positive InfoNCE; no detector score or threshold enters training',
        versions=versions(), source_hashes=source_hashes(), status='running')
    write_json(out/'manifest.json', manifest)
    try:
        manifest['equivalence'] = text_equivalence(split, pool('id_steer_dev'), data, device)
        h_id_dev = text_activations(split, pool('id_steer_dev'), device)
        h_ood_dev = text_activations(split, pool('ood_steer_dev'), device)
        median = float(position_norms(h_id_dev).median())
        radius, noise = cfg.radius * median, cfg.ood_noise_radius * median
        d, n_pos = dim(h_id_dev), positions(h_id_dev)
        manifest['radii'] = dict(activation_norm_median=median, radius=radius, noise_radius=noise,
            positions=n_pos, full_map_radius=radius * n_pos**.5, dim=d,
            units='injection space, per token (mean real tokens per query); un-augmented ID steering-dev')
        gen = torch.Generator().manual_seed(cfg.seed)
        noise_gen = torch.Generator(device=device).manual_seed(cfg.seed)
        random_unit = lambda n: F.normalize(torch.randn(n, d, generator=noise_gen, device=device), dim=1)
        swarms = nn.ModuleDict({n: SteeringSwarm(cfg.swarm_size, d, radius, torch.randn(cfg.swarm_size, d, generator=gen))
                                for n in SWARMS}).to(device)
        opt = torch.optim.Adam(swarms.parameters(), lr=cfg.lr)

        # Fixed un-augmented development batch that mirrors the training objective.
        n_dev = min(cfg.bs_id, len(h_id_dev) // 2)
        dev_id = h_id_dev[:n_dev].to(device)
        dev_ood = (h_ood_dev[:cfg.bs_ood].to(device) if cfg.ood_mode == 'clinc'
                   else offset(h_id_dev[n_dev:2 * n_dev].to(device), noise * random_unit(n_dev)))
        with torch.no_grad():
            dev_sims = [split.suffix_sim(h, cfg.sim_layer)[0] for h in (dev_id, dev_ood)]

        id_batches = text_batches(pool('id_steer_train'), st, cfg.bs_id, cfg.seed, cfg.workers)
        ood_batches = text_batches(pool('ood_steer_train' if cfg.ood_mode == 'clinc' else 'id_steer_train'),
                                   st, cfg.bs_ood, cfg.seed + 1, cfg.workers)
        acc = dict.fromkeys(SWARMS, 0.)
        with open(out/'train_log.jsonl', 'w') as log:
            for step in range(1, cfg.steps + 1):
                x_id, x_ood = to_device(next(id_batches), device), to_device(next(ood_batches), device)
                with torch.no_grad(), autocast():
                    if cfg.id_augment:  # anchor and clean reference are two dropout views of the same queries
                        with split.dropout():
                            h_id, h_ref = split.prefix(x_id), split.prefix(x_id)
                    else:
                        h_id = h_ref = split.prefix(x_id)
                    h_ood = split.prefix(x_ood)
                    if cfg.ood_mode == 'random':
                        h_ood = offset(h_ood, noise * random_unit(len(h_ood)))
                    sim_id, sim_ood = (split.suffix_sim(h, cfg.sim_layer)[0] for h in (h_ref, h_ood))
                opt.zero_grad(set_to_none=True)
                losses = text_swarm_losses(split, cfg, swarms, h_id, sim_id, h_ood, sim_ood, autocast, backward=True)
                for name, value in losses.items():
                    acc[name] += value
                opt.step()
                if step % cfg.log_every == 0 or step == cfg.steps:
                    n = (step - 1) % cfg.log_every + 1
                    with torch.no_grad():
                        dev = text_swarm_losses(split, cfg, swarms, dev_id, dev_sims[0], dev_ood, dev_sims[1], autocast)
                    rec = dict(step=step, seconds=round(time.perf_counter() - start, 1))
                    for name in SWARMS:
                        rec[f'{name}_loss'], rec[f'{name}_dev_loss'] = acc[name] / n, dev[name]
                    acc = dict.fromkeys(SWARMS, 0.)
                    log.write(json.dumps(rec) + '\n'); log.flush()
                    print('  '.join(f'{k} {v:.4g}' if isinstance(v, float) else f'{k} {v}' for k, v in rec.items()), flush=True)

        # Controls in the injection space; PCA sees ID steering-train only, the centroid is OOD-informed.
        h_train = text_activations(split, pool('id_steer_train'), device)
        reduced = (split.reduce(h_train) if isinstance(h_train, Tokens) else h_train).numpy().astype('float64')
        pca, pca_diagnostics = estimate_direction(reduced)
        centroid = position_mean(h_ood_dev) - position_mean(h_id_dev)
        vectors = {n: s.directions().cpu().numpy() for n, s in swarms.items()}
        vectors.update(random=F.normalize(torch.randn(cfg.swarm_size, d, generator=gen), dim=1).numpy(),
                       pca=pca[None], centroid=F.normalize(centroid, dim=0).numpy()[None])
        np.savez_compressed(out/'vectors.npz', **vectors, radius=radius, noise_radius=noise,
                            activation_norm_median=median)
        manifest['controls'] = dict(pca=pca_diagnostics, random='isotropic, matched count',
                                    centroid='ID->OOD steering-dev mean difference; OOD-informed (CLINC oos_val)')
        if (state_hash(st), state_hash(split.head)) != frozen:
            raise RuntimeError('Encoder or head weights changed during direction training')
        sanity = text_sanity_check(cfg, split, data, load_head(head_dir, cache), vectors, radius,
                                   h_id_dev, h_ood_dev, device)
        write_json(out/'sanity.json', sanity)
        manifest.update(status='completed', seconds=time.perf_counter() - start,
                        peak_gpu_memory_gb=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else None)
        write_json(out/'manifest.json', manifest)
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc), seconds=time.perf_counter() - start)
        write_json(out/'manifest.json', manifest)
        raise
    return sanity
