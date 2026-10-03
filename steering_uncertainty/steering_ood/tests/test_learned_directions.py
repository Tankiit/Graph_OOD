"""Learned directions: ported mechanics, split isolation and a generated end-to-end run.

Uses a tiny ResNet-shaped network and generated images, never benchmark claims.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from steering_ood import learned_directions as ld
from steering_ood.learned_directions import (SplitModel, SteerConfig, SteeringSwarm, infonce,
                                             offset, steer, swarm_losses)

PER_CLASS = dict(head=6, reference=8, direction=10, calibration=10, probe=2, id_test=4)


class TinyResNet(torch.nn.Module):
    """Same attribute layout as a torchvision ResNet whose fc was removed."""

    def __init__(self):
        super().__init__()
        nn = torch.nn
        block = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU())
        self.conv1, self.bn1 = nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4)
        self.relu, self.maxpool = nn.ReLU(), nn.MaxPool2d(2)
        self.layer1 = nn.Sequential(block(4, 4), block(4, 4))
        self.layer2 = nn.Sequential(block(4, 8), block(8, 8))
        self.layer3 = nn.Sequential(block(8, 8), block(8, 8))
        self.layer4 = nn.Sequential(block(8, 16), block(16, 16))
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return torch.flatten(self.avgpool(self.layer4(self.layer3(self.layer2(self.layer1(x))))), 1)


def tiny(seed=0):
    torch.manual_seed(seed)
    return TinyResNet().eval().requires_grad_(False), torch.nn.Linear(16, 2).requires_grad_(False)


class Recorded(list):
    """Dataset stand-in that remembers which indices were read."""

    def __init__(self, items):
        super().__init__(items)
        self.seen = set()

    def __getitem__(self, i):
        self.seen.add(int(i))
        return super().__getitem__(i)


def picture(rng, c=None):
    from PIL import Image
    pixels = rng.integers(0, 256, (16, 16, 3), dtype='uint8')
    if c is not None:  # fixture-only class cue
        pixels[:, :, c] = pixels[:, :, c] // 2 + 128
    return Image.fromarray(pixels)


def make_manifest(tmp_path, monkeypatch):
    """A CIFAR10-named main manifest over generated images, and a stand-in CIFAR100 train split.

    The stand-in holds 13 distinct images, one copy of an ID image and one repeated image.
    """
    from steering_ood.vision_data import image_hash
    rng = np.random.default_rng(5)
    items, rows = [], []
    for c in range(2):
        for role, n in PER_CLASS.items():
            for _ in range(n):
                image = picture(rng, c)
                rows.append(dict(id=f'image:fake:{len(items)}', source='train', index=len(items), role=role,
                                 label=c, pixel_sha256=image_hash(image)))
                items.append((image, c))
    distinct = [picture(rng) for _ in range(13)]
    id_ds, ood_ds = Recorded(items), Recorded([(i, 0) for i in distinct + [items[0][0], distinct[0]]])
    monkeypatch.setattr(ld, 'open_dataset', lambda spec, download=False: dict(cifar10=id_ds, cifar100=ood_ds)[spec['name']])
    splits = tmp_path / 'splits.json'
    splits.write_text(json.dumps(dict(sources=dict(train=dict(name='cifar10', root=str(tmp_path), split='train')), rows=rows)))
    return splits, rows, id_ds, ood_ds


@pytest.mark.parametrize('layer', ['layer1.0', 'layer3.0', 'layer4.1', 'pooled'])
def test_zero_offset_matches_encoder(layer):
    enc, head = tiny()
    x = torch.randn(3, 3, 16, 16)
    split = SplitModel(enc, head, layer)
    h = split.prefix(x)
    feat, logits = split.suffix(steer(h, torch.zeros(1, split.dim(h))))
    assert torch.allclose(feat, enc(x), atol=1e-6) and torch.allclose(logits, head(enc(x)), atol=1e-6)
    assert not any(p.requires_grad for p in split.parameters())
    with pytest.raises(ValueError, match='unknown layer'):
        SplitModel(enc, head, 'layer5.0')


def test_similarity_layer_must_be_downstream():
    enc, head = tiny()
    split = SplitModel(enc, head, 'layer3.0')
    h = split.prefix(torch.randn(2, 3, 16, 16))
    with pytest.raises(ValueError, match='upstream'):
        split.check_sim_layer('layer2.1')
    with pytest.raises(ValueError, match='unknown'):
        split.check_sim_layer('layer9.0')
    split.check_sim_layer('layer3.0'); split.check_sim_layer('pooled')
    own, feat, _ = split.suffix_sim(h, 'layer3.0')
    assert torch.equal(own, h.mean((2, 3)))
    later, _, _ = split.suffix_sim(h, 'layer4.0')
    assert torch.allclose(later, enc.layer4[0](enc.layer3[1](h)).mean((2, 3)), atol=1e-6)
    pooled, pooled_feat, _ = split.suffix_sim(h, 'pooled')
    assert torch.equal(pooled, pooled_feat) and torch.allclose(feat, pooled_feat, atol=1e-6)


def test_steer_applies_every_vector_at_every_position():
    torch.manual_seed(0)
    h, v = torch.randn(3, 8, 4, 4), torch.randn(2, 8)
    out = steer(h, v)
    assert out.shape == (6, 8, 4, 4)
    for k in range(2):  # rows k*B .. (k+1)*B carry vector k
        assert torch.allclose(out[3 * k:3 * k + 3] - h, v[k][None, :, None, None].expand_as(h), atol=1e-6)
    flat = h.mean((2, 3))
    assert torch.allclose(steer(flat, v)[3:] - flat, v[1].expand_as(flat), atol=1e-6)
    noise = 5 * torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    assert torch.allclose((offset(h, noise) - h).norm(dim=1), torch.full((3, 4, 4), 5.), atol=1e-5)
    assert torch.allclose((offset(flat, noise) - flat).norm(dim=1), torch.full((3,), 5.), atol=1e-5)


def test_swarm_stays_on_sphere():
    torch.manual_seed(0)
    swarm, target = SteeringSwarm(4, 8, 2.5), torch.randn(4, 8)
    start = swarm.directions().clone()
    opt = torch.optim.Adam(swarm.parameters(), lr=.5)
    for _ in range(3):
        opt.zero_grad(); (swarm() * target).sum().backward(); opt.step()
    assert not torch.allclose(swarm.directions(), start)
    assert torch.allclose(swarm().norm(dim=1), torch.full((4,), 2.5), atol=1e-5)
    assert torch.allclose(swarm.directions().norm(dim=1), torch.ones(4), atol=1e-6)


def test_infonce_matches_hand_computation():
    sim = torch.tensor([[[1., 0.], [0., 2.]]])  # K=1, B=2
    same, other = torch.tensor([[3., 0.], [0., 1.]]), torch.tensor([[1., 1.]])
    unit = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)
    a, s, o = unit(sim.numpy()[0]), unit(same.numpy()), unit(other.numpy())

    def expected(mask_own):
        losses = []
        for i in range(2):
            pos, neg = np.exp(a[i] @ o.T / .5), np.exp(a[i] @ s.T / .5)
            if mask_own:
                neg = np.delete(neg, i)
            losses.append(np.log(pos.sum() + neg.sum()) - np.log(pos.sum()))
        return np.mean(losses)

    assert infonce(sim, same, other, .5).item() == pytest.approx(expected(False), rel=1e-5)
    assert infonce(sim, same, other, .5, own_negative=False).item() == pytest.approx(expected(True), rel=1e-5)
    # Only the angle matters, and anchors on the target side score lower than anchors left at home.
    assert torch.allclose(infonce(3 * sim, same, other, .5), infonce(sim, same, other, .5), atol=1e-6)
    assert infonce(other.expand(1, 2, 2), same, other, .5) < infonce(same[None], same, other, .5)


@pytest.mark.parametrize('bad', [dict(ood_mode='svhn'), dict(amp='fp16'), dict(radius=0.),
                                 dict(swarm_size=0), dict(dev_fraction=1.), dict(detectors=())])
def test_config_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        SteerConfig(**bad).validate()


def test_steering_split_isolation(tmp_path, monkeypatch):
    splits, rows, _, ood_ds = make_manifest(tmp_path, monkeypatch)
    out = tmp_path / 'steer.json'
    ld.prepare_steering(splits, out, seed=3, dev_fraction=.2, n_ood_train=9, n_ood_dev=4)
    steering = json.loads(out.read_text())
    by_role = lambda role: [r for r in steering['rows'] if r['role'] == role]
    assert {r['role'] for r in steering['rows']} == set(ld.STEER_ROLES)
    # ID rows are exactly the direction role, split per class; nothing else from the main manifest.
    id_rows = by_role('id_steer_train') + by_role('id_steer_dev')
    assert {r['id'] for r in id_rows} == {r['id'] for r in rows if r['role'] == 'direction'}
    assert len(id_rows) == 20
    for c in range(2):
        assert sum(r['label'] == c for r in by_role('id_steer_dev')) == 2
    # OOD rows come only from the CIFAR100 train stand-in; every planted duplicate that was read is dropped.
    assert len(by_role('ood_steer_train')) == 9 and len(by_role('ood_steer_dev')) == 4
    assert steering['sources']['ood_steer'] == dict(name='cifar100', root=str(tmp_path), split='train')
    main = {r['pixel_sha256'] for r in rows}
    ood = [r['pixel_sha256'] for r in steering['rows'] if r['source'] == 'ood_steer']
    assert len(set(ood)) == 13 and not set(ood) & main
    assert len(steering['metadata']['dropped_duplicates']) == len(ood_ds.seen) - 13
    assert len({r['id'] for r in steering['rows']}) == len(steering['rows'])
    with pytest.raises(FileExistsError):
        ld.prepare_steering(splits, out)
    with pytest.raises(ValueError, match='too few'):
        ld.prepare_steering(splits, tmp_path / 'other.json', n_ood_train=10, n_ood_dev=4)


def test_views_are_augmented_and_changed_images_rejected():
    from torchvision.transforms import ToTensor
    from steering_ood.vision_data import image_hash
    image = picture(np.random.default_rng(1))
    row = dict(id='x', source='train', index=0, pixel_sha256=image_hash(image))
    view = ld.augment(ToTensor(), (16, 16))
    torch.manual_seed(0)
    a, b, clean = ld.Pool([row], {'train': [(image, 0)]}, (view, view, ToTensor()))[0]
    assert a.shape == b.shape == clean.shape == (3, 16, 16) and not torch.equal(a, b)
    assert torch.equal(clean, ToTensor()(image))
    with pytest.raises(ValueError, match='Source image changed'):
        ld.Pool([dict(row, pixel_sha256='0' * 64)], {'train': [(image, 0)]}, (ToTensor(),))[0]


def run_swarms(seed, vec_chunk):
    """Pooled steering between two separable Gaussian clusters."""
    torch.manual_seed(seed)
    split = SplitModel(*tiny(), 'pooled')
    cfg = SteerConfig(layer='pooled', swarm_size=4, vec_chunk=vec_chunk)
    eye = torch.eye(16)
    ids, ood = 3 * eye[0] + .3 * torch.randn(32, 16), 3 * eye[1] + .3 * torch.randn(32, 16)
    swarms = torch.nn.ModuleDict({n: SteeringSwarm(4, 16, 2.) for n in ld.SWARMS})
    opt = torch.optim.Adam(swarms.parameters(), lr=.1)
    autocast, history = (lambda: torch.autocast('cpu', enabled=False)), []
    for _ in range(60):
        opt.zero_grad()
        history.append(swarm_losses(split, cfg, swarms, ids, ids, ood, ood, autocast, backward=True))
        opt.step()
    return history, {n: s.directions() for n, s in swarms.items()}


def test_swarms_learn_the_separating_direction():
    history, dirs = run_swarms(0, vec_chunk=3)  # 4 vectors in chunks of 3 + 1
    for name in ld.SWARMS:
        assert history[-1][name] < history[0][name]
    centroid = torch.nn.functional.normalize(torch.eye(16)[1] - torch.eye(16)[0], dim=0)
    assert (dirs['id2ood'] @ centroid).mean() > 0 > (dirs['ood2id'] @ centroid).mean()
    again = run_swarms(0, vec_chunk=3)[1]
    whole = run_swarms(0, vec_chunk=4)[1]
    for name in ld.SWARMS:
        assert torch.equal(dirs[name], again[name])                    # same seed, same vectors
        assert torch.allclose(dirs[name], whole[name], atol=1e-4)      # chunking leaves the gradient exact


@pytest.mark.parametrize('mode', ['cifar100', 'random'])
def test_generated_end_to_end(tmp_path, monkeypatch, mode):
    pytest.importorskip('skorch'); pytest.importorskip('pytorch_ood')
    from torchvision.transforms import ToTensor
    from steering_ood.core import ROLES, save_cache
    from steering_ood.head import train_head
    from steering_ood.vision import state_hash
    splits, rows, id_ds, ood_ds = make_manifest(tmp_path, monkeypatch)
    enc, _ = tiny()
    monkeypatch.setattr(ld, 'build_encoder', lambda *a, **k: (enc, ToTensor(), dict(state_sha256=state_hash(enc))))
    # Cache of the same encoder over the same images, as run_pipeline.py would export it.
    with torch.no_grad():
        feats = enc(torch.stack([ToTensor()(image) for image, _ in id_ds])).numpy()
        ood_feats = enc(torch.rand(6, 3, 16, 16)).numpy()
    arrays = {}
    for role in ROLES[:-1]:
        index = [r['index'] for r in rows if r['role'] == role]
        arrays[f'{role}_x'] = feats[index]
        arrays[f'{role}_y'] = np.array([rows[i]['label'] for i in index], dtype='int64')
        arrays[f'{role}_ids'] = np.array([rows[i]['id'] for i in index])
    arrays.update(ood_test_x=ood_feats, ood_test_y=np.full(6, -1), ood_test_ids=np.array([f'ood:{i}' for i in range(6)]),
                  ood_test_groups=np.array(['a'] * 6))
    cache = tmp_path / 'cache.npz'
    save_cache(cache, arrays, dict(modality='vision', actual_encoder_state_sha256=state_hash(enc)))
    train_head(cache, tmp_path / 'head', epochs=1)

    cfg = SteerConfig(layer='layer3.0', ood_mode=mode, swarm_size=2, steps=2, bs_id=4, bs_ood=4, vec_chunk=1,
                      log_every=1, workers=0, n_ood_train=6, n_ood_dev=4, sanity_vectors=2, sanity_probes=4)
    steer_splits = tmp_path / 'steer.json'
    ld.prepare_steering(splits, steer_splits, cfg.seed, cfg.dev_fraction, cfg.n_ood_train, cfg.n_ood_dev)
    ood_ds.seen.clear()
    run = tmp_path / 'run'
    sanity = ld.train_directions(cfg, splits, steer_splits, cache, tmp_path / 'head', run)

    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'completed' and manifest['equivalence']['direct'] <= 1e-4
    assert manifest['augmentation'] is not None and manifest['source_commit'] == ld.SOURCE_COMMIT
    radii = manifest['radii']
    assert radii['dim'] == 8 and radii['positions'] == 64
    assert radii['radius'] == pytest.approx(.25 * radii['activation_norm_median'])
    assert radii['full_map_radius'] == pytest.approx(8 * radii['radius'])
    with np.load(run / 'vectors.npz') as f:
        for name in ('id2ood', 'ood2id', 'random'):
            assert f[name].shape == (2, 8) and np.allclose(np.linalg.norm(f[name], axis=1), 1, atol=1e-5)
        assert f['pca'].shape == f['centroid'].shape == (1, 8)
    log = [json.loads(line) for line in (run / 'train_log.jsonl').read_text().splitlines()]
    assert [r['step'] for r in log] == [1, 2] and all(np.isfinite(v) for r in log for v in r.values())
    assert json.loads((run / 'sanity.json').read_text())['sides'].keys() == {'id', 'ood'}
    for side, learned in (('id', 'id2ood'), ('ood', 'ood2id')):
        families = sanity['sides'][side]
        assert set(families) == {learned, 'random'}
        assert families[learned]['pooled_displacement'] > 0
        for det in cfg.detectors:
            clean = sanity['clean'][side][det]
            assert len(clean['scores']) == 4 and 0 <= clean['rate'] <= 1
            for family in families.values():
                assert 0 <= family[det]['rate'] <= 1 and np.isfinite(family[det]['mean_score'])
                # Conditioning uses the unsteered state of the same probes.
                assert family[det]['n_initially_correct'] == round(4 * (1 - clean['rate']))
    # CIFAR100 steering-train images are read only when they are the OOD side; dev is always scored.
    steering = json.loads(steer_splits.read_text())['rows']
    index = lambda role: {r['index'] for r in steering if r['role'] == role}
    assert index('ood_steer_dev') <= ood_ds.seen
    assert bool(index('ood_steer_train') & ood_ds.seen) == (mode == 'cifar100')


def test_token_steering_and_reduction():
    torch.manual_seed(0)
    h, v = torch.randn(3, 5, 8), torch.randn(2, 8)  # [B, T, D] ViT tokens
    out = steer(h, v)
    assert out.shape == (6, 5, 8) and torch.allclose(out[3:] - h, v[1].expand_as(h), atol=1e-6)
    noise = 5 * torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    assert torch.allclose((offset(h, noise) - h).norm(dim=-1), torch.full((3, 5), 5.), atol=1e-5)
    assert torch.equal(SplitModel.reduce(h), h[:, 0]) and SplitModel.dim(h) == 8 and SplitModel.positions(h) == 5
    assert SplitModel.position_norms(h).shape == (15,)
    assert torch.allclose(SplitModel.position_mean(h), h.reshape(-1, 8).mean(0))


def test_default_layer_follows_the_model():
    assert SteerConfig().layer == 'layer4.0' and SteerConfig(model='resnet50').layer == 'layer4.0'
    assert SteerConfig(model='vit_b16').layer == SteerConfig(model='dinov2_s').layer == 'blocks.9'
    assert SteerConfig(model='vit_b16', layer='pooled').layer == 'pooled'
    with pytest.raises(ValueError, match='model'):
        SteerConfig(model='bert').validate()


@pytest.mark.parametrize('model', list(ld.MODELS))
def test_real_architectures_split_exactly(model):
    """Untrained weights of every supported encoder: the cut model reproduces the encoder output."""
    pytest.importorskip('timm')
    from steering_ood.vision import build_encoder
    backend, name, _ = ld.MODELS[model]
    enc, transform, _ = build_encoder(backend, name, 'none')
    x = transform(picture(np.random.default_rng(2)).resize((32, 32))).unsqueeze(0)
    with torch.no_grad():
        direct = enc(x)
        head = torch.nn.Linear(direct.shape[1], 10).requires_grad_(False)
        for layer in (ld.DEFAULT_LAYERS[model], 'pooled'):
            split = SplitModel(enc, head, layer)
            h = split.prefix(x)
            feat, logits = split.suffix(steer(h, torch.zeros(1, split.dim(h))))
            assert torch.allclose(feat, direct, atol=1e-4, rtol=1e-4)
            assert torch.allclose(logits, head(direct), atol=1e-4, rtol=1e-4)
        split = SplitModel(enc, head, ld.DEFAULT_LAYERS[model])
        later = split.block_names[split.cut]  # the block right after the steering layer
        sim, feat, _ = split.suffix_sim(split.prefix(x), later)
        assert sim.dim() == 2 and sim.shape[0] == 1
        assert torch.allclose(feat, direct, atol=1e-4, rtol=1e-4)
