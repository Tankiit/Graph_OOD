"""Learned directions for text: split exactness on the real encoders, token masks, CLINC steering splits
and a generated end-to-end run. Needs the cached Hugging Face snapshots; models missing locally are skipped.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('sentence_transformers')
from steering_nlp import learned_directions as nl
from steering_nlp.learned_directions import (TextSplitModel, TextSteerConfig, Tokens, cat_tokens, offset,
                                             position_mean, position_norms, steer)

QUERIES = ['what is my balance', 'please book a table for four at the italian place tonight', 'hi']


def encoder(model):
    try:
        return nl.load_encoder(model)[0]
    except Exception as exc:  # snapshot not cached and no network
        pytest.skip(f'{model} unavailable: {exc!r}')


def features(st, texts=QUERIES):
    return nl.tokenizer_collate(st)(texts)


@pytest.mark.parametrize('model', list(nl.TEXT_MODELS))
def test_real_encoders_split_exactly(model):
    """Zero offset at the default layer and at pooled reproduces SentenceTransformer, padding included."""
    st = encoder(model)
    f = features(st)
    with torch.no_grad():
        direct = st(dict(f))['sentence_embedding']
        head = torch.nn.Linear(direct.shape[1], 150).requires_grad_(False)
        for layer in (nl.TEXT_DEFAULT_LAYERS[model], 'pooled'):
            split = TextSplitModel(st, head, layer)
            h = split.prefix(f)
            feat, logits = split.suffix(steer(h, torch.zeros(1, nl.dim(h))))
            assert torch.allclose(feat, direct, atol=1e-5)
            assert torch.allclose(logits, head(direct), atol=1e-4)
        split = TextSplitModel(st, head, nl.TEXT_DEFAULT_LAYERS[model])
        later = split.block_names[split.cut]
        sim, feat, _ = split.suffix_sim(split.prefix(f), later)
        assert sim.shape == (len(QUERIES), nl.dim(split.prefix(f))) and torch.allclose(feat, direct, atol=1e-5)
    assert torch.allclose(direct.norm(dim=1), torch.ones(len(QUERIES)), atol=1e-5)  # unit-norm encoders


def test_layers_and_defaults():
    st = encoder('minilm')
    split = TextSplitModel(st, torch.nn.Linear(384, 2), 'blocks.4')
    assert split.layers == [f'blocks.{i}' for i in range(6)] + ['pooled']
    with pytest.raises(ValueError, match='upstream'):
        split.check_sim_layer('blocks.2')
    with pytest.raises(ValueError, match='unknown layer'):
        TextSplitModel(st, torch.nn.Linear(384, 2), 'layer3.0')
    assert TextSteerConfig().layer == 'blocks.9' and TextSteerConfig(model='minilm').layer == 'blocks.4'
    assert TextSteerConfig(model='bge_large').layer == 'blocks.19'
    TextSteerConfig(model='minilm', ood_mode='random').validate()
    for bad in (dict(model='resnet18'), dict(ood_mode='cifar100'), dict(radius=0.)):
        with pytest.raises(ValueError):
            TextSteerConfig(**bad).validate()


def test_token_masks_ignore_padding():
    torch.manual_seed(0)
    h = torch.randn(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    tokens = Tokens(h, mask)
    assert torch.allclose(position_norms(tokens), torch.cat([h[0].norm(dim=-1), h[1, :2].norm(dim=-1)]), atol=1e-6)
    assert torch.allclose(position_mean(tokens), torch.cat([h[0], h[1, :2]]).mean(0), atol=1e-6)
    v = torch.randn(2, 3)
    out = steer(tokens, v)
    assert out.h.shape == (4, 4, 3) and torch.equal(out.mask, mask.repeat(2, 1))
    assert torch.allclose(out.h[2:] - h, v[1].expand_as(h), atol=1e-6)
    noise = 5 * torch.nn.functional.normalize(torch.randn(2, 3), dim=1)
    moved = Tokens(offset(tokens, noise).h - h, mask)  # displacement of every real token
    assert torch.allclose(position_norms(moved), torch.full((6,), 5.), atol=1e-5)
    padded = cat_tokens([tokens, Tokens(torch.randn(1, 6, 3), torch.ones(1, 6, dtype=torch.long))])
    assert padded.h.shape == (3, 6, 3) and padded.mask[:2, 4:].sum() == 0 and padded.h[:2, 4:].abs().sum() == 0


def test_dropout_views_differ_and_eval_is_deterministic():
    from steering_ood.vision import state_hash
    st = encoder('minilm')
    split = TextSplitModel(st, torch.nn.Linear(384, 2), 'blocks.4')
    f, before = features(st), state_hash(st)
    with torch.no_grad():
        clean = [split.prefix(f).h for _ in range(2)]
        with split.dropout():
            torch.manual_seed(0)
            a, b = split.prefix(f).h, split.prefix(f).h
    assert torch.equal(*clean) and not torch.allclose(a, b)
    assert not split.hf.training and state_hash(st) == before


def write_clinc(tmp_path, n_intents=2, per_intent=10):
    """A CLINC-shaped main manifest and raw file over generated queries."""
    rng = np.random.default_rng(3)
    words = ['balance', 'transfer', 'weather', 'alarm', 'flight', 'hotel', 'pizza', 'music', 'card', 'rate']
    query = lambda c, i: f"{words[c]} please {' '.join(rng.choice(words, 3))} number {i}"
    rows = []
    for c in range(n_intents):
        for role, n in (('head', per_intent), ('reference', per_intent), ('direction', per_intent),
                        ('calibration', per_intent), ('probe', 4), ('id_test', 4)):
            for _ in range(n):
                rows.append(dict(id=f'clinc:fake:{len(rows)}', text=query(c, len(rows)), label=c, role=role))
    for i in range(6):
        rows.append(dict(id=f'clinc:oos_test:{i}', text=f'tell me a joke about cats {i}', label=-1, role='ood_test'))
    splits, raw = tmp_path / 'clinc_splits.json', tmp_path / 'clinc_data_full.json'
    splits.write_text(json.dumps(dict(metadata={}, rows=rows)))
    oos = lambda name, n: [[f'{name} out of scope question number {i}', 'oos'] for i in range(n)]
    raw.write_text(json.dumps(dict(oos_train=oos('train', 10) + [[rows[0]['text'], 'oos']],
                                   oos_val=oos('val', 6) + [['train out of scope question number 0', 'oos']],
                                   oos_test=oos('test', 6))))
    return splits, raw, rows


def test_text_steering_split_isolation(tmp_path):
    splits, raw, rows = write_clinc(tmp_path)
    out = tmp_path / 'steer.json'
    nl.prepare_steering_text(splits, raw, out, seed=3, dev_fraction=.2)
    steering = json.loads(out.read_text())
    by_role = lambda role: [r for r in steering['rows'] if r['role'] == role]
    id_rows = by_role('id_steer_train') + by_role('id_steer_dev')
    assert {r['id'] for r in id_rows} == {r['id'] for r in rows if r['role'] == 'direction'}
    assert sum(r['label'] == 0 for r in by_role('id_steer_dev')) == 2
    # OOD only from oos_train / oos_val; the copy of an ID query and the repeated query are dropped.
    assert {r['id'].split(':')[1] for r in by_role('ood_steer_train')} == {'oos_train'}
    assert {r['id'].split(':')[1] for r in by_role('ood_steer_dev')} == {'oos_val'}
    assert len(by_role('ood_steer_train')) == 10 and len(by_role('ood_steer_dev')) == 6
    assert len(steering['metadata']['dropped_duplicates']) == 2
    assert not any('oos_test' in r['id'] for r in steering['rows'])
    with pytest.raises(FileExistsError):
        nl.prepare_steering_text(splits, raw, out)


@pytest.mark.parametrize('mode', ['clinc', 'random'])
def test_generated_end_to_end(tmp_path, mode):
    pytest.importorskip('skorch'); pytest.importorskip('pytorch_ood')
    from steering_ood.core import ROLES, save_cache
    from steering_ood.head import train_head
    st = encoder('minilm')
    splits, raw, rows = write_clinc(tmp_path)
    arrays = {}
    for role in ROLES:
        part = [r for r in rows if r['role'] == role]
        arrays[f'{role}_x'] = st.encode([r['text'] for r in part], convert_to_numpy=True).astype('float32')
        arrays[f'{role}_y'] = np.array([r['label'] for r in part], dtype='int64')
        arrays[f'{role}_ids'] = np.array([r['id'] for r in part])
    cache = tmp_path / 'minilm.npz'
    save_cache(cache, arrays, dict(modality='text', encoder=nl.TEXT_MODELS['minilm']))
    train_head(cache, tmp_path / 'head', epochs=1)
    cfg = TextSteerConfig(model='minilm', ood_mode=mode, swarm_size=2, steps=2, bs_id=4, bs_ood=4, vec_chunk=1,
                          log_every=1, sanity_vectors=2, sanity_probes=4)
    steer_splits = tmp_path / 'steer.json'
    nl.prepare_steering_text(splits, raw, steer_splits, cfg.seed, cfg.dev_fraction)
    run = tmp_path / 'run'
    sanity = nl.train_text_directions(cfg, splits, steer_splits, cache, tmp_path / 'head', run)
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'completed' and manifest['equivalence']['direct'] <= 1e-4
    assert manifest['radii']['dim'] == 384 and manifest['augmentation'] is not None
    with np.load(run / 'vectors.npz') as f:
        for name in ('id2ood', 'ood2id', 'random'):
            assert f[name].shape == (2, 384) and np.allclose(np.linalg.norm(f[name], axis=1), 1, atol=1e-5)
        assert f['pca'].shape == f['centroid'].shape == (1, 384)
    log = [json.loads(line) for line in (run / 'train_log.jsonl').read_text().splitlines()]
    assert [r['step'] for r in log] == [1, 2] and all(np.isfinite(v) for r in log for v in r.values())
    for side, learned in (('id', 'id2ood'), ('ood', 'ood2id')):
        assert set(sanity['sides'][side]) == {learned, 'random'}
        for det in cfg.detectors:
            assert len(sanity['clean'][side][det]['scores']) == 4
