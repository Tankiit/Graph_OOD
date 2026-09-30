"""CLINC150 partitions and a download-free synthetic integration fixture."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .core import ROLES, file_hash, save_cache, write_json


def prepare_clinc(source, output, seed=20260929):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    raw = json.loads(source.read_text())
    classes = sorted({label for _, label in raw["train"] if label != "oos"})
    if len(classes) != 150:
        raise ValueError("Use the official CLINC data_full.json with 150 ID classes")
    labels = {name: i for i, name in enumerate(classes)}
    rng, rows, text_roles = np.random.default_rng(seed), [], {}
    canonicalize = lambda t: " ".join(t.lower().split())
    counts = Counter(canonicalize(t) for split in ("train", "val", "test", "oos_test")
                     for t, _ in raw[split])
    duplicated = {t for t, count in counts.items() if count > 1}
    dropped = []

    def append(split, i, role):
        text, label = raw[split][i]
        # Exact duplicate text across roles compromises the split contract.
        canonical = canonicalize(text)
        if canonical in duplicated:
            dropped.append({"id": f"clinc:{split}:{i}", "assigned_role": role,
                            "reason": "duplicate normalized text; all copies excluded"})
            return
        text_roles[canonical] = role
        rows.append({"id": f"clinc:{split}:{i}", "text": text,
                     "label": -1 if role == "ood_test" else labels[label], "role": role})

    for label in classes:
        for split, counts in (("train", (("head", 60), ("reference", 20), ("direction", 20))),
                              ("val", (("calibration", 10), ("probe", 10))),
                              ("test", (("id_test", 30),))):
            idx = np.array([i for i, (_, y) in enumerate(raw[split]) if y == label])
            if len(idx) != sum(n for _, n in counts):
                raise ValueError(f"Unexpected count for {label} in {split}")
            rng.shuffle(idx)
            offset = 0
            for role, n in counts:
                for i in idx[offset:offset+n]:
                    append(split, int(i), role)
                offset += n
    for i in range(len(raw["oos_test"])):
        append("oos_test", i, "ood_test")
    write_json(output, {"metadata": {"dataset": "CLINC150 data_full.json", "seed": seed,
                                     "source_sha256": file_hash(source), "classes": classes,
                                     "oos_train_and_val": "excluded", "normalization": "none",
                                     "duplicate_policy": "drop all copies across used splits after role assignment",
                                     "dropped_duplicates": dropped,
                                     "role_counts": dict(Counter(r["role"] for r in rows)),
                                     "evaluation_variant": "deduplicated CLINC150; report modified test counts"},
                        "rows": rows})


def encode(split_file, output, model="sentence-transformers/all-mpnet-base-v2",
           revision=None, batch_size=64, device="cpu"):
    from sentence_transformers import SentenceTransformer
    split = json.loads(Path(split_file).read_text())
    encoder = SentenceTransformer(model, revision=revision, device=device)
    arrays = {}
    for role in ROLES:
        rows = [r for r in split["rows"] if r["role"] == role]
        arrays[f"{role}_x"] = encoder.encode([r["text"] for r in rows], batch_size=batch_size,
                                             convert_to_numpy=True, normalize_embeddings=False,
                                             show_progress_bar=True).astype("float32")
        arrays[f"{role}_y"] = np.array([r["label"] for r in rows], dtype="int64")
        arrays[f"{role}_ids"] = np.array([r["id"] for r in rows])
    metadata = dict(split["metadata"], modality="text", encoder=model, requested_revision=revision,
                    split_sha256=file_hash(split_file), fixture=False,
                    encoder_modules=str(encoder),
                    resolved_revision=getattr(getattr(getattr(encoder[0], "auto_model", None), "config", None), "_commit_hash", None))
    save_cache(output, arrays, metadata)


def synthetic_cache(output, seed=7, dim=6, classes=3):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(classes, dim)) * 1.5
    scale = np.ones(dim); scale[0] = 2
    arrays = {}
    sizes = dict(head=40, reference=24, direction=24, calibration=30, probe=8, id_test=30)
    for role, per_class in sizes.items():
        y = np.repeat(np.arange(classes), per_class)
        arrays[f"{role}_x"] = (means[y] + rng.normal(size=(len(y), dim))*scale).astype("float32")
        arrays[f"{role}_y"] = y
        arrays[f"{role}_ids"] = np.array([f"synthetic:{role}:{i}" for i in range(len(y))])
    arrays["ood_test_x"] = (rng.normal(size=(100, dim))*scale + 7).astype("float32")
    arrays["ood_test_y"] = np.full(100, -1, dtype="int64")
    arrays["ood_test_ids"] = np.array([f"synthetic:ood:{i}" for i in range(100)])
    save_cache(output, arrays, {"fixture": True, "seed": seed, "encoder": "synthetic features",
                               "normalization": "none", "world": "Gaussian class mixture"})


def project_cache(source, output, components=64):
    """Optional fixed ID-only PCA; fit on head training split, never on C/test."""
    from sklearn.decomposition import PCA
    from .core import load_cache, ROLES
    data = load_cache(source)
    if not 1 <= components <= min(data['head_x'].shape):
        raise ValueError('Invalid PCA dimension')
    pca = PCA(n_components=components, svd_solver='full', whiten=False).fit(data['head_x'])
    arrays = {k:v for k,v in data.items() if k!='metadata'}
    for role in ROLES:
        arrays[f'{role}_x'] = pca.transform(data[f'{role}_x']).astype('float32')
    arrays['projection_components'] = pca.components_
    arrays['projection_mean'] = pca.mean_
    metadata = dict(data['metadata'], parent_cache_sha256=file_hash(source),
                    projection=dict(kind='PCA', fit_role='head', components=components, whiten=False,
                                    explained_variance_ratio=pca.explained_variance_ratio_.tolist()),
                    path_units='Euclidean distance in fixed projected coordinates')
    save_cache(output,arrays,metadata)


def import_features(source, metadata_file, output):
    """Import a modality-agnostic NPZ with explicit split IDs and provenance."""
    with np.load(source,allow_pickle=False) as f:
        arrays={k:f[k] for k in f.files if k!='metadata_json'}
    metadata=json.loads(Path(metadata_file).read_text())
    required={'modality','encoder','split_provenance'}
    if not required<=set(metadata):
        raise ValueError(f'Metadata must include {sorted(required)}')
    metadata['imported_source_sha256']=file_hash(source)
    metadata['external_split_audit']='Caller-provided IDs checked for overlap; semantic/content duplicates not inferred from embeddings'
    save_cache(output,arrays,metadata)
