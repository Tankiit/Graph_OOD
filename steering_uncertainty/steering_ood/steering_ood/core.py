"""Scientific contracts, safe cache IO and reproducible output helpers."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np

ROLES = ("head", "reference", "direction", "calibration", "probe", "id_test", "ood_test")


def matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite feature matrix")
    return x


def threshold(scores, tau=0.05):
    scores = np.asarray(scores, dtype=float)
    if not 0 < tau < 1 or scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration scores or target rejection rate")
    rank = math.ceil((len(scores) + 1) * (1 - tau))
    return float(np.sort(scores)[rank - 1]) if rank <= len(scores) else float("inf")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jsonable(data), indent=2, allow_nan=False) + "\n")


def versions():
    result = {}
    for name in ("steering-ood", "numpy", "scipy", "scikit-learn", "torch", "skorch", "pytorch-ood", "sentence-transformers", "torchvision", "timm", "pillow"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def new_output(path):
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {p}; use a new run name")
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_cache(path):
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    seen, dim = set(), None
    for role in ROLES:
        x = matrix(data[f"{role}_x"])
        y, ids = data[f"{role}_y"], data[f"{role}_ids"]
        if y.shape != (len(x),) or ids.shape != y.shape or y.dtype.kind not in "iu":
            raise ValueError(f"Invalid labels/IDs for {role}")
        if dim is not None and x.shape[1] != dim:
            raise ValueError("Feature dimensions differ")
        dim = x.shape[1]
        current = set(ids.tolist())
        if len(current) != len(ids) or current & seen:
            raise ValueError("Split IDs overlap or repeat")
        seen |= current
        if role == "ood_test":
            if not np.all(y == -1):
                raise ValueError("OOD labels must be -1")
        elif np.any(y < 0):
            raise ValueError(f"OOD samples found in {role}")
    if "ood_test_groups" in data:
        groups=data["ood_test_groups"]
        if groups.shape != data["ood_test_y"].shape or groups.dtype.kind not in "US":
            raise ValueError("OOD group names must be one string per OOD example")
    classes = np.unique(data["head_y"])
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError("Head labels must be contiguous starting at zero")
    for role in ROLES[:-1]:
        if not set(data[f"{role}_y"]) <= set(classes):
            raise ValueError(f"Unknown ID labels in {role}")
    data["metadata"] = json.loads(str(data.pop("metadata_json")))
    return data


def save_cache(path, arrays, metadata):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, metadata_json=np.array(json.dumps(metadata)))
    load_cache(path)  # Validate before exposing an unusable cache.


def bootstrap_indices(y, rng):
    """Stratified bootstrap preserves every fitted class and its sample count."""
    return np.concatenate([rng.choice(idx, len(idx), replace=True)
                           for c in np.unique(y) for idx in [np.flatnonzero(y == c)]])


def variance_components(a):
    """Balanced empirical product-measure ANOVA; no pseudo-replicate CIs.

    a has axes reference, direction, probe. Decompose separately per probe,
    then average. Interaction is explicit and all components use ddof=0.
    """
    a = np.asarray(a, float)
    if a.ndim != 3 or not np.isfinite(a).all():
        raise ValueError("ANOVA requires a complete finite crossed array")
    grand = a.mean(axis=(0, 1), keepdims=True)
    ref = a.mean(axis=1, keepdims=True) - grand
    direction = a.mean(axis=0, keepdims=True) - grand
    interaction = a - grand - ref - direction
    return {"reference": float(np.mean(ref**2)),
            "direction": float(np.mean(direction**2)),
            "interaction": float(np.mean(interaction**2)),
            "total": float(np.mean((a-grand)**2))}


def source_hashes():
    root = Path(__file__).parent
    return {p.name:file_hash(p) for p in sorted(root.glob("*.py"))}
