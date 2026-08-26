from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr


def register_direction(cache_path: str, name: str, direction_path: str,
                       quantiles=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> dict:
    root = zarr.open_group(cache_path, mode="r")
    v = np.load(direction_path).astype(np.float32)
    v /= np.linalg.norm(v)
    coords = np.einsum("nld,d->nl", root["h_last"][:].astype(np.float32), v)
    result = {"direction": name, "quantiles": list(quantiles),
              "per_hidden_state": np.quantile(coords, quantiles, axis=0).T.tolist()}
    out = Path(cache_path) / "directions"
    out.mkdir(exist_ok=True)
    (out / f"{name}.quantiles.json").write_text(json.dumps(result))
    return result

