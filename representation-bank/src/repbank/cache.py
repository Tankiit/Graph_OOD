from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr


@dataclass
class CacheRow:
    h_last: np.ndarray       # [L+1, d]
    h_span: np.ndarray       # [3, S, d], zero padded
    span_mask: np.ndarray    # [S]
    logprobs: np.ndarray     # [S], NaN padded
    pair_id: str
    role: str
    label: float
    label_protocol: str
    prompt: str
    generation: str


class RepresentationBank:
    """Append-only Zarr v2 bank; rows remain flat and paired through pair_id."""

    def __init__(self, path: str | Path, *, n_layers: int, d_model: int,
                 span_cap: int, depth_fractions: list[float], model_id: str,
                 chunk_rows: int = 16, overwrite: bool = False):
        mode = "w" if overwrite else "a"
        self.root = zarr.open_group(str(path), mode=mode)
        attrs = self.root.attrs
        expected = {"schema_version": 1, "model_id": model_id, "n_layers": n_layers,
                    "d_model": d_model, "span_cap": span_cap,
                    "depth_fractions": depth_fractions,
                    "residual_definition": "embedding output + transformer block outputs"}
        if attrs and dict(attrs).get("model_id") != model_id:
            raise ValueError(f"cache belongs to {attrs.get('model_id')}, not {model_id}")
        attrs.update(expected)
        shapes = {
            "h_last": ((0, n_layers + 1, d_model), (chunk_rows, 1, d_model), "f2", 0),
            "h_span": ((0, len(depth_fractions), span_cap, d_model), (1, 1, span_cap, d_model), "f2", 0),
            "span_mask": ((0, span_cap), (chunk_rows, span_cap), "b1", False),
            "logprobs": ((0, span_cap), (chunk_rows, span_cap), "f4", np.nan),
        }
        for name, (shape, chunks, dtype, fill) in shapes.items():
            if name not in self.root:
                self.root.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype, fill_value=fill)
        self.meta_path = Path(path) / "rows.jsonl"
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: CacheRow) -> int:
        index = self.root["h_last"].shape[0]
        for name in ("h_last", "h_span", "span_mask", "logprobs"):
            array = self.root[name]
            array.resize(index + 1, *array.shape[1:])
            array[index] = getattr(row, name)
        meta = {k: getattr(row, k) for k in
                ("pair_id", "role", "label", "label_protocol", "prompt", "generation")}
        with self.meta_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        return index

