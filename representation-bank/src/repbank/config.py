from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    hf_id: str
    tinker_id: str | None = None
    base_model: bool = False
    trust_remote_code: bool = True
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    device_map: str = "auto"


class CaptureConfig(BaseModel):
    depth_fractions: list[float] = Field(default_factory=lambda: [0.2, 0.5, 0.8])
    span_cap: int = 32
    last_token_all_layers: bool = True
    chunk_rows: int = 16

    def model_layer_indices(self, n_layers: int) -> list[int]:
        # Fractions refer to transformer blocks, indexed 0..n_layers-1.
        return [min(n_layers - 1, max(0, round(f * (n_layers - 1)))) for f in self.depth_fractions]


class GenerationConfig(BaseModel):
    max_new_tokens: int = 200
    temperature: float = 0.0
    do_sample: bool = False


class CacheConfig(BaseModel):
    path: Path
    overwrite: bool = False


class RunConfig(BaseModel):
    model: ModelConfig
    capture: CaptureConfig = CaptureConfig()
    generation: GenerationConfig = GenerationConfig()
    cache: CacheConfig

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

