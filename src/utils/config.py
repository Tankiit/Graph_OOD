from typing import Any, Dict, Optional, Union
from pathlib import Path
import yaml


class Config(dict):
    """Dictionary subclass supporting attribute-style access (e.g. cfg.model.name)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        for k, v in dict(*args, **kwargs).items():
            self[k] = self._wrap(v)

    def _wrap(self, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, Config):
            return Config(value)
        elif isinstance(value, list):
            return [self._wrap(v) for v in value]
        elif isinstance(value, tuple):
            return tuple(self._wrap(v) for v in value)
        return value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        """Convert Config recursively back to standard Python dict."""
        out = {}
        for k, v in self.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [item.to_dict() if isinstance(item, Config) else item for item in v]
            else:
                out[k] = v
        return out


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into base dictionary."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path_or_dict: Union[str, Path, Dict[str, Any]], overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load configuration from a YAML file or dictionary with optional overrides."""
    if isinstance(path_or_dict, (str, Path)):
        p = Path(path_or_dict)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    elif isinstance(path_or_dict, dict):
        raw = path_or_dict.copy()
    else:
        raise ValueError(f"Invalid config source: {type(path_or_dict)}")

    if overrides:
        raw = merge_dicts(raw, overrides)

    return Config(raw)


def save_config(config: Union[Config, Dict[str, Any]], path: Union[str, Path]) -> None:
    """Save configuration to a YAML file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    dict_data = config.to_dict() if isinstance(config, Config) else config
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict_data, f, default_flow_style=False, sort_keys=False)
