"""Configuration loading (YAML + command-line overrides)."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Config(dict):
    """dict with attribute access: cfg.train.epochs"""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _parse_value(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> None:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, Config())
        node[parts[-1]] = _wrap(_parse_value(value))


def load_config(path: str | os.PathLike | None = None, overrides: Iterable[str] = ()) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = _wrap(copy.deepcopy(raw))
    apply_overrides(cfg, overrides)
    env_data = os.environ.get("QASR_DATA_DIR")
    if env_data:
        cfg.paths.data_dir = env_data
    return cfg


def resolve(path_like: str | os.PathLike) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path_like)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def data_dir(cfg: Config) -> Path:
    return resolve(cfg.paths.data_dir)


def checkpoints_dir(cfg: Config) -> Path:
    d = resolve(cfg.paths.checkpoints_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir(cfg: Config) -> Path:
    d = resolve(cfg.paths.results_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d
