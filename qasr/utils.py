"""Small shared helpers: logging, seeding, device selection, timing."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "qasr") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt="%H:%M:%S",
                            stream=sys.stdout)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device(pref: str = "auto"):
    import torch

    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("config asks for CUDA but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_cuda_dll_dirs() -> None:
    """On Windows, let onnxruntime-gpu / ctranslate2 find the CUDA + cuDNN DLLs
    that ship inside the torch wheel (torch/lib)."""
    if os.name != "nt":
        return
    try:
        import torch

        lib = Path(torch.__file__).parent / "lib"
        if lib.is_dir():
            os.add_dll_directory(str(lib))
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception:  # pragma: no cover - best effort
        pass


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class Timer:
    """Accumulates wall-clock timings per named stage (milliseconds)."""

    def __init__(self) -> None:
        self.records: dict[str, list[float]] = {}

    @contextmanager
    def __call__(self, name: str, sync_cuda: bool = False):
        if sync_cuda:
            _cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync_cuda:
                _cuda_sync()
            self.records.setdefault(name, []).append((time.perf_counter() - t0) * 1000.0)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for k, v in self.records.items():
            a = np.asarray(v)
            out[k] = {"n": int(a.size), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
                      "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}
        return out


def _cuda_sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
