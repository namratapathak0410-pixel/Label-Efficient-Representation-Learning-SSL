"""Small helpers: config loading, seeding, device selection, result paths, CSV logging."""
from __future__ import annotations

import logging
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def get_logger(name: str = "ssl") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(name)


def deep_update(base: dict, new: dict) -> dict:
    out = dict(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config. A config may contain `base: other.yaml` (same folder) to inherit values."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_update(load_config(path.parent / base), cfg)
    return cfg


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def get_paths(cfg: dict) -> dict:
    """All output folders of one experiment (created on demand)."""
    name = cfg["experiment_name"]
    res = ROOT / "results" / name
    ck = ROOT / "checkpoints" / name
    d = dict(
        results=res, checkpoints=ck, eda=res / "eda", splits=res / "splits", runs=res / "runs",
        plots=res / "plots", mae=ck / "mae", cache=ROOT / "data" / "cache",
    )
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed: int, deterministic: bool = False) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Use the GPU if there is one, otherwise fall back to the CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hardware_summary() -> dict:
    import psutil
    import torch

    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


def append_row_csv(path: str | Path, row: dict, key: str = "run_id") -> None:
    """Append a result row; if a row with the same `key` exists it is replaced (re-runs)."""
    path = Path(path)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        if key in old.columns and key in row:
            old = old[old[key] != row[key]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
