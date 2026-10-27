"""Config, paths, seeding, logging."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_logger(name: str = "mc") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def paths(cfg: dict) -> dict:
    exp = cfg["experiment"]
    p = dict(
        raw=ROOT / "data" / "raw",
        cache=ROOT / "data" / "cache" / f"{cfg['img_size']}px",
        splits=ROOT / "results" / exp / "splits",
        ckpt=ROOT / "checkpoints" / exp,
        runs=ROOT / "results" / exp / "runs",
        results=ROOT / "results" / exp,
    )
    for k, v in p.items():
        v.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def device() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available - this pipeline is meant to run on the GPU.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolved_datasets(cfg: dict) -> dict:
    """slot -> dataset name actually used (written by `prepare`)."""
    f = paths(cfg)["cache"] / "resolved_datasets.json"
    if not f.exists():
        raise SystemExit("Run `python run.py prepare` first.")
    return read_json(f)
