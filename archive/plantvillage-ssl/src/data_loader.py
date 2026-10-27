"""Dataset scanning, split loading, label-budget subsets and DataLoaders."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .preprocessing import eval_transform, train_transform
from .utils import get_paths, resolve

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def scan_imagefolder(root) -> tuple[pd.DataFrame, list[str]]:
    """Scan <root>/<class>/<image>. Returns a DataFrame (relpath, class_name, label) and the class list."""
    root = resolve(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}\n"
            "Run:  python scripts/prepare_dataset.py --dataset plantvillage   (or --dataset synthetic to test)"
        )
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if {"color", "grayscale"} <= set(classes):
        raise RuntimeError(
            f"{root} contains the sub-folders color/grayscale/segmented. Point data.root to the 'color' folder "
            "(or re-run scripts/prepare_dataset.py which does this for you)."
        )
    rows = []
    for ci, c in enumerate(classes):
        for p in sorted((root / c).rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                rows.append((p.relative_to(root).as_posix(), c, ci))
    df = pd.DataFrame(rows, columns=["relpath", "class_name", "label"])
    if df.empty:
        raise RuntimeError(f"No images found under {root}")
    return df, classes


class ImageDataset(Dataset):
    """Reads images from disk (or from an in-RAM cache of pre-resized copies)."""

    def __init__(self, df: pd.DataFrame, root, transform, cache_size: int | None = None, with_labels: bool = True):
        self.root = resolve(root)
        self.paths = [self.root / p for p in df["relpath"]]
        self.labels = df["label"].astype(int).values if with_labels else None
        self.transform = transform
        self.cache = None
        if cache_size:
            self.cache = [self._load(p, cache_size) for p in tqdm(self.paths, desc="caching images", leave=False)]

    @staticmethod
    def _load(path, size=None):
        with Image.open(path) as im:
            im = im.convert("RGB")
            if size:
                im = im.resize((size, size), Image.BICUBIC)
            else:
                im.load()
            return im.copy()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.cache[i] if self.cache is not None else self._load(self.paths[i])
        x = self.transform(img)
        return (x, int(self.labels[i])) if self.labels is not None else x


def split_csv_path(cfg) -> Path:
    return get_paths(cfg)["splits"] / "split.csv"


def load_split(cfg) -> pd.DataFrame:
    p = split_csv_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run:  python main.py split --config <config>")
    return pd.read_csv(p)


def class_names_from_split(df: pd.DataFrame) -> list[str]:
    return df.drop_duplicates("label").sort_values("label")["class_name"].tolist()


def budget_subset(train_df: pd.DataFrame, pct: float, seed: int) -> pd.DataFrame:
    """Stratified label-budget subset of the TRAIN split.

    * at least 1 image per class,
    * deterministic given (pct, seed) -> every model sees exactly the same labelled images,
    * nested: the 5 % subset is contained in the 10 % subset of the same seed.
    """
    if pct >= 100:
        return train_df.copy()
    parts = []
    for lab, g in train_df.groupby("label"):
        rng = np.random.RandomState(seed * 100003 + int(lab))
        order = rng.permutation(len(g))
        k = max(1, int(math.ceil(pct / 100.0 * len(g))))
        parts.append(g.iloc[order[:k]])
    return pd.concat(parts).sort_index()


def make_loader(df, cfg, train: bool, with_labels: bool = True, batch_size: int | None = None, transform=None,
                drop_last: bool = False, shuffle: bool | None = None) -> DataLoader:
    d = cfg["data"]
    size = d["img_size"]
    tf = transform or (train_transform(size) if train else eval_transform(size))
    ds = ImageDataset(df, d["root"], tf, cache_size=d.get("cache_size") if d.get("cache_in_ram") else None,
                      with_labels=with_labels)
    bs = batch_size or cfg["train"]["batch_size"]
    nw = d.get("num_workers", 0)
    import torch

    return DataLoader(ds, batch_size=bs, shuffle=train if shuffle is None else shuffle, num_workers=nw,
                      drop_last=drop_last, pin_memory=torch.cuda.is_available(),
                      persistent_workers=nw > 0)
