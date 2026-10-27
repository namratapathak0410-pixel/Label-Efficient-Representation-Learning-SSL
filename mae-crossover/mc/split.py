"""Leakage-safe train/val/test split and nested label budgets.

1. Exact and near-duplicate images (Hamming distance <= threshold between 256-bit dHashes) are joined into groups
   (union-find). A group never crosses train/val/test.
2. Groups are split stratified by class (StratifiedGroupKFold): one fold = test, one fold = val, rest = train.
3. Label budgets: for each seed, every class's train images are shuffled once; budget b% takes the first
   max(1, round(b% * n_class)) of them. Subsets are therefore nested (5% inside 10% inside ...) and identical for
   every model trained with that seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data import load_cache
from .utils import get_logger, paths, resolved_datasets, write_json

log = get_logger()


def duplicate_groups(hashes: np.ndarray, threshold: int, chunk: int = 512) -> np.ndarray:
    n = len(hashes)
    words = np.ascontiguousarray(hashes).view(np.uint64)  # (n, 4)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for s in range(0, n, chunk):
        a = words[s:s + chunk]
        dist = np.bitwise_count(a[:, None, :] ^ words[None, :, :]).sum(-1)
        ii, jj = np.nonzero(dist <= threshold)
        for i, j in zip(ii + s, jj):
            if j > i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def make_split(name: str, cfg: dict) -> pd.DataFrame:
    p = paths(cfg)
    out = p["splits"] / f"{name}_split.csv"
    if out.exists():
        return pd.read_csv(out)
    sc = cfg["split"]
    c = load_cache(name, cfg)
    labels = c["labels"]
    groups = duplicate_groups(c["dhash"], sc["dhash_threshold"])
    k = round(1 / sc["test_frac"])
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=sc["split_seed"])
    folds = list(sgkf.split(np.zeros(len(labels)), labels, groups))
    split = np.full(len(labels), "train", dtype=object)
    split[folds[0][1]] = "test"
    split[folds[1][1]] = "val"
    df = pd.DataFrame(dict(idx=np.arange(len(labels)), label=labels, group=groups, split=split, id=c["ids"]))
    # integrity checks
    spans = df.groupby("group")["split"].nunique()
    assert (spans == 1).all(), "a duplicate group crosses splits"
    df.to_csv(out, index=False)
    sizes = df.groupby("group").size()
    write_json(p["splits"] / f"{name}_split_report.json", dict(
        dataset=name, n=len(df), counts=df["split"].value_counts().to_dict(),
        n_groups=int(len(sizes)), n_duplicate_groups=int((sizes > 1).sum()),
        n_images_in_duplicate_groups=int(sizes[sizes > 1].sum()), groups_spanning_splits=int((spans > 1).sum()),
        dhash_threshold=sc["dhash_threshold"],
        min_train_per_class=int(df[df.split == "train"].groupby("label").size().min()),
        min_test_per_class=int(df[df.split == "test"].groupby("label").size().min())))
    log.info("[%s] split: %s, %d duplicate groups", name, df["split"].value_counts().to_dict(), (sizes > 1).sum())
    return df


def budget_indices(df: pd.DataFrame, budget_pct: float, seed: int) -> np.ndarray:
    """Cache indices of the labelled training subset for this budget and seed (nested across budgets)."""
    tr = df[df["split"] == "train"]
    rng = np.random.default_rng(10_000 + seed)
    out = []
    for lab in sorted(tr["label"].unique()):
        ids = tr.loc[tr["label"] == lab, "idx"].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        out.append(ids[:max(1, int(round(budget_pct / 100 * len(ids))))])
    return np.sort(np.concatenate(out))


def split_all(cfg: dict):
    for name in resolved_datasets(cfg).values():
        df = make_split(name, cfg)
        tr = df[df.split == "train"]
        rows = []
        for s in cfg["train"]["seeds"]:
            prev = set()
            for b in sorted(cfg["train"]["budgets"]):
                ids = budget_indices(df, b, s)
                assert prev <= set(ids), "budgets are not nested"
                prev = set(ids)
                per_class = tr[tr.idx.isin(ids)].groupby("label").size()
                rows.append(dict(seed=s, budget_pct=b, n_labeled=len(ids), min_per_class=int(per_class.min()),
                                 median_per_class=float(per_class.median())))
        pd.DataFrame(rows).to_csv(paths(cfg)["splits"] / f"{name}_budgets.csv", index=False)
