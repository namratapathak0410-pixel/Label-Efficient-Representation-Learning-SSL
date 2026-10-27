"""Image index (corruption check + hashes), duplicate grouping and the leakage-free train/val/test split.

Why group-aware splitting?
PlantVillage contains several photos / augmented copies of the SAME leaf. If one copy lands in train
and another in test, the test accuracy is inflated (data leakage). We therefore
  1. compute an MD5 of every file (exact duplicates) and a 256-bit "difference hash" (near duplicates),
  2. put all duplicates / near-duplicates into one GROUP (union-find),
  3. split GROUPS (stratified by class) so a group never straddles train / val / test.
The split is created ONCE, before any pretraining or model selection, and is saved to results/<exp>/splits/split.csv.
"""
from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .data_loader import scan_imagefolder
from .utils import get_logger, get_paths, resolve

log = get_logger()
MAX_BUCKET = 3000  # buckets bigger than this (e.g. hundreds of identical blank images) are skipped in the near-dup search


# ----------------------------------------------------------------------------- hashing
def _hash_one(args):
    root, rel = args
    try:
        data = (root / rel).read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
            im.draft("L", (128, 128))  # fast reduced-size JPEG decoding (no-op for PNG)
            g = im.convert("L").resize((17, 16), Image.BILINEAR)  # forces a real decode -> detects truncated files
        a = np.asarray(g, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()  # 16 x 16 = 256 bits (difference hash)
        return dict(relpath=rel, ok=True, md5=md5, hash=np.packbits(bits).tobytes().hex(), w=w, h=h, error="")
    except Exception as e:  # corrupted / unreadable file
        return dict(relpath=rel, ok=False, md5="", hash="", w=0, h=0, error=str(e)[:200])


def build_index(cfg, force: bool = False) -> pd.DataFrame:
    """Scan the dataset once; cache result (so EDA and split share it)."""
    paths = get_paths(cfg)
    root = resolve(cfg["data"]["root"])
    cache = paths["cache"] / f"index_{cfg['data']['name']}.csv"
    df, _ = scan_imagefolder(root)
    if cache.exists() and not force:
        old = pd.read_csv(cache, keep_default_na=False)
        if len(old) == len(df) and set(old["relpath"]) == set(df["relpath"]):
            old["ok"] = old["ok"].astype(str).str.lower().eq("true")
            return old
    log.info("Indexing %d images (corruption check + hashing); this is done once ...", len(df))
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(tqdm(ex.map(_hash_one, [(root, r) for r in df["relpath"]]), total=len(df)))
    out = df.merge(pd.DataFrame(res), on="relpath")
    out.to_csv(cache, index=False)
    return out


# ----------------------------------------------------------------------------- duplicate groups
def find_groups(df: pd.DataFrame, threshold: int = 10):
    """Union-find over exact duplicates (same MD5) and near duplicates (hash Hamming distance <= threshold).

    Near-duplicate search uses locality-sensitive buckets: split the 256 bits into threshold+1 bands;
    two hashes that differ in <= threshold bits must be identical in at least one band (pigeonhole).
    """
    n = len(df)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    first, exact_extra = {}, 0
    for i, m in enumerate(df["md5"].values):
        if m in first:
            union(first[m], i)
            exact_extra += 1
        else:
            first[m] = i

    bits = np.unpackbits(np.array([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in df["hash"]]), axis=1)
    bounds = np.linspace(0, bits.shape[1], threshold + 2).astype(int)
    near_pairs, skipped = 0, 0
    for b in range(threshold + 1):
        keys = np.packbits(bits[:, bounds[b]:bounds[b + 1]], axis=1)
        buckets = defaultdict(list)
        for i in range(n):
            buckets[keys[i].tobytes()].append(i)
        for idx in buckets.values():
            if len(idx) < 2:
                continue
            if len(idx) > MAX_BUCKET:
                skipped += 1
                continue
            B = bits[idx]
            for a in range(len(idx) - 1):
                d = (B[a] != B[a + 1:]).sum(1)
                for j in np.nonzero(d <= threshold)[0]:
                    union(idx[a], idx[a + 1 + j])
                    near_pairs += 1
    groups = np.array([find(i) for i in range(n)])
    sizes = pd.Series(groups).value_counts()
    tmp = pd.DataFrame({"g": groups, "label": df["label"].values})
    cross = int((tmp.groupby("g")["label"].nunique() > 1).sum())
    stats = dict(
        n_images=n, exact_duplicate_images=exact_extra, near_dup_pairs_found=near_pairs,
        n_groups=int(len(sizes)), images_in_multi_image_groups=int(sizes[sizes > 1].sum()),
        largest_group=int(sizes.max()), groups_with_mixed_classes=cross, skipped_giant_buckets=skipped,
        near_dup_threshold_bits=threshold,
    )
    return groups, stats


# ----------------------------------------------------------------------------- split
def _subsample(df: pd.DataFrame, max_per_class, max_total, seed) -> pd.DataFrame:
    if not max_per_class and not max_total:
        return df
    rng = np.random.RandomState(seed)
    parts = []
    for _, g in df.groupby("label"):
        if max_total:
            k = max(3, int(round(max_total * len(g) / len(df))))
        else:
            k = max_per_class
        k = min(k, len(g))
        parts.append(g.iloc[rng.permutation(len(g))[:k]])
    return pd.concat(parts).reset_index(drop=True)


def group_stratified_split(df: pd.DataFrame, ratios, seed: int) -> pd.Series:
    """Assign every GROUP to train/val/test, class by class, so class proportions are preserved."""
    rng = np.random.RandomState(seed)
    gl = df.groupby("group").agg(label=("label", lambda s: s.mode().iloc[0]), n=("label", "size")).reset_index()
    assign = {}
    for _, g in gl.groupby("label"):
        g = g.iloc[rng.permutation(len(g))]
        cum = g["n"].cumsum().values - g["n"].values / 2.0  # midpoint of each group in the cumulative count
        frac = cum / g["n"].sum()
        for grp, f in zip(g["group"], frac):
            assign[grp] = "train" if f < ratios[0] else ("val" if f < ratios[0] + ratios[1] else "test")
    return df["group"].map(assign)


def verify_no_leakage(df: pd.DataFrame) -> dict:
    per_group = df.groupby("group")["split"].nunique()
    assert (per_group == 1).all(), "A duplicate group spans several splits!"
    sets = {s: set(df.loc[df.split == s, "md5"]) for s in ("train", "val", "test")}
    overlap = {f"{a}&{b}": len(sets[a] & sets[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    assert all(v == 0 for v in overlap.values()), f"Exact duplicates across splits: {overlap}"
    return {"groups_spanning_splits": 0, "exact_md5_overlap": overlap}


def build_split(cfg, force: bool = False) -> pd.DataFrame:
    paths = get_paths(cfg)
    out_csv = paths["splits"] / "split.csv"
    if out_csv.exists() and not force:
        log.info("Split already exists: %s (use --force to recreate)", out_csv)
        return pd.read_csv(out_csv)

    idx = build_index(cfg)
    bad = idx[~idx["ok"]]
    bad[["relpath", "class_name", "error"]].to_csv(paths["eda"] / "corrupted_images.csv", index=False)
    good = idx[idx["ok"]].reset_index(drop=True)
    log.info("%d images OK, %d corrupted (removed)", len(good), len(bad))

    groups, dstats = find_groups(good, cfg["split"].get("near_dup_threshold", 10))
    good["group"] = groups
    seed = cfg["split"]["seed"]
    good = _subsample(good, cfg["data"].get("max_per_class"), cfg["data"].get("max_total"), seed)
    good["split"] = group_stratified_split(good, cfg["split"]["ratios"], seed)
    leak = verify_no_leakage(good)

    keep = ["relpath", "class_name", "label", "split", "group", "md5"]
    good[keep].to_csv(out_csv, index=False)

    counts = good.pivot_table(index="class_name", columns="split", values="relpath", aggfunc="count", fill_value=0)
    counts.to_csv(paths["splits"] / "split_counts_per_class.csv")
    missing = [c for s in ("train", "val", "test") for c in counts.index[counts.get(s, 0) == 0]]
    report = dict(
        n_total=len(good), n_corrupted_removed=len(bad), per_split=good["split"].value_counts().to_dict(),
        duplicate_stats_full_dataset=dstats, leakage_checks=leak, split_seed=seed, ratios=cfg["split"]["ratios"],
        classes_missing_in_some_split=sorted(set(missing)),
    )
    (paths["splits"] / "split_report.json").write_text(json.dumps(report, indent=2))
    log.info("Split written: %s", good["split"].value_counts().to_dict())
    if report["classes_missing_in_some_split"]:
        log.warning("Classes missing from some split: %s", report["classes_missing_in_some_split"])
    return good[keep]
