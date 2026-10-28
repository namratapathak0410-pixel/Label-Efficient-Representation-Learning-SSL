"""Bigger unlabelled pool for MAE in experiment 2 (PREREGISTRATION_V2.md).

pool = the train images of the 7,000-image sample + extra images of the same dataset, cfg.pool.size in total.
Leakage control:
  1. extras are drawn only from images NOT in the 7,000-image sample, so no val/test image can be in the pool;
  2. any extra within the dHash threshold of a val/test image, or of its horizontal/vertical/double flip, is dropped;
  3. labels of pool images are never read by pretraining.
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .data import SOURCES, dhash256, load_cache, load_square
from .split import make_split
from .utils import get_logger, paths, resolved_datasets, write_json

log = get_logger()


def pool_file(name: str, cfg: dict):
    return paths(cfg)["cache"] / f"{name}_pool{cfg['pool']['size']}.npz"


def _reference_hashes(images: np.ndarray) -> np.ndarray:
    """dHash of val/test images and of their flips (the same photo is often present mirrored or flipped)."""
    views = [images, images[:, :, ::-1], images[:, ::-1], images[:, ::-1, ::-1]]
    return np.concatenate([np.stack([dhash256(np.ascontiguousarray(x)) for x in v]) for v in views])


def _near(hashes: np.ndarray, ref: np.ndarray, threshold: int, chunk: int = 256) -> np.ndarray:
    a_words = np.ascontiguousarray(hashes).view(np.uint64)
    r_words = np.ascontiguousarray(ref).view(np.uint64)
    out = np.zeros(len(hashes), dtype=bool)
    for s in range(0, len(hashes), chunk):
        d = np.bitwise_count(a_words[s:s + chunk, None, :] ^ r_words[None, :, :]).sum(-1)
        out[s:s + chunk] = (d <= threshold).any(1)
    return out


def build_pool(name: str, cfg: dict):
    out = pool_file(name, cfg)
    if out.exists():
        try:
            with open(out, "rb") as f:
                n = np.load(f)["images"].shape[0]
            log.info("[%s] pool exists (%d extra images)", name, n)
            return out
        except Exception as e:
            log.warning("[%s] pool unreadable (%s) - rebuilding", name, e)
            out.unlink()
    pc, p = cfg["pool"], paths(cfg)
    c, df = load_cache(name, cfg), make_split(name, cfg)
    n_extra = pc["size"] - int((df.split == "train").sum())
    _, _, loaders, ids = SOURCES[name](p["raw"])
    used = set(map(str, c["ids"]))
    cand = np.array([i for i, x in enumerate(ids) if x not in used])
    log.info("[%s] %d images outside the sample, %d extras wanted", name, len(cand), n_extra)
    if len(cand) < int(n_extra * 1.1):
        raise RuntimeError(f"{name}: only {len(cand)} images outside the sample, need ~{n_extra}")
    rng = np.random.default_rng(pc["seed"])
    take = cand[rng.permutation(len(cand))[: int(n_extra * 1.08) + 200]]
    imgs, keep, bad = [], [], 0
    for i in tqdm(take, desc=f"decode pool {name}", mininterval=10):
        try:
            imgs.append(load_square(loaders[i], cfg["img_size"]))
            keep.append(i)
        except Exception:
            bad += 1
    imgs = np.stack(imgs)
    hashes = np.stack([dhash256(x) for x in imgs])
    held = df.loc[df.split.isin(["val", "test"]), "idx"].to_numpy()
    clash = _near(hashes, _reference_hashes(c["images"][held]), cfg["split"]["dhash_threshold"])
    sel = np.nonzero(~clash)[0][:n_extra]
    if len(sel) < n_extra:
        raise RuntimeError(f"{name}: {len(sel)} extras left after removing near-duplicates, need {n_extra}")
    tmp = out.with_name(out.stem + ".partial.npz")
    np.savez(tmp, images=imgs[sel], dhash=hashes[sel], ids=np.asarray([ids[keep[i]] for i in sel]))
    tmp.replace(out)
    write_json(p["cache"] / f"{name}_pool_info.json", dict(
        dataset=name, pool_size=pc["size"], n_extra=int(len(sel)), n_train_in_pool=int((df.split == "train").sum()),
        n_outside_sample=int(len(cand)), n_sampled=int(len(take)), n_unreadable=bad,
        n_removed_as_near_duplicate_of_val_or_test=int(clash.sum()), dhash_threshold=cfg["split"]["dhash_threshold"],
        n_overlap_with_sample=len(set(map(str, [ids[keep[i]] for i in sel])) & used)))
    log.info("[%s] pool built: %d extras (%d removed as near-duplicates of val/test) -> %s", name, len(sel),
             int(clash.sum()), out)
    return out


def build_all(cfg: dict):
    res = resolved_datasets(cfg)
    for slot in cfg["pool"]["extra_slots"]:
        build_pool(res[slot], cfg)


def unlabelled_images(cfg: dict, dataset: str, use_pool: bool) -> np.ndarray:
    """Images (uint8, N,H,W,3) the MAE may see: train split of the sample, plus the extras when use_pool."""
    df, c = make_split(dataset, cfg), load_cache(dataset, cfg)
    imgs = c["images"][df.loc[df.split == "train", "idx"].to_numpy()]
    if use_pool:
        with open(pool_file(dataset, cfg), "rb") as f:
            z = np.load(f)
            imgs = np.concatenate([imgs, z["images"]])
    return imgs
