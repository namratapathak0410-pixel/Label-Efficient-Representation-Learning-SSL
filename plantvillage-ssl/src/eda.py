"""Exploratory data analysis: counts, class plot, samples, corrupted + duplicate images, imbalance, leakage risk."""
from __future__ import annotations

import json

from .split_dataset import build_index, find_groups
from .utils import get_logger, get_paths, resolve
from .visualization import plot_class_distribution, plot_sample_grid

log = get_logger()


def run_eda(cfg):
    paths = get_paths(cfg)
    out = paths["eda"]
    root = resolve(cfg["data"]["root"])
    idx = build_index(cfg)

    counts = idx.groupby("class_name").size()
    counts.rename("n_images").to_csv(out / "class_counts.csv")
    plot_class_distribution(counts, out / "class_distribution.png")

    good = idx[idx["ok"]].reset_index(drop=True)
    plot_sample_grid(good, root, out / "sample_images.png")

    bad = idx[~idx["ok"]]
    bad[["relpath", "class_name", "error"]].to_csv(out / "corrupted_images.csv", index=False)

    groups, dstats = find_groups(good, cfg["split"].get("near_dup_threshold", 10))
    good["group"] = groups
    dup = good[good.duplicated("group", keep=False)].sort_values("group")
    dup[["group", "relpath", "class_name"]].to_csv(out / "duplicate_groups.csv", index=False)

    imb = float(counts.max() / counts.min())
    summary = dict(
        dataset_root=str(root), n_images=int(len(idx)), n_classes=int(len(counts)),
        n_corrupted=int(len(bad)), min_class=[counts.idxmin(), int(counts.min())],
        max_class=[counts.idxmax(), int(counts.max())], imbalance_ratio_max_over_min=round(imb, 2),
        image_size_most_common=str(good.groupby(["w", "h"]).size().sort_values(ascending=False).head(3).to_dict()),
        duplicates=dstats,
        notes=[
            "Imbalance ratio > ~3 means accuracy alone is misleading -> macro-F1 is reported.",
            "Images that belong to a duplicate group are kept together in one split (see split_dataset.py) "
            "to prevent train/test leakage.",
        ],
    )
    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("EDA written to %s", out)
    log.info("images=%d classes=%d corrupted=%d imbalance(max/min)=%.2f", len(idx), len(counts), len(bad), imb)
    log.info("duplicates: %s", dstats)
    return summary
