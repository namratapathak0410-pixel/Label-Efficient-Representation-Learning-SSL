"""Command line entry point.

  python run.py prepare     download + cache the datasets (and ImageNetV2 for the distance measures)
  python run.py split       leakage-safe splits and nested label budgets
  python run.py benchmark   measure GPU speed and print the time plan for this config
  python run.py distance    ImageNet-to-target distances (before any training)
  python run.py pretrain    MAE per dataset (and per seed if mae.per_seed)
  python run.py tune        learning rate per (dataset, arm) on validation, seed 0
  python run.py train       the full grid, seed-major, resumable
  python run.py analyze     tables, plots, REPORT.md
  python run.py all         everything above in order

Experiment 2 (use --config config_v2.yaml):
  python run.py pool        build the 20,000-image unlabelled pools (E2, F)
  python run.py v2          pool -> MAE seed 0 -> tune -> grid, with an overall time-left estimate in the log
  python run.py analyze2    tables, plots and REPORT_V2.md for the 7-arm experiment

Add --config configs/smoke.yaml for a quick end-to-end test.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

from mc.utils import get_logger, load_config, resolved_datasets, write_json, paths

log = get_logger()


def benchmark(cfg):
    import torch

    from mc.mae import pretrain_one
    from mc.split import budget_indices
    from mc.train import Data, schedule, train_one

    names = list(resolved_datasets(cfg).values())
    data = Data(cfg, names[0])
    speed = {}
    for arm in ("A_cnn_scratch", "C_vit_imagenet", "D_vit_scratch"):
        train_one(cfg, data, arm, 100, 0, 1e-3, tag="bench", max_steps=20)  # warm-up (cudnn autotune)
        r = train_one(cfg, data, arm, 100, 0, 1e-3, tag="bench", max_steps=150)
        speed[arm] = r["seconds"] / r["steps"]
        log.info("[bench] %s: %.1f ms/step (%.0f img/s incl. validation)", arm, 1e3 * speed[arm],
                 r["images_per_second"])
    speed["E_vit_mae"] = speed["D_vit_scratch"]
    del data
    torch.cuda.empty_cache()
    m = pretrain_one(cfg, names[0], 0, max_steps=100)
    mae_step = m["pretrain_seconds"] / 100
    log.info("[bench] MAE: %.1f ms/step, peak %.2f GB", 1e3 * mae_step, m["peak_gpu_mem_gb"])

    t = cfg["train"]
    from mc.split import make_split

    df = make_split(names[0], cfg)
    n_train = int((df.split == "train").sum())
    steps_per_run = {b: schedule(cfg, len(budget_indices(df, b, 0)))[1] for b in t["budgets"]}
    grid_s = sum(speed[a] * steps_per_run[b] for a in cfg["arms"] for b in t["budgets"]) * len(names) * len(t["seeds"])
    tune_steps = schedule(cfg, len(budget_indices(df, t["tune_budget"], 0)))[1]
    tune_s = sum(speed[a] * tune_steps * 3 for a in cfg["arms"]) * len(names)
    n_mae = len(names) * (len(t["seeds"]) if cfg["mae"]["per_seed"] else 1)
    mae_s = n_mae * cfg["mae"]["epochs"] * max(1, n_train // cfg["mae"]["batch_size"]) * mae_step
    plan = dict(ms_per_step={k: 1e3 * v for k, v in speed.items()}, mae_ms_per_step=1e3 * mae_step,
                steps_per_run=steps_per_run, n_train=n_train,
                hours_worst_case=dict(mae=mae_s / 3600, tune=tune_s / 3600, grid=grid_s / 3600,
                                      total=(mae_s + tune_s + grid_s) / 3600),
                note="worst case = no early stopping; real runs usually stop earlier")
    write_json(paths(cfg)["results"] / "benchmark.json", plan)
    log.info("[bench] plan (hours, worst case): %s", plan["hours_worst_case"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "split", "benchmark", "distance", "pretrain", "tune", "train",
                                      "analyze", "all", "pool", "v2", "analyze2"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--budgets", type=float, nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config)
    t0 = time.time()
    stages = ["prepare", "split", "distance", "pretrain", "tune", "train", "analyze"] if a.stage == "all" else [a.stage]
    for st in stages:
        log.info("===== %s (%s) =====", st, cfg["experiment"])
        if st == "prepare":
            from mc.data import prepare
            prepare(cfg)
        elif st == "split":
            from mc.split import split_all
            split_all(cfg)
        elif st == "benchmark":
            benchmark(cfg)
        elif st == "distance":
            from mc.distance import compute
            compute(cfg)
        elif st == "pretrain":
            from mc.mae import pretrain_all
            pretrain_all(cfg, a.seeds, a.datasets)
        elif st == "tune":
            from mc.train import tune
            tune(cfg, a.datasets, a.arms)
        elif st == "train":
            from mc.train import grid
            grid(cfg, a.seeds, a.datasets, a.arms, a.budgets)
        elif st == "pool":
            from mc.pool import build_all
            build_all(cfg)
        elif st == "v2":
            from mc.pipeline2 import run
            run(cfg)
        elif st == "analyze2":
            from mc.analyze2 import analyze2
            analyze2(cfg)
        elif st == "analyze":
            from mc.analyze import analyze
            analyze(cfg)
    log.info("done in %.1f min", (time.time() - t0) / 60)


if __name__ == "__main__":
    sys.exit(main())
