"""Command line entry point.

  python main.py check      --config configs/cpu.yaml   # hardware, dataset, existing files
  python main.py benchmark  --config configs/cpu.yaml   # seconds / epoch estimate per model on THIS machine
  python main.py eda        --config configs/cpu.yaml
  python main.py split      --config configs/cpu.yaml
  python main.py pretrain   --config configs/cpu.yaml   # MAE on unlabeled train images
  python main.py train      --config configs/cpu.yaml [--models A_cnn_scratch ...] [--budgets 10 100] [--seeds 0 1]
  python main.py aggregate  --config configs/cpu.yaml   # table + plots + REPORT.md
  python main.py all        --config configs/cpu.yaml   # eda -> split -> pretrain -> train -> aggregate
"""
from __future__ import annotations

import argparse
import sys
import time

from src.utils import ROOT, get_logger, get_paths, hardware_summary, load_config, resolve

log = get_logger()


def cmd_check(cfg):
    import json

    print("== Hardware ==")
    print(json.dumps(hardware_summary(), indent=2))
    paths = get_paths(cfg)
    root = resolve(cfg["data"]["root"])
    print("\n== Dataset ==")
    if root.exists():
        n_cls = len([d for d in root.iterdir() if d.is_dir()])
        print(f"found: {root}  ({n_cls} class folders)")
    else:
        print(f"NOT FOUND: {root}\n -> run: python scripts/prepare_dataset.py --dataset plantvillage")
    print("\n== Project state ==")
    for name, p in [("split", paths["splits"] / "split.csv"), ("MAE encoder", paths["mae"] / "encoder.pt"),
                    ("runs.csv", paths["results"] / "runs.csv"), ("report", paths["results"] / "REPORT.md")]:
        print(f"{name:12s}: {'yes' if p.exists() else 'not yet'}  ({p.relative_to(ROOT)})")
    import torch

    if not torch.cuda.is_available():
        print("\nNo GPU detected -> CPU mode. Use configs/cpu.yaml (small models) or run configs/gpu.yaml on Colab.")


def cmd_benchmark(cfg):
    """Time a few training steps of every model on random data to estimate the cost of the grid."""
    import torch

    from src.models import MODEL_KINDS, build_model
    from src.utils import get_device

    dev = get_device()
    bs, size = cfg["train"]["batch_size"], cfg["data"]["img_size"]
    cfg = {**cfg, "models": {**cfg["models"], "strict_pretrained": False}}
    x = torch.randn(bs, 3, size, size, device=dev)
    y = torch.randint(0, 10, (bs,), device=dev)
    seen = set()
    print(f"device={dev}  batch={bs}  img={size}")
    for kind, info in MODEL_KINDS.items():
        key = (info["family"],)
        if key in seen and kind != "A_cnn_scratch":
            continue
        seen.add(key)
        model, _ = build_model("D_vit_scratch" if info["family"] == "vit" else "A_cnn_scratch", cfg, 10)
        model.to(dev).train()
        opt = torch.optim.AdamW(model.parameters(), 1e-4)
        for i in range(4):
            if i == 1:
                t0 = time.perf_counter()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        dt = (time.perf_counter() - t0) / 3
        fam = cfg["models"]["cnn_arch"] if info["family"] == "cnn" else cfg["models"]["vit_arch"]
        print(f"{fam:28s} {bs / dt:7.1f} img/s  -> 1000 training images ~ {1000 / (bs / dt):.0f} s per epoch")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "benchmark", "eda", "split", "pretrain", "train", "aggregate", "all"])
    ap.add_argument("--config", default="configs/cpu.yaml")
    ap.add_argument("--models", nargs="*", default=None, help="subset of model kinds (see config)")
    ap.add_argument("--budgets", nargs="*", type=float, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="redo even if the output already exists")
    a = ap.parse_args()
    cfg = load_config(a.config)

    if a.command == "check":
        cmd_check(cfg)
    elif a.command == "benchmark":
        cmd_benchmark(cfg)
    elif a.command == "eda":
        from src.eda import run_eda

        run_eda(cfg)
    elif a.command == "split":
        from src.split_dataset import build_split

        build_split(cfg, force=a.force)
    elif a.command == "pretrain":
        from src.mae_pretraining import pretrain

        pretrain(cfg, force=a.force)
    elif a.command == "train":
        from src.train_classifier import run_grid

        run_grid(cfg, a.models, a.budgets, a.seeds, a.force)
    elif a.command == "aggregate":
        from src.report import aggregate

        aggregate(cfg)
    elif a.command == "all":
        from src.eda import run_eda
        from src.mae_pretraining import pretrain
        from src.report import aggregate
        from src.split_dataset import build_split
        from src.train_classifier import run_grid

        run_eda(cfg)
        build_split(cfg, force=a.force)
        pretrain(cfg, force=a.force)
        run_grid(cfg, a.models, a.budgets, a.seeds, a.force)
        aggregate(cfg)


if __name__ == "__main__":
    sys.exit(main())
