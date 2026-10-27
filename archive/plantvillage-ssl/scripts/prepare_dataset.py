"""Download / prepare a dataset into  data/raw/<name>/<class>/<image>  (ImageFolder layout).

Examples
  python scripts/prepare_dataset.py --dataset plantvillage          # Kaggle (needs internet, ~1 GB)
  python scripts/prepare_dataset.py --dataset plantvillage --src "C:/Users/me/Downloads/plantvillage dataset"
  python scripts/prepare_dataset.py --dataset eurosat               # via torchvision, no account needed (~90 MB)
  python scripts/prepare_dataset.py --dataset cifar10               # via torchvision (~170 MB)
  python scripts/prepare_dataset.py --dataset synthetic             # fake images, only to test the pipeline
  python scripts/prepare_dataset.py --dataset imagefolder --src D:/my_images --name mydata

After preparing, point `data.root` / `data.name` in your YAML config to the printed folder.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def count_images(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if p.suffix.lower() in IMG_EXT)


def find_class_root(start: Path) -> Path:
    """Find the folder whose sub-folders are the classes (prefers a folder called 'color' for PlantVillage)."""
    cands = [p for p in [start, *start.rglob("*")] if p.is_dir()]
    colors = [p for p in cands if p.name.lower() == "color"]
    if colors:
        return colors[0]
    best = None
    for p in cands:
        subs = [d for d in p.iterdir() if d.is_dir()]
        if len(subs) >= 2 and all(any(f.suffix.lower() in IMG_EXT for f in d.iterdir() if f.is_file()) for d in subs):
            if best is None or len(subs) > len([d for d in best.iterdir() if d.is_dir()]):
                best = p
    if best is None:
        sys.exit(f"Could not find a class-folder layout under {start}")
    return best


def copy_tree(src: Path, name: str) -> Path:
    dst = RAW / name
    if dst.exists():
        print(f"{dst} already exists - leaving it untouched (delete it to re-create).")
        return dst
    print(f"Copying {src} -> {dst} ...")
    shutil.copytree(src, dst)
    return dst


def plantvillage(src: str | None):
    if src is None:
        try:
            import kagglehub
        except ImportError:
            sys.exit("pip install kagglehub   (or download the zip manually from Kaggle and use --src)")
        print("Downloading abdallahalidev/plantvillage-dataset from Kaggle ...")
        src = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")
    return copy_tree(find_class_root(Path(src)), "plantvillage")


def eurosat():
    from torchvision.datasets import EuroSAT

    ds = EuroSAT(root=str(RAW / "_torchvision"), download=True)
    return copy_tree(Path(ds.root), "eurosat")


def cifar10():
    from PIL import Image
    from torchvision.datasets import CIFAR10

    dst = RAW / "cifar10"
    if dst.exists():
        print(f"{dst} already exists")
        return dst
    ds = CIFAR10(root=str(RAW / "_torchvision"), train=True, download=True)
    for c in ds.classes:
        (dst / c).mkdir(parents=True, exist_ok=True)
    for i, (img, y) in enumerate(ds):
        img.save(dst / ds.classes[y] / f"{i:05d}.png")
    return dst


def synthetic(n_classes=6, per_class=40, size=64):
    from PIL import Image, ImageDraw

    dst = RAW / "synthetic"
    if dst.exists():
        print(f"{dst} already exists")
        return dst
    rng = np.random.RandomState(0)
    for c in range(n_classes):
        d = dst / f"class_{c}"
        d.mkdir(parents=True, exist_ok=True)
        base = rng.randint(40, 215, 3)
        for i in range(per_class):
            arr = np.clip(base + rng.randn(size, size, 3) * 18, 0, 255).astype(np.uint8)
            im = Image.fromarray(arr)
            dr = ImageDraw.Draw(im)
            for _ in range(c + 1):  # class c has c+1 dark "lesions"
                x, y = rng.randint(8, size - 8, 2)
                dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(30, 20, 10))
            im.save(d / f"img_{i:03d}.png")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["plantvillage", "eurosat", "cifar10", "synthetic", "imagefolder"])
    ap.add_argument("--src", default=None, help="existing folder (plantvillage manual download / imagefolder)")
    ap.add_argument("--name", default=None, help="target folder name for --dataset imagefolder")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if a.dataset == "plantvillage":
        out = plantvillage(a.src)
    elif a.dataset == "eurosat":
        out = eurosat()
    elif a.dataset == "cifar10":
        out = cifar10()
    elif a.dataset == "synthetic":
        out = synthetic()
    else:
        if not a.src or not a.name:
            sys.exit("--dataset imagefolder needs --src and --name")
        out = copy_tree(find_class_root(Path(a.src)), a.name)
    n_cls = len([d for d in out.iterdir() if d.is_dir()])
    print(f"\nReady: {out}\n  classes: {n_cls}   images: {count_images(out)}")
    print(f'  -> in your config set:  data.name: {out.name}   data.root: data/raw/{out.name}')


if __name__ == "__main__":
    main()
