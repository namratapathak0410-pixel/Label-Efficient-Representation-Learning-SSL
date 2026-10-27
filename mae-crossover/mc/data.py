"""Download datasets, draw a fixed-size stratified sample, decode once into a uint8 cache.

Every dataset is reduced to the same number of images (cfg.n_per_dataset) so that the MAE unlabelled pool and the
label budgets have the same absolute size in every domain - the domain is the thing that varies, not the data volume.
Images are resized (shorter side -> img_size, bicubic, antialiased) and centre-cropped to a square.
"""
from __future__ import annotations

import tarfile
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .utils import get_logger, paths, write_json

log = get_logger()
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ----------------------------------------------------------------------------- item lists
# A source returns (class_names, labels[np.int64], loaders[list of callables returning a PIL image], ids[list str])

def _from_files(files, labels, class_names):
    loaders = [(lambda f=f: Image.open(f)) for f in files]
    return list(class_names), np.asarray(labels, dtype=np.int64), loaders, [str(f) for f in files]


def _from_arrays(arrays, labels, class_names, prefix):
    loaders = [(lambda a=a: Image.fromarray(a)) for a in arrays]
    return list(class_names), np.asarray(labels, dtype=np.int64), loaders, [f"{prefix}:{i}" for i in range(len(arrays))]


def _class_folders(root: Path):
    """Find the directory whose sub-folders are classes (most image-holding sub-folders wins)."""
    best, best_n = None, 1
    for d in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        subs = [s for s in d.iterdir() if s.is_dir() and any(f.suffix.lower() in IMG_EXT for f in s.iterdir())]
        if len(subs) > best_n:
            best, best_n = d, len(subs)
    if best is None:
        raise RuntimeError(f"No class folders found under {root}")
    return best


def _folder_dataset(root: Path):
    classes = sorted(s.name for s in root.iterdir() if s.is_dir())
    files, labels = [], []
    for i, c in enumerate(classes):
        fs = sorted(f for f in (root / c).iterdir() if f.suffix.lower() in IMG_EXT)
        files += fs
        labels += [i] * len(fs)
    return _from_files(files, labels, classes)


def src_plantvillage(raw: Path):
    import kagglehub

    root = Path(kagglehub.dataset_download("abdallahalidev/plantvillage-dataset"))
    color = [p for p in root.rglob("color") if p.is_dir()]
    if not color:
        raise RuntimeError("PlantVillage download has no 'color' folder")
    return _folder_dataset(color[0])  # only colour images; grayscale/segmented are copies of the same leaves


def src_cifar10(raw: Path):
    from torchvision.datasets import CIFAR10

    tr = CIFAR10(root=str(raw / "torchvision"), train=True, download=True)
    te = CIFAR10(root=str(raw / "torchvision"), train=False, download=True)
    arr = np.concatenate([tr.data, te.data])
    lab = np.concatenate([tr.targets, te.targets])
    return _from_arrays(arr, lab, tr.classes, "cifar10")


def src_cifar10_kaggle(raw: Path):
    import kagglehub

    return _folder_dataset(_class_folders(Path(kagglehub.dataset_download("shaurov/cifar-10-dataset"))))


def src_eurosat(raw: Path):
    from torchvision.datasets import EuroSAT

    ds = EuroSAT(root=str(raw / "torchvision"), download=True)
    files, labels = zip(*ds.samples)
    return _from_files(files, labels, ds.classes)


def src_eurosat_kaggle(raw: Path):
    import kagglehub

    root = Path(kagglehub.dataset_download("apollo2506/eurosat-dataset"))
    rgb = [p for p in root.rglob("EuroSAT") if p.is_dir()]
    return _folder_dataset(_class_folders(rgb[0] if rgb else root))


def src_pets(raw: Path):
    from torchvision.datasets import OxfordIIITPet

    files, labels, classes = [], [], None
    for split in ("trainval", "test"):
        ds = OxfordIIITPet(root=str(raw / "torchvision"), split=split, target_types="category", download=True)
        files += list(ds._images)
        labels += list(ds._labels)
        classes = ds.classes
    return _from_files(files, labels, classes)


def src_pets_kaggle(raw: Path):
    import kagglehub

    root = Path(kagglehub.dataset_download("devdgohil/the-oxfordiiit-pet-dataset"))
    files = sorted(f for f in root.rglob("*.jpg"))
    names = ["_".join(f.stem.split("_")[:-1]) for f in files]  # Abyssinian_12.jpg -> Abyssinian
    classes = sorted(set(names))
    return _from_files(files, [classes.index(n) for n in names], classes)


def src_dtd(raw: Path):
    from torchvision.datasets import DTD

    files, labels, classes = [], [], None
    for split in ("train", "val", "test"):
        ds = DTD(root=str(raw / "torchvision"), split=split, partition=1, download=True)
        files += list(ds._image_files)
        labels += list(ds._labels)
        classes = ds.classes
    return _from_files(files, labels, classes)


def src_flowers102(raw: Path):
    from torchvision.datasets import Flowers102

    files, labels = [], []
    for split in ("train", "val", "test"):
        ds = Flowers102(root=str(raw / "torchvision"), split=split, download=True)
        files += list(ds._image_files)
        labels += list(ds._labels)
    return _from_files(files, labels, [f"flower_{i:03d}" for i in range(102)])


def src_stl10(raw: Path):
    from torchvision.datasets import STL10

    tr = STL10(root=str(raw / "torchvision"), split="train", download=True)
    te = STL10(root=str(raw / "torchvision"), split="test", download=True)
    arr = np.concatenate([tr.data, te.data]).transpose(0, 2, 3, 1)
    return _from_arrays(arr, np.concatenate([tr.labels, te.labels]), tr.classes, "stl10")


def src_imagenetv2(raw: Path):
    """ImageNetV2 matched-frequency (10,000 images, 10 per ImageNet class) - a public stand-in for ImageNet,
    used only as the reference distribution for the domain-distance measures."""
    from huggingface_hub import hf_hub_download

    out = raw / "imagenetv2"
    folder = out / "imagenetv2-matched-frequency-format-val"
    if not folder.exists():
        tar = hf_hub_download("vaishaal/ImageNetV2", "imagenetv2-matched-frequency.tar.gz", repo_type="dataset",
                              local_dir=str(out))
        with tarfile.open(tar) as t:
            t.extractall(out)
    classes = sorted((d.name for d in folder.iterdir() if d.is_dir()), key=int)
    files, labels = [], []
    for c in classes:
        fs = sorted((folder / c).iterdir())
        files += fs
        labels += [int(c)] * len(fs)
    return _from_files(files, labels, classes)


SOURCES = {
    "plantvillage": src_plantvillage, "cifar10": src_cifar10, "cifar10_kaggle": src_cifar10_kaggle,
    "eurosat": src_eurosat, "eurosat_kaggle": src_eurosat_kaggle, "pets": src_pets, "pets_kaggle": src_pets_kaggle,
    "dtd": src_dtd, "flowers102": src_flowers102, "stl10": src_stl10, "imagenetv2": src_imagenetv2,
}


# ----------------------------------------------------------------------------- decoding
def load_square(loader, size: int) -> np.ndarray:
    with loader() as im:
        im = im.convert("RGB")
        w, h = im.size
        s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC, reducing_gap=3.0)
        w, h = im.size
        l, t = (w - size) // 2, (h - size) // 2
        return np.asarray(im.crop((l, t, l + size, t + size)), dtype=np.uint8)


def dhash256(img: np.ndarray) -> np.ndarray:
    """256-bit difference hash (16x16 horizontal gradients of a 17x16 greyscale thumbnail)."""
    g = np.asarray(Image.fromarray(img).convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).reshape(-1))  # 32 bytes


def build_cache(name: str, cfg: dict, n: int | None) -> Path:
    p = paths(cfg)
    out = p["cache"] / f"{name}.npz"
    if out.exists():
        try:
            with open(out, "rb") as f:  # own handle, so Windows can delete the file if it is broken
                _ = np.load(f)["labels"].shape
            log.info("[%s] cache exists: %s", name, out)
            return out
        except Exception as e:  # e.g. a write that was interrupted
            log.warning("[%s] cache is unreadable (%s) - rebuilding", name, e)
            out.unlink()
    log.info("[%s] listing / downloading ...", name)
    classes, labels, loaders, ids = SOURCES[name](p["raw"])
    log.info("[%s] %d images, %d classes available", name, len(labels), len(classes))
    idx = np.arange(len(labels))
    if n is not None and len(idx) > n:
        # sample a few extra so that dropping corrupted files still leaves n images
        take = min(len(idx) - 1, int(n * 1.02) + len(classes))
        idx, _ = train_test_split(idx, train_size=take, stratify=labels, random_state=cfg["split"]["split_seed"])
        idx = np.sort(idx)
    imgs, labs, keep_ids, bad = [], [], [], []
    for i in tqdm(idx, desc=f"decode {name}", mininterval=5):
        try:
            imgs.append(load_square(loaders[i], cfg["img_size"]))
            labs.append(labels[i])
            keep_ids.append(ids[i])
        except Exception as e:  # corrupted / unreadable file
            bad.append(f"{ids[i]}: {e}")
    imgs, labs = np.stack(imgs), np.asarray(labs)
    if n is not None and len(labs) > n:  # trim the extras back to exactly n, stratified
        sel, _ = train_test_split(np.arange(len(labs)), train_size=n, stratify=labs,
                                  random_state=cfg["split"]["split_seed"])
        sel = np.sort(sel)
        imgs, labs, keep_ids = imgs[sel], labs[sel], [keep_ids[i] for i in sel]
    present = np.unique(labs)
    remap = {old: new for new, old in enumerate(present)}  # drop classes that vanished (never happens at n=7000)
    labs = np.asarray([remap[v] for v in labs], dtype=np.int64)
    classes = [classes[i] for i in present]
    hashes = np.stack([dhash256(x) for x in imgs])
    tmp = out.with_name(out.stem + ".partial.npz")  # write-then-rename: an interrupted write never looks finished
    np.savez(tmp, images=imgs, labels=labs, class_names=np.asarray(classes), ids=np.asarray(keep_ids),
             dhash=hashes)
    tmp.replace(out)
    write_json(p["cache"] / f"{name}_info.json", dict(
        name=name, n_images=int(len(labs)), n_classes=len(classes), n_available=int(len(labels)),
        n_corrupted=len(bad), corrupted=bad[:50], img_size=cfg["img_size"],
        class_counts={c: int((labs == i).sum()) for i, c in enumerate(classes)}))
    log.info("[%s] cached %d images (%d classes, %d unreadable) -> %s", name, len(labs), len(classes), len(bad), out)
    return out


def load_cache(name: str, cfg: dict) -> dict:
    z = np.load(paths(cfg)["cache"] / f"{name}.npz")
    return {k: z[k] for k in z.files}


def prepare(cfg: dict) -> dict:
    """Resolve each domain slot to the first candidate dataset that downloads and decodes successfully."""
    p = paths(cfg)
    resolved = {}
    for slot, candidates in cfg["datasets"].items():
        for name in candidates:
            try:
                build_cache(name, cfg, cfg["n_per_dataset"])
                info = (p["cache"] / f"{name}_info.json").read_text()
                if '"n_images"' in info:
                    resolved[slot] = name
                    break
            except Exception as e:
                log.warning("[%s] %s failed (%s: %s) - trying the next candidate", slot, name, type(e).__name__, e)
        else:
            raise SystemExit(f"No dataset could be prepared for domain '{slot}': {candidates}")
    try:
        build_cache("imagenetv2", cfg, None)
    except Exception as e:
        log.warning("ImageNetV2 (reference set for distances) failed: %s", e)
    write_json(p["cache"] / "resolved_datasets.json", resolved)
    log.info("Datasets used: %s", resolved)
    return resolved
