"""Domain distance between ImageNet and each target dataset, computed BEFORE any fine-tuning result exists.

Reference set: ImageNetV2 matched-frequency (10k images, public stand-in for ImageNet).
Feature extractor: the frozen ImageNet-1k ViT used by arm C (timm default cls-token pooling, same img_size).
Only TRAIN-split images of the target are used; validation is used only to score the linear probe.

  fid                 Frechet distance between Gaussian fits of the two feature clouds (label-free)
  otdd                optimal-transport dataset distance (Alvarez-Melis & Fusi 2020) with a diagonal-Gaussian label
                      cost, in a 64-d PCA space, on 1,500 random samples per side (label-aware)
  imagenet_confidence mean max-softmax of the ImageNet classifier on target images (label-free; high = ImageNet-like)
  linear_probe_acc    validation accuracy of logistic regression on frozen ImageNet features, 100 % train labels
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import timm
import torch
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .data import load_cache
from .gpu_data import GPUImages
from .split import make_split
from .utils import device, get_logger, paths, resolved_datasets, write_json

log = get_logger()


@torch.no_grad()
def features(model, images: np.ndarray, bs=256):
    dev = device()
    g = GPUImages(images, np.zeros(len(images), dtype=np.int64), dev)
    feats, conf = [], []
    for i in range(0, len(g), bs):
        x, _ = g.batch(torch.arange(i, min(i + bs, len(g)), device=dev), None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.forward_head(model.forward_features(x), pre_logits=True)
            logits = model.head(f)
        feats.append(f.float().cpu())
        conf.append(logits.float().softmax(-1).max(-1).values.cpu())
    return torch.cat(feats).numpy().astype(np.float64), torch.cat(conf).numpy()


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def otdd(xa, ya, xb, yb, n=1500, seed=0):
    import ot

    rng = np.random.default_rng(seed)

    def stats(x, y):
        return {c: (x[y == c].mean(0), x[y == c].std(0) + 1e-6) for c in np.unique(y)}

    sa, sb = stats(xa, ya), stats(xb, yb)
    ia = rng.choice(len(xa), min(n, len(xa)), replace=False)
    ib = rng.choice(len(xb), min(n, len(xb)), replace=False)
    ca, cb = sorted(sa), sorted(sb)
    ma, va = np.stack([sa[c][0] for c in ca]), np.stack([sa[c][1] for c in ca])
    mb, vb = np.stack([sb[c][0] for c in cb]), np.stack([sb[c][1] for c in cb])
    # W2^2 between diagonal Gaussians of every (target class, ImageNet class) pair
    label_cost = ((ma[:, None] - mb[None]) ** 2).sum(-1) + ((va[:, None] - vb[None]) ** 2).sum(-1)
    pa = np.searchsorted(ca, ya[ia])
    pb = np.searchsorted(cb, yb[ib])
    feat_cost = ((xa[ia][:, None] - xb[ib][None]) ** 2).sum(-1)
    m = feat_cost + label_cost[pa][:, pb]
    w = np.full(len(ia), 1 / len(ia)), np.full(len(ib), 1 / len(ib))
    return float(np.sqrt(ot.emd2(w[0], w[1], m, numItermax=2_000_000)))


def compute(cfg):
    p = paths(cfg)
    out = p["results"] / "domain_distance.csv"
    model = timm.create_model(cfg["models"]["vit_arch"], pretrained=True, img_size=cfg["img_size"]).to(device()).eval()
    ref = load_cache("imagenetv2", cfg)
    f_ref, conf_ref = features(model, ref["images"])
    rows = []
    for slot, name in resolved_datasets(cfg).items():
        c, df = load_cache(name, cfg), make_split(name, cfg)
        tr = df.loc[df.split == "train", "idx"].to_numpy()
        va = df.loc[df.split == "val", "idx"].to_numpy()
        f_tr, conf_tr = features(model, c["images"][tr])
        f_va, _ = features(model, c["images"][va])
        pca = PCA(64, random_state=0).fit(np.concatenate([f_tr, f_ref]))
        sc = StandardScaler().fit(f_tr)
        probe = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(f_tr), c["labels"][tr])
        rows.append(dict(
            domain=slot, dataset=name, fid=frechet(f_tr, f_ref),
            otdd=otdd(pca.transform(f_tr), c["labels"][tr], pca.transform(f_ref), ref["labels"]),
            imagenet_confidence=float(conf_tr.mean()),
            linear_probe_acc=float((probe.predict(sc.transform(f_va)) == c["labels"][va]).mean()),
            n_train=len(tr), n_classes=int(len(np.unique(c["labels"])))))
        log.info("[distance] %s", rows[-1])
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    write_json(p["results"] / "domain_distance_meta.json", dict(
        reference="ImageNetV2 matched-frequency", n_reference=len(f_ref), reference_self_confidence=float(conf_ref.mean()),
        extractor=cfg["models"]["vit_arch"], img_size=cfg["img_size"]))
    return d
