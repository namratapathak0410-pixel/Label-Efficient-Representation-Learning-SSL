"""The five compared models (A-E) built with timm.

A  CNN from scratch                      (random init)
B  CNN, ImageNet-pretrained, fine-tuned  (supervised pretraining on ANOTHER domain: ImageNet-1k)
C  ViT, ImageNet-pretrained, fine-tuned
D  ViT from scratch
E  ViT whose encoder was pretrained with MAE on the UNLABELLED TRAIN images of the target dataset

ImageNet pretraining vs target-domain self-supervised pretraining (NOT equivalent):
  * ImageNet pretraining uses ~1.3 M *labelled* images of everyday objects -> the labels (and a huge amount of
    extra data) come from outside the project. It tests "does a big external labelled dataset help?".
  * MAE pretraining uses only the unlabelled training images of the SAME domain (leaves) and no external data.
    It tests "can unlabelled in-domain data replace/compete with external labelled data?".
  They differ in data source, data amount, label usage and compute cost, so differences between C and E
  cannot be attributed to the pretraining *algorithm* alone.
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn

from .utils import get_logger

log = get_logger()

MODEL_KINDS = {
    "A_cnn_scratch": dict(family="cnn", pretrained=False, mae=False, pretraining="None (random init)", short="A"),
    "B_cnn_pretrained": dict(family="cnn", pretrained=True, mae=False, pretraining="ImageNet supervised", short="B"),
    "C_vit_pretrained": dict(family="vit", pretrained=True, mae=False, pretraining="ImageNet supervised", short="C"),
    "D_vit_scratch": dict(family="vit", pretrained=False, mae=False, pretraining="None (random init)", short="D"),
    "E_vit_mae": dict(family="vit", pretrained=False, mae=True, pretraining="Target-domain MAE (self-supervised)",
                      short="E"),
}


def model_display_name(kind: str, cfg: dict) -> str:
    info = MODEL_KINDS[kind]
    arch = cfg["models"]["cnn_arch"] if info["family"] == "cnn" else cfg["models"]["vit_arch"]
    arch = arch.replace("_patch16_224", "/16")
    suffix = {"A": "scratch", "B": "ImageNet FT", "C": "ImageNet FT", "D": "scratch", "E": "MAE FT"}[info["short"]]
    return f"{info['short']}: {arch} ({suffix})"


def create_vit(arch: str, num_classes: int, img_size: int, drop_path: float = 0.0, pretrained: bool = False):
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes, img_size=img_size,
                             drop_path_rate=drop_path)


def create_cnn(arch: str, num_classes: int, pretrained: bool = False):
    return timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)


def mae_encoder_path(cfg: dict) -> Path:
    from .utils import get_paths

    return get_paths(cfg)["mae"] / "encoder.pt"


def build_model(kind: str, cfg: dict, num_classes: int):
    """Returns (model, info). info['pretrained_loaded'] tells whether pretrained weights were really loaded."""
    info = MODEL_KINDS[kind]
    m, d = cfg["models"], cfg["data"]
    loaded = None
    if info["family"] == "cnn":
        if info["pretrained"]:
            try:
                model = create_cnn(m["cnn_arch"], num_classes, pretrained=True)
                loaded = True
            except Exception as e:
                if m.get("strict_pretrained", True):
                    raise RuntimeError(f"Could not load ImageNet weights for {m['cnn_arch']}: {e}") from e
                log.warning("Pretrained weights unavailable (%s) -> RANDOM INIT used (recorded in results)", e)
                model, loaded = create_cnn(m["cnn_arch"], num_classes, False), False
        else:
            model = create_cnn(m["cnn_arch"], num_classes, False)
    else:
        dp = m.get("vit_drop_path", 0.0)
        if info["pretrained"]:
            try:
                model = create_vit(m["vit_arch"], num_classes, d["img_size"], dp, pretrained=True)
                loaded = True
            except Exception as e:
                if m.get("strict_pretrained", True):
                    raise RuntimeError(f"Could not load ImageNet weights for {m['vit_arch']}: {e}") from e
                log.warning("Pretrained weights unavailable (%s) -> RANDOM INIT used (recorded in results)", e)
                model, loaded = create_vit(m["vit_arch"], num_classes, d["img_size"], dp, False), False
        else:
            model = create_vit(m["vit_arch"], num_classes, d["img_size"], dp, False)
        if info["mae"]:
            ck = mae_encoder_path(cfg)
            if not ck.exists():
                raise FileNotFoundError(f"MAE encoder checkpoint missing: {ck}\nRun: python main.py pretrain --config <config>")
            state = torch.load(ck, map_location="cpu")
            res = model.load_state_dict(state, strict=False)
            bad_missing = [k for k in res.missing_keys if not k.startswith(("head", "fc_norm"))]
            assert not res.unexpected_keys and not bad_missing, f"MAE weights do not fit: {res}"
            loaded = True
    return model, dict(pretrained_loaded=loaded)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Size of the parameters stored as float32 (MB)."""
    return count_params(model) * 4 / 1e6
