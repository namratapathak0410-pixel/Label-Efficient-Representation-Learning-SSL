"""The four compared arms. C, D and E share one ViT architecture; only the initialisation differs.

  A_cnn_scratch  ResNet (cfg.models.cnn_arch), random init
  C_vit_imagenet ViT, ImageNet-1k supervised weights (DeiT), position embeddings resized to img_size
  D_vit_scratch  same ViT, random init
  E_vit_mae      same ViT, encoder from MAE pretraining on the target dataset's unlabelled TRAIN images only
"""
from __future__ import annotations

from pathlib import Path

import timm
import torch

ARMS = {
    "A_cnn_scratch": dict(family="cnn", init="scratch", label="ResNet scratch"),
    "C_vit_imagenet": dict(family="vit", init="imagenet", label="ViT ImageNet-1k"),
    "D_vit_scratch": dict(family="vit", init="scratch", label="ViT scratch"),
    "E_vit_mae": dict(family="vit", init="mae", label="ViT target-MAE"),
}


def create_vit(cfg: dict, num_classes: int, pretrained: bool = False, drop_path: float = 0.0):
    m = cfg["models"]
    return timm.create_model(m["vit_arch"], pretrained=pretrained, num_classes=num_classes, img_size=cfg["img_size"],
                             drop_path_rate=drop_path, global_pool=m["vit_pool"], fc_norm=False)


def mae_encoder_path(cfg: dict, dataset: str, seed: int) -> Path:
    from .utils import paths

    s = seed if cfg["mae"]["per_seed"] else 0
    return paths(cfg)["ckpt"] / dataset / f"mae_seed{s}" / "encoder.pt"


def build(arm: str, cfg: dict, num_classes: int, dataset: str, seed: int):
    a = ARMS[arm]
    dp = cfg["models"]["vit_drop_path"]
    if a["family"] == "cnn":
        return timm.create_model(cfg["models"]["cnn_arch"], pretrained=False, num_classes=num_classes), {}
    if a["init"] == "imagenet":
        model = create_vit(cfg, num_classes, pretrained=True, drop_path=dp)  # raises if weights cannot load
        return model, {"weights": cfg["models"]["vit_arch"] + " (timm, ImageNet-1k)"}
    model = create_vit(cfg, num_classes, pretrained=False, drop_path=dp)
    if a["init"] == "mae":
        p = mae_encoder_path(cfg, dataset, seed)
        if not p.exists():
            raise FileNotFoundError(f"MAE encoder missing: {p} (run `pretrain` first)")
        sd = torch.load(p, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [k for k in missing if not k.startswith(("head", "fc_norm"))]
        if bad or unexpected:
            raise RuntimeError(f"MAE encoder does not match the ViT: missing={bad} unexpected={unexpected}")
        return model, {"weights": str(p)}
    return model, {}


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def param_groups(model, arm: str, cfg: dict, lr: float, wd: float):
    """AdamW groups: no weight decay on biases/norms/tokens; layer-wise lr decay for pretrained ViTs (C and E)."""
    ld = cfg["train"]["layer_decay"] if ARMS[arm]["init"] in ("imagenet", "mae") else 1.0
    depth = len(model.blocks) if hasattr(model, "blocks") else 0
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = p.ndim <= 1 or n.endswith(".bias") or "pos_embed" in n or "cls_token" in n
        if ld < 1.0:
            if n.startswith(("cls_token", "pos_embed", "patch_embed")):
                layer = 0
            elif n.startswith("blocks."):
                layer = int(n.split(".")[1]) + 1
            else:
                layer = depth + 1
            scale = ld ** (depth + 1 - layer)
        else:
            layer, scale = 0, 1.0
        key = (layer, no_wd)
        g = groups.setdefault(key, {"params": [], "weight_decay": 0.0 if no_wd else wd, "lr_scale": scale})
        g["params"].append(p)
    for g in groups.values():
        g["lr"] = lr * g["lr_scale"]
    return list(groups.values())
