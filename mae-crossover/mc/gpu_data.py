"""Dataset held on the GPU as uint8 + per-sample augmentation on the GPU (no JPEG decoding / DataLoader workers).

Augmentation is identical for every model and every dataset (declared in PREREGISTRATION.md):
  fine-tuning : random resized crop (scale 0.35-1, ratio 3/4-4/3), horizontal flip, brightness/contrast +-20 %
  MAE         : random resized crop (scale 0.2-1), horizontal flip (as in He et al. 2022)
  evaluation  : none (the cached image is already the centre crop)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from .utils import IMAGENET_MEAN, IMAGENET_STD


class GPUImages:
    def __init__(self, images: np.ndarray, labels: np.ndarray, dev: torch.device):
        self.x = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous().to(dev)  # uint8 N,3,H,W
        self.y = torch.from_numpy(labels).long().to(dev)
        self.mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.y)

    def norm(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return (x_uint8.float() / 255.0 - self.mean) / self.std

    def batch(self, idx: torch.Tensor, aug: str | None, gen: torch.Generator | None = None):
        x = self.x[idx].float() / 255.0
        if aug == "finetune":
            x = random_resized_crop(x, (0.35, 1.0), gen)
            x = color_jitter(x, 0.2, 0.2, gen)
        elif aug == "mae":
            x = random_resized_crop(x, (0.2, 1.0), gen)
        return (x - self.mean) / self.std, self.y[idx]


def _rand(n, gen, dev):
    return torch.rand(n, device=dev, generator=gen)


def random_resized_crop(x: torch.Tensor, scale, gen, ratio=(3 / 4, 4 / 3)) -> torch.Tensor:
    """Per-sample RandomResizedCrop + horizontal flip via one affine grid_sample."""
    n, dev = x.shape[0], x.device
    area = scale[0] + (scale[1] - scale[0]) * _rand(n, gen, dev)
    logr = math.log(ratio[0]) + (math.log(ratio[1]) - math.log(ratio[0])) * _rand(n, gen, dev)
    r = torch.exp(logr)
    w = torch.sqrt(area * r).clamp(max=1.0)  # crop size as a fraction of the image side
    h = torch.sqrt(area / r).clamp(max=1.0)
    cx = (1 - w) * (2 * _rand(n, gen, dev) - 1)  # centre in [-1+w, 1-w] (grid coordinates)
    cy = (1 - h) * (2 * _rand(n, gen, dev) - 1)
    flip = torch.where(_rand(n, gen, dev) < 0.5, -1.0, 1.0)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = w * flip
    theta[:, 0, 2] = cx
    theta[:, 1, 1] = h
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def color_jitter(x: torch.Tensor, brightness: float, contrast: float, gen) -> torch.Tensor:
    n, dev = x.shape[0], x.device
    b = 1 + brightness * (2 * _rand(n, gen, dev) - 1)
    c = 1 + contrast * (2 * _rand(n, gen, dev) - 1)
    x = x * b.view(n, 1, 1, 1)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - m) * c.view(n, 1, 1, 1) + m).clamp(0, 1)
