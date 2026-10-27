"""Image transforms (train / eval / MAE) with the reason for every choice.

Why these augmentations for LEAF-DISEASE images?
* RandomResizedCrop(scale 0.6-1.0, ratio 0.9-1.1): small scale/position changes make the model robust to
  how the leaf is framed. We keep scale >= 0.6 because a very small crop could cut away the lesion
  (the only evidence for the class) while keeping the label.
* Horizontal AND vertical flip: a leaf has no canonical "up"; flips do not change disease texture.
* Light brightness / contrast jitter (15 %): simulates camera/lighting differences.
* NO hue / saturation jitter and NO strong colour augmentation: leaf colour (yellowing, browning,
  dark spots) IS the disease signal.
* NO heavy blur, cutout or rotation by large angles that could erase lesions or add black corners.

Evaluation preprocessing = plain resize to (img_size, img_size) + normalisation (no random operation).
We resize instead of centre-cropping so that lesions near the border are never removed.

Normalisation: ImageNet mean/std for ALL models. ImageNet-pretrained networks expect it, and using the
same statistics everywhere keeps the comparison fair (and avoids test-set statistics leaking in).
"""
from __future__ import annotations

from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BICUBIC = InterpolationMode.BICUBIC


def train_transform(img_size: int):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.9, 1.1), interpolation=BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=BICUBIC),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def mae_transform(img_size: int):
    """MAE pretraining: random crop (a bit more aggressive than fine-tuning, as in the MAE paper) + flip."""
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.4, 1.0), ratio=(0.9, 1.1), interpolation=BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def denormalize(x):
    """Tensor (B,3,H,W) normalised -> [0,1] for plotting."""
    import torch

    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)
