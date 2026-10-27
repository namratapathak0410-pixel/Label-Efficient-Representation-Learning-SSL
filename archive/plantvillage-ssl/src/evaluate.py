"""Prediction + metrics (accuracy, macro P/R/F1, per-class report, confusion matrix, ECE, Brier)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .calibration import brier_score, ece_score, softmax_np


@torch.no_grad()
def predict(model, loader, device, amp: bool = False):
    """Return (logits [N,C] float32, labels [N]) for a labelled loader."""
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(x)
        logits.append(out.float().cpu())
        ys.append(y)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def macro_f1(y_true, y_pred) -> float:
    """Macro-F1 over the classes that occur in y_true (used for validation / early stopping)."""
    labs = np.unique(y_true)
    return float(precision_recall_fscore_support(y_true, y_pred, labels=labs, average="macro", zero_division=0)[2])


def compute_metrics(logits: np.ndarray, y: np.ndarray, class_names: list[str]) -> dict:
    probs = softmax_np(logits)
    pred = probs.argmax(1)
    labs = np.unique(y)
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labs, average="macro", zero_division=0)
    pc_p, pc_r, pc_f, sup = precision_recall_fscore_support(y, pred, labels=labs, average=None, zero_division=0)
    per_class = pd.DataFrame({"class": [class_names[i] for i in labs], "precision": pc_p, "recall": pc_r,
                              "f1": pc_f, "support": sup})
    cm = confusion_matrix(y, pred, labels=range(len(class_names)))
    return dict(
        acc=float(accuracy_score(y, pred)), macro_precision=float(p), macro_recall=float(r), macro_f1=float(f),
        ece=ece_score(probs, y), brier=brier_score(probs, y),
        mean_confidence=float(probs.max(1).mean()),
        per_class=per_class, confusion_matrix=cm,
    )
