"""Calibration metrics: ECE, Brier score, reliability-diagram data, temperature scaling.

Accuracy only asks "is the arg-max class right?". Calibration asks "when the model says 90 % confident,
is it right 90 % of the time?". A network can be 99 % accurate and still over-confident, so we report both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def reliability_table(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        rows.append(dict(bin_low=lo, bin_high=hi, count=n,
                         confidence=float(conf[m].mean()) if n else np.nan,
                         accuracy=float(correct[m].mean()) if n else np.nan))
    return pd.DataFrame(rows)


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error: weighted mean of |accuracy - confidence| over confidence bins."""
    t = reliability_table(probs, labels, n_bins)
    t = t[t["count"] > 0]
    return float((t["count"] / t["count"].sum() * (t["accuracy"] - t["confidence"]).abs()).sum())


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multi-class Brier score: mean over samples of sum_k (p_k - 1[k = y])^2 (lower is better, range 0-2)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Post-hoc temperature scaling (Guo et al., 2017). Fitted on VALIDATION logits only."""
    import torch

    lg = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0).item())
