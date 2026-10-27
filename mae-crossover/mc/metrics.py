"""Classification and calibration metrics computed from logits."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def softmax_np(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = logits / t
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(probs, y, n_bins=15, adaptive=False) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    if adaptive:  # equal-mass bins
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = ids == b
        if m.any():
            total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)


def reliability_bins(probs, y, n_bins=15):
    conf, acc = probs.max(1), (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    return [dict(bin=b, count=int((ids == b).sum()),
                 conf=float(conf[ids == b].mean()) if (ids == b).any() else None,
                 acc=float(acc[ids == b].mean()) if (ids == b).any() else None) for b in range(n_bins)]


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature that minimises NLL on the given (validation) logits."""
    z, t = torch.tensor(logits, dtype=torch.float64), torch.tensor(y)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0))


def evaluate_logits(logits: np.ndarray, y: np.ndarray, n_classes: int, temperature: float = 1.0) -> dict:
    probs = softmax_np(logits, temperature)
    pred = probs.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, labels=np.arange(n_classes), average="macro",
                                                  zero_division=0)
    _, rc, _, _ = precision_recall_fscore_support(y, pred, labels=np.arange(n_classes), average=None, zero_division=0)
    onehot = np.eye(n_classes)[y]
    return dict(
        accuracy=float((pred == y).mean()), macro_precision=float(p), macro_recall=float(r), macro_f1=float(f1),
        min_class_recall=float(rc.min()), nll=float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean()),
        brier=float(((probs - onehot) ** 2).sum(1).mean()), ece=ece(probs, y), adaptive_ece=ece(probs, y, adaptive=True),
        mean_confidence=float(probs.max(1).mean()))


def full_report(logits_val, y_val, logits_test, y_test, n_classes) -> dict:
    t = fit_temperature(logits_val, y_val)
    out = {f"test_{k}": v for k, v in evaluate_logits(logits_test, y_test, n_classes).items()}
    ts = evaluate_logits(logits_test, y_test, n_classes, t)
    out.update({f"test_ts_{k}": ts[k] for k in ("nll", "brier", "ece", "adaptive_ece")})
    out["temperature"] = t
    out.update({f"val_{k}": v for k, v in evaluate_logits(logits_val, y_val, n_classes).items()
                if k in ("accuracy", "macro_f1", "nll", "ece")})
    return out


def per_class(logits, y, n_classes):
    pred = logits.argmax(1)
    p, r, f1, s = precision_recall_fscore_support(y, pred, labels=np.arange(n_classes), zero_division=0)
    return dict(precision=p.tolist(), recall=r.tolist(), f1=f1.tolist(), support=s.tolist(),
                confusion=confusion_matrix(y, pred, labels=np.arange(n_classes)).tolist())
