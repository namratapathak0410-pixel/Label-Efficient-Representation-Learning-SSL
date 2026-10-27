"""All plotting functions (matplotlib, saved to PNG; no display needed)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from .utils import resolve  # noqa: E402


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_class_distribution(counts: pd.Series, path):
    counts = counts.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(counts)), 5))
    ax.bar(range(len(counts)), counts.values, color="#3b78b5")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=90, fontsize=7)
    ax.set_ylabel("number of images")
    ax.set_title(f"Class distribution ({len(counts)} classes, {int(counts.sum())} images)")
    _save(fig, path)


def plot_sample_grid(df: pd.DataFrame, root, path, n_classes: int = 12, per_class: int = 3, seed: int = 0):
    rng = np.random.RandomState(seed)
    classes = sorted(df["class_name"].unique())
    pick = [classes[i] for i in sorted(rng.choice(len(classes), min(n_classes, len(classes)), replace=False))]
    fig, axes = plt.subplots(len(pick), per_class, figsize=(2.0 * per_class, 1.9 * len(pick)))
    axes = np.atleast_2d(axes)
    for r, c in enumerate(pick):
        sub = df[df["class_name"] == c]
        sel = sub.iloc[rng.permutation(len(sub))[:per_class]]
        for k in range(per_class):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(sel):
                ax.imshow(Image.open(resolve(root) / sel.iloc[k]["relpath"]).convert("RGB"))
            if k == 0:
                ax.set_title(c[:28], fontsize=7, loc="left")
    _save(fig, path)


def plot_training_curves(hist: pd.DataFrame, path, title: str = ""):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train")
    v = hist.dropna(subset=["val_loss"])
    ax[0].plot(v["epoch"], v["val_loss"], "o-", label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("cross-entropy"); ax[0].legend(); ax[0].set_title("Loss")
    ax[1].plot(hist["epoch"], hist["train_acc"], label="train")
    ax[1].plot(v["epoch"], v["val_acc"], "o-", label="val")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend(); ax[1].set_title("Accuracy")
    fig.suptitle(title, fontsize=9)
    _save(fig, path)


def plot_confusion_matrix(cm: np.ndarray, names: list[str], path, normalize: bool = True, title: str = ""):
    m = cm.astype(float)
    if normalize:
        m = m / np.maximum(m.sum(1, keepdims=True), 1)
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(5, 0.28 * n + 3), max(4.5, 0.28 * n + 2.5)))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    fs = 6 if n > 20 else 8
    ax.set_xticklabels(names, rotation=90, fontsize=fs); ax.set_yticklabels(names, fontsize=fs)
    if n <= 15:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{m[i, j]:.2f}" if normalize else int(m[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if m[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title or "Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def plot_reliability(rel: pd.DataFrame, ece: float, path, title: str = ""):
    r = rel.dropna(subset=["accuracy"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    ax[0].plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax[0].bar((r["bin_low"] + r["bin_high"]) / 2, r["accuracy"], width=(r["bin_high"] - r["bin_low"]) * 0.95,
              color="#3b78b5", alpha=0.8, label="accuracy in bin")
    ax[0].set_xlabel("confidence"); ax[0].set_ylabel("accuracy"); ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title(f"Reliability diagram (ECE = {ece:.4f})", fontsize=9)
    ax[1].bar((rel["bin_low"] + rel["bin_high"]) / 2, rel["count"], width=(rel["bin_high"] - rel["bin_low"]) * 0.95,
              color="gray")
    ax[1].set_xlabel("confidence"); ax[1].set_ylabel("# samples"); ax[1].set_title("Confidence histogram", fontsize=9)
    fig.suptitle(title, fontsize=8)
    _save(fig, path)


def plot_metric_vs_budget(num: pd.DataFrame, metric: str, ylabel: str, path, label_fn, scale: float = 100.0):
    """num has columns model_kind, budget_pct, <metric>_mean, <metric>_std (from report.py)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for kind, g in num.groupby("model_kind"):
        g = g.sort_values("budget_pct")
        m = g[f"{metric}_mean"].values * scale
        s = np.nan_to_num(g[f"{metric}_std"].values) * scale
        ax.errorbar(g["budget_pct"], m, yerr=s, marker="o", capsize=3, label=label_fn(kind))
    ax.set_xscale("log")
    ticks = sorted(num["budget_pct"].unique())
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t:g}%" for t in ticks])
    ax.set_xlabel("labelled training data (log scale)"); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    ax.set_title(f"{ylabel} vs label budget (mean ± std over seeds)", fontsize=9)
    _save(fig, path)
