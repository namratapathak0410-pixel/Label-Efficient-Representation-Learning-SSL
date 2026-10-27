"""Turns results/<exp>/runs.csv into the pre-registered analyses, plots and REPORT.md.

Primary : gap(b) = macro-F1(E) - macro-F1(C), paired by seed (same labelled subset), per dataset and budget;
          crossover budget = where the mean gap changes sign (log-budget interpolation);
          association between the domain distance and the mean gap (Spearman, exact permutation p) and a
          leave-one-dataset-out prediction of the mean gap.
Secondary: winner map, ranking agreement across datasets (Kendall's W, Friedman), calibration vs budget,
          E vs D (does MAE help a ViT at all).
"""
from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .models import ARMS
from .train import collect
from .utils import get_logger, paths, write_json

log = get_logger()
METRIC = "test_macro_f1"
DIST_COLS = ["fid", "otdd", "imagenet_confidence", "linear_probe_acc"]
COLORS = {"A_cnn_scratch": "#7a7a7a", "C_vit_imagenet": "#1f77b4", "D_vit_scratch": "#ff7f0e", "E_vit_mae": "#2ca02c"}


def t_ci(x, level=0.95):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def paired_gap(df, a, b, metric=METRIC):
    """Per dataset & budget: mean(a-b) over seeds where both runs exist, with a 95 % t-interval."""
    piv = df.pivot_table(index=["dataset", "budget_pct", "seed"], columns="arm", values=metric)
    rows = []
    for (d, bud), g in piv.groupby(level=[0, 1]):
        if a not in g or b not in g:
            continue
        diff = (g[a] - g[b]).dropna().to_numpy()
        if len(diff) == 0:
            continue
        lo, hi = t_ci(diff)
        rows.append(dict(dataset=d, budget_pct=bud, gap=diff.mean(), ci_lo=lo, ci_hi=hi, n_seeds=len(diff),
                         p_value=stats.ttest_1samp(diff, 0).pvalue if len(diff) > 1 else np.nan))
    return pd.DataFrame(rows)


def crossover(g: pd.DataFrame) -> dict:
    """Budget at which the mean gap (E-C) changes sign, interpolated in log10(budget)."""
    g = g.sort_values("budget_pct")
    x, y = np.log10(g["budget_pct"].to_numpy()), g["gap"].to_numpy()
    sig = (g["ci_lo"] > 0) | (g["ci_hi"] < 0)
    for i in range(len(y) - 1):
        if np.sign(y[i]) != np.sign(y[i + 1]) and y[i] != 0:
            xc = x[i] + (x[i + 1] - x[i]) * (-y[i]) / (y[i + 1] - y[i])
            supported = bool(sig.iloc[: i + 1].any() and sig.iloc[i + 1:].any())
            return dict(crossover_budget=float(10 ** xc), direction="E overtakes C" if y[i + 1] > 0 else
                        "C overtakes E", crossover_supported_by_ci=supported, censored="")
    if (y > 0).all():
        return dict(crossover_budget=np.nan, direction="E ahead at every budget", crossover_supported_by_ci=False,
                    censored=f"<{g['budget_pct'].min():g}")
    return dict(crossover_budget=np.nan, direction="C ahead at every budget", crossover_supported_by_ci=False,
                censored=f">{g['budget_pct'].max():g}")


def mean_gap_log(g):
    """Average gap over the log-budget axis (trapezoid), one number per dataset."""
    g = g.sort_values("budget_pct")
    x = np.log10(g["budget_pct"].to_numpy())
    return float(np.trapezoid(g["gap"], x) / (x[-1] - x[0])) if len(x) > 1 else float(g["gap"].iloc[0])


def exact_spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    rho = stats.spearmanr(x, y).statistic
    perms = [stats.spearmanr(x, np.asarray(p)).statistic for p in itertools.permutations(y)]
    return float(rho), float(np.mean(np.abs(perms) >= abs(rho) - 1e-12))


def loo_prediction(x, y):
    """Leave-one-dataset-out linear prediction of y from x vs. predicting the mean of the others."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    err, base = [], []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        k = np.polyfit(x[m], y[m], 1)
        err.append(abs(np.polyval(k, x[i]) - y[i]))
        base.append(abs(y[m].mean() - y[i]))
    return float(np.mean(err)), float(np.mean(base))


def kendall_w(ranks: np.ndarray) -> float:
    """ranks: (m judges = datasets, n items = arms)."""
    m, n = ranks.shape
    s = ((ranks.sum(0) - m * (n + 1) / 2) ** 2).sum()
    return float(12 * s / (m ** 2 * (n ** 3 - n)))


def analyze(cfg):
    p = paths(cfg)
    res, figs = p["results"], p["results"] / "plots"
    figs.mkdir(exist_ok=True)
    df = collect(cfg)
    if df.empty:
        raise SystemExit("No finished runs yet.")
    dist_f = res / "domain_distance.csv"
    dist = pd.read_csv(dist_f) if dist_f.exists() else pd.DataFrame()
    datasets = sorted(df.dataset.unique(), key=lambda d: (dist.set_index("dataset").loc[d, "fid"]
                                                          if len(dist) and d in set(dist.dataset) else 0))
    arms = [a for a in cfg["arms"] if a in set(df.arm)]
    out = {}

    # ---- summary table
    agg = {}
    for col in [METRIC, "test_accuracy", "test_ece", "test_nll", "test_brier", "test_ts_ece", "test_ts_nll",
                "train_seconds"]:
        agg[col + "_mean"] = (col, "mean")
        agg[col + "_std"] = (col, "std")
    agg["n_seeds"] = ("seed", "nunique")
    agg["n_labeled"] = ("n_labeled", "mean")
    summ = df.groupby(["dataset", "arm", "budget_pct"]).agg(**agg).reset_index()
    summ.to_csv(res / "summary.csv", index=False)

    # ---- primary: E - C gap, crossover, association with distance
    gap = paired_gap(df, "E_vit_mae", "C_vit_imagenet")
    gap.to_csv(res / "gap_E_minus_C.csv", index=False)
    cross = []
    for d, g in gap.groupby("dataset"):
        cross.append(dict(dataset=d, mean_gap_log=mean_gap_log(g), **crossover(g)))
    cross = pd.DataFrame(cross)
    if len(dist):
        cross = cross.merge(dist, on="dataset", how="left")
    cross.to_csv(res / "crossover.csv", index=False)
    assoc = []
    if len(dist) and len(cross) >= 3:
        for col in DIST_COLS:
            ok = cross.dropna(subset=[col])
            rho, p_exact = exact_spearman(ok[col], ok["mean_gap_log"])
            loo, base = loo_prediction(ok[col], ok["mean_gap_log"])
            assoc.append(dict(distance=col, n_datasets=len(ok), spearman_rho=rho, exact_p=p_exact,
                              loo_abs_error=loo, loo_baseline_abs_error=base))
        pd.DataFrame(assoc).to_csv(res / "distance_association.csv", index=False)
    out["n_runs"] = len(df)

    # ---- secondary: E - D (does target MAE help a ViT at all)
    paired_gap(df, "E_vit_mae", "D_vit_scratch").to_csv(res / "gap_E_minus_D.csv", index=False)
    paired_gap(df, "E_vit_mae", "A_cnn_scratch").to_csv(res / "gap_E_minus_A.csv", index=False)

    # ---- winner map + ranking agreement
    piv = summ.pivot_table(index=["budget_pct", "dataset"], columns="arm", values=METRIC + "_mean")
    winners = piv.idxmax(axis=1).rename("winner").reset_index()
    winners.to_csv(res / "winners.csv", index=False)
    agree = []
    for col, better in [(METRIC + "_mean", "high"), ("test_ece_mean", "low"), ("test_ts_ece_mean", "low")]:
        pv = summ.pivot_table(index=["budget_pct", "dataset"], columns="arm", values=col)
        for bud, g in pv.groupby(level=0):
            g = g.dropna()
            if len(g) < 2 or g.shape[1] < 3:
                continue
            r = (g.rank(axis=1, ascending=(better == "low"))).to_numpy()
            fr = stats.friedmanchisquare(*g.to_numpy().T) if len(g) >= 2 else None
            agree.append(dict(metric=col.replace("_mean", ""), budget_pct=bud, n_datasets=len(g), kendall_w=kendall_w(r),
                              friedman_p=float(fr.pvalue) if fr else np.nan,
                              same_winner_everywhere=bool(len(set(g.idxmax(axis=1) if better == "high"
                                                                 else g.idxmin(axis=1))) == 1)))
    pd.DataFrame(agree).to_csv(res / "ranking_agreement.csv", index=False)

    # ---- plots
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        for a in arms:
            s = summ[(summ.dataset == d) & (summ.arm == a)].sort_values("budget_pct")
            ax.errorbar(s.budget_pct, s[METRIC + "_mean"], yerr=s[METRIC + "_std"], marker="o", ms=4, capsize=3,
                        label=ARMS[a]["label"], color=COLORS[a])
        ax.set_xscale("log"); ax.set_title(d); ax.set_xlabel("labelled training images (%)")
        ax.set_xticks(cfg["train"]["budgets"]); ax.set_xticklabels([f"{b:g}" for b in cfg["train"]["budgets"]])
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("test macro-F1"); axes[0][-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "learning_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for d in datasets:
        g = gap[gap.dataset == d].sort_values("budget_pct")
        ax.plot(g.budget_pct, g.gap, marker="o", label=d)
        ax.fill_between(g.budget_pct, g.ci_lo, g.ci_hi, alpha=0.15)
    ax.axhline(0, color="k", lw=0.8); ax.set_xscale("log")
    ax.set_xticks(cfg["train"]["budgets"]); ax.set_xticklabels([f"{b:g}" for b in cfg["train"]["budgets"]])
    ax.set_xlabel("labelled training images (%)"); ax.set_ylabel("macro-F1: target-MAE minus ImageNet")
    ax.set_title("Where does target-domain MAE overtake ImageNet?"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(figs / "gap_E_minus_C.png", dpi=150); plt.close(fig)

    if len(dist) and len(cross):
        fig, axes = plt.subplots(1, len(DIST_COLS), figsize=(4 * len(DIST_COLS), 3.5))
        for ax, col in zip(axes, DIST_COLS):
            ax.scatter(cross[col], cross.mean_gap_log)
            for _, r in cross.iterrows():
                ax.annotate(r.dataset, (r[col], r.mean_gap_log), fontsize=8)
            ax.axhline(0, color="k", lw=0.8); ax.set_xlabel(col); ax.grid(alpha=0.3)
        axes[0].set_ylabel("mean gap E-C (log-budget avg)")
        fig.tight_layout(); fig.savefig(figs / "distance_vs_gap.png", dpi=150); plt.close(fig)

    for col, name in [("test_ece", "ece"), ("test_ts_ece", "ece_after_temperature"), ("test_nll", "nll")]:
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False)
        for ax, d in zip(axes[0], datasets):
            for a in arms:
                s = summ[(summ.dataset == d) & (summ.arm == a)].sort_values("budget_pct")
                ax.errorbar(s.budget_pct, s[col + "_mean"], yerr=s[col + "_std"], marker="o", ms=4, capsize=3,
                            label=ARMS[a]["label"], color=COLORS[a])
            ax.set_xscale("log"); ax.set_title(d); ax.grid(alpha=0.3); ax.set_xlabel("labelled (%)")
        axes[0][0].set_ylabel(name); axes[0][-1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figs / f"{name}_vs_budget.png", dpi=150); plt.close(fig)

    wm = winners.pivot(index="dataset", columns="budget_pct", values="winner").reindex(datasets)
    fig, ax = plt.subplots(figsize=(1.3 * wm.shape[1] + 2, 0.6 * len(wm) + 1.2))
    codes = {a: i for i, a in enumerate(arms)}
    ax.imshow(wm.map(lambda v: codes.get(v, -1)).to_numpy(dtype=float), cmap="tab10", vmin=0, vmax=9, aspect="auto")
    for i, d in enumerate(wm.index):
        for j, b in enumerate(wm.columns):
            v = wm.loc[d, b]
            ax.text(j, i, ARMS[v]["label"] if isinstance(v, str) else "", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(wm.shape[1])); ax.set_xticklabels([f"{b:g}%" for b in wm.columns])
    ax.set_yticks(range(len(wm))); ax.set_yticklabels(wm.index); ax.set_title("Best arm (mean test macro-F1)")
    fig.tight_layout(); fig.savefig(figs / "winner_map.png", dpi=150); plt.close(fig)

    write_report(cfg, summ, gap, cross, pd.DataFrame(assoc), pd.DataFrame(agree), dist, df)
    write_json(res / "analysis_meta.json", out)
    log.info("analysis written to %s", res)


def _md(df: pd.DataFrame, floatfmt=".3f") -> str:
    if df is None or df.empty:
        return "_not available yet_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(format(v, floatfmt) if isinstance(v, (float, np.floating)) and not np.isnan(v)
                                       else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_report(cfg, summ, gap, cross, assoc, agree, dist, df):
    p = paths(cfg)
    s = summ.copy()
    s["macro_f1"] = s.apply(lambda r: f"{r[METRIC + '_mean']:.3f} ± {r[METRIC + '_std']:.3f}"
                            if r.n_seeds > 1 else f"{r[METRIC + '_mean']:.3f}", axis=1)
    s["ECE"] = s.apply(lambda r: f"{r['test_ece_mean']:.3f}", axis=1)
    table = s.pivot_table(index=["dataset", "budget_pct"], columns="arm", values="macro_f1", aggfunc="first")
    seeds = sorted(int(s) for s in df.seed.unique())
    txt = f"""# Results: when does target-domain MAE overtake ImageNet initialisation?

Generated automatically from `runs.csv` ({len(df)} fine-tuning runs, seeds {seeds}). Every number below comes from
the logs; nothing is typed by hand. Test sets were evaluated once per run after restoring the best-validation weights.

## 1. Domain distance from ImageNet (computed before training)
{_md(dist)}

## 2. Primary result: macro-F1 gap, target-MAE (E) minus ImageNet (C), paired by seed
{_md(gap)}

### Crossover budget and mean gap per dataset
{_md(cross[[c for c in ['dataset', 'mean_gap_log', 'crossover_budget', 'censored', 'direction',
                       'crossover_supported_by_ci'] if c in cross]])}

### Does distance predict the gap? (exploratory: one point per dataset)
{_md(assoc)}

With n datasets the smallest attainable exact p-value is 2/n! (= 0.083 for n = 4); treat this as descriptive.

## 3. Test macro-F1 (mean ± sd over seeds)
{_md(table.reset_index())}

## 4. Does the winner replicate across datasets?
{_md(agree)}

## Figures
`plots/learning_curves.png`, `plots/gap_E_minus_C.png`, `plots/distance_vs_gap.png`, `plots/winner_map.png`,
`plots/ece_vs_budget.png`, `plots/ece_after_temperature_vs_budget.png`, `plots/nll_vs_budget.png`;
MAE reconstructions in `checkpoints/<experiment>/<dataset>/mae_seed*/reconstructions.png`.
"""
    (p["results"] / "REPORT.md").write_text(txt, encoding="utf-8")
