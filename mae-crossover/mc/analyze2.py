"""Analysis of experiment 2 (PREREGISTRATION_V2.md): 7 arms, hypotheses H6-H11, tables, plots, REPORT_V2.md.

Same conventions as analyze.py (gaps paired by seed, 95 % t-intervals, crossover on a log-budget axis, exact Spearman).
Every number in the report is computed here from runs.csv; nothing is typed by hand.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .analyze import (DIST_COLS, METRIC, _md, crossover, exact_spearman, kendall_w, loo_prediction, mean_gap_log,
                      paired_gap, t_ci)
from .models import ARMS
from .train import collect
from .utils import ROOT, get_logger, paths, write_json

log = get_logger()
COLORS = {"A_cnn_scratch": "#7a7a7a", "A2_cnn34_scratch": "#333333", "C_vit_imagenet": "#1f77b4",
          "D_vit_scratch": "#ff7f0e", "E_vit_mae": "#2ca02c", "E2_vit_mae_big": "#9467bd",
          "F_vit_imagenet_mae": "#d62728"}
OLD4 = ["A_cnn_scratch", "C_vit_imagenet", "D_vit_scratch", "E_vit_mae"]
PAIRS = [("E2_vit_mae_big", "E_vit_mae"), ("E2_vit_mae_big", "C_vit_imagenet"), ("F_vit_imagenet_mae", "C_vit_imagenet"),
         ("F_vit_imagenet_mae", "E2_vit_mae_big"), ("F_vit_imagenet_mae", "E_vit_mae"),
         ("A2_cnn34_scratch", "A_cnn_scratch"), ("E2_vit_mae_big", "A2_cnn34_scratch"), ("E_vit_mae", "C_vit_imagenet")]


def short(arm):
    return arm.split("_")[0]


def seed_mean_gap(df, a, b, dataset):
    """Per seed: gap a-b averaged over budgets -> (mean, ci_lo, ci_hi, n_seeds)."""
    piv = df[df.dataset == dataset].pivot_table(index=["budget_pct", "seed"], columns="arm", values=METRIC)
    if a not in piv or b not in piv:
        return None
    d = (piv[a] - piv[b]).dropna().groupby(level="seed").mean().to_numpy()
    lo, hi = t_ci(d)
    return float(d.mean()), float(lo), float(hi), len(d)


def hypotheses(cfg, df, summ, gaps, dist, assoc_f):
    rows = []
    big = sorted(set(df[df.arm == "E2_vit_mae_big"].dataset))

    # H6: E2 > E, averaged over budgets and seeds, in each pooled dataset
    parts = []
    for d in big:
        r = seed_mean_gap(df, "E2_vit_mae_big", "E_vit_mae", d)
        parts.append(f"{d} {r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]" if r else f"{d} n/a")
    ok = [seed_mean_gap(df, "E2_vit_mae_big", "E_vit_mae", d) for d in big]
    cells = gaps[("E2_vit_mae_big", "E_vit_mae")]
    rows.append(("H6", "E2 > E (budget- and seed-averaged) in each of PlantVillage, CIFAR-10, EuroSAT",
                 "; ".join(parts) + f"; E2>E in {(cells.gap > 0).sum()}/{len(cells)} cells",
                 "supported" if ok and all(r and r[0] > 0 for r in ok) else "not supported"))

    # H7: E2-C larger than E-C at every budget, and E2 overtakes C somewhere (CI excludes 0)
    g2, g1 = gaps[("E2_vit_mae_big", "C_vit_imagenet")], gaps[("E_vit_mae", "C_vit_imagenet")]
    m = g2.merge(g1, on=["dataset", "budget_pct"], suffixes=("_e2", "_e"))
    larger = int((m.gap_e2 > m.gap_e).sum())
    over = g2[g2.ci_lo > 0]
    cells_txt = ", ".join(f"{r.dataset} {r.budget_pct:g}% ({r.gap:+.3f})" for r in over.itertuples()) or "none"
    best = g2.loc[g2.gap.idxmax()]
    rows.append(("H7", "E2-C > E-C at every budget; E2 overtakes C (CI excludes 0) somewhere",
                 f"E2-C > E-C in {larger}/{len(m)} cells; cells with E2 significantly above C: {cells_txt}; "
                 f"best E2-C gap {best.gap:+.3f} ({best.dataset}, {best.budget_pct:g}%)",
                 "supported" if larger == len(m) and len(over) else
                 "partly supported" if larger == len(m) or len(over) else "not supported"))

    # H8: F beats C in more than half of the 20 cells; gain larger for farther domains
    gf = gaps[("F_vit_imagenet_mae", "C_vit_imagenet")]
    nwin, sig = int((gf.gap > 0).sum()), int((gf.ci_lo > 0).sum())
    txt = f"F>C in {nwin}/{len(gf)} cells ({sig} with CI above 0, {int((gf.ci_hi < 0).sum())} with CI below 0)"
    if assoc_f is not None and len(assoc_f):
        txt += "; Spearman of mean F-C gap with distance: " + ", ".join(
            f"{r.distance} {r.spearman_rho:+.2f} (p={r.exact_p:.2f})" for r in assoc_f.itertuples())
    rows.append(("H8", "F > C in more than half of the 20 cells; gain larger in farther domains", txt,
                 "supported" if nwin > len(gf) / 2 else "not supported"))

    # H9: F >= E2 in the three pooled datasets
    parts, okk = [], []
    for d in big:
        r = seed_mean_gap(df, "F_vit_imagenet_mae", "E2_vit_mae_big", d)
        if r:
            parts.append(f"{d} {r[0]:+.3f} [{r[1]:+.3f}, {r[2]:+.3f}]")
            okk.append(r[0] >= 0)
    rows.append(("H9", "F >= E2 (budget- and seed-averaged) in the three pooled datasets", "; ".join(parts),
                 "supported" if okk and all(okk) else "not supported"))

    # H10: ResNet-34 vs ResNet-18
    ga = gaps[("A2_cnn34_scratch", "A_cnn_scratch")]
    small = int((ga.gap.abs() < 0.03).sum())
    piv = summ.pivot_table(index=["dataset", "budget_pct"], columns="arm", values=METRIC + "_mean")
    mae_best = piv[[c for c in ("E_vit_mae", "E2_vit_mae_big") if c in piv]].max(axis=1)
    flips = int(((mae_best > piv["A_cnn_scratch"]) != (mae_best > piv["A2_cnn34_scratch"])).sum())
    rows.append(("H10", "A2-A within +-0.03 in most cells; MAE-vs-ResNet verdict unchanged",
                 f"|A2-A| < 0.03 in {small}/{len(ga)} cells (range {ga.gap.min():+.3f} to {ga.gap.max():+.3f}); "
                 f"best MAE arm beats ResNet-18 in {int((mae_best > piv['A_cnn_scratch']).sum())}/{len(piv)} cells, "
                 f"ResNet-34 in {int((mae_best > piv['A2_cnn34_scratch']).sum())}/{len(piv)}; verdict flips in {flips} cells",
                 "supported" if small > len(ga) / 2 and flips == 0 else "partly supported" if small > len(ga) / 2
                 else "not supported"))

    # H11: winner among A, C, D, E unchanged by the wider lr grid
    old = piv[OLD4]
    win = old.idxmax(axis=1)
    notc = win[win != "C_vit_imagenet"]
    rows.append(("H11", "Wider lr grid does not change the winner among A, C, D, E (v1: C won 20/20)",
                 f"C wins {(win == 'C_vit_imagenet').sum()}/{len(win)} cells" +
                 ("" if notc.empty else "; others: " + ", ".join(f"{d} {b:g}% -> {short(a)}" for (d, b), a in notc.items())),
                 "supported" if notc.empty else "not supported"))
    return pd.DataFrame(rows, columns=["id", "hypothesis", "result", "verdict"])


def analyze2(cfg):
    p = paths(cfg)
    res, figs = p["results"], p["results"] / "plots"
    figs.mkdir(exist_ok=True)
    df = collect(cfg)
    if df.empty:
        raise SystemExit("No finished runs yet.")
    dist = pd.read_csv(ROOT / "results" / cfg["reuse_from"] / "domain_distance.csv")
    datasets = sorted(df.dataset.unique(), key=lambda d: dist.set_index("dataset").loc[d, "fid"])
    arms = [a for a in cfg["arms"] if a in set(df.arm)]

    agg = {}
    for col in [METRIC, "test_accuracy", "test_ece", "test_nll", "test_ts_ece", "train_seconds"]:
        agg[col + "_mean"], agg[col + "_std"] = (col, "mean"), (col, "std")
    agg["n_seeds"], agg["lr"] = ("seed", "nunique"), ("lr", "first")
    summ = df.groupby(["dataset", "arm", "budget_pct"]).agg(**agg).reset_index()
    summ.to_csv(res / "summary.csv", index=False)

    gaps = {}
    for a, b in PAIRS:
        g = paired_gap(df, a, b)
        gaps[(a, b)] = g
        g.to_csv(res / f"gap_{short(a)}_minus_{short(b)}.csv", index=False)

    cross = []
    for (a, b) in [("E_vit_mae", "C_vit_imagenet"), ("E2_vit_mae_big", "C_vit_imagenet"),
                   ("F_vit_imagenet_mae", "C_vit_imagenet")]:
        for d, g in gaps[(a, b)].groupby("dataset"):
            cross.append(dict(pair=f"{short(a)}-{short(b)}", dataset=d, mean_gap_log=mean_gap_log(g),
                              **crossover(g)))
    cross = pd.DataFrame(cross).merge(dist, on="dataset", how="left")
    cross.to_csv(res / "crossover.csv", index=False)

    assoc_rows = []
    for pair in cross.pair.unique():
        sub = cross[cross.pair == pair]
        if len(sub) < 3:
            continue
        for col in DIST_COLS:
            rho, pe = exact_spearman(sub[col], sub["mean_gap_log"])
            loo, base = loo_prediction(sub[col], sub["mean_gap_log"])
            assoc_rows.append(dict(pair=pair, distance=col, n_datasets=len(sub), spearman_rho=rho, exact_p=pe,
                                   loo_abs_error=loo, loo_baseline_abs_error=base))
    assoc = pd.DataFrame(assoc_rows)
    assoc.to_csv(res / "distance_association.csv", index=False)

    # winners and ranking agreement, over all arms available in a cell
    piv = summ.pivot_table(index=["budget_pct", "dataset"], columns="arm", values=METRIC + "_mean")
    winners = piv.idxmax(axis=1).rename("winner").reset_index()
    winners.to_csv(res / "winners.csv", index=False)
    agree = []
    for col, better in [(METRIC + "_mean", "high"), ("test_ece_mean", "low"), ("test_ts_ece_mean", "low")]:
        pv = summ.pivot_table(index=["budget_pct", "dataset"], columns="arm", values=col)
        for bud, g in pv.groupby(level=0):
            g = g.dropna(axis=1)  # arms present in every dataset (E2 is missing for Pets)
            if len(g) >= 2 and g.shape[1] >= 3:
                r = g.rank(axis=1, ascending=(better == "low")).to_numpy()
                fr = stats.friedmanchisquare(*g.to_numpy().T)
                agree.append(dict(metric=col.replace("_mean", ""), budget_pct=bud, n_datasets=len(g),
                                  n_arms=g.shape[1], kendall_w=kendall_w(r), friedman_p=float(fr.pvalue)))
    agree = pd.DataFrame(agree)
    agree.to_csv(res / "ranking_agreement.csv", index=False)

    lrc = res / "lr_changes_vs_v1.csv"
    lr_changes = pd.read_csv(lrc) if lrc.exists() else pd.DataFrame()
    hyp = hypotheses(cfg, df, summ, gaps, dist, assoc[assoc.pair == "F-C"] if len(assoc) else None)
    hyp.to_csv(res / "hypotheses_v2.csv", index=False)

    # calibration summary per arm
    cal = summ.groupby("arm").agg(ece=("test_ece_mean", "mean"), ece_after_ts=("test_ts_ece_mean", "mean"),
                                  nll=("test_nll_mean", "mean"), macro_f1=(METRIC + "_mean", "mean")).reindex(arms)
    cal.to_csv(res / "calibration_by_arm.csv")

    _plots(cfg, summ, gaps, cross, datasets, arms, winners, figs)
    _report(cfg, df, summ, gaps, cross, assoc, agree, hyp, lr_changes, cal, dist, arms)
    write_json(res / "analysis_meta.json", dict(n_runs=len(df), arms=arms))
    log.info("analysis written to %s", res)


def _plots(cfg, summ, gaps, cross, datasets, arms, winners, figs):
    ticks = cfg["train"]["budgets"]

    def xaxis(ax):
        ax.set_xscale("log"); ax.set_xticks(ticks); ax.set_xticklabels([f"{b:g}" for b in ticks]); ax.grid(alpha=0.3)

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.8), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        for a in arms:
            s = summ[(summ.dataset == d) & (summ.arm == a)].sort_values("budget_pct")
            if len(s):
                ax.errorbar(s.budget_pct, s[METRIC + "_mean"], yerr=s[METRIC + "_std"], marker="o", ms=3, capsize=2,
                            label=ARMS[a]["label"], color=COLORS[a], ls="--" if a == "A2_cnn34_scratch" else "-")
        ax.set_title(d); ax.set_xlabel("labelled training images (%)"); xaxis(ax)
    axes[0][0].set_ylabel("test macro-F1"); axes[0][-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figs / "learning_curves_v2.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.6), squeeze=False)
    for ax, d in zip(axes[0], datasets):
        for (a, b), col in [(("E_vit_mae", "C_vit_imagenet"), COLORS["E_vit_mae"]),
                            (("E2_vit_mae_big", "C_vit_imagenet"), COLORS["E2_vit_mae_big"]),
                            (("F_vit_imagenet_mae", "C_vit_imagenet"), COLORS["F_vit_imagenet_mae"])]:
            g = gaps[(a, b)]
            g = g[g.dataset == d].sort_values("budget_pct")
            if len(g):
                ax.plot(g.budget_pct, g.gap, marker="o", ms=3, color=col, label=f"{short(a)} - C")
                ax.fill_between(g.budget_pct, g.ci_lo, g.ci_hi, alpha=0.12, color=col)
        ax.axhline(0, color="k", lw=0.8); ax.set_title(d); ax.set_xlabel("labelled training images (%)"); xaxis(ax)
    axes[0][0].set_ylabel("macro-F1 gap to ImageNet ViT (C)"); axes[0][-1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(figs / "gap_vs_imagenet_v2.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    cal = summ.groupby("arm").agg(raw=("test_ece_mean", "mean"), ts=("test_ts_ece_mean", "mean")).reindex(arms)
    x = np.arange(len(arms))
    ax.bar(x - 0.2, cal.raw, 0.4, label="raw ECE"); ax.bar(x + 0.2, cal.ts, 0.4, label="after temperature scaling")
    ax.set_xticks(x); ax.set_xticklabels([short(a) for a in arms]); ax.set_ylabel("mean ECE over all cells")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(figs / "calibration_v2.png", dpi=150); plt.close(fig)

    wm = winners.pivot(index="dataset", columns="budget_pct", values="winner").reindex(datasets)
    fig, ax = plt.subplots(figsize=(1.4 * wm.shape[1] + 2, 0.6 * len(wm) + 1.2))
    codes = {a: i for i, a in enumerate(arms)}
    ax.imshow(wm.map(lambda v: codes.get(v, -1)).to_numpy(dtype=float), cmap="tab10", vmin=0, vmax=9, aspect="auto")
    for i, d in enumerate(wm.index):
        for j, b in enumerate(wm.columns):
            v = wm.loc[d, b]
            ax.text(j, i, short(v) if isinstance(v, str) else "", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(wm.shape[1])); ax.set_xticklabels([f"{b:g}%" for b in wm.columns])
    ax.set_yticks(range(len(wm))); ax.set_yticklabels(wm.index); ax.set_title("Best arm (mean test macro-F1)")
    fig.tight_layout(); fig.savefig(figs / "winner_map_v2.png", dpi=150); plt.close(fig)

    sub = cross[cross.pair == "F-C"]
    if len(sub) >= 3:
        fig, axes = plt.subplots(1, len(DIST_COLS), figsize=(4 * len(DIST_COLS), 3.4))
        for ax, col in zip(axes, DIST_COLS):
            ax.scatter(sub[col], sub.mean_gap_log)
            for r in sub.itertuples():
                ax.annotate(r.dataset, (getattr(r, col), r.mean_gap_log), fontsize=8)
            ax.axhline(0, color="k", lw=0.8); ax.set_xlabel(col); ax.grid(alpha=0.3)
        axes[0].set_ylabel("mean gap F-C (log-budget avg)")
        fig.tight_layout(); fig.savefig(figs / "distance_vs_gap_FC_v2.png", dpi=150); plt.close(fig)


def _report(cfg, df, summ, gaps, cross, assoc, agree, hyp, lr_changes, cal, dist, arms):
    s = summ.copy()
    s["f1"] = s.apply(lambda r: f"{r[METRIC + '_mean']:.3f} ± {r[METRIC + '_std']:.3f}" if r.n_seeds > 1
                      else f"{r[METRIC + '_mean']:.3f}", axis=1)
    table = s.pivot_table(index=["dataset", "budget_pct"], columns="arm", values="f1", aggfunc="first")[arms]
    table.columns = [short(a) for a in arms]
    gtxt = ""
    for (a, b), g in gaps.items():
        gtxt += f"\n### {short(a)} minus {short(b)}  ({ARMS[a]['label']} vs {ARMS[b]['label']})\n{_md(g)}\n"
    changes = ""
    if len(lr_changes):
        ch = lr_changes[lr_changes.changed]
        changes = (f"The wider grid changed the chosen learning rate in {len(ch)} of {len(lr_changes)} v1 (dataset, arm) "
                   f"cells; those cells were re-run.\n\n{_md(lr_changes)}")
    txt = f"""# Results of experiment 2: bigger MAE pool, ImageNet + MAE, ResNet-34

Generated automatically from `runs.csv` ({len(df)} fine-tuning runs, {df.seed.nunique()} seeds, arms: {', '.join(short(a) for a in arms)}).
Pre-registered in `PREREGISTRATION_V2.md` before any of this ran. E2 = MAE on a 20,000-image pool (not Pets), F =
ImageNet ViT + continued MAE on the pool, A2 = ResNet-34. Arms A, C, D, E use the v1 runs wherever the learning rate
was unchanged by the wider grid.

## Hypotheses (H6-H11)
{_md(hyp)}

## Learning-rate grid
{changes or '_not available yet_'}

## Mean test macro-F1 (± sd over seeds)
{_md(table.reset_index())}

## Calibration and mean F1 by arm (mean over all cells)
{_md(cal.reset_index())}

## Crossover and mean gap to the ImageNet ViT
{_md(cross[['pair', 'dataset', 'mean_gap_log', 'crossover_budget', 'censored', 'direction', 'crossover_supported_by_ci']])}

### Distance association of the mean gap (descriptive; n = 3 or 4 datasets)
{_md(assoc)}

## Winners and ranking agreement
{_md(agree)}

## All paired gaps (macro-F1, paired by seed)
{gtxt}
## Figures
`plots/learning_curves_v2.png`, `plots/gap_vs_imagenet_v2.png`, `plots/calibration_v2.png`, `plots/winner_map_v2.png`,
`plots/distance_vs_gap_FC_v2.png`.
"""
    (paths(cfg)["results"] / "REPORT_V2.md").write_text(txt, encoding="utf-8")
