"""Builds every figure (paper/figures/*.pdf) and data table (paper/tables/*.tex) of the paper from the result
files of the main experiment, so that no number in a figure or table is typed by hand.
Usage (from mae-crossover):  .venv\\Scripts\\python.exe paper\\make_figures.py
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "crossover_v1"
CKPT = ROOT / "checkpoints" / "crossover_v1"
FIG = Path(__file__).parent / "figures"
TAB = Path(__file__).parent / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

DATASETS = ["plantvillage", "eurosat", "cifar10", "pets"]
NAMES = {"plantvillage": "PlantVillage", "eurosat": "EuroSAT", "cifar10": "CIFAR-10", "pets": "Oxford Pets"}
DOMAIN = {"plantvillage": "plant", "eurosat": "satellite", "cifar10": "objects", "pets": "animals"}
NATIVE = {"plantvillage": "256", "eurosat": "64", "cifar10": "32", "pets": "var."}  # native image side in pixels
ARMS = ["A_cnn_scratch", "C_vit_imagenet", "D_vit_scratch", "E_vit_mae"]
ARM_LABEL = {"A_cnn_scratch": "A: ResNet-18 scratch", "C_vit_imagenet": "C: ViT ImageNet",
             "D_vit_scratch": "D: ViT scratch", "E_vit_mae": "E: ViT MAE (in-domain)"}
ARM_SHORT = {"A_cnn_scratch": "A", "C_vit_imagenet": "C", "D_vit_scratch": "D", "E_vit_mae": "E"}
COLOR = {"A_cnn_scratch": "#7f7f7f", "C_vit_imagenet": "#1f77b4", "D_vit_scratch": "#ff7f0e", "E_vit_mae": "#2ca02c"}
MARK = {"A_cnn_scratch": "s", "C_vit_imagenet": "o", "D_vit_scratch": "^", "E_vit_mae": "D"}
DCOLOR = {"plantvillage": "#2ca02c", "eurosat": "#8c564b", "cifar10": "#1f77b4", "pets": "#d62728"}
BUDGETS = [5, 10, 20, 50, 100]

plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.1, "lines.markersize": 3.2,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.01})

summ = pd.read_csv(RES / "summary.csv")
runs = pd.read_csv(RES / "runs.csv")
gap_c = pd.read_csv(RES / "gap_E_minus_C.csv")
gap_d = pd.read_csv(RES / "gap_E_minus_D.csv")
gap_a = pd.read_csv(RES / "gap_E_minus_A.csv")
cross = pd.read_csv(RES / "crossover.csv").set_index("dataset")
assoc = pd.read_csv(RES / "distance_association.csv")
rank = pd.read_csv(RES / "ranking_agreement.csv")
dist = pd.read_csv(RES / "domain_distance.csv").set_index("dataset")


def save(fig, name, **kw):
    fig.savefig(FIG / name, **kw)
    if os.environ.get("PREVIEW_DIR"):  # optional PNG copy for checking the layout
        fig.savefig(Path(os.environ["PREVIEW_DIR"]) / name.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)


def log_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.minorticks_off()
    ax.grid(alpha=0.25, lw=0.4)


# ---------- Fig: learning curves (macro-F1 vs budget, one panel per dataset) ----------
fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.75), sharey=False)
for ax, d in zip(axes, DATASETS):
    for a in ARMS:
        s = summ[(summ.dataset == d) & (summ.arm == a)].sort_values("budget_pct")
        m, sd = s.test_macro_f1_mean.values, s.test_macro_f1_std.values
        ax.plot(s.budget_pct, m, marker=MARK[a], color=COLOR[a], label=ARM_LABEL[a])
        ax.fill_between(s.budget_pct, m - sd, m + sd, color=COLOR[a], alpha=0.18, lw=0)
    log_axis(ax)
    ax.set_title(NAMES[d])
    ax.set_xlabel("label budget (% of train labels)")
axes[0].set_ylabel("test macro-F1")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(w_pad=0.6)
save(fig, "learning_curves.pdf")

# ---------- Fig: paired gap E - C with 95 % CI ----------
fig, ax = plt.subplots(figsize=(3.45, 2.0))
offs = dict(zip(DATASETS, [-0.045, -0.015, 0.015, 0.045]))
for d in DATASETS:
    g = gap_c[gap_c.dataset == d].sort_values("budget_pct")
    x = g.budget_pct.values * 10 ** offs[d]
    ax.errorbar(x, g.gap, yerr=[g.gap - g.ci_lo, g.ci_hi - g.gap], marker="o", color=DCOLOR[d], capsize=1.8,
                elinewidth=0.7, label=NAMES[d])
ax.axhline(0, color="k", lw=0.7, ls="--")
log_axis(ax)
ax.set_xlabel("label budget (% of train labels)")
ax.set_ylabel("macro-F1 gap  E $-$ C")
ax.legend(frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.8)
save(fig, "gap_E_minus_C.pdf")

# ---------- Fig: domain distance vs mean log-budget gap ----------
DIST = [("fid", "FID\n(higher = farther)"), ("otdd", "OTDD\n(higher = farther)"),
        ("imagenet_confidence", "ImageNet confidence\n(lower = farther)"),
        ("linear_probe_acc", "linear-probe accuracy\n(lower = farther)")]
fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.75), sharey=True)
for ax, (k, lab) in zip(axes, DIST):
    r = assoc[assoc.distance == k].iloc[0]
    for d in DATASETS:
        ax.scatter(dist.loc[d, k], cross.loc[d, "mean_gap_log"], color=DCOLOR[d], s=16, zorder=3,
                   label=NAMES[d])
    ax.set_xlabel(lab)
    ax.set_title(fr"$\rho$ = {r.spearman_rho:+.1f}, exact $p$ = {r.exact_p:.2f}")
    ax.grid(alpha=0.25, lw=0.4)
    ax.axhline(0, color="k", lw=0.6, ls="--")
axes[0].set_ylabel("mean gap E $-$ C")
h, l = axes[0].get_legend_handles_labels()
fig.tight_layout(w_pad=0.5, rect=(0, 0, 1, 0.9))
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
save(fig, "distance_vs_gap.pdf")

# ---------- Fig: calibration (ECE before / after temperature scaling, mean over datasets) ----------
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.6), sharey=True)
for ax, col, title in [(axes[0], "test_ece_mean", "before temp. scaling"),
                       (axes[1], "test_ts_ece_mean", "after temp. scaling")]:
    for a in ARMS:
        s = summ[summ.arm == a].groupby("budget_pct")[col].mean()
        ax.plot(s.index, s.values, marker=MARK[a], color=COLOR[a], label=ARM_SHORT[a])
    log_axis(ax)
    ax.set_title(title)
    ax.set_xlabel("label budget (%)")
axes[0].set_ylabel("test ECE (mean of 4 datasets)")
axes[1].legend(frameon=False, loc="upper right", ncol=2)
fig.tight_layout(w_pad=0.4)
save(fig, "calibration.pdf")

# ---------- Fig: MAE reconstructions (first two examples of each dataset, seed 0) ----------
# reconstructions.png is an 8-column grid of 150 px tiles: rows masked / reconstruction / original
X0, PITCH, TILE, ROWS_Y = 20, 174, 150, [31, 207, 383]
tiles = []
for d in DATASETS:
    im = np.asarray(Image.open(CKPT / d / "mae_seed0" / "reconstructions.png").convert("RGB"))
    for j in range(1):  # one image per domain, so the figure fits one column
        x = X0 + j * PITCH
        tiles.append((d, [im[y:y + TILE, x:x + TILE] for y in ROWS_Y]))
fig, axes = plt.subplots(3, 4, figsize=(3.45, 2.7))
for c, (d, col) in enumerate(tiles):
    for r in range(3):
        ax = axes[r, c]
        ax.imshow(col[r])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0, c].set_title(NAMES[d])
for r, lab in enumerate(["masked", "reconstr.", "original"]):
    axes[r, 0].set_ylabel(lab, fontsize=6.5)
fig.subplots_adjust(wspace=0.04, hspace=0.04)
save(fig, "mae_reconstructions.pdf", dpi=200)


# ---------- Tables ----------
def f3(x):
    return f"{x:.3f}"


def signed(x):
    return f"{x:+.3f}".replace("-", "$-$")


# Datasets, splits and measured distances
rows = []
for d in DATASETS:
    rep = json.load(open(RES / "splits" / f"{d}_split_report.json"))
    c = rep["counts"]
    rows.append(f"{NAMES[d]} & {DOMAIN[d]} & {int(dist.loc[d, 'n_classes'])} & {NATIVE[d]} & "
                f"{c['train']}/{c['val']}/{c['test']} & {rep['n_duplicate_groups']} & "
                f"{dist.loc[d, 'fid']:.0f} & {dist.loc[d, 'otdd']:.1f} & {dist.loc[d, 'imagenet_confidence']:.2f} & "
                f"{dist.loc[d, 'linear_probe_acc']:.2f} \\\\")
(TAB / "datasets.tex").write_text("\n".join(rows) + "\n")

# Main macro-F1 table (mean +- sd over 3 seeds), best arm in bold
rows = []
for d in DATASETS:
    for i, b in enumerate(BUDGETS):
        s = summ[(summ.dataset == d) & (summ.budget_pct == b)].set_index("arm")
        best = s.test_macro_f1_mean.idxmax()
        cells = []
        for a in ARMS:
            v = f"{s.loc[a, 'test_macro_f1_mean']:.3f}\\,{{\\tiny$\\pm${s.loc[a, 'test_macro_f1_std']:.3f}}}"
            cells.append(f"\\textbf{{{v}}}" if a == best else v)
        g = gap_c[(gap_c.dataset == d) & (gap_c.budget_pct == b)].iloc[0]
        star = "$^{*}$" if (g.ci_hi < 0 or g.ci_lo > 0) else ""
        name = f"\\multirow{{5}}{{*}}{{\\rotatebox{{90}}{{{NAMES[d]}}}}}" if i == 0 else ""
        rows.append(f"{name} & {b} & " + " & ".join(cells) + f" & {signed(g.gap)}{star} \\\\")
    rows.append("\\midrule" if d != DATASETS[-1] else "")
(TAB / "macro_f1.tex").write_text("\n".join(r for r in rows if r) + "\n")


# Pairwise gaps summary: E-C, E-D, E-A per dataset
def gap_summary(df, d):
    g = df[df.dataset == d]
    sig = ((g.ci_lo > 0) | (g.ci_hi < 0)).sum()
    return f"{signed(g.gap.min())} to {signed(g.gap.max())}", f"{sig}/5"


rows = []
for d in DATASETS:
    ec = gap_summary(gap_c, d)
    ed = gap_summary(gap_d, d)
    ea = gap_summary(gap_a, d)
    rows.append(f"{NAMES[d]} & {signed(cross.loc[d, 'mean_gap_log'])} & ${cross.loc[d, 'censored']}$\\,\\% & "
                f"{ed[0]} & {ed[1]} & {ea[0]} & {ea[1]} \\\\")
(TAB / "pairwise.tex").write_text("\n".join(rows) + "\n")

# Distance association
DNAME = {"fid": "FID", "otdd": "OTDD", "imagenet_confidence": "ImageNet confidence",
         "linear_probe_acc": "Linear-probe accuracy"}
PRED = {"fid": "$+$", "otdd": "$+$", "imagenet_confidence": "$-$", "linear_probe_acc": "$-$"}
rows = [f"{DNAME[r.distance]} & {PRED[r.distance]} & {signed(r.spearman_rho)} & "
        f"{r.exact_p:.3f} & {r.loo_abs_error:.3f} & {r.loo_baseline_abs_error:.3f} \\\\"
        for r in assoc.itertuples()]
(TAB / "association.tex").write_text("\n".join(rows) + "\n")

# Ranking agreement across datasets
rows = []
for b in BUDGETS:
    cells = []
    for m in ["test_macro_f1", "test_ece", "test_ts_ece"]:
        r = rank[(rank.metric == m) & (rank.budget_pct == b)].iloc[0]
        cells.append(f"{r.kendall_w:.3f} & {r.friedman_p:.3f}")
    rows.append(f"{b} & " + " & ".join(cells) + " \\\\")
(TAB / "ranking.tex").write_text("\n".join(rows) + "\n")

# ---------- Study 2 (results/crossover_v2): bigger MAE pool (E2), ImageNet + MAE (F), ResNet-34 (A2) ----------
import sys

sys.path.insert(0, str(ROOT))
from mc.analyze import mean_gap_log  # noqa: E402  (same log-budget average as the analysis)

RES2 = ROOT / "results" / "crossover_v2"
summ2 = pd.read_csv(RES2 / "summary.csv")
runs2 = pd.read_csv(RES2 / "runs.csv")
G2 = {k: pd.read_csv(RES2 / f"gap_{k}.csv") for k in
      ["E_minus_C", "E2_minus_C", "F_minus_C", "E2_minus_E", "A2_minus_A", "F_minus_E2"]}
S2COL = {"E_minus_C": ("#2ca02c", "E: MAE, 5k pool"), "E2_minus_C": ("#9467bd", "E2: MAE, 20k pool"),
         "F_minus_C": ("#d62728", "F: ImageNet + MAE")}

fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.75), sharey=False)
for ax, d in zip(axes, DATASETS):
    for k, (col, lab) in S2COL.items():
        g = G2[k][G2[k].dataset == d].sort_values("budget_pct")
        if len(g):
            ax.plot(g.budget_pct, g.gap, marker="o", color=col, label=lab)
            ax.fill_between(g.budget_pct, g.ci_lo, g.ci_hi, color=col, alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.7, ls="--")
    log_axis(ax)
    ax.set_title(NAMES[d] + (" (no extra images)" if d == "pets" else ""))
    ax.set_xlabel("label budget (% of train labels)")
axes[0].set_ylabel("macro-F1 gap to C")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(w_pad=0.6)
save(fig, "study2_gaps.pdf")


def nsig(g, side):
    return int((g.ci_lo > 0).sum() if side > 0 else (g.ci_hi < 0).sum())


rows = []
for d in DATASETS:
    sel = {k: v[v.dataset == d] for k, v in G2.items()}
    if len(sel["E2_minus_E"]):
        e2 = (f"{signed(mean_gap_log(sel['E2_minus_E']))} & {nsig(sel['E2_minus_E'], 1)}/{nsig(sel['E2_minus_E'], -1)} & "
              f"{signed(mean_gap_log(sel['E2_minus_C']))}")
    else:
        e2 = "-- & -- & --"
    f = sel["F_minus_C"]
    a = sel["A2_minus_A"]
    rows.append(f"{NAMES[d]} & {signed(mean_gap_log(sel['E_minus_C']))} & {e2} & {signed(mean_gap_log(f))} & "
                f"{int((f.gap > 0).sum())}/{nsig(f, -1)} & {signed(a.gap.min())} to {signed(a.gap.max())} \\\\")
(TAB / "study2.tex").write_text("\n".join(rows) + "\n")

# ---------- Numbers quoted in the text (printed for checking) ----------
print("STUDY 2: runs", len(runs2), " winners:", summ2.loc[summ2.groupby(["dataset", "budget_pct"])
      .test_macro_f1_mean.idxmax()].arm.value_counts().to_dict())
for k, g in G2.items():
    print(f"  {k}: positive {(g.gap > 0).sum()}/{len(g)}, CI>0 {nsig(g, 1)}, CI<0 {nsig(g, -1)}, "
          f"range {g.gap.min():+.3f}..{g.gap.max():+.3f}")
print("  F-C cells:", G2["F_minus_C"].round(3)[["dataset", "budget_pct", "gap", "ci_lo", "ci_hi"]].values.tolist())
print("  mean ECE:", summ2.groupby("arm")[["test_ece_mean", "test_ts_ece_mean"]].mean().round(3).to_dict())
print("  early stop:", runs2.groupby("arm").stopped_early.mean().round(2).to_dict())
lrc = pd.read_csv(RES2 / "lr_changes_vs_v1.csv")
print("  lr changed:", int(lrc.changed.sum()), "/", len(lrc))
grid = {"s": [2.5e-4, 4e-3], "p": [1.25e-4, 2e-3]}
ch = json.load(open(RES2 / "lr_choice.json"))
edge = [k for k, v in ch.items() if v["lr"] in (grid["s"] if ("A" == k.split("|")[1][0] or k.split("|")[1].startswith("D")) else grid["p"])]
print("  lr at new grid edge:", len(edge), "/", len(ch), edge)
for d in ["plantvillage", "eurosat", "cifar10"]:
    loss = [pd.read_csv(ROOT / "checkpoints" / "crossover_v2" / d / f"mae_big_seed{s}" / "mae_loss.csv").iloc[-1, -1]
            for s in range(3)]
    print(f"  MAE 20k {d}: final loss {np.mean(loss):.3f}")
for d in DATASETS:
    loss = [pd.read_csv(ROOT / "checkpoints" / "crossover_v2" / d / f"imagenet_mae_seed{s}" / "mae_loss.csv").iloc[-1, -1]
            for s in range(3)]
    print(f"  MAE from ImageNet {d}: final loss {np.mean(loss):.3f}")

best_raw = summ.loc[summ.groupby(["dataset", "budget_pct"]).test_ece_mean.idxmin()].arm.value_counts()
best_ts = summ.loc[summ.groupby(["dataset", "budget_pct"]).test_ts_ece_mean.idxmin()].arm.value_counts()
winners = summ.loc[summ.groupby(["dataset", "budget_pct"]).test_macro_f1_mean.idxmax()].arm.value_counts()
sig_d = ((gap_d.ci_lo > 0) | (gap_d.ci_hi < 0)).sum()
sig_a = ((gap_a.ci_lo > 0) | (gap_a.ci_hi < 0)).sum()
print("macro-F1 winners:", winners.to_dict())
print("lowest raw ECE:", best_raw.to_dict(), " lowest TS-ECE:", best_ts.to_dict())
print("E-D significant:", sig_d, "/20; positive:", (gap_d.gap > 0).sum(), " E-A significant:", sig_a, "/20")
print("E-A significant cells:", gap_a[(gap_a.ci_lo > 0) | (gap_a.ci_hi < 0)][["dataset", "budget_pct", "gap"]].values.tolist())
print("early-stop rate:", runs.groupby("arm").stopped_early.mean().round(3).to_dict())
print("fine-tuning hours:", round(runs.train_seconds.sum() / 3600, 2), " runs:", len(runs))
print("mean ECE all budgets:", summ.groupby("arm")[["test_ece_mean", "test_ts_ece_mean"]].mean().round(3).to_dict())
print("ranking W ranges:", rank.groupby("metric").kendall_w.agg(["min", "max"]).to_dict())
print("ranking p max:", rank.groupby("metric").friedman_p.agg(["min", "max"]).round(3).to_dict())
for d in DATASETS:
    loss = [pd.read_csv(CKPT / d / f"mae_seed{s}" / "mae_loss.csv").iloc[-1, -1] for s in range(3)]
    info = [json.load(open(CKPT / d / f"mae_seed{s}" / "pretrain_info.json"))["pretrain_seconds"] / 60 for s in range(3)]
    print(f"MAE {d}: final loss {np.mean(loss):.3f} (sd {np.std(loss, ddof=1):.3f}), minutes {np.mean(info):.1f}")
print("figures ->", FIG, " tables ->", TAB)
