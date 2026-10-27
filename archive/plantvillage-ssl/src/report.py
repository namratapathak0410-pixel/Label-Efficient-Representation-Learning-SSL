"""Aggregate runs.csv -> comparison table (CSV + Markdown), plots, best-model selection, REPORT.md.

Everything is computed from results/<exp>/runs.csv. Experiments that have not run are shown as 'pending'.
Best-model selection uses VALIDATION scores only (the test set is only reported, never used to choose).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .models import MODEL_KINDS, model_display_name
from .utils import get_logger, get_paths
from .visualization import plot_confusion_matrix, plot_metric_vs_budget

log = get_logger()

NUM_COLS = ["test_acc", "test_macro_f1", "test_ece", "test_brier", "test_ece_tempscaled", "val_acc", "val_macro_f1",
            "val_ece", "train_time_s", "pretrain_time_s", "total_time_s", "model_size_mb", "params_M", "n_labeled"]


def _pm(m, s, scale=100.0, nd=2):
    if pd.isna(m):
        return "pending"
    return f"{m * scale:.{nd}f} ± {s * scale:.{nd}f}" if not pd.isna(s) else f"{m * scale:.{nd}f} (1 seed)"


def _time(sec):
    if pd.isna(sec):
        return "pending"
    return f"{sec / 60:.1f} min" if sec >= 90 else f"{sec:.0f} s"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def aggregate(cfg: dict):
    paths = get_paths(cfg)
    runs_csv = paths["results"] / "runs.csv"
    if not runs_csv.exists():
        raise FileNotFoundError("No results yet (results/<exp>/runs.csv missing). Run `python main.py train` first.")
    runs = pd.read_csv(runs_csv).drop_duplicates("run_id", keep="last")
    runs["total_time_s"] = runs["train_time_s"] + runs["pretrain_time_s"].fillna(0)
    kinds = [k for k in cfg["models"]["kinds"]]
    budgets = list(cfg["train"]["budgets"])

    g = runs.groupby(["model_kind", "budget_pct"])
    mean = g[NUM_COLS].mean().add_suffix("_mean")
    std = g[NUM_COLS].std().add_suffix("_std")
    num = pd.concat([mean, std, g.size().rename("n_seeds")], axis=1).reset_index()
    num.to_csv(paths["results"] / "summary_numeric.csv", index=False)

    rows = []
    for k in kinds:
        for b in budgets:
            r = num[(num.model_kind == k) & (num.budget_pct == b)]
            base = dict(Model=model_display_name(k, cfg), **{"Pretraining method": MODEL_KINDS[k]["pretraining"]},
                        **{"Label budget": f"{b:g}%"})
            if r.empty:
                rows.append({**base, "Test accuracy (%)": "pending", "Macro-F1 (%)": "pending", "ECE (%)": "pending",
                             "Training time": "pending", "Model size": "pending", "MAE pretrain time": "pending",
                             "Seeds": 0})
                continue
            r = r.iloc[0]
            pre = r["pretrain_time_s_mean"]
            rows.append({**base,
                         "Test accuracy (%)": _pm(r["test_acc_mean"], r["test_acc_std"]),
                         "Macro-F1 (%)": _pm(r["test_macro_f1_mean"], r["test_macro_f1_std"]),
                         "ECE (%)": _pm(r["test_ece_mean"], r["test_ece_std"]),
                         "Training time": _time(r["train_time_s_mean"]),
                         "Model size": f"{r['model_size_mb_mean']:.1f} MB ({r['params_M_mean']:.1f} M params)",
                         "MAE pretrain time": _time(pre) if MODEL_KINDS[k]["mae"] else "n/a",
                         "Seeds": int(r["n_seeds"])})
    table = pd.DataFrame(rows)
    table.to_csv(paths["results"] / "results_table.csv", index=False)
    (paths["results"] / "results_table.md").write_text(_md(table))

    # ---- plots
    label_fn = lambda k: model_display_name(k, cfg)  # noqa: E731
    plot_metric_vs_budget(num, "test_macro_f1", "Macro-F1 (%)", paths["plots"] / "macro_f1_vs_budget.png", label_fn)
    plot_metric_vs_budget(num, "test_acc", "Test accuracy (%)", paths["plots"] / "accuracy_vs_budget.png", label_fn)
    plot_metric_vs_budget(num, "test_ece", "ECE (%)", paths["plots"] / "ece_vs_budget.png", label_fn)

    # ---- best model per criterion (VALIDATION scores)
    top = max(budgets)
    best = {}

    def pick(df, col, largest=True):
        df = df.dropna(subset=[col])
        if df.empty:
            return None
        r = df.loc[df[col].idxmax() if largest else df[col].idxmin()]
        return r["model_kind"]

    at_top = num[num.budget_pct == top]
    best["classification (highest validation macro-F1 at %g%% labels)" % top] = pick(at_top, "val_macro_f1_mean")
    complete = num.groupby("model_kind").filter(lambda d: set(d.budget_pct) >= set(budgets))
    eff = complete.groupby("model_kind")["val_macro_f1_mean"].mean().reset_index()
    best["label efficiency (highest validation macro-F1 averaged over all budgets)"] = (
        pick(eff, "val_macro_f1_mean") if not eff.empty else None)
    cal = complete.groupby("model_kind")["val_ece_mean"].mean().reset_index()
    best["calibration (lowest validation ECE averaged over all budgets)"] = (
        pick(cal, "val_ece_mean", False) if not cal.empty else None)
    best["compute (lowest total training time incl. pretraining at %g%% labels)" % top] = pick(
        at_top, "total_time_s_mean", False)
    best["model size (smallest)"] = pick(at_top, "model_size_mb_mean", False)
    best_named = {c: (model_display_name(k, cfg) if k else "pending") for c, k in best.items()}
    (paths["results"] / "best_models.json").write_text(json.dumps(best_named, indent=2))

    # ---- confusion matrix of the best-classification model at the top budget (summed over seeds)
    cm_note = "pending"
    kbest = best["classification (highest validation macro-F1 at %g%% labels)" % top]
    if kbest:
        cms = [np.load(paths["runs"] / rid / "confusion_matrix.npy")
               for rid in runs[(runs.model_kind == kbest) & (runs.budget_pct == top)].run_id
               if (paths["runs"] / rid / "confusion_matrix.npy").exists()]
        if cms:
            from .data_loader import class_names_from_split, load_split

            names = class_names_from_split(load_split(cfg))
            plot_confusion_matrix(sum(cms), names, paths["plots"] / "confusion_matrix_best.png",
                                  title=f"{model_display_name(kbest, cfg)} @ {top:g}% labels (test, summed over seeds)")
            cm_note = "results/%s/plots/confusion_matrix_best.png" % cfg["experiment_name"]

    # ---- REPORT.md (numbers only; interpretation is left to the author)
    md =[f"# Results report - experiment `{cfg['experiment_name']}`", "",
          "Generated automatically from `runs.csv`. Mean ± std over seeds (`Seeds` column). "
          "Values marked *pending* have not been run. ECE/accuracy/F1 are in %.", "",
          "## Comparison table (test set)", "", _md(table), "",
          "## Best model per criterion (selected on VALIDATION scores, not on test)", ""]
    md += [f"- **{c}:** {v}" for c, v in best_named.items()]
    md += ["", "## Plots", "",
           f"- Macro-F1 vs label budget: `plots/macro_f1_vs_budget.png`",
           f"- Accuracy vs label budget: `plots/accuracy_vs_budget.png`",
           f"- ECE vs label budget: `plots/ece_vs_budget.png`",
           f"- Confusion matrix of best model: `{cm_note}`", "",
           "## Calibration after temperature scaling (extra, T fitted on validation)", ""]
    ts = num[["model_kind", "budget_pct", "test_ece_mean", "test_ece_tempscaled_mean"]].copy()
    ts.insert(0, "Model", ts.pop("model_kind").map(lambda k: model_display_name(k, cfg)))
    ts["test_ece_mean"] = (ts["test_ece_mean"] * 100).round(2)
    ts["test_ece_tempscaled_mean"] = (ts["test_ece_tempscaled_mean"] * 100).round(2)
    ts.columns = ["Model", "Budget %", "ECE % (raw)", "ECE % (temp-scaled)"]
    md += [_md(ts), "",
           "## Interpretation", "",
           "_To be written by the author after inspecting the numbers above. Do not draw conclusions that the "
           "table does not support (check the std over seeds and the number of seeds first)._", ""]
    (paths["results"] / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    log.info("Report written: %s", paths["results"] / "REPORT.md")
    print("\n" + _md(table) + "\n")
    for c, v in best_named.items():
        print(f"best {c}: {v}")
    return table, best_named
