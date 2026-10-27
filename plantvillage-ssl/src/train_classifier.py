"""Supervised fine-tuning / training of one (model, label budget, seed) run + full evaluation + logging.

Model selection rule: early stopping and best-checkpoint choice use the VALIDATION macro-F1 only.
The test set is evaluated exactly once per run, AFTER the best checkpoint has been restored, and is never
used to choose epochs, hyper-parameters or the "best model" (report.py ranks models on validation scores).
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn

from .calibration import ece_score, brier_score, fit_temperature, reliability_table, softmax_np
from .data_loader import budget_subset, class_names_from_split, load_split, make_loader
from .evaluate import compute_metrics, macro_f1, predict
from .models import MODEL_KINDS, build_model, count_params, model_size_mb
from .utils import append_row_csv, get_device, get_logger, get_paths, set_seed
from .visualization import plot_confusion_matrix, plot_reliability, plot_training_curves

log = get_logger()


def _param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") or "pos_embed" in n or "cls_token" in n else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def _lr_factor(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return max(0.01, 0.5 * (1 + math.cos(math.pi * prog)))


def run_id_for(kind: str, budget, seed: int) -> str:
    return f"{kind}_b{budget:g}_s{seed}"


def train_one_run(cfg: dict, kind: str, budget: float, seed: int, force: bool = False) -> dict:
    paths = get_paths(cfg)
    rid = run_id_for(kind, budget, seed)
    run_dir = paths["runs"] / rid
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not force:
        log.info("[skip] %s already done", rid)
        return json.loads(metrics_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)

    tc = cfg["train"]
    set_seed(seed)
    device = get_device()
    use_amp = device.type == "cuda"

    split = load_split(cfg)
    class_names = class_names_from_split(split)
    C = len(class_names)
    train_full = split[split["split"] == "train"]
    sub = budget_subset(train_full, budget, seed)
    val_df, test_df = split[split["split"] == "val"], split[split["split"] == "test"]
    assert set(sub["relpath"]) <= set(train_full["relpath"])  # labelled images come from the train split only

    n = len(sub)
    bs = min(tc["batch_size"], n)
    drop_last = n > bs and n % bs == 1  # a last batch of size 1 breaks BatchNorm
    train_loader = make_loader(sub, cfg, train=True, batch_size=bs, drop_last=drop_last)
    val_loader = make_loader(val_df, cfg, train=False, batch_size=max(bs, 64))
    test_loader = make_loader(test_df, cfg, train=False, batch_size=max(bs, 64))

    steps_per_epoch = len(train_loader)
    epochs = min(tc["max_epochs"], max(tc["epochs"], math.ceil(tc["min_updates"] / steps_per_epoch)))
    eval_every = max(1, epochs // tc["target_evals"])
    total_steps, warm = epochs * steps_per_epoch, tc["warmup_epochs"] * steps_per_epoch

    model, info = build_model(kind, cfg, C)
    model.to(device)
    lr = tc["lr"][kind]
    opt = torch.optim.AdamW(_param_groups(model, tc["weight_decay"]), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = nn.CrossEntropyLoss(label_smoothing=tc.get("label_smoothing", 0.0))
    proc = psutil.Process()
    peak_rss = proc.memory_info().rss
    if use_amp:
        torch.cuda.reset_peak_memory_stats()

    log.info("[%s] n_labeled=%d epochs=%d bs=%d lr=%g device=%s", rid, n, epochs, bs, lr, device)
    hist, best_f1, best_epoch, bad, best_state, step = [], -1.0, 0, 0, None, 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        tl, tcorrect, tn = 0.0, 0, 0
        for x, y in train_loader:
            for g in opt.param_groups:
                g["lr"] = lr * _lr_factor(step, total_steps, warm)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = crit(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tc.get("grad_clip"):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt); scaler.update(); step += 1
            tl += loss.item() * len(y); tn += len(y); tcorrect += (out.argmax(1) == y).sum().item()
        row = dict(epoch=epoch, train_loss=tl / tn, train_acc=tcorrect / tn, val_loss=np.nan, val_acc=np.nan,
                   val_macro_f1=np.nan)
        peak_rss = max(peak_rss, proc.memory_info().rss)
        if epoch % eval_every == 0 or epoch == epochs:
            vl, vy = predict(model, val_loader, device, use_amp)
            vp = vl.argmax(1)
            row.update(val_loss=float(nn.functional.cross_entropy(torch.tensor(vl), torch.tensor(vy))),
                       val_acc=float((vp == vy).mean()), val_macro_f1=macro_f1(vy, vp))
            if row["val_macro_f1"] > best_f1 + 1e-6:
                best_f1, best_epoch, bad = row["val_macro_f1"], epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            log.info("[%s] ep %3d  train_loss %.4f  val_acc %.4f  val_F1 %.4f  (best %.4f @%d)", rid, epoch,
                     row["train_loss"], row["val_acc"], row["val_macro_f1"], best_f1, best_epoch)
            if bad >= tc["patience"]:
                log.info("[%s] early stopping at epoch %d", rid, epoch)
                hist.append(row)
                break
        hist.append(row)
    train_time = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if use_amp else peak_rss / 1e6

    # ---- restore best-on-validation weights, save checkpoint, final evaluation
    model.load_state_dict(best_state)
    ck_dir = paths["checkpoints"] / "runs"
    ck_dir.mkdir(parents=True, exist_ok=True)
    ck_path = ck_dir / f"{rid}.pt"
    torch.save(best_state, ck_path)

    val_logits, val_y = predict(model, val_loader, device, use_amp)
    test_logits, test_y = predict(model, test_loader, device, use_amp)
    vm = compute_metrics(val_logits, val_y, class_names)
    tm = compute_metrics(test_logits, test_y, class_names)

    T = fit_temperature(val_logits, val_y)  # fitted on validation only
    tp_ts = softmax_np(test_logits / T)
    test_ece_ts, test_brier_ts = ece_score(tp_ts, test_y), brier_score(tp_ts, test_y)

    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(run_dir / "history.csv", index=False)
    tm["per_class"].to_csv(run_dir / "per_class_metrics.csv", index=False)
    np.save(run_dir / "confusion_matrix.npy", tm["confusion_matrix"])
    np.savez_compressed(run_dir / "test_logits.npz", logits=test_logits.astype(np.float32), labels=test_y)
    plot_training_curves(hist_df, run_dir / "curves.png", rid)
    rel = reliability_table(softmax_np(test_logits), test_y)
    plot_reliability(rel, tm["ece"], run_dir / "reliability.png", rid)
    if C <= 40:
        plot_confusion_matrix(tm["confusion_matrix"], class_names, run_dir / "confusion_matrix.png", title=rid)

    pre_time = 0.0
    if MODEL_KINDS[kind]["mae"]:
        pi = paths["mae"] / "pretrain_info.json"
        pre_time = json.loads(pi.read_text())["pretrain_time_s"] if pi.exists() else float("nan")

    arch = cfg["models"]["cnn_arch"] if MODEL_KINDS[kind]["family"] == "cnn" else cfg["models"]["vit_arch"]
    result = dict(
        run_id=rid, model_kind=kind, arch=arch, pretraining=MODEL_KINDS[kind]["pretraining"],
        budget_pct=budget, n_labeled=n, seed=seed, img_size=cfg["data"]["img_size"],
        epochs_planned=epochs, epochs_run=len(hist), best_epoch=best_epoch, batch_size=bs, lr=lr,
        val_acc=vm["acc"], val_macro_f1=vm["macro_f1"], val_ece=vm["ece"], val_brier=vm["brier"],
        test_acc=tm["acc"], test_macro_precision=tm["macro_precision"], test_macro_recall=tm["macro_recall"],
        test_macro_f1=tm["macro_f1"], test_ece=tm["ece"], test_brier=tm["brier"],
        test_mean_confidence=tm["mean_confidence"], temperature=T, test_ece_tempscaled=test_ece_ts,
        test_brier_tempscaled=test_brier_ts,
        train_time_s=train_time, pretrain_time_s=pre_time, params_M=count_params(model) / 1e6,
        model_size_mb=model_size_mb(model), checkpoint_mb=ck_path.stat().st_size / 1e6, peak_mem_mb=peak_mem,
        device=str(device), pretrained_loaded=info["pretrained_loaded"],
    )
    metrics_path.write_text(json.dumps(result, indent=2, default=float))
    append_row_csv(paths["results"] / "runs.csv", result)
    log.info("[%s] done: val_F1 %.4f | test_acc %.4f test_F1 %.4f ECE %.4f | %.1f s", rid, vm["macro_f1"],
             tm["acc"], tm["macro_f1"], tm["ece"], train_time)
    return result


def run_grid(cfg: dict, kinds=None, budgets=None, seeds=None, force: bool = False):
    kinds = kinds or cfg["models"]["kinds"]
    budgets = budgets or cfg["train"]["budgets"]
    seeds = seeds if seeds is not None else cfg["train"]["seeds"]
    todo = [(k, b, s) for k in kinds for b in budgets for s in seeds]
    log.info("%d runs planned (%d models x %d budgets x %d seeds)", len(todo), len(kinds), len(budgets), len(seeds))
    for i, (k, b, s) in enumerate(todo, 1):
        log.info("===== run %d / %d =====", i, len(todo))
        train_one_run(cfg, k, b, s, force=force)
