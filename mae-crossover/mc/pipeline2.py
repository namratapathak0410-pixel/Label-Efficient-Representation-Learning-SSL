"""Experiment 2 driver (PREREGISTRATION_V2.md): pool -> MAE seed 0 -> learning-rate tuning -> grid -> analysis.

One process sees the whole plan, so after every finished task it can log an overall estimate of the time left:
  ETA | stage train | 83 tasks left | elapsed 4.2 h | ~11.3 h left | finish ~Fri 29 Oct 05:10
The estimate = sum of expected seconds of every unfinished task (from the measured v1 run times), multiplied by the
ratio of actual to expected time measured so far in this process, separately for MAE, tuning and fine-tuning tasks.
It also writes logs/STATUS.txt (just the latest line). Runs and encoders that already exist are skipped, so the
script can be stopped and restarted at any time.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import pandas as pd

from .mae import pretrain_one
from .models import ARMS, PRETRAINED_INITS, mae_encoder_path
from .pool import build_all
from .train import Data, lr_choice_path, reuse_run, run_id, train_one, collect
from .utils import ROOT, get_logger, paths, read_json, resolved_datasets, write_json

log = get_logger()
MAE_SECONDS = 1300.0       # v1 measured 1238-1330 s for 15,200 steps
OVERHEAD = 4.0             # model build + test evaluation per fine-tuning run
SCRATCH_RATIO = {"A2_cnn34_scratch": 1.9}  # ResNet-34 vs ResNet-18 cost, refined by the measured ratio below


def applies(cfg, arm, dataset) -> bool:
    v = ARMS[arm].get("variant")
    if v is None:
        return True
    res = resolved_datasets(cfg)
    return dataset in [res[s] for s in cfg["mae_variants"][v]["pool_slots"]]


class Estimates:
    def __init__(self, cfg):
        v1 = pd.read_csv(ROOT / "results" / cfg["reuse_from"] / "runs.csv")
        self.base = v1.groupby(["arm", "budget_pct"]).train_seconds.mean()
        self.v1_lr: dict = {}

    def finetune(self, arm, budget) -> float:
        g = lambda a: float(self.base[(a, float(budget))])
        if arm == "A2_cnn34_scratch":
            s = SCRATCH_RATIO[arm] * g("A_cnn_scratch")
        elif arm == "F_vit_imagenet_mae":
            s = (g("C_vit_imagenet") + g("E_vit_mae")) / 2
        elif arm == "E2_vit_mae_big":
            s = g("E_vit_mae")
        else:
            s = g(arm)
        return s + OVERHEAD


def _v1_lr(cfg, est: Estimates, rid: str):
    if rid not in est.v1_lr:
        f = ROOT / "results" / cfg["reuse_from"] / "runs" / rid / "metrics.json"
        est.v1_lr[rid] = read_json(f)["lr"] if f.exists() else None
    return est.v1_lr[rid]


def lr_grid(cfg, arm):
    return cfg["train"]["lr_grid"]["pretrained" if ARMS[arm]["init"] in PRETRAINED_INITS else "scratch"]


def remaining_tasks(cfg, est: Estimates, choice: dict) -> list[tuple[str, float]]:
    """[(kind, expected seconds)] for everything not yet on disk. Work that can be copied from v1 counts as free."""
    names, runs, t = list(resolved_datasets(cfg).values()), paths(cfg)["runs"], cfg["train"]
    out, pre_seen = [], set()

    def need_pretrain(var, ds, seed):
        if (var, ds, seed) not in pre_seen and not mae_encoder_path(cfg, ds, seed, var).exists():
            pre_seen.add((var, ds, seed))
            out.append(("pretrain", MAE_SECONDS))

    for var in cfg["mae_variants"]:
        for ds in names:
            if any(applies(cfg, a, ds) and ARMS[a].get("variant") == var for a in cfg["arms"]):
                need_pretrain(var, ds, 0)
    for ds in names:
        for arm in cfg["arms"]:
            if not applies(cfg, arm, ds):
                continue
            for lr in lr_grid(cfg, arm):
                rid = run_id(ds, arm, t["tune_budget"], 0, f"tune-lr{lr:g}")
                v1 = _v1_lr(cfg, est, rid)
                if (runs / rid / "metrics.json").exists() or (v1 is not None and abs(v1 - lr) < 1e-12):
                    continue
                out.append(("tune", est.finetune(arm, t["tune_budget"]) - OVERHEAD))
    for s in t["seeds"]:
        for ds in names:
            for b in t["budgets"]:
                for arm in cfg["arms"]:
                    if not applies(cfg, arm, ds):
                        continue
                    rid = run_id(ds, arm, b, s)
                    if (runs / rid / "metrics.json").exists():
                        continue
                    lr = choice.get(f"{ds}|{arm}", {}).get("lr")
                    v1 = _v1_lr(cfg, est, rid)
                    if (lr is None and v1 is not None) or (lr is not None and v1 is not None and abs(v1 - lr) < 1e-12):
                        continue  # reusable from v1 (assumed, until tuning has decided)
                    if ARMS[arm].get("variant"):
                        need_pretrain(ARMS[arm]["variant"], ds, s)
                    out.append(("train", est.finetune(arm, b)))
    return out


class Tracker:
    def __init__(self, cfg):
        self.cfg, self.est, self.t0 = cfg, Estimates(cfg), time.time()
        self.cal = {k: [0.0, 0.0, 0] for k in ("pretrain", "tune", "train")}  # actual s, expected s, count
        self.n0 = None

    def factor(self, kind) -> float:
        a, e, n = self.cal[kind]
        return a / e if n >= 2 and e > 0 else 1.0

    def finished(self, kind, expected, actual):
        c = self.cal[kind]
        c[0] += actual; c[1] += expected; c[2] += 1

    def report(self, stage, choice):
        tasks = remaining_tasks(self.cfg, self.est, choice)
        if self.n0 is None:
            self.n0 = len(tasks)
        left = sum(s * self.factor(k) for k, s in tasks)
        fin = datetime.now() + timedelta(seconds=left)
        by = {k: sum(1 for kk, _ in tasks if kk == k) for k in self.cal}
        line = (f"ETA | stage {stage} | {len(tasks)} tasks left (MAE {by['pretrain']}, tuning {by['tune']}, "
                f"fine-tuning {by['train']}) | elapsed {(time.time() - self.t0) / 3600:.1f} h | "
                f"~{left / 3600:.1f} h left | finish ~{fin:%a %d %b %H:%M}")
        log.info(line)
        (ROOT / "logs").mkdir(exist_ok=True)
        (ROOT / "logs" / "STATUS.txt").write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {line}\n", encoding="ascii",
                                                  errors="replace")


def _timed(tr, kind, expected, fn):
    t0 = time.perf_counter()
    r = fn()
    dt = time.perf_counter() - t0
    if dt > 5 or kind == "pretrain":  # copies from v1 take no time and are not evidence about speed
        tr.finished(kind, expected, dt)
    return r


def run(cfg, stage="all"):
    tr, est = Tracker(cfg), None
    est = tr.est
    names = list(resolved_datasets(cfg).values())
    t = cfg["train"]
    choice = read_json(lr_choice_path(cfg)) if lr_choice_path(cfg).exists() else {}
    log.info("===== experiment 2 (%s): %d arms =====", cfg["experiment"], len(cfg["arms"]))
    tr.report("start", choice)

    log.info("----- stage pool -----")
    build_all(cfg)

    log.info("----- stage MAE seed 0 -----")
    for var in cfg["mae_variants"]:
        for ds in names:
            if any(applies(cfg, a, ds) and ARMS[a].get("variant") == var for a in cfg["arms"]):
                if not mae_encoder_path(cfg, ds, 0, var).exists():
                    _timed(tr, "pretrain", MAE_SECONDS, lambda: pretrain_one(cfg, ds, 0, variant=var))
                    tr.report("MAE seed 0", choice)

    log.info("----- stage tune (validation macro-F1, 20 %% budget, seed 0) -----")
    for ds in names:
        data = None
        for arm in cfg["arms"]:
            key = f"{ds}|{arm}"
            if key in choice or not applies(cfg, arm, ds):
                continue
            scores = {}
            for lr in lr_grid(cfg, arm):
                if data is None:
                    data = Data(cfg, ds)
                r = _timed(tr, "tune", est.finetune(arm, t["tune_budget"]) - OVERHEAD, lambda: train_one(
                    cfg, data, arm, t["tune_budget"], 0, lr, tag=f"tune-lr{lr:g}", evaluate_test=False))
                scores[lr] = r["best_val_macro_f1"]
                log.info("[tune] %s %s lr=%g val macro-F1 %.4f", ds, arm, lr, scores[lr])
                tr.report("tune", choice)
            choice[key] = dict(lr=max(scores, key=scores.get), val_macro_f1=scores, budget=t["tune_budget"])
            write_json(lr_choice_path(cfg), choice)
        del data
    _record_lr_changes(cfg, choice)

    log.info("----- stage train (seed-major) -----")
    data = None
    for s in t["seeds"]:
        for ds in names:
            for b in t["budgets"]:
                for arm in cfg["arms"]:
                    if not applies(cfg, arm, ds):
                        continue
                    if (paths(cfg)["runs"] / run_id(ds, arm, b, s) / "metrics.json").exists():
                        continue
                    var = ARMS[arm].get("variant")
                    if var and not mae_encoder_path(cfg, ds, s, var).exists():
                        _timed(tr, "pretrain", MAE_SECONDS, lambda: pretrain_one(cfg, ds, s, variant=var))
                    if data is None or data.name != ds:
                        data = None
                        import torch
                        torch.cuda.empty_cache()
                        data = Data(cfg, ds)
                    lr = choice[f"{ds}|{arm}"]["lr"]
                    r = _timed(tr, "train", est.finetune(arm, b), lambda: train_one(cfg, data, arm, b, s, lr))
                    log.info("[train] %s %s b=%g s=%d lr=%g  test macro-F1 %.4f  ECE %.3f  (%.0fs)", ds, arm, b, s, lr,
                             r["test_macro_f1"], r["test_ece"], r["train_seconds"])
                    tr.report("train", choice)
    collect(cfg)
    log.info("ETA | all fine-tuning finished | elapsed %.1f h", (time.time() - tr.t0) / 3600)


def _record_lr_changes(cfg, choice):
    """Which of the v1 arms got a different learning rate on the wider grid (their runs are then re-done)."""
    v1 = read_json(ROOT / "results" / cfg["reuse_from"] / "lr_choice.json")
    rows = [dict(dataset=k.split("|")[0], arm=k.split("|")[1], lr_v1=v["lr"], lr_v2=choice[k]["lr"],
                 changed=abs(v["lr"] - choice[k]["lr"]) > 1e-12) for k, v in v1.items() if k in choice]
    pd.DataFrame(rows).to_csv(paths(cfg)["results"] / "lr_changes_vs_v1.csv", index=False)
    log.info("learning rate changed for %d of %d v1 (dataset, arm) cells", sum(r["changed"] for r in rows), len(rows))
