"""Fine-tuning one (dataset, arm, budget, seed) run, the learning-rate selection, and the full grid.

Protocol (identical for every arm; declared in PREREGISTRATION.md):
  AdamW (wd 0.05, none on biases/norms/tokens), batch 128, bf16 autocast, grad-clip 1.0, linear warm-up + cosine.
  Steps = clamp(epochs_equiv * ceil(n_labelled / batch), min_steps, max_steps).
  Validation macro-F1 is measured cfg.train.eval_points times; stop after `patience` evaluations without improvement
  and restore the best-validation weights. The test set is evaluated exactly once, after that restore.
  Learning rate per (dataset, arm) is chosen on validation macro-F1 at the tuning budget with seed 0 only.
"""
from __future__ import annotations

import copy
import math
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import load_cache
from .gpu_data import GPUImages
from .metrics import evaluate_logits, full_report, per_class
from .models import ARMS, PRETRAINED_INITS, build, n_params, param_groups
from .split import budget_indices, make_split
from .utils import ROOT, device, get_logger, paths, read_json, resolved_datasets, set_seed, write_json

log = get_logger()


class Data:
    """One dataset on the GPU, with fixed val/test index tensors."""

    def __init__(self, cfg, name):
        dev = device()
        self.name, self.df = name, make_split(name, cfg)
        c = load_cache(name, cfg)
        self.n_classes = len(c["class_names"])
        self.gpu = GPUImages(c["images"], c["labels"], dev)
        self.val = torch.tensor(self.df.loc[self.df.split == "val", "idx"].to_numpy(), device=dev)
        self.test = torch.tensor(self.df.loc[self.df.split == "test", "idx"].to_numpy(), device=dev)


@torch.no_grad()
def predict(model, data: Data, idx: torch.Tensor, bs: int = 512) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    outs, ys = [], []
    for i in range(0, len(idx), bs):
        x, y = data.gpu.batch(idx[i:i + bs], None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs.append(model(x).float())
        ys.append(y)
    return torch.cat(outs).cpu().numpy(), torch.cat(ys).cpu().numpy()


def schedule(cfg, n_lab):
    t = cfg["train"]
    bs = min(t["batch_size"], n_lab)
    steps = int(min(t["max_steps"], max(t["min_steps"], t["epochs_equiv"] * math.ceil(n_lab / bs))))
    return bs, steps


def run_id(dataset, arm, budget, seed, tag="main"):
    return f"{tag}__{dataset}__{arm}__b{budget:g}__s{seed}"


def reuse_run(cfg, rid: str, lr: float):
    """Copy a finished run of the earlier experiment (cfg.reuse_from) if it used exactly this learning rate.
    Identical data, split, labelled subset, seed and protocol, so the run is the same experiment, not a new one."""
    src_exp = cfg.get("reuse_from")
    if not src_exp:
        return None
    src = ROOT / "results" / src_exp / "runs" / rid
    if not (src / "metrics.json").exists():
        return None
    m = read_json(src / "metrics.json")
    if abs(m["lr"] - lr) > 1e-12:
        return None
    shutil.copytree(src, paths(cfg)["runs"] / rid)
    return m


def train_one(cfg, data: Data, arm: str, budget: float, seed: int, lr: float, tag: str = "main",
              evaluate_test: bool = True, max_steps: int | None = None) -> dict:
    rid = run_id(data.name, arm, budget, seed, tag)
    out = paths(cfg)["runs"] / rid
    if (out / "metrics.json").exists() and max_steps is None:
        return read_json(out / "metrics.json")
    if max_steps is None:
        m = reuse_run(cfg, rid, lr)
        if m is not None:
            return m
    t_cfg, dev = cfg["train"], device()
    set_seed(seed)
    lab = torch.tensor(budget_indices(data.df, budget, seed), device=dev)
    bs, steps = schedule(cfg, len(lab))
    if max_steps is not None:
        steps = max_steps
    model, info = build(arm, cfg, data.n_classes, data.name, seed)
    model = model.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(param_groups(model, arm, cfg, lr, t_cfg["weight_decay"]), lr=lr, fused=True)
    warm = max(1, int(t_cfg["warmup_frac"] * steps))
    eval_every = max(1, steps // t_cfg["eval_points"])
    gen = torch.Generator(device=dev).manual_seed(seed)
    best, best_state, bad, hist = -1.0, None, 0, []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    step, perm, pos = 0, None, len(lab)
    stopped_early = False
    while step < steps:
        if pos + bs > len(lab):
            perm, pos = lab[torch.randperm(len(lab), device=dev, generator=gen)], 0
        idx = perm[pos:pos + bs]
        pos += bs
        f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
        for g in opt.param_groups:
            g["lr"] = lr * g["lr_scale"] * f
        model.train()
        x, y = data.gpu.batch(idx, "finetune", gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x.contiguous(memory_format=torch.channels_last)), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % eval_every == 0 or step == steps:
            lv, yv = predict(model, data, data.val)
            m = evaluate_logits(lv, yv, data.n_classes)
            hist.append(dict(step=step, train_loss=float(loss), val_macro_f1=m["macro_f1"], val_acc=m["accuracy"],
                             val_nll=m["nll"]))
            if m["macro_f1"] > best + 1e-6:
                best, bad = m["macro_f1"], 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= t_cfg["patience"] and max_steps is None:
                    stopped_early = True
                    break
    torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    if max_steps is not None:
        return dict(seconds=train_s, steps=step, images_per_second=step * bs / train_s)
    model.load_state_dict(best_state)
    lv, yv = predict(model, data, data.val)
    res = dict(run_id=rid, tag=tag, dataset=data.name, arm=arm, family=ARMS[arm]["family"], init=ARMS[arm]["init"],
               budget_pct=budget, seed=seed, lr=lr, n_labeled=int(len(lab)), batch_size=bs, planned_steps=steps,
               steps_run=step, stopped_early=stopped_early, train_seconds=train_s,
               peak_gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9, n_params=n_params(model),
               best_val_macro_f1=best, **info)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    if evaluate_test:
        lt, yt = predict(model, data, data.test)
        res.update(full_report(lv, yv, lt, yt, data.n_classes))
        np.savez_compressed(out / "logits.npz", val=lv.astype(np.float16), val_y=yv, test=lt.astype(np.float16),
                            test_y=yt)
        write_json(out / "per_class.json", per_class(lt, yt, data.n_classes))
    write_json(out / "metrics.json", res)
    del model, opt, best_state
    torch.cuda.empty_cache()
    return res


# ----------------------------------------------------------------------------- learning-rate selection
def lr_choice_path(cfg):
    return paths(cfg)["results"] / "lr_choice.json"


def tune(cfg, datasets=None, arms=None):
    t = cfg["train"]
    choice = read_json(lr_choice_path(cfg)) if lr_choice_path(cfg).exists() else {}
    for name in datasets or resolved_datasets(cfg).values():
        data = Data(cfg, name)
        for arm in arms or cfg["arms"]:
            key = f"{name}|{arm}"
            if key in choice:
                continue
            grid = t["lr_grid"]["pretrained" if ARMS[arm]["init"] in PRETRAINED_INITS else "scratch"]
            scores = {}
            for lr in grid:
                r = train_one(cfg, data, arm, t["tune_budget"], 0, lr, tag=f"tune-lr{lr:g}", evaluate_test=False)
                scores[lr] = r["best_val_macro_f1"]
                log.info("[tune] %s %s lr=%g val macro-F1 %.4f", name, arm, lr, scores[lr])
            choice[key] = dict(lr=max(scores, key=scores.get), val_macro_f1=scores, budget=t["tune_budget"])
            write_json(lr_choice_path(cfg), choice)
        del data
        torch.cuda.empty_cache()
    return choice


# ----------------------------------------------------------------------------- main grid
def grid(cfg, seeds=None, datasets=None, arms=None, budgets=None):
    """Seed-major order: every dataset gets seed 0 before any dataset gets seed 1 (a stop leaves balanced data)."""
    t = cfg["train"]
    choice = read_json(lr_choice_path(cfg))
    names = datasets or list(resolved_datasets(cfg).values())
    arms = arms or cfg["arms"]
    budgets = budgets or t["budgets"]
    todo = [(s, n, b, a) for s in (seeds or t["seeds"]) for n in names for b in budgets for a in arms]
    log.info("grid: %d runs", len(todo))
    done_t0, n_done, data = time.perf_counter(), 0, None
    for k, (s, n, b, a) in enumerate(todo):
        if (paths(cfg)["runs"] / run_id(n, a, b, s) / "metrics.json").exists():
            continue
        if a == "E_vit_mae":
            from .mae import pretrain_one

            pretrain_one(cfg, n, s)  # no-op when this seed's encoder already exists
        if data is None or data.name != n:
            data = None
            torch.cuda.empty_cache()
            data = Data(cfg, n)
        r = train_one(cfg, data, a, b, s, choice[f"{n}|{a}"]["lr"])
        n_done += 1
        rate = (time.perf_counter() - done_t0) / n_done
        left = sum(1 for (s2, n2, b2, a2) in todo[k + 1:]
                   if not (paths(cfg)["runs"] / run_id(n2, a2, b2, s2) / "metrics.json").exists())
        log.info("[%d/%d] %s %s b=%g s=%d  test macro-F1 %.4f  ECE %.3f  (%.0fs; ~%.1f h left)", k + 1, len(todo),
                 n, a, b, s, r["test_macro_f1"], r["test_ece"], r["train_seconds"], left * rate / 3600)
    collect(cfg)


def collect(cfg) -> pd.DataFrame:
    rows = [read_json(p) for p in paths(cfg)["runs"].glob("main__*/metrics.json")]
    df = pd.DataFrame(rows)
    if len(df):
        df.sort_values(["dataset", "arm", "budget_pct", "seed"]).to_csv(paths(cfg)["results"] / "runs.csv",
                                                                         index=False)
    return df
