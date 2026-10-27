"""Masked Autoencoder pretraining (He et al., 2022) on the unlabelled TRAIN images of one dataset.

The encoder is exactly the ViT used by arms C/D/E (timm, same img_size), so its weights load 1:1 into arm E.
Labels, validation and test images are never touched. Pretraining length is a fixed number of epochs over a pool of
identical size in every dataset, so every domain gets the same MAE compute.
Deviations from the reference code: shallow decoder (cfg.mae.decoder_depth, small-data MAE studies find 1-2 blocks
sufficient), encoder position embeddings start from fixed 2-D sin-cos values and stay trainable.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from .data import load_cache
from .gpu_data import GPUImages
from .models import create_vit, mae_encoder_path
from .split import make_split
from .utils import device, get_logger, resolved_datasets, set_seed, write_json

log = get_logger()


def _sincos_1d(dim, pos):
    omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=np.float64) / (dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def sincos_2d(dim, grid):
    g = np.arange(grid, dtype=np.float32)
    gw, gh = np.meshgrid(g, g)
    emb = np.concatenate([_sincos_1d(dim // 2, gh), _sincos_1d(dim // 2, gw)], axis=1)
    emb = np.concatenate([np.zeros([1, dim]), emb], axis=0)  # cls token slot
    return torch.from_numpy(emb).float().unsqueeze(0)


class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim, decoder_depth, decoder_heads, mask_ratio, norm_pix_loss=True):
        super().__init__()
        self.encoder = encoder
        self.patch = encoder.patch_embed.patch_size[0]
        self.grid = encoder.patch_embed.grid_size[0]
        self.mask_ratio, self.norm_pix_loss = mask_ratio, norm_pix_loss
        dim = encoder.embed_dim
        with torch.no_grad():
            encoder.pos_embed.copy_(sincos_2d(dim, self.grid))
        self.decoder_embed = nn.Linear(dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.register_buffer("decoder_pos", sincos_2d(decoder_dim, self.grid))
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_dim, decoder_heads, mlp_ratio=4.0, qkv_bias=True) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch ** 2 * 3)
        nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, imgs):
        b, p, h = imgs.shape[0], self.patch, self.grid
        x = imgs.reshape(b, 3, h, p, h, p)
        return torch.einsum("nchpwq->nhwpqc", x).reshape(b, h * h, p * p * 3)

    def unpatchify(self, x):
        b, p, h = x.shape[0], self.patch, self.grid
        x = torch.einsum("nhwpqc->nchpwq", x.reshape(b, h, h, p, p, 3))
        return x.reshape(b, 3, h * p, h * p)

    def forward(self, imgs):
        e = self.encoder
        x = e.patch_embed(imgs) + e.pos_embed[:, 1:, :]
        b, n, d = x.shape
        keep = int(n * (1 - self.mask_ratio))
        ids_shuffle = torch.argsort(torch.rand(b, n, device=x.device), dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        x = torch.gather(x, 1, ids_shuffle[:, :keep, None].expand(-1, -1, d))
        mask = torch.ones(b, n, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)  # 1 = hidden
        cls = (e.cls_token + e.pos_embed[:, :1, :]).expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in e.blocks:
            x = blk(x)
        x = self.decoder_embed(e.norm(x))
        dd = x.shape[-1]
        full = torch.cat([x[:, 1:, :], self.mask_token.expand(b, n - keep, -1)], dim=1)
        full = torch.gather(full, 1, ids_restore[..., None].expand(-1, -1, dd))
        x = torch.cat([x[:, :1, :], full], dim=1) + self.decoder_pos
        for blk in self.decoder_blocks:
            x = blk(x)
        pred = self.decoder_pred(self.decoder_norm(x))[:, 1:, :]
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = (((pred.float() - target) ** 2).mean(-1) * mask).sum() / mask.sum()  # hidden patches only
        return loss, pred, mask


def _lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@torch.no_grad()
def save_reconstructions(model, imgs, mean, std, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, pred, mask = model(imgs)
    pred = pred.float()
    target = model.patchify(imgs)
    if model.norm_pix_loss:
        m, v = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
        pred = pred * (v + 1e-6) ** 0.5 + m
    mk = mask[..., None]
    rows = [model.unpatchify(target * (1 - mk)), model.unpatchify(pred * mk + target * (1 - mk)), imgs]
    n = imgs.shape[0]
    fig, ax = plt.subplots(3, n, figsize=(1.6 * n, 5))
    for r, (title, t) in enumerate(zip(["masked input", "reconstruction", "original"], rows)):
        t = (t * std + mean).clamp(0, 1)
        for i in range(n):
            ax[r, i].imshow(t[i].permute(1, 2, 0).float().cpu().numpy())
            ax[r, i].axis("off")
        ax[r, 0].set_title(title, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def pretrain_one(cfg: dict, dataset: str, seed: int, max_steps: int | None = None) -> dict:
    enc_path = mae_encoder_path(cfg, dataset, seed)
    out = enc_path.parent
    out.mkdir(parents=True, exist_ok=True)
    if enc_path.exists() and max_steps is None:
        return {}
    mc, dev = cfg["mae"], device()
    set_seed(1000 + seed)
    df = make_split(dataset, cfg)
    c = load_cache(dataset, cfg)
    tr_idx = df.loc[df.split == "train", "idx"].to_numpy()  # unlabelled pool = train split only
    data = GPUImages(c["images"][tr_idx], np.zeros(len(tr_idx), dtype=np.int64), dev)  # labels are NOT given
    encoder = create_vit(cfg, num_classes=0)
    model = MAE(encoder, mc["decoder_dim"], mc["decoder_depth"], mc["decoder_heads"], mc["mask_ratio"]).to(dev)
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or any(k in n_ for k in ("pos_embed", "cls_token", "mask_token")) else decay).append(p)
    bs = mc["batch_size"]
    base_lr = mc["blr"] * bs / 256
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": mc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=base_lr, betas=(0.9, 0.95), fused=True)
    spe = max(1, len(data) // bs)
    total = mc["epochs"] * spe if max_steps is None else max_steps
    warm = min(mc["warmup_epochs"] * spe, total // 10)
    gen = torch.Generator(device=dev).manual_seed(1000 + seed)
    last = out / "last.pt"
    step, hist, elapsed = 0, [], 0.0
    if last.exists() and max_steps is None:
        ck = torch.load(last, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, hist, elapsed = ck["step"], ck["hist"], ck["elapsed"]
        log.info("[%s] resuming MAE seed %d at step %d/%d", dataset, seed, step, total)
    log.info("[%s] MAE seed %d: %d unlabelled train images, %d steps (%d epochs), mask %.2f", dataset, seed, len(data),
             total, total // spe, mc["mask_ratio"])
    model.train()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    losses = []
    while step < total:
        perm = torch.randperm(len(data), device=dev, generator=gen)
        for i in range(spe):
            if step >= total:
                break
            x, _ = data.batch(perm[i * bs:(i + 1) * bs], "mae", gen)
            for g in opt.param_groups:
                g["lr"] = _lr(step, total, warm, base_lr)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = model(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.detach())
            step += 1
            if step % spe == 0 or step == total:
                hist.append(dict(step=step, epoch=step / spe, loss=float(torch.stack(losses).mean())))
                losses = []
                ep = step // spe
                if max_steps is None and (ep % max(1, mc["epochs"] // 10) == 0 or step == total):
                    log.info("[%s] MAE epoch %d/%d loss %.4f", dataset, ep, mc["epochs"], hist[-1]["loss"])
                if max_steps is None and ep % 50 == 0:
                    torch.cuda.synchronize()
                    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, hist=hist,
                                    elapsed=elapsed + time.perf_counter() - t0), last)
    torch.cuda.synchronize()
    elapsed += time.perf_counter() - t0
    info = dict(dataset=dataset, seed=seed, steps=total, epochs=total / spe, n_unlabelled=len(data),
                batch_size=bs, base_lr=base_lr, mask_ratio=mc["mask_ratio"], decoder_depth=mc["decoder_depth"],
                decoder_dim=mc["decoder_dim"], pretrain_seconds=elapsed, images_per_second=total * bs / elapsed,
                final_loss=hist[-1]["loss"] if hist else None, uses_labels=False, uses_val_or_test=False,
                peak_gpu_mem_gb=torch.cuda.max_memory_allocated() / 1e9)
    if max_steps is not None:
        return info
    torch.save(model.encoder.state_dict(), enc_path)
    pd.DataFrame(hist).to_csv(out / "mae_loss.csv", index=False)
    write_json(out / "pretrain_info.json", info)
    try:
        x, _ = data.batch(torch.arange(8, device=dev), None)
        save_reconstructions(model, x, data.mean, data.std, out / "reconstructions.png")
    except Exception as e:  # a figure must never lose a finished pretraining
        log.warning("reconstruction figure failed: %s", e)
    last.unlink(missing_ok=True)
    log.info("[%s] MAE seed %d done in %.1f min -> %s", dataset, seed, elapsed / 60, enc_path)
    del model, data
    torch.cuda.empty_cache()
    return info


def pretrain_all(cfg: dict, seeds=None, datasets=None):
    """Default: seed 0 only (needed by `tune`); the grid pretrains the other seeds just before they are needed."""
    seeds = seeds or [cfg["train"]["seeds"][0]]
    for s in seeds:
        for name in datasets or resolved_datasets(cfg).values():
            pretrain_one(cfg, name, s)
