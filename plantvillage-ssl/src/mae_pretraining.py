"""Masked Autoencoder (He et al., 2022) pretraining on the UNLABELLED images of the TRAIN split.

Idea (plain words): hide 75 % of the image patches, let a ViT encoder see only the 25 % that remain, and ask a
small decoder to repaint the hidden patches. To do this well the encoder must learn what leaves look like
(texture, veins, lesion shapes) - without any label. We then reuse the encoder for classification.

Deviations from the original code (documented for the thesis): positional embeddings are initialised with
fixed 2-D sin-cos values but left trainable in the encoder; the architecture sizes are set in the YAML config.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from .data_loader import ImageDataset, load_split
from .models import create_vit
from .preprocessing import denormalize, mae_transform
from .utils import get_device, get_logger, get_paths, set_seed

log = get_logger()


# ----------------------------------------------------------------------------- sin-cos position embeddings
def _sincos_1d(dim, pos):
    omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=np.float64) / (dim / 2.0))
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def sincos_2d(dim, grid, cls_token=True):
    g = np.arange(grid, dtype=np.float32)
    gw, gh = np.meshgrid(g, g)
    emb = np.concatenate([_sincos_1d(dim // 2, gh), _sincos_1d(dim // 2, gw)], axis=1)
    if cls_token:
        emb = np.concatenate([np.zeros([1, dim]), emb], axis=0)
    return torch.from_numpy(emb).float().unsqueeze(0)


# ----------------------------------------------------------------------------- model
class MAE(nn.Module):
    def __init__(self, encoder, decoder_dim=128, decoder_depth=2, decoder_heads=4, mask_ratio=0.75,
                 norm_pix_loss=True):
        super().__init__()
        self.encoder = encoder
        self.patch = encoder.patch_embed.patch_size[0]
        self.grid = encoder.patch_embed.grid_size[0]
        self.num_patches = encoder.patch_embed.num_patches
        self.mask_ratio, self.norm_pix_loss = mask_ratio, norm_pix_loss
        D = encoder.embed_dim

        with torch.no_grad():
            encoder.pos_embed.copy_(sincos_2d(D, self.grid, True))
        self.decoder_embed = nn.Linear(D, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.register_buffer("decoder_pos", sincos_2d(decoder_dim, self.grid, True))
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_dim, decoder_heads, mlp_ratio=4.0, qkv_bias=True) for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch ** 2 * 3)
        nn.init.normal_(self.mask_token, std=0.02)

    # image <-> patches
    def patchify(self, imgs):
        B, _, H, W = imgs.shape
        p, h, w = self.patch, H // self.patch, W // self.patch
        x = imgs.reshape(B, 3, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(B, h * w, p * p * 3)

    def unpatchify(self, x):
        p, h = self.patch, self.grid
        B = x.shape[0]
        x = x.reshape(B, h, h, p, p, 3)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(B, 3, h * p, h * p)

    def random_masking(self, x):
        B, N, D = x.shape
        keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)  # 1 = removed, 0 = kept
        return x_keep, mask, ids_restore

    def forward_encoder(self, imgs):
        e = self.encoder
        x = e.patch_embed(imgs) + e.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x)
        cls = (e.cls_token + e.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in e.blocks:
            x = blk(x)
        return e.norm(x), mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        B, L, D = x.shape
        n_mask = ids_restore.shape[1] + 1 - L
        full = torch.cat([x[:, 1:, :], self.mask_token.expand(B, n_mask, -1)], dim=1)
        full = torch.gather(full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = torch.cat([x[:, :1, :], full], dim=1) + self.decoder_pos
        for blk in self.decoder_blocks:
            x = blk(x)
        return self.decoder_pred(self.decoder_norm(x))[:, 1:, :]

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        loss = (((pred.float() - target) ** 2).mean(-1) * mask).sum() / mask.sum()  # loss only on hidden patches
        return loss, pred, mask


# ----------------------------------------------------------------------------- helpers
def _lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    return 1e-6 + (base - 1e-6) * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


@torch.no_grad()
def save_reconstructions(model, imgs, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    _, pred, mask = model(imgs)
    target = model.patchify(imgs)
    if model.norm_pix_loss:
        mean, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
        pred = pred * (var + 1e-6) ** 0.5 + mean
    m = mask.unsqueeze(-1)
    recon = model.unpatchify(pred * m + target * (1 - m))
    masked = model.unpatchify(target * (1 - m))
    rows = [denormalize(masked), denormalize(recon), denormalize(imgs)]
    n = imgs.shape[0]
    fig, ax = plt.subplots(3, n, figsize=(1.6 * n, 5))
    for r, (name, t) in enumerate(zip(["masked input", "MAE reconstruction", "original"], rows)):
        for i in range(n):
            ax[r, i].imshow(t[i].permute(1, 2, 0).cpu().numpy())
            ax[r, i].axis("off")
        ax[r, 0].set_title(name, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- main routine
def pretrain(cfg, force: bool = False):
    paths = get_paths(cfg)
    out = paths["mae"]
    enc_path, last_path = out / "encoder.pt", out / "last.pt"
    if enc_path.exists() and not force:
        log.info("MAE encoder already exists: %s (use --force to retrain)", enc_path)
        return enc_path
    mc = cfg["mae"]
    set_seed(mc["seed"])
    device = get_device()
    split = load_split(cfg)
    train_df = split[split["split"] == "train"]  # ONLY train images; labels are never read (with_labels=False)
    d = cfg["data"]
    ds = ImageDataset(train_df, d["root"], mae_transform(d["img_size"]),
                      cache_size=d.get("cache_size") if d.get("cache_in_ram") else None, with_labels=False)
    bs = min(mc["batch_size"], len(ds))
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(ds) >= bs * 2,
                                         num_workers=d.get("num_workers", 0), pin_memory=device.type == "cuda")
    encoder = create_vit(cfg["models"]["vit_arch"], num_classes=0, img_size=d["img_size"], drop_path=0.0)
    model = MAE(encoder, mc["decoder_dim"], mc["decoder_depth"], mc["decoder_heads"], mc["mask_ratio"],
                mc["norm_pix_loss"]).to(device)
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim <= 1 or "pos_embed" in n_ or "cls_token" in n_ or "mask_token" in n_ else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": mc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=mc["lr"], betas=(0.9, 0.95))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps_per_epoch = len(loader)
    total, warm = mc["epochs"] * steps_per_epoch, mc["warmup_epochs"] * steps_per_epoch

    start_epoch, elapsed, hist = 0, 0.0, []
    if last_path.exists() and not force:
        ck = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch, elapsed, hist = ck["epoch"], ck["elapsed"], ck["hist"]
        log.info("Resuming MAE pretraining from epoch %d", start_epoch)

    log.info("MAE pretraining: %d unlabeled train images, mask ratio %.2f, %d epochs, device=%s",
             len(ds), mc["mask_ratio"], mc["epochs"], device)
    step = start_epoch * steps_per_epoch
    t0 = time.perf_counter()
    for epoch in range(start_epoch, mc["epochs"]):
        model.train()
        losses = []
        for x in loader:
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, total, warm, mc["lr"])
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, _, _ = model(x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            losses.append(loss.item()); step += 1
        hist.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), lr=opt.param_groups[0]["lr"]))
        if (epoch + 1) % max(1, mc["epochs"] // 20) == 0 or epoch == mc["epochs"] - 1:
            log.info("MAE epoch %d/%d  loss %.4f", epoch + 1, mc["epochs"], hist[-1]["loss"])
        if (epoch + 1) % 10 == 0:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=epoch + 1,
                            elapsed=elapsed + time.perf_counter() - t0, hist=hist), last_path)
    elapsed += time.perf_counter() - t0

    torch.save(model.encoder.state_dict(), enc_path)
    pd.DataFrame(hist).to_csv(out / "mae_loss.csv", index=False)
    info = dict(pretrain_time_s=elapsed, epochs=mc["epochs"], n_unlabeled_images=len(ds), mask_ratio=mc["mask_ratio"],
                device=str(device), vit_arch=cfg["models"]["vit_arch"], img_size=d["img_size"],
                final_loss=hist[-1]["loss"] if hist else None, uses_labels=False,
                encoder_params=sum(p.numel() for p in model.encoder.parameters()))
    (out / "pretrain_info.json").write_text(json.dumps(info, indent=2))
    try:
        x = next(iter(loader))[:8].to(device)
        save_reconstructions(model, x, out / "reconstructions.png")
        import matplotlib.pyplot as plt

        h = pd.DataFrame(hist)
        plt.figure(figsize=(5, 3.4)); plt.plot(h["epoch"], h["loss"]); plt.xlabel("epoch"); plt.ylabel("MAE loss")
        plt.title("MAE pretraining loss"); plt.tight_layout(); plt.savefig(out / "mae_loss.png", dpi=120); plt.close()
    except Exception as e:  # plotting must never kill a finished pretraining
        log.warning("Could not save reconstruction figure: %s", e)
    log.info("MAE encoder saved: %s (%.1f min)", enc_path, elapsed / 60)
    return enc_path
