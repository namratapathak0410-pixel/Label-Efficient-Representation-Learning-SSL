# Progress log

What was done, in order. Times are IST. Newest entries at the bottom. Commit hashes are on `main`.

## 2026-10-27

| Time | What | Result / commit |
|---|---|---|
| ~11:25 | Connected `D:\SSL` to the GitHub repo, checked out `main` and `feature/dataset-preparation` | - |
| ~11:30 | Pushed test file `hi.py` | `588d49b` |
| ~11:40 | Set repo-local git email to `sohamguha20@gmail.com` (the GitHub account email) | - |
| ~11:45 | Pushed first exploratory pipeline `plantvillage-ssl/` | `3053af5` |
| ~12:30 | Read the proposal `Plant_Disease_SSL_Research_Proposal_and_Literature_Review.docx` | - |
| ~12:40 | Ran a 20-agent literature workflow: 4 research sweeps, 7 novelty candidates, 2 adversarial checks each, final ranking | Main novelty chosen: "When does MAE overtake ImageNet, and can we predict it?" (see `NOVELTY.md`) |
| ~13:30 | Found the GPU is an **RTX 4050 Laptop 6 GB** (not a 4060). Python 3.11.9, no PyTorch yet | Plan adjusted to ViT-Tiny / ResNet-18 |
| ~14:00 | Made the design decisions with Soham (see `DECISIONS.md`) | - |
| ~14:10 | Created `mae-crossover/` with a virtual environment `.venv` | - |
| ~14:10-14:50 | Wrote the new pipeline from scratch: `run.py`, `mc/*.py`, `config.yaml`, `configs/smoke.yaml` | - |
| 14:50 | Installed PyTorch 2.11.0+cu128, timm 1.0.30 and libraries; GPU visible to PyTorch | - |
| 14:52 | Unit checks passed: ImageNet weights load, MAE encoder loads into arm E without any mismatch, GPU crop is exact, temperature scaling works, duplicate finder works | - |
| 14:55 | Wrote `PREREGISTRATION.md` (hypotheses + analysis plan) | - |
| 14:55 | Started `python run.py prepare` (downloads) | - |
| 14:57 | Moved old code to `archive/`, removed `hi.py`, committed pipeline + pre-registration **before any training** | `23e0617` |
| 15:08 | PlantVillage downloaded from Kaggle (no login needed), 54,305 images, 38 classes; 7,000 cached, 0 unreadable | - |
| 16:09 | CIFAR-10 downloaded (slow server, ~1 h), 60,000 images; 7,000 cached | - |
| 16:10 | EuroSAT downloaded, 27,000 images; 7,000 cached | - |
| 16:10 | Oxford Pets downloading (~790 MB at ~0.5 MB/s) | - |
| 16:15 | Started these documentation files (`docs/`) at Soham's request | `66a1f3e` |
| 16:17 | Oxford Pets cached: 7,349 available, 7,000 used, 37 classes, 0 unreadable | - |
| 16:17 | ImageNetV2 download started, then interrupted when the Claude session ended | - |
| ~16:40 | Checked caches: all 4 datasets complete. Class counts in the 7,000-image samples: CIFAR-10 700 each; EuroSAT 519-778; Pets 176-191; PlantVillage 19-710 (very imbalanced: smallest class gets 1 labelled image at 5 %) | - |
| ~16:40 | Restarted `prepare` (finished datasets are skipped) to fetch ImageNetV2 | in progress |
| ~16:45 | Wrote `docs/THEORY.md` (full theory, every number explained). Verified parameter counts by hand against PyTorch: ViT-Tiny 5,507,914 and ResNet-18 11,181,642 (10 classes) | this commit |

## Next
1. Finish downloads (Pets, ImageNetV2).
2. Smoke test (`configs/smoke.yaml`) to catch bugs end to end.
3. Speed benchmark, then lock the time plan.
4. `split` -> `distance` -> `pretrain` -> `tune` -> `train` -> `analyze`.
