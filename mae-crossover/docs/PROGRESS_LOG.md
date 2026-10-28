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
| ~16:45 | Wrote `docs/THEORY.md` (full theory, every number explained). Verified parameter counts by hand against PyTorch: ViT-Tiny 5,507,914 and ResNet-18 11,181,642 (10 classes) | `eefd787` |
| 16:28 | ImageNetV2 downloaded and cached (10,000 images, 1,000 classes, 0 unreadable). No fallback dataset was needed for any domain | - |
| 16:28 | Smoke test, attempt 1: **failed** - `pets.npz` was corrupt (its write was cut off when the earlier Claude session ended, but the file existed so it looked finished) | bug found |
| 16:30 | Fix 1: caches are now written to a `.partial.npz` file and renamed only when complete; an unreadable cache is detected and rebuilt automatically | - |
| 16:32 | Fix 1 exposed bug 2: Windows kept the broken file open, so it could not be deleted and the fallback switched Pets to the Kaggle copy. Fix 2: open the file with our own handle. Deleted the Kaggle copy, rebuilt Pets from the official source | - |
| 16:33-16:42 | Smoke test, attempt 2: **all 8 stages passed end to end** in 8.9 min (64 tiny runs; numbers meaningless by design) | - |
| 16:43 | Main splits created (identical to smoke splits, same seed 42): ~5,000 / 1,000 / 1,000 per dataset. Duplicate groups: PlantVillage 0, CIFAR-10 0, EuroSAT 1 (78 near-uniform sea/lake images), Pets 34 (69 images) | - |
| 16:45 | Speed benchmark: ResNet-18 126 ms/step, ViT 115-117 ms/step, MAE 85 ms/step (peak 1.9 GB GPU). Worst-case plan **16.1 h** (MAE 2.2 h, tuning 1.3 h, grid 12.7 h) vs 48 h available | `results/crossover_v1/benchmark.json` |
| 16:46 | Started `distance` (FID, OTDD, ImageNet confidence, linear probe) | - |
| 16:45 | Committed fixes, splits, benchmark | `8269a4c` |
| 16:47 | Distances done. FID: PlantVillage 433, EuroSAT 318, Pets 273, CIFAR-10 267. ImageNet confidence: Pets 0.51, PlantVillage 0.30, CIFAR-10 0.29, EuroSAT 0.14. OTDD: PlantVillage 30.6, Pets 29.3, CIFAR-10 28.3, EuroSAT 26.4. Linear probe: PlantVillage 0.90, Pets 0.86, EuroSAT 0.86, CIFAR-10 0.83. **The measures disagree on the ordering** | `results/crossover_v1/domain_distance.csv` |
| 16:48 | Soham chose to spend spare time on MAE 800 epochs. Recorded as pre-registration amendment 1 together with the measured distances, committed **before** any MAE or main training | `8daa385` |
| 16:49 | Wrote `run_experiment.ps1` and launched it in its own window | - |
| 16:50 | Log file was unreadable (Windows PowerShell's Tee-Object writes UTF-16, Python appends ASCII). Stopped after ~1 min of MAE, deleted the partial MAE folder, fixed the script to write ASCII, relaunched | - |
| 16:50 | **Main experiment running**: MAE pretraining PlantVillage seed 0, 15,200 steps. Progress in `logs/experiment.log` | running |
| (times above 16:40 corrected to git/log timestamps; earlier entries marked ~ are approximate) | | |
| 18:13 | MAE pretraining seed 0 done for all 4 datasets (20.5-20.6 min each, 82.5 min total) | `checkpoints/crossover_v1/*/mae_seed0/` |
| 19:11 | Learning-rate selection done (48 runs, 58.4 min) | `results/crossover_v1/lr_choice.json` |
| 19:11 | Main grid started (240 runs, seed-major) | - |
| 22:21 | Seed 0 finished (runs 1-80) | - |

## 2026-10-28

| Time | What | Result / commit |
|---|---|---|
| 22:24 (10-27) - 06:47 | MAE pretraining for seeds 1 and 2 ran inside the grid, just before arm E needed each one (8 x 21.7-22.2 min) | `checkpoints/crossover_v1/*/mae_seed{1,2}/` |
| 07:27 | Main grid done: 240/240 runs, no failures. 9.3 h of fine-tuning plus ~2.9 h of MAE | `results/crossover_v1/runs.csv` |
| 07:27 | `analyze` done; `run_experiment.ps1` auto-committed and pushed the results. Whole experiment 16:50 -> 07:27 = **14.6 h** (planned worst case ~18 h) | `c3ff9cb` |
| 07:40 | Read the results against the pre-registered hypotheses; wrote `docs/FINDINGS.md`. Short version: ImageNet ViT wins every cell, the MAE gap shrinks with labels in every domain but never crosses (H1 partly, H4 rejected), MAE > scratch ViT (H3 mostly), MAE ~ ResNet scratch | this commit |

| ~08:00-11:30 | Walked Soham through the whole project, part by part (problem, arms, data, distances, MAE, fine-tuning, running) | - |
| ~11:30 | Wrote the IEEE two-column paper in `paper/`. All figures and data tables are generated from the results by `paper/make_figures.py`, and all 34 references were checked against arXiv or the publisher. Not yet compiled: there is no LaTeX on the laptop, and installing Tectonic was blocked by the permission system, so it was checked statically instead (braces, environments, citations, labels, files) | this commit |

## Next
1. Compile the paper (Overleaf zip, or allow a local LaTeX install) and check that it is 5-7 pages
2. Look at the plots (`results/crossover_v1/plots/`) and MAE reconstructions.
2. Decide with Soham how to present the post-hoc extrapolation (FINDINGS.md section 3) and whether any follow-up is worth running in the remaining time.
3. Thesis write-up: methods from PREREGISTRATION.md and THEORY.md, results from REPORT.md and FINDINGS.md.
