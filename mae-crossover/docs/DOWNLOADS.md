# Downloads

Everything downloaded to this machine for the project, where it lives, and why. None of the data is in git
(`data/` and `checkpoints/` are git-ignored); anyone can re-create it with `python run.py prepare`.

## Software (into `mae-crossover/.venv`, ~3 GB)
| Package | Version | Why |
|---|---|---|
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 | Deep learning on the RTX 4050 (CUDA 12.8 build) |
| timm | 1.0.30 | ResNet-18 and DeiT-Tiny (ViT) models and ImageNet weights |
| numpy, pandas, scipy, scikit-learn | 2.4.6, 3.1.0, 1.17.1, 1.9.1 | Data handling, splits, statistics, linear probe |
| matplotlib | 3.11.2 | Plots |
| pot | 0.9.7 | Optimal transport for the OTDD distance |
| kagglehub | 1.0.3 | PlantVillage download |
| huggingface_hub | 2.3.0 | ImageNetV2 and DeiT weights download |
| pyyaml, pillow, tqdm | - | Config, image decoding, progress bars |

## Datasets
| Dataset | Source | Images available | Used | Stored at | Size | Licence / note |
|---|---|---|---|---|---|---|
| PlantVillage (colour) | Kaggle `abdallahalidev/plantvillage-dataset` (no login needed) | 54,305 / 38 classes | 7,000 | `C:\Users\Soham\.cache\kagglehub\...` | 2.5 GB (zip contains colour, grayscale, segmented; only colour used) | CC BY-SA (Hughes & Salathe 2015) |
| CIFAR-10 | torchvision (official Toronto server) | 60,000 / 10 classes | 7,000 | `data/raw/torchvision/` | 170 MB | Krizhevsky 2009 |
| EuroSAT RGB | torchvision | 27,000 / 10 classes | 7,000 | `data/raw/torchvision/` | 90 MB | MIT (Helber et al. 2019) |
| Oxford-IIIT Pets | torchvision (official Oxford server) | 7,349 / 37 classes | 7,000 | `data/raw/torchvision/` | ~790 MB | CC BY-SA 4.0 (Parkhi et al. 2012) |
| ImageNetV2 matched-frequency | Hugging Face `vaishaal/ImageNetV2` | 10,000 / 1,000 classes | all | `data/raw/imagenetv2/` | ~1.2 GB | Reference set for the domain-distance measures only (Recht et al. 2019) |

Soham's Kaggle links for CIFAR-10, EuroSAT and Pets point to copies of the same datasets; torchvision downloads the
original releases directly (no account, verified checksums). The Kaggle copies are kept as automatic fallbacks.

## Pretrained weights
| Weights | Source | Used by |
|---|---|---|
| `deit_tiny_patch16_224` (ImageNet-1k, Touvron et al. 2021) | Hugging Face via timm | Arm C, and the frozen feature extractor for domain distances |

## Derived files (created by the pipeline)
| File | What |
|---|---|
| `data/cache/160px/<dataset>.npz` | The 7,000 sampled images per dataset, decoded to 160x160 uint8, with labels, ids and 256-bit dHashes (~540 MB each) |
| `data/cache/160px/<dataset>_info.json` | Class counts, number of corrupted files, how many were available |
| `data/cache/160px/resolved_datasets.json` | Which dataset was actually used for each domain (after fallbacks) |

## Download log
| Date / time | Item | Outcome |
|---|---|---|
| 10-27 14:50 | Python packages | OK |
| 10-27 14:51 | DeiT-Tiny weights | OK |
| 10-27 15:08 | PlantVillage | OK, 0 unreadable images |
| 10-27 16:09 | CIFAR-10 | OK (server slow, ~1 h) |
| 10-27 16:10 | EuroSAT | OK |
| 10-27 16:17 | Oxford Pets | OK, 0 unreadable images |
| 10-27 16:17 | ImageNetV2 | interrupted (terminal session ended), restarted ~16:40 |
| 10-28 11:09 | Tectonic 0.15 (LaTeX engine, session temp folder) | Downloaded, but fails on Windows while setting up its LaTeX format (`expl3` error), so not used |
| 10-28 ~11:40 | TinyTeX v2026.03 (TeX Live 2025), zip release, installed by Soham into `%APPDATA%\TinyTeX` | OK. The default `.exe` release was blocked by Windows Smart App Control; the `.zip` release works |
| 10-28 ~11:45 | LaTeX packages `ieeetran multirow pgf cite courier` via `tlmgr`, from the TeX Live 2025 archive (`ftp.math.utah.edu/pub/tex/historic/systems/texlive/2025/tlnet-final`) | OK. The current repository is TeX Live 2026 and refuses the older install |
| 10-28 ~11:50 | `pypdfium2` (pip, into `.venv`) to render PDF pages for checking the layout | OK. `pymupdf` was tried first and removed because Smart App Control blocks its DLL |
| 10-28 ~12:50 | `pandas` 3.1.0 -> **2.3.3** (pip, into `.venv`) | `import pandas` failed with "An Application Control policy has blocked this file" (pandas 3.1.0 `hashtable` DLL, which had worked on 10-27). 2.3.3 imports fine |
| 10-28 13:06 | Extra unlabelled images for the MAE pools (PlantVillage 15,001, CIFAR-10 15,000, EuroSAT 15,000), decoded from the already-downloaded datasets | OK, `data/cache/160px/*_pool20000.npz`, ~1.2 GB each; no new download |

## Compiling the paper
`%APPDATA%\TinyTeX\bin\windows\pdflatex.exe` main, then `bibtex.exe main`, then `pdflatex` twice (run in `paper/`).
