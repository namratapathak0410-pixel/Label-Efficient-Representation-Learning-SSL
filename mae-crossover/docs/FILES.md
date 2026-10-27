# What every file does

## Repository root (`D:\SSL`)
| Path | What |
|---|---|
| `README.md` | Original project overview (Namrata) |
| `Plant_Disease_SSL_Research_Proposal_and_Literature_Review.docx` | Research proposal and literature review (local, not in git) |
| `archive/plantvillage-ssl/` | First exploratory pipeline (PlantVillage only, 5 models). Kept for reference, not used |
| `archive/plantVillage.ipynb` | Namrata's Colab notebook: dataset audit, splits, first ResNet-50 run |
| `mae-crossover/` | **The current project** |

## `mae-crossover/`
| Path | What | Read it when |
|---|---|---|
| `run_experiment.ps1` | Runs pretrain -> tune -> train -> analyze in its own window, logs to `logs/experiment.log`, resumes after interruptions, commits and pushes results at the end | Starting or resuming the main experiment |
| `run.py` | Command line: `prepare`, `split`, `benchmark`, `distance`, `pretrain`, `tune`, `train`, `analyze`, `all` | You want to run anything |
| `config.yaml` | All settings of the main experiment (datasets, models, budgets, seeds, MAE, training) | You want to know or change a setting |
| `configs/smoke.yaml` | Tiny settings for a few-minute end-to-end test (results meaningless) | Checking the code works |
| `PREREGISTRATION.md` | Hypotheses and analysis plan, committed before training | Writing the methods chapter |
| `README.md` | Setup and how to run | First time |
| `requirements.txt` | Python packages | Setting up another machine |
| `.gitignore` | Keeps data, checkpoints, per-run files and the venv out of git | - |

### `mae-crossover/mc/` (the code)
| File | What it does | Main functions |
|---|---|---|
| `utils.py` | Config loading, folder paths, seeding, GPU setup, JSON helpers | `load_config`, `paths`, `set_seed`, `device` |
| `data.py` | Downloads each dataset (with fallbacks), draws the stratified 7,000-image sample, decodes every image once to 160x160 and saves a `.npz` cache with duplicate hashes | `prepare`, `build_cache`, `load_cache`, `dhash256` |
| `split.py` | Finds duplicate groups, makes the leakage-safe train/val/test split, builds the nested label budgets | `make_split`, `budget_indices`, `duplicate_groups` |
| `gpu_data.py` | Keeps a dataset in GPU memory; random crop, flip and colour jitter on the GPU | `GPUImages`, `random_resized_crop` |
| `models.py` | Builds arms A/C/D/E; layer-wise learning-rate decay for pretrained ViTs | `ARMS`, `build`, `param_groups` |
| `mae.py` | Masked Autoencoder: model, pretraining loop, reconstruction figure | `MAE`, `pretrain_one`, `pretrain_all` |
| `train.py` | One fine-tuning run with early stopping, learning-rate selection, the full grid | `train_one`, `tune`, `grid`, `collect` |
| `metrics.py` | Accuracy, macro P/R/F1, min class recall, NLL, Brier, ECE, adaptive ECE, temperature scaling | `full_report`, `fit_temperature` |
| `distance.py` | Distance between ImageNet and each dataset: FID, OTDD, ImageNet confidence, linear probe | `compute` |
| `analyze.py` | Gap E-C, crossover, distance association, winner map, ranking agreement, plots, `REPORT.md` | `analyze` |

### `mae-crossover/docs/` (this documentation)
| File | What |
|---|---|
| `PROGRESS_LOG.md` | Chronological log of what was done |
| `DECISIONS.md` | Every design decision, who made it, why |
| `DOWNLOADS.md` | Everything downloaded, where it is stored, sizes, licences |
| `FILES.md` | This file |
| `WHY_EACH_STEP.md` | Plain-English reason for every pipeline step |
| `NOVELTY.md` | Research gap and novelty, with the literature check |
| `GLOSSARY.md` | ViT, MAE, ECE and other terms explained simply |
| `THEORY.md` | Full theory: problem, related work, design, data flow, ResNet/ViT/MAE internals with every number explained, metrics, distances, statistics, limitations |

### Created when running (not in git unless noted)
| Path | What |
|---|---|
| `data/` | Raw downloads and the 160 px caches |
| `checkpoints/crossover_v1/<dataset>/mae_seed<k>/` | MAE encoder, loss curve, reconstruction picture, info JSON |
| `results/crossover_v1/splits/` | Split CSVs, split reports, budget tables (in git) |
| `results/crossover_v1/runs/<run_id>/` | Per run: metrics, training history, logits, per-class metrics (not in git, large) |
| `results/crossover_v1/*.csv, REPORT.md, plots/` | Final tables, figures and report (in git) |
