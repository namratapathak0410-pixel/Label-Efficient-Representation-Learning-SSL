# MAE crossover: when does target-domain self-supervision beat ImageNet?

M.Tech research pipeline (THEsoham, namratapathak0410-pixel). One question, tested across four domains:

> At which label budget does a ViT pretrained with a **Masked Autoencoder on the target dataset's own unlabelled
> images** overtake the same ViT initialised from **ImageNet**, and can a measured distance from ImageNet predict it?

Hypotheses and the full protocol are fixed in [PREREGISTRATION.md](PREREGISTRATION.md) (committed before training).

Project documentation, kept up to date as work happens: [progress log](docs/PROGRESS_LOG.md) ·
[decisions](docs/DECISIONS.md) · [downloads](docs/DOWNLOADS.md) · [what each file does](docs/FILES.md) ·
[why each step](docs/WHY_EACH_STEP.md) · [novelty](docs/NOVELTY.md) · [glossary](docs/GLOSSARY.md) ·
[**full theory**](docs/THEORY.md)

## Arms
| ID | Model | Starts from |
|---|---|---|
| A | ResNet-18 | random weights |
| C | ViT-Tiny (DeiT) | ImageNet-1k supervised weights |
| D | ViT-Tiny | random weights |
| E | ViT-Tiny | MAE pretraining on the target's unlabelled train images |

Datasets: PlantVillage (plant disease), CIFAR-10 (objects), EuroSAT (satellite), Oxford-IIIT Pets (animals), 7,000
images each, 160 px. Label budgets 5/10/20/50/100 %, seeds 0/1/2.

### Experiment 2 (arms A2, E2, F)
Pre-registered in [PREREGISTRATION_V2.md](PREREGISTRATION_V2.md). Adds ResNet-34 (A2), MAE on a 20,000-image unlabelled
pool (E2) and an ImageNet ViT with continued MAE on that pool (F), and widens the learning-rate grid. Run with
`powershell -ExecutionPolicy Bypass -File run_experiment_v2.ps1`; `logs/STATUS.txt` shows the estimated time left.

## Setup (Windows, NVIDIA GPU)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Run
```powershell
python run.py prepare      # downloads + caches datasets (PlantVillage via Kaggle, others via torchvision) and ImageNetV2
python run.py split        # leakage-safe splits, nested label budgets
python run.py benchmark    # GPU speed -> results/<exp>/benchmark.json with the time plan
python run.py distance     # ImageNet-to-target distances, before training
python run.py pretrain     # MAE, seed 0 (other seeds are pretrained automatically by `train`)
python run.py tune         # learning rate per dataset and arm on validation
python run.py train        # 240 runs, seed-major, resumable (just re-run after a crash or reboot)
python run.py analyze      # results/<exp>/REPORT.md, CSVs, plots/
```
`python run.py all` runs everything; `--config configs/smoke.yaml` is a few-minute end-to-end test.
Subsets: `python run.py train --seeds 0 --datasets eurosat --arms C_vit_imagenet E_vit_mae --budgets 5 100`.

## Code map
| File | What it does |
|---|---|
| `mc/data.py` | download, stratified 7,000-image sample, decode once to a uint8 cache |
| `mc/split.py` | duplicate groups (dHash), stratified group split, nested label budgets |
| `mc/gpu_data.py` | dataset kept on the GPU, per-sample augmentation on the GPU |
| `mc/models.py` | arms A/C/D/E, layer-wise lr decay |
| `mc/mae.py` | Masked Autoencoder pretraining |
| `mc/train.py` | one fine-tuning run, lr selection, the grid |
| `mc/metrics.py` | accuracy, macro-F1, NLL, Brier, ECE, adaptive ECE, temperature scaling |
| `mc/distance.py` | FID, OTDD, ImageNet confidence, linear probe |
| `mc/analyze.py` | gap E-C, crossover, distance association, winner map, ranking agreement, REPORT.md |

## Outputs
`results/<exp>/`: `runs.csv` (one row per run), `summary.csv`, `gap_E_minus_C.csv`, `crossover.csv`,
`domain_distance.csv`, `distance_association.csv`, `ranking_agreement.csv`, `lr_choice.json`, `benchmark.json`,
`REPORT.md`, `plots/`. Per-run logs, logits and per-class metrics in `results/<exp>/runs/<run_id>/` (not in git).
MAE encoders and reconstruction images in `checkpoints/<exp>/<dataset>/mae_seed<k>/` (not in git).

The previous exploratory code (`plantvillage-ssl/`, `plantVillage.ipynb`) is kept in `archive/`.
