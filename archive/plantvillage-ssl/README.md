# Label-Efficient Representation Learning with Self-Supervised Learning (MAE)

Does **target-domain MAE pretraining** improve plant-disease classification **and prediction calibration**
when only a few labelled images are available, compared with training from scratch and with ImageNet-pretrained models?

The code is dataset-agnostic (any folder-per-class image dataset: PlantVillage, EuroSAT, CIFAR-10, Oxford Pets, ...).
Nothing in this repository contains results: every number in the final report is produced by running the experiments.

| Model | What it is | Pretraining |
|---|---|---|
| A | CNN (ResNet18 on CPU / ResNet50 on GPU) | none (scratch) |
| B | same CNN | ImageNet supervised, fine-tuned |
| C | ViT (Tiny on CPU / Base-16 on GPU) | ImageNet supervised, fine-tuned |
| D | same ViT | none (scratch) |
| E | same ViT | **MAE on the unlabelled train images of the target dataset**, fine-tuned |

---

## 1. Project structure

```
plantvillage-ssl/
├── main.py                  # command line entry point (check / eda / split / pretrain / train / aggregate / all)
├── requirements.txt
├── configs/
│   ├── smoke.yaml           # 2-minute pipeline test on fake images
│   ├── cpu.yaml             # YOUR DEFAULT (no GPU): ResNet18, ViT-Tiny, 96 px, <=100 images/class
│   ├── pilot_1000.yaml      # 1,000-image pilot study (CPU)
│   ├── colab_t4.yaml        # free Colab GPU: ViT-Small, 128 px
│   └── gpu.yaml             # full protocol: ResNet50 + ViT-B/16 at 224 px
├── scripts/prepare_dataset.py   # download / convert datasets into data/raw/<name>/<class>/*.jpg
├── src/
│   ├── data_loader.py       # scan folders, label-budget subsets, DataLoaders, RAM cache
│   ├── preprocessing.py     # augmentations (with the reasons) + normalisation
│   ├── split_dataset.py     # corruption check, exact + near-duplicate groups, leakage-free stratified split
│   ├── eda.py               # class counts, plots, samples, duplicates, imbalance
│   ├── models.py            # models A-E
│   ├── mae_pretraining.py   # Masked Autoencoder implementation + pretraining loop
│   ├── train_classifier.py  # training / early stopping / evaluation of one run + run grid
│   ├── evaluate.py          # accuracy, macro P/R/F1, per-class report, confusion matrix
│   ├── calibration.py       # ECE, Brier, reliability bins, temperature scaling
│   ├── visualization.py     # all plots
│   └── report.py            # table, plots, best-model selection, REPORT.md
├── data/raw/                # datasets (git-ignored)
├── checkpoints/             # model weights (git-ignored)
├── results/<experiment>/    # CSV, JSON, plots, REPORT.md (git-ignored; copy what you need into the thesis)
├── notebooks/colab_run.ipynb
└── docs/                    # ROADMAP.md, METHODOLOGY.md (thesis text), LIMITATIONS.md
```

---

## 2. Setup in VS Code (no GPU)

1. Install **Python 3.10 - 3.12** (python.org; tick *Add to PATH* on Windows) and **VS Code** with the extensions
   *Python* and *Jupyter*.
2. Open the folder: `File > Open Folder... > plantvillage-ssl`.
3. Open a terminal (`Ctrl + ~`) and create the environment:

```bash
python -m venv .venv
# Windows PowerShell:   .venv\Scripts\Activate.ps1      (if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
# macOS / Linux:        source .venv/bin/activate

# small CPU-only PyTorch (a few hundred MB instead of several GB):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
4. `Ctrl+Shift+P` -> *Python: Select Interpreter* -> choose `.venv`.
5. Check the machine:

```bash
python main.py check --config configs/cpu.yaml
```

### Test the whole pipeline first (about 2-5 minutes, fake images)

```bash
python scripts/prepare_dataset.py --dataset synthetic
python main.py all --config configs/smoke.yaml
```
If this ends with a results table in `results/smoke_test/REPORT.md`, your installation works.
(The smoke config allows running without internet; the real configs *require* the ImageNet weights, which timm
downloads automatically on first use.)

---

## 3. Get the dataset (you have not downloaded it yet)

**Option A - automatic (needs internet, ~1 GB; works without a Kaggle login for public datasets):**
```bash
python scripts/prepare_dataset.py --dataset plantvillage
```
**Option B - manual:** download the zip from
<https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset>, unzip it anywhere, then
```bash
python scripts/prepare_dataset.py --dataset plantvillage --src "C:/path/to/unzipped/folder"
```
The script finds the `color/` folder (the Kaggle archive also contains `grayscale/` and `segmented/` - use only
`color`, otherwise you would count every leaf three times) and copies it to `data/raw/plantvillage/<class>/*.jpg`.

Other datasets from your list:
```bash
python scripts/prepare_dataset.py --dataset eurosat      # 27k satellite images, 10 classes (CPU friendly)
python scripts/prepare_dataset.py --dataset cifar10
python scripts/prepare_dataset.py --dataset imagefolder --src "D:/oxford_pets/images_by_breed" --name pets
```
For a new dataset copy `configs/cpu.yaml`, change `experiment_name`, `data.name`, `data.root`.

---

## 4. Run the experiments (CPU)

First see how fast your machine is:
```bash
python main.py benchmark --config configs/cpu.yaml     # prints images/second per architecture
```
Then run step by step (every step is resumable; finished runs are skipped):
```bash
python main.py eda       --config configs/cpu.yaml     # results/plantvillage_cpu/eda/*
python main.py split     --config configs/cpu.yaml     # leakage-free train/val/test split
python main.py pretrain  --config configs/cpu.yaml     # MAE on the unlabelled TRAIN images -> model E
python main.py train     --config configs/cpu.yaml     # all models x budgets x seeds
python main.py aggregate --config configs/cpu.yaml     # table, plots, REPORT.md
```
Run only part of the grid (good for a first look, e.g. 2 budgets, 1 seed):
```bash
python main.py train --config configs/cpu.yaml --models A_cnn_scratch E_vit_mae --budgets 10 100 --seeds 0
```
1,000-image pilot: `python main.py all --config configs/pilot_1000.yaml`.

**Realistic expectation on a CPU:** the full CPU grid is 5 models x 5 budgets x 3 seeds = 75 runs. Use `benchmark`
to estimate it, start with `--seeds 0`, and let the machine run overnight. Laptops sleep: disable sleep mode
or run in Colab. If it is too slow, lower `data.max_per_class`, `mae.epochs`, or use `configs/pilot_1000.yaml`.

---

## 5. Using a GPU later (Google Colab)

1. <https://colab.research.google.com> -> *Runtime > Change runtime type > T4 GPU*.
2. Upload this folder to Google Drive (without `data/`, `results/`, `checkpoints/`) or push it to GitHub.
3. Open `notebooks/colab_run.ipynb` (it mounts Drive, installs requirements, downloads the dataset, runs the pipeline).
4. Choose the config: **`configs/colab_t4.yaml`** (ViT-Small, 128 px, 300 img/class) for the free T4;
   **`configs/gpu.yaml`** (ViT-B/16, 224 px, all images) needs >= 16 GB GPU memory and many hours (MAE pretraining
   of ViT-B on 38 k images is the expensive part) - Colab Pro / a lab GPU.
5. Save `results/` and `checkpoints/mae/` to Drive - Colab storage is deleted when the session ends.

---

## 6. Outputs

```
results/<experiment>/
├── eda/        class_counts.csv, class_distribution.png, sample_images.png, corrupted_images.csv,
│               duplicate_groups.csv, eda_summary.json
├── splits/     split.csv (every image -> train/val/test + duplicate group), split_report.json
├── runs.csv    ONE ROW PER RUN with all metrics, times, model size, memory, hyper-parameters
├── runs/<run_id>/  history.csv, curves.png, reliability.png, confusion_matrix.png, per_class_metrics.csv,
│                   test_logits.npz, metrics.json
├── results_table.csv / .md     Model | Pretraining | Label budget | Test acc | Macro-F1 | ECE | Training time | Model size
├── summary_numeric.csv, best_models.json, REPORT.md
└── plots/      macro_f1_vs_budget.png, accuracy_vs_budget.png, ece_vs_budget.png, confusion_matrix_best.png
```
MAE outputs: `checkpoints/<experiment>/mae/` (encoder.pt, reconstructions.png, mae_loss.csv, pretrain_info.json).

---

## 7. Reproducibility and integrity rules built into the code

* **One split, made first.** `split.csv` is created before any pretraining or tuning. Exact duplicates (MD5) and
  near-duplicates (256-bit difference hash) are grouped, and a group is never split across train/val/test.
* **Test set is never touched** by MAE pretraining (it only reads the train split, without labels), augmentation
  choices, hyper-parameters or early stopping. It is evaluated once per run after the best *validation* checkpoint is restored.
* **Best-model selection uses validation scores** (`report.py`), not test scores.
* **Label budgets** are stratified, nested and identical for all models given the same seed.
* Seeds are fixed (`set_seed`); results are mean ± std over the seeds in the config. GPU runs may still differ
  slightly between machines (non-deterministic CUDA kernels).
* Hyper-parameters, runtime, device, memory and whether pretrained weights were actually loaded are logged in `runs.csv`.
* The table shows `pending` for every experiment that has not been run. Nothing is hard-coded.

More: `docs/ROADMAP.md` (step-by-step plan), `docs/METHODOLOGY.md` (thesis-ready text), `docs/LIMITATIONS.md`.
