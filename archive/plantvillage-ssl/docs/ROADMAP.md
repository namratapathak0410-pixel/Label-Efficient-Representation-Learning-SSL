# Step-by-step implementation roadmap

## Phase 0 - Setup (day 1)
1. Install Python, VS Code, create `.venv`, install requirements (README section 2).
2. `python main.py check` and `python main.py benchmark` -> know how fast your CPU is.
3. `python scripts/prepare_dataset.py --dataset synthetic` then `python main.py all --config configs/smoke.yaml`
   -> proves the pipeline works end to end.

## Phase 1 - Data (day 1-2)
4. Download PlantVillage (`scripts/prepare_dataset.py --dataset plantvillage`).
5. `python main.py eda` -> read `eda_summary.json`: image count, number of classes, imbalance ratio, corrupted files,
   duplicate groups. **Write down the real numbers** (the prompt says "approximately 54,000 images / 38 classes";
   the Kaggle `color` folder should be verified, not assumed).
6. `python main.py split` -> read `split_report.json` (counts per split, duplicate groups, leakage checks).
   From now on do not touch the split; never recreate it after you have seen test results.

## Phase 2 - Baselines (day 2-5)
7. `python main.py train --models A_cnn_scratch B_cnn_pretrained C_vit_pretrained D_vit_scratch --seeds 0 --budgets 10 100`
   (a first look at two budgets).
8. Check `runs/<id>/curves.png`: is validation loss going down? Does early stopping trigger?
9. Tune learning rates only by comparing **validation** macro-F1 (edit `train.lr` in the config, change
   `experiment_name` for each tuning attempt so results do not mix). Apply the same small tuning effort to every model,
   otherwise the comparison is unfair.

## Phase 3 - MAE (day 3-7)
10. `python main.py pretrain` -> look at `checkpoints/<exp>/mae/reconstructions.png` (does the model paint plausible
    leaf texture?) and `mae_loss.png` (is the loss still falling? then train longer: raise `mae.epochs`).
11. `python main.py train --models E_vit_mae` for the same budgets/seeds as the baselines.

## Phase 4 - Full grid and report (day 7-14)
12. `python main.py train` (all models x budgets x seeds; resumable, run overnight).
13. `python main.py aggregate` -> `REPORT.md`, plots, `best_models.json`.
14. Inspect: confusion matrix (which diseases are confused?), per-class metrics, reliability diagrams,
    temperature-scaled ECE.

## Phase 5 - Analysis and writing (week 3+)
15. Answer the research question from the table **with the std over seeds** in mind (differences smaller than the std
    are not evidence).
16. Complementary experiments if time permits (never at the cost of the core grid):
    - longer MAE pretraining / different mask ratio (50/75/90 %) - ablation, selected on validation;
    - second SSL method (SimCLR or DINO) on the same ViT;
    - MAE pretraining on ImageNet-sized external data vs target-domain data (separates "SSL algorithm" from "data source");
    - linear probing (frozen encoder) in addition to fine-tuning;
    - another dataset (`--dataset eurosat`) to support the claim "any dataset".
17. Write: `docs/METHODOLOGY.md` is the thesis methodology (fill the bracketed values from your runs),
    `docs/LIMITATIONS.md` is the limitations section.

## Where a defensible contribution can come from (not "MAE on plants")
MAE and other SSL methods have already been applied to plant-disease images, so do **not** claim novelty for that.
Defensible, checkable contributions are:
- a **controlled** comparison at identical architecture, split, label subsets and budget of compute;
- the **label-efficiency curve** (1-100 %) for scratch vs ImageNet vs in-domain MAE with seeds;
- the **calibration** analysis (ECE/Brier before and after temperature scaling), which is much less studied than accuracy;
- leakage-aware evaluation (duplicate groups) - many plant-disease papers report inflated numbers because of
  near-duplicate leaves across splits; quantify the effect by also running one naive random split and comparing.
Do a literature search (Google Scholar: "self-supervised plant disease", "masked autoencoder plant", "label-efficient
plant disease recognition", "calibration plant disease") and list what is already covered **before** you fix your claim.
