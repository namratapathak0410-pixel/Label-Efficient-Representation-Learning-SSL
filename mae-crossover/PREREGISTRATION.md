# Pre-registration: When does target-domain MAE overtake ImageNet initialisation?

Committed to the repository **before any fine-tuning result was produced**; the git timestamp of this file is the
record. Any later change to this file is visible in the git history and must be reported in the thesis as a deviation.

## Research question
For one fixed Vision Transformer (DeiT-Tiny architecture, 5.7 M parameters, 160 px input), at which label budget does
self-supervised MAE pretraining on the target dataset's own unlabelled training images overtake supervised ImageNet-1k
initialisation, and does that crossover depend on a measured distance between ImageNet and the target domain?

## Arms (same data, splits, label subsets, augmentation and training protocol)
| ID | Model | Initialisation |
|---|---|---|
| A | ResNet-18 | random |
| C | ViT (DeiT-Tiny) | ImageNet-1k supervised weights (timm `deit_tiny_patch16_224`), position embeddings resized to 160 px |
| D | same ViT | random |
| E | same ViT | MAE (He et al. 2022) on the unlabelled train split of the same dataset, one pretraining per seed |

## Datasets (one per domain, 7,000 images each, stratified sample)
plant: PlantVillage (colour) - objects: CIFAR-10 - satellite: EuroSAT (RGB) - animals: Oxford-IIIT Pets.
If a download fails, the next listed candidate in `config.yaml` is used and this is reported.
Every image is resized (shorter side 160, bicubic) and centre-cropped to 160x160. CIFAR-10 (32 px) and EuroSAT
(64 px) are therefore upsampled; this is a declared limitation.

## Split and leakage control
70/15/15 train/val/test with `StratifiedGroupKFold` (seed 42), where groups join exact and near-duplicate images
(256-bit dHash, Hamming <= 10). MAE sees only train images, without labels. Test images are evaluated once per run,
after restoring the best-validation checkpoint.

## Label budgets and seeds
5, 10, 20, 50, 100 % of each class's training images (at least 1 per class), nested, identical for all arms for a
given seed. Seeds 0, 1, 2; a seed fixes the labelled subset, the initialisation of new layers, data order and the
MAE pretraining of arm E. Runs are executed seed-major so that an interruption leaves every dataset with the same seeds.

## Training protocol
Fine-tuning: AdamW (wd 0.05), batch 128, bf16, grad-clip 1.0, 5 % warm-up + cosine, steps =
clamp(100 x ceil(n/128), 600, 4000), 25 validation checks, early stopping after 8 checks without improvement of
validation macro-F1. Layer-wise lr decay 0.75 for C and E. Augmentation for all arms: random resized crop
(0.35-1), horizontal flip, brightness/contrast +-20 %. Learning rate per (dataset, arm) chosen on validation macro-F1
at the 20 % budget, seed 0, from {5e-4, 1e-3, 2e-3} (scratch) or {2.5e-4, 5e-4, 1e-3} (pretrained).
MAE: mask 75 %, decoder 2 blocks x 128-d, norm-pixel loss, 400 epochs over the ~4,900 train images, batch 256,
lr 1.5e-4 x 256/256, wd 0.05, 20 warm-up epochs. Identical MAE compute in every domain.
(**Amended to 800 epochs / 40 warm-up epochs before training - see Amendments.**)

## Domain distance (computed from frozen ImageNet features before training)
Reference: ImageNetV2 matched-frequency. Measures: FID, OTDD (diagonal-Gaussian label cost, 64-d PCA),
mean ImageNet max-softmax confidence, linear-probe validation accuracy.

## Hypotheses
- **H1 (crossover):** the macro-F1 gap E - C increases with the label budget in at least one domain, i.e. ImageNet
  initialisation is ahead at small budgets and target MAE closes or reverses the gap at larger budgets.
- **H2 (distance):** the mean gap E - C (averaged over log-budget) is larger for domains farther from ImageNet
  (higher FID / OTDD, lower ImageNet confidence and linear-probe accuracy).
- **H3 (MAE vs scratch):** E > D at every budget in every domain.
- **H4 (replicability):** the best arm is not the same in every domain and budget (Kendall's W < 1).
- **H5 (calibration, secondary):** the ranking of arms by ECE differs from the ranking by macro-F1.

## Analysis plan
- Primary metric: test macro-F1. Gap E - C paired by seed; mean and 95 % t-interval over seeds.
- Crossover budget: sign change of the mean gap, interpolated on log10(budget); reported as censored ("<5" or
  ">100") if there is no sign change; called supported only if the 95 % intervals exclude 0 on both sides.
- H2: Spearman rho between each distance and the mean gap, exact permutation p-value; leave-one-dataset-out
  linear prediction vs a predict-the-mean baseline. With 4 datasets this is **exploratory** (minimum attainable p =
  0.083); it is reported as descriptive evidence, not as a confirmed law.
- Replicability: winner per (dataset, budget); Kendall's W and Friedman test of arm ranks across datasets, per budget,
  for macro-F1, ECE and ECE after temperature scaling (temperature fitted on validation only).
- Secondary metrics: accuracy, macro precision/recall, minimum class recall, NLL, Brier, ECE (15 bins), adaptive ECE.
- All results, including those that contradict the hypotheses, are reported.

## Amendments (made before any main-experiment training)
| # | Date (IST) | Change | Reason | Results existing at the time |
|---|---|---|---|---|
| 1 | 2026-10-27 ~16:55 | MAE pretraining 400 -> **800 epochs**, warm-up 20 -> **40 epochs** (~15,200 steps per pretraining) | The speed benchmark showed the full plan needs ~16 h worst case of the 48 h; 800 epochs is the standard MAE schedule and avoids under-training arm E. Chosen by Soham | Only the smoke test (meaningless by design) and the domain distances (fixed, unaffected by this change). No MAE pretraining or main fine-tuning run had started |

## Domain distances as measured (recorded before training)
FID / OTDD / ImageNet confidence / linear-probe accuracy: PlantVillage 433.3 / 30.62 / 0.298 / 0.899;
CIFAR-10 266.5 / 28.25 / 0.294 / 0.833; EuroSAT 318.1 / 26.42 / 0.139 / 0.858; Pets 272.9 / 29.30 / 0.510 / 0.861.
The measures do not agree on the ordering (e.g. OTDD places EuroSAT closest, ImageNet confidence farthest); H2 is
therefore evaluated separately for each measure, as planned.

## Known limitations declared in advance
Small ViT and 160 px (compute budget: one RTX 4050 Laptop GPU, 6 GB, 48 h); DeiT weights were trained at 224 px;
labelled validation set is larger than the smallest training budgets (same for all arms); 4 domains only for H2;
one MAE recipe (no masking-ratio search).
