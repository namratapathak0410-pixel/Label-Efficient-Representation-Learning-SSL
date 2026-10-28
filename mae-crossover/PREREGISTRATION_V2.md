# Pre-registration, part 2 (experiment `crossover_v2`)

Written and committed **before any v2 training**. It extends `PREREGISTRATION.md` (v1, already run; results in
`results/crossover_v1/` and `docs/FINDINGS.md`). Nothing in v1 is changed. The v1 finding that motivates this part:
the ImageNet ViT (C) beat target-domain MAE (E) in all 20 (dataset, budget) cells, but E was trained on only ~5,000
unlabelled images, far less than self-supervised learning is meant to use.

## Question
Does the answer change when MAE gets a realistic amount of unlabelled data, and can ImageNet and in-domain MAE be combined?

## New arms (everything else is identical to v1: data, splits, budgets, seeds, fine-tuning protocol, metrics)
| Arm | What | Datasets |
|---|---|---|
| **E2** `E2_vit_mae_big` | ViT-Tiny, random init, MAE-pretrained on a **20,000-image** unlabelled pool, then fine-tuned | PlantVillage, CIFAR-10, EuroSAT |
| **F** `F_vit_imagenet_mae` | ViT-Tiny from ImageNet-1k weights, **continued** MAE pretraining on the same pool, then fine-tuned | all four |
| **A2** `A2_cnn34_scratch` | ResNet-34, random init (a stronger scratch CNN than A = ResNet-18) | all four |

Arms A, C, D, E stay as in v1. Oxford Pets has no extra unlabelled images (all ~7,350 are in the 7,000 sample), so
its pool stays the ~5,000 train images: E2 is not run there (it would equal E), and F uses the 5,000-image pool.

## MAE on the bigger pool
- **Pool** = the ~5,000 train images of the sample + extra images of the same dataset, 20,000 in total.
  Extras are drawn at random from the images **not in the 7,000-image sample**, so no val or test image can be in the
  pool. In addition, any extra within Hamming distance <= 10 (256-bit dHash) of a val/test image, or of its
  horizontal, vertical or double flip, is dropped. Labels of pool images are never read.
- **Compute is matched to v1:** the same 15,200 optimiser steps, batch 256, warm-up 760 steps (= v1's 40 epochs x
  19 steps). Only the data changes: 20,000 images give ~195 epochs instead of 800.
- E2 uses the v1 MAE recipe (mask 75 %, decoder 2 x 128, norm-pixel loss, lr 1.5e-4). Per seed, a separate pretraining.
- F starts from the timm DeiT-Tiny ImageNet-1k weights (position embeddings kept, not reset) and uses a **smaller MAE
  learning rate, 5e-5** (one third of v1's), so that the ImageNet features are adapted rather than overwritten.
  This value is a declared choice and was **not tuned**; a different value could change F's result.
- Fine-tuning of E2 and F: identical to C and E (AdamW, LLRD 0.75, same augmentation, same stopping rule).

## Learning-rate grid widened for all arms (the v1 limitation: 11 of 16 chosen rates were on the grid edge)
Grid at the 20 % budget, seed 0, validation macro-F1 (unchanged rule):
scratch {2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3}; pretrained {1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3}.
v1 tuning runs for the three old values are reused (identical settings). For arms A, C, D, E: if the new choice equals
the v1 choice, the v1 runs are reused; if it differs, the 15 runs of that (dataset, arm) are re-run with the new rate
and the new numbers replace the old ones in the v2 analysis. Both versions are kept (v1 results are not overwritten).

## Hypotheses (directions stated before seeing any v2 result)
- **H6 (more unlabelled data helps MAE):** E2 > E, averaged over budgets and seeds, in each of PlantVillage, CIFAR-10, EuroSAT.
- **H7 (crossover with realistic unlabelled data):** the gap E2 - C is larger (less negative) than E - C at every
  budget, and E2 overtakes C (gap > 0 with a 95 % interval excluding 0) at some budget in at least one dataset.
  Most likely in PlantVillage, where the v1 gap was already only -0.020 at 100 %.
- **H8 (ImageNet + in-domain MAE):** F beats C (mean test macro-F1) in more than half of all 20 (dataset, budget)
  cells, and the F - C gain is larger for domains farther from ImageNet (Spearman with each of the four distances;
  exploratory, n = 4).
- **H9 (F vs E2):** F >= E2 in the three datasets where both exist.
- **H10 (CNN size):** replacing ResNet-18 by ResNet-34 changes macro-F1 by less than 0.03 in most cells and does not
  change who wins against the MAE arms.
- **H11 (grid):** widening the learning-rate grid does not change the winner among A, C, D, E in any cell
  (v1: C won 20 of 20).

## Analysis plan (same conventions as v1)
Primary metric test macro-F1; gaps paired by seed (same labelled subset), mean and 95 % t-interval over the 3 seeds;
crossover by sign change on a log-budget axis (censored if none); association with the four domain distances by exact
Spearman (descriptive, n = 3 or 4). Pairs: E2-E, E2-C, F-C, F-E2, A2-A, E2-A2, E-C (re-checked with the new grid).
Calibration (ECE, ECE after temperature scaling) is reported for every arm. All 7 hypotheses and all results are
reported, including those that contradict them. With 3 seeds and several comparisons, individual "significant" cells
are descriptive, not confirmatory.

## Declared limitations
One MAE recipe; F's MAE learning rate untuned; the pool is 20,000 images, not the whole dataset; hashing cannot
detect all near-duplicates (e.g. PlantVillage photographs of the same leaf from another angle may be in the pool and
in val/test, as already possible within v1's own splits); Oxford Pets has no extra images; same 6 GB GPU, ViT-Tiny, 160 px.

## Time plan (estimate, before running)
MAE pretraining 21 runs x ~21.5 min = 7.5 h; tuning ~1.8 h; fine-tuning of new arms ~9 h (A2 is slow, ~5 h);
re-runs of changed cells 0-3 h. Total about 19-22 h, against 48 h available.
