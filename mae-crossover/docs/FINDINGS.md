# Findings of the main experiment (crossover_v1)

Plain-English reading of `results/crossover_v1/REPORT.md` (240 runs, 4 datasets x 4 arms x 5 budgets x 3 seeds),
checked against the pre-registered hypotheses in `PREREGISTRATION.md`. Every number comes from the CSVs in
`results/crossover_v1/`. Section 3 is **post-hoc** (not pre-registered) and is labelled as such.

Arms: A = ResNet-18 scratch, C = ViT-Tiny ImageNet, D = ViT-Tiny scratch, E = ViT-Tiny MAE on the target domain.

## 1. Main numbers: test macro-F1, mean of 3 seeds

| dataset | budget | A | C | D | E | gap E - C |
|---|---|---|---|---|---|---|
| PlantVillage | 5 % | 0.548 | **0.806** | 0.500 | 0.567 | -0.239 |
| | 100 % | 0.970 | **0.982** | 0.911 | 0.962 | -0.020 |
| EuroSAT | 5 % | 0.764 | **0.934** | 0.714 | 0.751 | -0.183 |
| | 100 % | 0.949 | **0.982** | 0.918 | 0.933 | -0.049 |
| CIFAR-10 | 5 % | 0.382 | **0.868** | 0.314 | 0.425 | -0.443 |
| | 100 % | 0.746 | **0.932** | 0.565 | 0.736 | -0.196 |
| Pets | 5 % | 0.117 | **0.771** | 0.073 | 0.151 | -0.620 |
| | 100 % | 0.659 | **0.890** | 0.286 | 0.638 | -0.252 |

All budgets are in `REPORT.md` section 3.

## 2. Hypotheses (pre-registered)

| | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | ImageNet ahead at small budgets; MAE closes **or** reverses the gap at larger budgets | **Partly supported: the gap closes, but there is no crossover** | Gap E - C shrinks steadily with the budget in all 4 domains (e.g. PlantVillage -0.239 -> -0.020). Every gap is negative with a 95 % CI excluding 0, including PlantVillage at 100 % (-0.020, CI -0.037 to -0.003). Crossover is censored ">100 %" everywhere |
| H2 | The gap is smaller for domains farther from ImageNet | **Direction partly as predicted, not significant** | Mean gaps: PlantVillage -0.118, EuroSAT -0.110 (far domains) vs CIFAR-10 -0.317, Pets -0.489 (ImageNet-like content). Spearman with the mean gap: FID +0.6 (p = 0.42) and ImageNet confidence -0.8 (p = 0.33) point the predicted way; OTDD -0.4 points the wrong way; linear probe 0. Only ImageNet confidence beats the predict-the-mean baseline in leave-one-out (0.124 vs 0.193). With 4 datasets this is exploratory, as declared |
| H3 | MAE beats scratch ViT (E > D) at every budget in every domain | **Mostly supported** | E > D in all 20 cells by point estimate; CI excludes 0 in 13 of 20. Large on Pets (+0.35 at 100 %) and CIFAR-10 (+0.17). On EuroSAT, MAE adds almost nothing (+0.01 to +0.04, none significant) |
| H4 | The best arm is not the same in every domain and budget (Kendall's W < 1) | **Rejected** | ImageNet ViT (C) wins all 20 dataset x budget cells. Arm ranks by macro-F1 agree across datasets: W = 0.90 to 1.00, Friedman p <= 0.013. Usual order C > E ~ A > D |
| H5 | Ranking by calibration (ECE) differs from ranking by macro-F1 | **Partly supported** | Before temperature scaling, C also has the lowest ECE in 17 of 20 cells. After temperature scaling, all arms are within about 0.01-0.06 ECE and their ranking no longer agrees across datasets (W = 0.05 to 0.53, no Friedman p < 0.05); scratch ViT D is best calibrated in 7 cells |

### Other observations
- **MAE ViT is roughly equal to ResNet-18 from scratch (E vs A).** Differences stay within about +-0.05, and only
  2 of 20 are significant. On ~5,000 images, MAE pretraining gives the ViT about what the CNN's built-in locality
  bias gives the ResNet, but not more.
- **A ViT from scratch is the weakest arm everywhere.** On Pets at 100 % it reaches only 0.286, which is the ViT
  "data hunger" in action.
- **ImageNet fine-tuning converges fastest.** Arm C stopped early in 67 % of runs, against 20 % for D and E and 8 %
  for A.

## 3. Post-hoc (NOT pre-registered): extrapolating the crossover

The gap E - C is almost a straight line in log10(budget). For each dataset, a straight-line fit to its 5 points
gives Pearson r between 0.963 and 0.999. Extending each line to gap = 0 gives the budget at which MAE would catch
ImageNet, as a percentage of the current 5,000-image training set:

| dataset | gap closes per 10x labels | extrapolated crossover | FID |
|---|---|---|---|
| PlantVillage | 0.167 | ~117 % | 433 (farthest) |
| EuroSAT | 0.108 | ~241 % | 318 |
| Pets | 0.295 | ~933 % | 273 |
| CIFAR-10 | 0.187 | ~1114 % | 267 (closest) |

The order of the extrapolated crossovers matches the FID order exactly. This is the only ordering out of 24 that
does so, so the exact p-value is 0.083 (two-sided).

**Caveats:**
- This analysis was chosen after seeing the data.
- It extrapolates beyond the measured range: more than 100 % would also mean more unlabelled MAE data.
- It uses n = 4.
- CIFAR-10 is upsampled from 32 px, which may hurt MAE there.

It is a hypothesis for future work, not a result. The pre-registered distance test (mean gap) is section 2, H2.

## 4. One-sentence summary
With ~5,000 images per domain, ImageNet initialisation beats in-domain MAE at every label budget in all four
domains. MAE's disadvantage shrinks steadily as labels grow, and it shrinks fastest in domains far from ImageNet:
PlantVillage at 100 % labels is within 2 F1 points. MAE clearly beats a scratch ViT and roughly matches a
scratch ResNet-18.
