# Research gap and novelty

Result of a 20-agent literature workflow on 2026-10-27 (4 parallel literature sweeps -> 7 candidate novelty claims ->
2 independent skeptical reviewers per claim searching for prior work -> ranking). Several papers were seen only as
abstracts or snippets: **open every paper before citing it.**

## Chosen contribution
> A controlled, leakage-safe study of four ways to start a small image classifier - scratch ResNet, scratch ViT,
> ImageNet-1k ViT, and ViT MAE-pretrained only on the target's unlabelled images - across label budgets and across
> domains whose distance from ImageNet we **measure**. We test where target-domain MAE overtakes ImageNet
> initialisation (the crossover budget), whether that crossover shifts with measured domain distance, and whether the
> winning strategy replicates across domains, by accuracy and by calibration.

## Why it is a gap
- Cole et al. (CVPR 2022) studied when in-domain self-supervision helps, but for contrastive learning on ResNet-50 with
  qualitative ("near/far") domain labels - not MAE on ViTs, not a measured distance.
- Ericsson et al. (CVPR 2021) list the link between domain distance and transfer as open.
- El-Nouby et al. (2021) and Zhou et al. (2023) show target-only masked pretraining works, but do not map a crossover
  over label budgets across domains.
- Sambyal et al. (2023) study calibration vs data size, but for CNNs on medical images only.

## Claims we must NOT make
| Claim | Why not |
|---|---|
| "First to use MAE for plant disease" | Wang et al. 2024 (Agronomy, MAE + CBAM) |
| "First to compare scratch / ImageNet / self-supervised under few labels on PlantVillage" | Gokten et al. 2026 (SimCLR, 1/10/100 %) - verify |
| "Target-only MAE beats scratch ViT" as a new finding | El-Nouby 2021, Zhou 2023, SDMAE, Das 2024 |
| "First leakage-safe PlantVillage split" | Guerrero-Jaramillo et al. 2026 |
| "The pipeline is the contribution" | VTAB, BSCD-FSL, USB, LabelBench exist; the pipeline is the instrument |
| A strong distance-predicts-crossover law from 4 datasets | 4 points cannot establish it; reported as exploratory |

## Must-cite (closest work)
Cole et al. 2022 (arXiv 2105.05837); Ericsson et al. 2021 (2011.13377); El-Nouby et al. 2021 (2112.10740);
Zhou et al. 2023 (2203.05573); He et al. 2022 MAE (2111.06377); Mao et al. 2022 SDMAE (2212.05677);
Das et al. WACV 2024 (2310.20704); Sambyal et al. 2023 (2309.13132); Minderer et al. 2021 (2106.07998);
Guo et al. 2017 (1706.04599); Hammoud et al. ECCV 2024 (2403.13808); Alvarez-Melis & Fusi 2020 (OTDD);
Dehghani et al. 2021 Benchmark Lottery (2107.07002); Demsar 2006 (JMLR); Xu et al. 2022 (Front. Plant Sci.);
Wang et al. 2024 (Agronomy); Wang et al. 2024 CRE (2407.14911); Gokten et al. 2026 (Pak. J. Agric. Sci.).

Full workflow output (all links, verdicts): kept locally, summarised here.
