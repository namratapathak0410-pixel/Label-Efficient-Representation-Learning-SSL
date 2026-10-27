# Decision log

Every design decision, who made it, and why. "Soham" = decided by Soham; "Claude" = proposed by Claude and accepted.
If a decision changes later, the old row stays and a new row is added (so the history is honest).

| # | Date | Decision | By | Why |
|---|---|---|---|---|
| 1 | 10-27 | Drop the ImageNet-pretrained CNN (old model B) | Soham | Focus the study on ViT initialisations; the CNN is a from-scratch reference only |
| 2 | 10-27 | Keep a ViT trained from scratch (arm D) | Soham (Claude recommended) | Only D isolates what MAE adds: D and E differ in nothing except MAE pretraining. Also hypothesis H2 of the original proposal |
| 3 | 10-27 | Label budgets as % of training labels: 5, 10, 20, 50, 100 | Soham ("start at 5%") | 1 % would mean < 1 image per class in Pets/PlantVillage at our dataset size; images per class are logged for each run anyway |
| 4 | 10-27 | Build a general multi-domain pipeline; plants are one of several domains | Soham | Goal is to test whether the best strategy replicates across domains |
| 5 | 10-27 | Four domains only: PlantVillage, CIFAR-10, EuroSAT, Oxford Pets | Soham | Soham's choice of datasets; Claude had suggested 8 for a stronger H2 - with 4, the distance correlation is declared exploratory |
| 6 | 10-27 | If a dataset fails to download, use another dataset from a different domain | Soham | Fallback list in `config.yaml` (e.g. EuroSAT -> Kaggle copy -> DTD textures) |
| 7 | 10-27 | Main novelty: "When does MAE overtake ImageNet, and can we predict it?" | Soham | Survived the adversarial literature check; see `NOVELTY.md` |
| 8 | 10-27 | Whole experiment must finish within **48 hours** on the local GPU | Soham | Deadline. Drives decisions 9-13 |
| 9 | 10-27 | ViT size: **ViT-Tiny** (DeiT-Tiny architecture, 5.5 M params) | Claude | ViT-Small would not fit 240 runs + 12 MAE pretrainings into 48 h on a 6 GB laptop GPU |
| 10 | 10-27 | CNN: **ResNet-18** from scratch (11 M params) | Claude | Closest standard ResNet to ViT-Tiny in size; Soham's earlier underfitting with ResNet-18 was with frozen pretrained layers, here every layer trains |
| 11 | 10-27 | ImageNet weights for C: **DeiT-Tiny, ImageNet-1k only** | Soham agreed | Cleaner than the default ImageNet-21k weights; same architecture as D and E |
| 12 | 10-27 | Image size **160 px** for every dataset | Soham ("160 if it fits") | 224 px is ~2x slower; 160 keeps the 48 h budget. Same size everywhere makes domains comparable |
| 13 | 10-27 | 7,000 images per dataset (stratified sample) | Claude | Same unlabelled MAE pool size and label counts in every domain, so the domain is what varies. Pets has only 7,349 images in total |
| 14 | 10-27 | Split 70/15/15, duplicate-aware (dHash, Hamming <= 10), seed 42 | Claude | Prevents near-identical images (same leaf) appearing in train and test |
| 15 | 10-27 | 3 seeds; seed controls labelled subset, init, data order and E's MAE pretraining | Claude | Mean +- sd and paired tests need >= 3; one MAE per seed so E's variance is honest |
| 16 | 10-27 | Seed-major run order (all datasets seed 0, then seed 1, then 2) | Claude | If time runs out, every dataset still has the same number of seeds |
| 17 | 10-27 | Learning rate chosen per (dataset, arm) on validation at 20 % budget, seed 0 | Claude | One fixed lr would favour some arms; tuning on validation (never test) is fair and declared |
| 18 | 10-27 | Same augmentation for all arms and datasets (crop 0.35-1, flip, brightness/contrast 20 %) | Claude | Domain-specific augmentation would confound the comparison |
| 19 | 10-27 | MAE: 75 % mask, 2-block decoder, 400 epochs, identical in every domain | Claude | Standard He et al. setting; shallow decoder is enough for small data (Das 2024, SDMAE) |
| 20 | 10-27 | Domain distance measured with FID, OTDD, ImageNet confidence and linear probe, using ImageNetV2 as the ImageNet reference | Claude | Novelty requires a *measured* distance; ImageNetV2 is a free 10k-image stand-in for ImageNet |
| 21 | 10-27 | Pre-register hypotheses and commit before training | Claude | Timestamped proof that hypotheses were not written after seeing the results |
| 22 | 10-27 | All work on `main` branch | Soham | Soham and Namrata work on it together |
| 23 | 10-27 | Old code moved to `archive/`, not deleted; `hi.py` removed | Claude (Soham: "change the entire pipeline") | Keeps the history; test file was not part of the project |
| 24 | 10-27 | Keep separate, continuously updated docs in `docs/` and push regularly | Soham | Traceability for the thesis |
