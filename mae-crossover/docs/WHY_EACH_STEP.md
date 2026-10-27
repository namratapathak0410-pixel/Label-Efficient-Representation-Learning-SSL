# Why each step exists

The pipeline order is `prepare -> split -> benchmark -> distance -> pretrain -> tune -> train -> analyze`.

## 1. `prepare`: download and cache
- **What:** downloads each dataset, takes a stratified random sample of 7,000 images, resizes every image to 160x160 once and stores all pixels in one file per dataset.
- **Why 7,000 everywhere:** if one dataset had 50,000 images and another 7,000, MAE would get far more unlabelled data in one domain and we could not tell whether a difference comes from the *domain* or from the *amount of data*.
- **Why decode once:** reading and resizing JPEGs on Windows is slower than the GPU. Decoding once and keeping the pixels in GPU memory makes each training run several times faster - this is what makes 240 runs fit in 48 hours.
- **Why a fallback list:** if a server is down, the pipeline uses the next dataset from a different domain instead of stopping (Soham's rule).

## 2. `split`: train / validation / test, without leakage
- **What:** finds near-identical images (same leaf photographed twice, augmented copies) with a 256-bit "difference hash", keeps every group of near-duplicates in the same split, then splits 70/15/15 keeping class proportions.
- **Why:** PlantVillage is known to contain several photos of the same leaf. If one is in training and its twin in test, accuracy is inflated (the model recognises the leaf, not the disease).
- **Label budgets:** for each seed, each class's training images are shuffled once and the first 5 %, 10 %, ... are labelled. The 5 % set is inside the 10 % set, and so on (nested), and every arm gets exactly the same images, so differences come from the model, not from luckier labels.

## 3. `benchmark`: measure speed before committing 48 hours
- **What:** times a few hundred training steps for each arm and for MAE, then estimates the total hours.
- **Why:** if the estimate is above the budget we shrink the plan *before* starting, not halfway.

## 4. `distance`: how far is each dataset from ImageNet?
- **What:** passes ImageNetV2 images and each dataset's training images through the frozen ImageNet ViT and compares the two feature clouds (FID, OTDD), plus how confident the ImageNet classifier is on the target images and how well a simple linear classifier on ImageNet features does.
- **Why before training:** the distance must be fixed before we see any result, so it cannot be chosen to fit the outcome. This measured distance is the "can we predict it" part of the novelty.

## 5. `pretrain`: MAE on the unlabelled training images
- **What:** hides 75 % of each image's 16x16 patches and trains the ViT to redraw them; 800 passes over the ~5,000 training images (raised from 400 before training, amendment 1); one pretraining per seed.
- **Why only training images:** using validation or test images (even without labels) would leak information about the test set.
- **Why identical settings in every domain:** same compute and same recipe, so the domain is the only thing that changes.

## 6. `tune`: pick the learning rate fairly
- **What:** for each dataset and arm, tries 3 learning rates at the 20 % budget (seed 0) and keeps the one with the best *validation* macro-F1.
- **Why:** a single learning rate suits some arms better than others (pretrained models want smaller ones). Choosing on validation, never on test, keeps the comparison fair and honest.

## 7. `train`: the main grid
- **What:** 4 datasets x 4 arms x 5 budgets x 3 seeds = 240 fine-tuning runs. Each run trains, checks validation 25 times, stops early if validation stops improving, restores the best checkpoint and evaluates the test set once.
- **Why seed-major order:** all datasets get seed 0 first. If time runs out, results stay balanced.
- **Why resumable:** a laptop can crash or restart; finished runs are skipped when you re-run.

## 8. `analyze`: answer the research question
- **Gap E - C:** at each budget, the macro-F1 of target-MAE minus ImageNet, paired by seed (same labelled images), with a 95 % interval.
- **Crossover:** the budget where that gap changes sign; only called real if the intervals on both sides exclude zero.
- **Distance vs gap:** does the gap grow with distance from ImageNet? With 4 datasets this is exploratory and is reported as such.
- **Winner map and ranking agreement:** is the best arm the same everywhere? (Soham's replicability question.)
- **Calibration:** ECE, NLL, Brier before and after temperature scaling - is the most accurate model also the most trustworthy?
