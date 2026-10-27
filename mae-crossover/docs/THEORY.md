# Theory and design of the study

This document explains the whole project from first principles: the problem, the related work, what we do, how we
do it, and the theory behind every model, number and measure. Numbers marked ✓ were checked against the actual code
(e.g. parameter counts printed by PyTorch).

Contents
1. Problem statement
2. Research question and hypotheses
3. Related work
4. What we are doing (study design)
5. How the data flows through the pipeline
6. The datasets and their numbers
7. ResNet (convolutional networks)
8. Vision Transformer (ViT)
9. ImageNet pretraining and DeiT
10. Masked Autoencoders (MAE)
11. Fine-tuning: optimiser, schedule and regularisation
12. Evaluation metrics (accuracy and calibration)
13. Domain distance measures
14. Statistical analysis
15. Compute budget
16. Limitations

---

## 1. Problem statement

Deep image classifiers need many labelled images. Labelling is slow and expensive - for plant diseases it needs an
expert - while *unlabelled* images of the same domain are cheap. Two common ways to cope with few labels are:

1. **Transfer from ImageNet:** start from a network already trained on 1.3 million labelled everyday photos.
2. **Self-supervised pretraining on the target domain:** first learn from the unlabelled images of *our* domain
   (here with a Masked Autoencoder), then fine-tune on the few labels.

ImageNet transfer brings a huge amount of knowledge, but about the *wrong* kind of images if the target domain is far
from everyday photos (satellite images, leaf lesions). Target-domain pretraining sees the *right* kind of images, but
far fewer of them. Which one wins is not obvious, and it plausibly depends on (a) how many labels we have and (b) how
different the domain is from ImageNet. Practitioners currently have no measured rule for this choice.

## 2. Research question and hypotheses

> **RQ.** For one fixed Vision Transformer, at which label budget does MAE pretraining on the target dataset's own
> unlabelled images overtake ImageNet initialisation, and does that crossover shift with a measured distance between
> ImageNet and the target domain?

| | Hypothesis | Tested by |
|---|---|---|
| H1 | The gap E - C changes with the label budget, so a crossover exists in at least one domain | sign change of the paired gap with 95 % intervals |
| H2 | The gap E - C is larger for domains farther from ImageNet | Spearman correlation with four distance measures; leave-one-dataset-out prediction |
| H3 | E beats D (MAE helps a ViT compared with no pretraining) | paired gap E - D |
| H4 | The best arm is not the same in every domain and budget | winner map, Kendall's W, Friedman test |
| H5 | The ranking by calibration differs from the ranking by accuracy | rankings by ECE vs macro-F1 |

A result against a hypothesis is still a result and is reported (see `PREREGISTRATION.md`).

## 3. Related work

**Self-supervised learning and MAE.** He et al. (2022) introduced Masked Autoencoders: hide most image patches,
reconstruct them, keep the encoder. It scales well and transfers well when pretrained on ImageNet-sized data.
El-Nouby et al. (2021) showed masked/denoising pretraining works even when pretraining only on the (smaller) target
dataset. Zhou et al. (2023) found the same for medical images ("self pre-training"). Mao et al. (2022, SDMAE) and Das
et al. (WACV 2024) adapt MAE to small datasets and find that a shallow decoder is enough.

**Plant disease recognition with limited labels.** Xu et al. (2022) used large plant-specific transfer (PlantCLEF)
for few-shot disease recognition. Wang et al. (2024, Agronomy) combined MAE with an attention module (CBAM) on plant
leaves. Wang et al. (2024, CRE / GPID-22) combined masked reconstruction with contrastive learning on 205k plant
images. Gokten et al. (2026) compared scratch, ImageNet and SimCLR self-supervision on PlantVillage at 1/10/100 %
labels. So "self-supervision for plant disease" is **not** new.

**When does in-domain pretraining help?** Cole et al. (CVPR 2022) asked when contrastive self-supervision works, and
found domain and data size matter - but for ResNet-50 and contrastive learning, with domains described qualitatively.
Ericsson et al. (CVPR 2021) showed self-supervised ImageNet models transfer unevenly across domains and named
domain distance as an open question. Kornblith et al. (2019) showed better ImageNet models usually transfer better,
but less so for fine-grained or distant domains. Raghu et al. (2019) found ImageNet transfer gives limited gains on
medical images. Hammoud et al. (ECCV 2024) and the OTDD line of work (Alvarez-Melis & Fusi 2020) measure dataset
distance to predict transfer.

**Calibration.** Guo et al. (2017) showed modern networks are over-confident and introduced temperature scaling.
Minderer et al. (2021) found ViTs are often better calibrated than CNNs at full data. Sambyal et al. (2023) studied
calibration vs training-set size for CNNs on medical images.

**Replicability of benchmark rankings.** Dehghani et al. (2021, "The Benchmark Lottery") showed method rankings
change with the choice of datasets. Demsar (2006) set the standard for comparing methods over several datasets
(Friedman test, ranks).

**The gap we address.** No study we found measures, for MAE on ViTs with *target-only* pretraining, the label budget
at which it overtakes ImageNet initialisation across domains *and* relates that crossover to a *measured* domain
distance, *and* checks whether the winner replicates by both accuracy and calibration. Details and the claims we must
avoid are in `NOVELTY.md`.

## 4. What we are doing (study design)

A **controlled experiment**: everything is held fixed except the factor being studied.

| Factor | Levels | Held fixed |
|---|---|---|
| Initialisation (arm) | A ResNet-18 scratch · C ViT ImageNet · D ViT scratch · E ViT target-MAE | architecture of C/D/E, training recipe, augmentation, data |
| Label budget | 5, 10, 20, 50, 100 % | the labelled images are the same for every arm (per seed) |
| Domain | plant, objects, satellite, animals | 7,000 images each, 160 px, 70/15/15 split |
| Seed | 0, 1, 2 | - |

Total: 4 datasets x 4 arms x 5 budgets x 3 seeds = **240 fine-tuning runs**, plus 4 x 3 = **12 MAE pretrainings**,
plus 4 x 4 x 3 = **48 learning-rate selection runs**.

**Why each comparison matters:**
- **E vs C** - the main question: in-domain self-supervision vs ImageNet knowledge.
- **E vs D** - does MAE add anything at all? D and E are identical except for MAE pretraining.
- **C vs D** - how much ImageNet is worth for this domain.
- **A vs D** - CNN vs ViT without any pretraining: how much the CNN's built-in assumptions help with little data.

## 5. How the data flows through the pipeline

```
 Internet (Kaggle, torchvision, Hugging Face)
        |
        v
 [prepare]  full dataset  --stratified sample-->  7,000 images
            each image: RGB -> shorter side 160 px (bicubic) -> centre crop 160x160 -> uint8 array
            + 256-bit dHash fingerprint
            saved once: data/cache/160px/<dataset>.npz   (~540 MB: 7,000 x 160 x 160 x 3 bytes)
        |
        v
 [split]    near-duplicate groups (dHash, Hamming <= 10)  ->  StratifiedGroupKFold
            ~4,900 train | ~1,050 val | ~1,050 test        (groups never cross)
            label budgets per seed: nested subsets of train (5 % inside 10 % inside ... 100 %)
        |
        +--> [distance]  frozen ImageNet DeiT-Tiny features of train images vs ImageNetV2
        |                -> FID, OTDD, ImageNet confidence, linear-probe accuracy (before any training)
        |
        +--> [pretrain]  arm E only: ALL ~4,900 train images, labels removed
        |                GPU: random crop 0.2-1 + flip -> hide 75 % of patches -> reconstruct -> encoder.pt
        |
        v
 [tune / train]  whole dataset uploaded once to GPU memory as uint8 (~540 MB)
            each step: pick 128 labelled train indices -> on GPU: random resized crop 0.35-1, flip,
            brightness/contrast -> normalise with ImageNet mean/std -> model -> cross-entropy -> AdamW
            every 1/25 of the run: predict val set -> macro-F1 -> keep best weights / early stop
            end: restore best weights -> predict val + test once -> metrics.json, logits.npz
        |
        v
 [analyze]  runs.csv -> gaps, crossover, distance correlation, winner map, calibration -> REPORT.md + plots
```

**Normalisation.** Every pixel value x in [0, 1] becomes (x - mean) / std per colour channel, with the ImageNet
statistics mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225). The same normalisation is used for every arm
so that inputs are identical; for C it is required because its weights were learned with it.

**Why the GPU-resident design:** decoding a JPEG and resizing it costs ~2-5 ms on the CPU, which for 128-image
batches is slower than the GPU step itself. Decoding once and doing augmentation as one batched GPU operation
(an affine `grid_sample`) removes that bottleneck. The augmentation is mathematically the same as torchvision's
RandomResizedCrop + horizontal flip (verified: an identity crop reproduces the image to 2e-5).

## 6. The datasets and their numbers

| Domain | Dataset | Native size | Classes | Available | Images per class in our 7,000 | Distance from ImageNet (expected) |
|---|---|---|---|---|---|---|
| plant | PlantVillage (colour) | 256 x 256 | 38 | 54,305 | 19 - 710 (imbalanced) | medium: leaves exist in ImageNet, diseases do not; lab background |
| objects | CIFAR-10 | 32 x 32 | 10 | 60,000 | 700 each | close in content, but tiny images (5x upsampled) |
| satellite | EuroSAT RGB | 64 x 64 | 10 | 27,000 | 519 - 778 | far: top-down views do not exist in ImageNet |
| animals | Oxford-IIIT Pets | ~500 x 375 | 37 | 7,349 | 176 - 191 | very close: ImageNet contains these breeds |

"Expected" distances are our prior guesses; the pipeline *measures* them (section 13) and only the measured values
are used in the analysis.

**What the budgets mean in images.** With ~70 % of 7,000 in train (~4,900):

| Budget | CIFAR-10 / class | EuroSAT / class | Pets / class | PlantVillage / class (smallest .. largest) |
|---|---|---|---|---|
| 5 % | ~24 | ~18 - 27 | ~6 | 1 .. ~25 |
| 10 % | ~49 | ~36 - 54 | ~13 | 1 .. ~50 |
| 20 % | ~98 | ~73 - 109 | ~25 | ~3 .. ~100 |
| 50 % | ~245 | ~180 - 270 | ~63 | ~7 .. ~250 |
| 100 % | ~490 | ~360 - 545 | ~126 | ~13 .. ~500 |

Every class always gets at least 1 labelled image. The same percentage means very different numbers of images per
class across datasets - this is why the analysis also reports `n_labeled` and why macro-F1 (every class counts
equally) is the primary metric.

**Why 160 px.** Compute of a ViT grows with the number of patches N = (image size / 16)^2, and attention grows with
N^2. At 224 px N = 196; at 160 px N = 100: about 2x fewer patches and ~3-4x less attention work. 160 is a multiple of
16, so patches tile the image exactly.

## 7. ResNet (convolutional networks)

**Convolution.** A small filter (e.g. 3x3 pixels x 64 channels) slides over the image and computes a weighted sum at
every position. The same weights are used everywhere (*weight sharing*), and each output depends only on a small
neighbourhood (*locality*). These two built-in assumptions - **inductive biases** - say "a pattern means the same
thing wherever it appears" and "nearby pixels belong together". They are true for natural images, so a CNN needs
less data to learn.

**Receptive field.** Stacking layers and downsampling lets deeper filters "see" larger regions, building up from
edges to textures to object parts.

**Residual connection (He et al. 2016).** Each block computes y = F(x) + x. The "+ x" shortcut lets gradients flow
straight through, so very deep networks can be trained; the block only has to learn a *correction* F(x).

**Batch normalisation.** After each convolution, activations are standardised with the batch mean and variance, then
rescaled by learned parameters. It stabilises and speeds up training. With very small labelled sets it still works
here because batches are drawn with augmentation.

**ResNet-18 layout (at 160 px input):**

| Stage | Operation | Output size | Channels |
|---|---|---|---|
| stem | 7x7 conv, stride 2 + 3x3 max-pool, stride 2 | 40 x 40 | 64 |
| layer1 | 2 basic blocks (two 3x3 convs each) | 40 x 40 | 64 |
| layer2 | 2 basic blocks, first with stride 2 | 20 x 20 | 128 |
| layer3 | 2 basic blocks, stride 2 | 10 x 10 | 256 |
| layer4 | 2 basic blocks, stride 2 | 5 x 5 | 512 |
| head | global average pool -> linear 512 -> classes | - | - |

"18" = 1 stem conv + 8 blocks x 2 convs + 1 linear layer. **Parameters: 11,181,642 with a 10-class head ✓**
(11,176,512 in the body + 512 x 10 + 10 in the head). For 37/38 classes the head adds ~14k more.

**Why ResNet-18 and not 34/50.** ResNet-34 has 21.8 M and ResNet-50 25.6 M parameters. Our ViT has 5.5 M. ResNet-18
is the smallest standard ResNet and the closest in size; from scratch on ~4,900 images or fewer, bigger CNNs mostly
add over-fitting risk. Arm A is a *reference point* (a well-understood scratch baseline), not a competitor we tune
to win.

## 8. Vision Transformer (ViT)

Dosovitskiy et al. (2021) showed that the Transformer, built for language, works on images if the image is turned
into a sequence of "words".

**Step 1 - patches.** The 160x160 image is cut into 16x16 patches: 160/16 = 10 per side, **N = 100 patches**. Each
patch is 16 x 16 x 3 = **768 numbers**.

**Step 2 - patch embedding.** Each patch is flattened and multiplied by a learned matrix to a vector of width
**D = 192** (ViT-Tiny). Implemented as a 16x16 convolution with stride 16. Parameters: 768 x 192 + 192 = 147,648.

**Step 3 - class token and position embedding.** A learned extra token ("[CLS]", 192 numbers) is prepended -> 101
tokens. Because attention ignores order, a learned **position embedding** (101 x 192 = 19,392 numbers) is added so
the model knows where each patch was.

**Step 4 - 12 Transformer blocks.** Each block does:
```
x = x + Attention(LayerNorm(x))      # patches exchange information
x = x + MLP(LayerNorm(x))            # each patch is processed on its own
```
- **Self-attention.** Each token makes a query q, key k and value v (three 192 -> 192 projections). The attention
  weight between tokens i and j is softmax(q_i · k_j / sqrt(d)); each token's output is the weighted average of all
  values. So every patch can look at every other patch in **one** layer - a global receptive field from the start.
- **Multi-head.** ViT-Tiny splits the 192 dims into **3 heads of 64**, so it can attend to 3 different things at once.
- **MLP.** 192 -> 768 -> 192 with a GELU non-linearity (expansion ratio 4).
- **LayerNorm** standardises each token's 192 numbers (instead of over the batch, unlike BatchNorm).
- **Residual "+ x"** as in ResNet.

Parameters per block: attention 192x576+576 (qkv) + 192x192+192 (output) = 148,224; MLP 192x768+768 + 768x192+192 =
295,872; two LayerNorms 768. **Total 444,864 per block, x 12 = 5,338,368.**

**Step 5 - pooling and head.** We average the 100 patch tokens (global average pooling, `vit_pool: avg`) after a
final LayerNorm (384 parameters) and apply a linear layer 192 -> classes (192 x 10 + 10 = 1,930 for 10 classes).

**Total: 5,338,368 + 147,648 + 192 + 19,392 + 384 + 1,930 = 5,507,914 parameters ✓** (exactly what PyTorch reports
for our 10-class model).

**What the names mean.**

| Model | Width D | Depth | Heads | MLP | Parameters |
|---|---|---|---|---|---|
| ViT-Tiny (ours) | 192 | 12 | 3 | 768 | 5.7 M (5.5 M at 160 px, 10 classes) |
| ViT-Small | 384 | 12 | 6 | 1,536 | 22 M |
| ViT-Base | 768 | 12 | 12 | 3,072 | 86 M |

"patch16" = 16x16 patches; "224" = the resolution it was pretrained at.

**Why ViTs are "data hungry".** A ViT has almost no built-in assumptions about images: nothing tells it that
neighbouring patches are related - it must *learn* locality from data. With enough data this flexibility wins; with
little data a ViT trained from scratch (arm D) usually loses to a CNN (arm A). This is exactly why pretraining (C or
E) matters so much for ViTs, and why D is the right control for E.

**Stochastic depth (drop path 0.1).** During fine-tuning each residual branch is randomly skipped with a probability
rising linearly to 0.1 in the last block - a regulariser that reduces over-fitting.

## 9. ImageNet pretraining and DeiT

**ImageNet-1k:** 1.28 million training photos, 1,000 classes (animals, objects, vehicles, food...). A network trained
to classify them learns general features - edges, textures, shapes, object parts - that transfer to other tasks.

**DeiT (Touvron et al. 2021):** a recipe that trains ViTs on ImageNet-1k *alone* (no extra data) with strong
augmentation and regularisation for 300 epochs. `deit_tiny_patch16_224` reaches ~72 % ImageNet top-1 accuracy. We
use the plain (non-distilled) version.

**Why ImageNet-1k only (not 21k).** Many public ViT weights are first trained on ImageNet-21k (14 million images).
Using those would give C far more external data, making the C vs E comparison less clean. With DeiT, C's advantage is
"1.28 M labelled external images" - clearly defined.

**Resolution change 224 -> 160.** DeiT learned 14x14 = 196 position embeddings. At 160 px there are 10x10 = 100
patches, so timm resizes the 14x14 grid of position embeddings to 10x10 by bicubic interpolation. This is standard
practice but slightly disturbs the pretrained model - a small handicap for C that we declare as a limitation (it
works *against* ImageNet, so if C still wins, the conclusion is stronger).

## 10. Masked Autoencoders (MAE)

**Idea.** Images are highly redundant: you can guess a hidden patch from its neighbours. If we hide most patches,
the model can only fill them in by understanding the content (shape of a leaf, layout of a field). That understanding
is what we want for classification - and it needs no labels.

**Procedure for one image (our numbers):**
1. 100 patches -> randomly keep **25 % = 25 patches**, hide 75 (`mask_ratio 0.75`).
2. The **encoder** (the same ViT-Tiny as C/D/E) processes only the 25 visible patches + [CLS] = 26 tokens. This
   makes MAE cheap: ~4x fewer tokens than normal training.
3. A small **decoder** gets the 25 encoded patches plus 75 copies of a learned "mask token" (each with its position
   embedding), and predicts the pixels of every patch. Decoder: width 128, **2 blocks**, 4 heads, then a linear layer
   128 -> 768 (= 16x16x3 pixels). About 0.52 M parameters, thrown away after pretraining.
4. **Target:** each hidden patch's pixels, normalised to zero mean and unit variance within the patch
   (`norm_pix_loss`). He et al. found this improves the learned features (the model focuses on structure, not
   brightness).
5. **Loss:** mean squared error, computed **only on the 75 hidden patches**:
   L = (1/|M|) Σ_{i in M} ||pred_i - target_i||² / 768.

**Why 75 %.** He et al. found 75 % best for ImageNet: low ratios make the task too easy (copy a neighbour), very high
ratios make it impossible. We keep the standard value (one recipe for all domains; no per-domain tuning).

**Position embeddings.** MAE starts the encoder's position embeddings from fixed 2-D sine-cosine patterns (a smooth
encoding of row and column), as in the original.

**Schedule.** 800 epochs over ~5,000 images, batch 256 -> 19 steps per epoch, **15,200 steps** (~22 min each on
the RTX 4050 at 85 ms/step; originally 400 epochs, raised to 800 before training - amendment 1). Learning rate
blr x batch / 256 = 1.5e-4 x 256 / 256 = **1.5e-4**, 40 warm-up epochs, then cosine decay; AdamW with
betas (0.9, 0.95) and weight decay 0.05 (He et al.). Augmentation: random resized crop (scale 0.2-1) + flip only.

**Why the same epochs everywhere.** All datasets have the same number of training images, so 400 epochs = the same
number of steps = the same MAE compute in every domain. Differences between domains then cannot come from one domain
getting more pretraining.

**One MAE per seed.** E's three seeds use three independently pretrained encoders, so E's error bars include the
randomness of pretraining, not only of fine-tuning.

## 11. Fine-tuning: optimiser, schedule and regularisation

| Setting | Value | Meaning / reason |
|---|---|---|
| Loss | cross-entropy | standard for classification; no label smoothing (it would distort calibration) |
| Optimiser | AdamW | Adam with decoupled weight decay; standard for ViTs and fine for ResNets |
| Weight decay | 0.05 | pulls weights towards zero (regularisation); not applied to biases, norms, tokens, position embeddings |
| Batch size | 128 (or all labelled images if fewer) | fits the 6 GB GPU comfortably |
| Precision | bfloat16 autocast | ~2x faster on RTX 40xx with no loss scaling needed |
| Gradient clipping | norm 1.0 | prevents rare huge updates |
| Warm-up | first 5 % of steps | learning rate rises linearly from ~0, avoiding early instability |
| Schedule | cosine decay to 0 | smooth decrease, standard |
| Steps | clamp(100 x ceil(n/128), 600, 4000) | ~100 epochs of the labelled data, at least 600 updates, at most 4,000 |
| Validation checks | 25 per run | macro-F1 on the validation set |
| Early stopping | 8 checks without improvement | saves time; the best-validation weights are restored |
| Layer-wise lr decay | 0.75 (C and E only) | block k (of 12) gets lr x 0.75^(13 - k); the patch embedding gets lr x 0.75^13 = 0.024 x lr ✓. Keeps general early-layer knowledge, adapts late layers more |
| Learning rate | chosen per (dataset, arm) on validation | scratch {5e-4, 1e-3, 2e-3}, pretrained {2.5e-4, 5e-4, 1e-3} |

**Steps per budget (example ~4,900 train images):** 100 % -> 39 batches per epoch -> 3,900 steps;
50 % -> 20 -> 2,000; 20 % -> 8 -> 800; 10 % and 5 % -> 600 (the minimum, i.e. many more passes over the few images).

**Fine-tuning augmentation:** random resized crop (area 35-100 %, aspect 3/4-4/3), horizontal flip, brightness and
contrast +-20 %. Hue/saturation changes are excluded on purpose: colour carries diagnostic information (leaf lesions,
land cover).

## 12. Evaluation metrics

Let y be the true class, ŷ the predicted class, p the predicted probability vector (softmax of the logits z).

| Metric | Formula / definition | Why |
|---|---|---|
| Accuracy | fraction with ŷ = y | intuitive, but dominated by large classes |
| Precision (class c) | TP / (TP + FP) | of predictions "c", how many were right |
| Recall (class c) | TP / (TP + FN) | of true "c", how many were found |
| F1 (class c) | 2PR / (P + R) | balance of both |
| **Macro-F1 (primary)** | mean of F1 over classes | every class counts equally - essential for imbalanced PlantVillage |
| Min class recall | worst class recall | shows a disease that is always missed |
| NLL | mean of -log p_y | penalises confident mistakes heavily; proper scoring rule |
| Brier | mean of Σ_c (p_c - 1[y=c])² | squared error of probabilities; proper scoring rule |
| **ECE** | Σ_bins (n_b/n) · abs(acc_b - conf_b), 15 equal-width bins of confidence | average gap between how sure and how right |
| Adaptive ECE | same with equal-count bins | less sensitive to empty bins |

**Temperature scaling.** Divide logits by one number T before softmax: p = softmax(z / T). T > 1 softens
over-confident predictions; T is fitted by minimising NLL on the **validation** set; predictions (argmax) do not
change, only confidence. We report test ECE/NLL/Brier both before and after. If an arm's calibration advantage
disappears after temperature scaling, the advantage was only a simple over/under-confidence.

## 13. Domain distance measures

All are computed with the frozen ImageNet DeiT-Tiny (the starting point of arm C), i.e. "how different does this
dataset look *to the ImageNet model*". Reference = ImageNetV2 (10,000 images, 10 per ImageNet class).

| Measure | How | Larger means |
|---|---|---|
| **FID** | fit a Gaussian (mean μ, covariance Σ) to each feature cloud; d² = abs(μ1 - μ2)² + Tr(Σ1 + Σ2 - 2(Σ1Σ2)^½) | farther from ImageNet |
| **OTDD** | optimal transport between the two labelled datasets; cost between two samples = feature distance² + distance² between their classes' Gaussians (diagonal); 64-d PCA, 1,500 samples each | farther (uses labels) |
| **ImageNet confidence** | mean of the ImageNet classifier's top probability on target images | *closer* (looks like ImageNet) |
| **Linear-probe accuracy** | logistic regression on frozen features, trained on train labels, scored on validation | *closer* (ImageNet features already fit) |

Using four measures avoids betting on one; whether they agree is itself informative. They are computed **before**
training, so they cannot be chosen to fit the results.

## 14. Statistical analysis

**Paired gap.** For dataset d, budget b, seed s: gap = F1(E) - F1(C). Pairing by seed means both arms used the same
labelled images, which removes the "lucky subset" noise. Mean over 3 seeds with a 95 % t-interval:
mean ± t(0.975, 2) · sd / sqrt(3), where t(0.975, 2) = 4.30 - wide, because 3 seeds is few; this is honest.

**Crossover budget.** Budgets are spaced roughly evenly on a log scale (5, 10, 20, 50, 100). Where the mean gap
changes sign between two budgets, the crossover is found by linear interpolation in log10(budget). If the gap never
changes sign the crossover is "censored" (reported as "<5 %" or ">100 %"). A crossover is called *supported* only if
the intervals exclude 0 on both sides.

**Mean gap.** One number per dataset: the average gap over the log-budget axis (trapezoid rule), used for H2.

**Correlation with distance.** Spearman's rho (rank correlation). With 4 datasets there are 4! = 24 possible
orderings, so the exact permutation p-value can be no smaller than 2/24 = **0.083**. H2 therefore cannot reach
p < 0.05 with 4 domains - it is reported as **exploratory**. The leave-one-dataset-out test predicts each dataset's
mean gap from the other three and compares the error with simply predicting their average.

**Ranking agreement.** For each budget, arms are ranked within each dataset. **Kendall's W** = 12S / (m²(n³ - n)),
with m datasets, n arms, S the squared deviation of rank sums: W = 1 means every dataset ranks the arms identically,
W = 0 no agreement. The **Friedman test** asks whether arms differ systematically across datasets (Demsar 2006).

## 15. Compute budget

Hardware: NVIDIA RTX 4050 Laptop GPU, 6 GB, Ada architecture (supports bfloat16). Deadline: 48 h for everything.

| Item | Count | Notes |
|---|---|---|
| MAE pretrainings | 12 | 15,200 steps each (~22 min), encoder on 26 tokens |
| LR selection runs | 48 | at the 20 % budget, seed 0 |
| Main fine-tuning runs | 240 | 600-4,000 steps each, early stopping |
| GPU memory | dataset ~0.54 GB + model/activations | well within 6 GB |

Exact timings come from `python run.py benchmark` (`results/crossover_v1/benchmark.json`) and are recorded in
`PROGRESS_LOG.md`.

## 16. Limitations (declared in advance)

- Small ViT (Tiny) and 160 px because of the compute budget; conclusions may differ for larger models.
- DeiT was pretrained at 224 px; position embeddings are interpolated to 160 px (handicaps C slightly).
- CIFAR-10 (32 px) and EuroSAT (64 px) are upsampled to 160 px.
- One dataset per domain and only 4 domains: H2 is exploratory.
- 3 seeds give wide confidence intervals.
- The labelled validation set (~1,050 images) is larger than the smallest training budgets; this is unrealistic for
  true low-label settings but identical for every arm, so the comparison stays fair.
- One MAE recipe (75 % mask, 800 epochs); no search over MAE settings.
- PlantVillage has lab backgrounds and is known to be "easy"; it may not reflect field conditions.
