# Methodology (thesis-ready draft)

> Fill every `[ ... ]` from your own logs (`eda_summary.json`, `split_report.json`, `runs.csv`, config file).
> Do not write numbers you have not measured.

## 3.1 Research question and hypotheses
**RQ.** Does target-domain Masked Autoencoder (MAE) pretraining improve classification performance and prediction
calibration of a Vision Transformer when only a small fraction of the training labels is available, compared with
(i) training the same architecture from scratch and (ii) fine-tuning ImageNet-pretrained models?

**H1.** At small label budgets, E (MAE-pretrained ViT) achieves higher macro-F1 than D (ViT from scratch).
**H2.** E is competitive with or better than C (ImageNet-pretrained ViT) at the smallest budgets, despite using no external data.
**H3.** E has lower Expected Calibration Error than D. (Hypotheses can be rejected; report the outcome either way.)

## 3.2 Dataset
The experiments use the [PlantVillage / other] dataset [source, version/date downloaded]. After download we verified that
the folder contains [N] images in [K] classes (class sizes from [min] to [max]; imbalance ratio [r]); [m] corrupted files
were removed. Only the `color` images were used. The "[config name]" configuration uses at most [x] images per class
([N_used] images in total).

## 3.3 Data splitting and leakage control
Before any pretraining or model selection the images were divided into train / validation / test ([70/15/15] %).
Because the dataset contains several photographs or augmented copies of the same leaf, exact duplicates (MD5) and near-duplicates
(Hamming distance <= [10] bits between 256-bit difference hashes) were grouped, and whole groups were assigned to
a single split, stratified by class. The split report shows [g] duplicate groups and zero groups spanning splits.
The split (seed [42]) was fixed for all experiments. The test set was not used for self-supervised pretraining,
augmentation design, hyper-parameter selection, early stopping or model selection.

## 3.4 Label budgets
Supervised training used b in {[1, 5, 10, 20, 100]} % of the labelled training images, sampled stratified by class
with at least one image per class. For each seed the subsets are nested and identical for every model. The validation
and test labels were never used for training. [For the pilot with 1,000 images only [~700] images are in the training
split; pretraining used only these, because the held-out validation/test images were reserved.]

## 3.5 Models
| ID | Architecture | Initialisation |
|---|---|---|
| A | [ResNet18/ResNet50] | random |
| B | [same] | ImageNet-1k supervised weights (timm) |
| C | [ViT-Tiny/16 / ViT-B/16] | ImageNet weights (timm), position embeddings interpolated to [img_size] px |
| D | same as C | random |
| E | same as C | MAE encoder pretrained on the unlabelled training images |

**ImageNet pretraining vs target-domain self-supervised pretraining.** ImageNet pretraining uses about 1.3 million
labelled images of other domains, i.e. external data and external supervision. MAE pretraining uses only the unlabelled training
images of the target domain, which is far less data but in-domain and label-free. They answer different questions
and require different resources; we therefore never treat them as equivalent and discuss their difference in the results.

**MAE.** The encoder is a ViT that processes only the visible patches; [75] % of the 16x16 patches are masked at random. A
lightweight Transformer decoder ([depth] blocks, width [dim]) reconstructs the pixels of the masked patches, with
per-patch normalised targets and mean-squared error computed on masked patches only (He et al., 2022). Pretraining ran for
[100] epochs with AdamW (lr [..], weight decay 0.05, cosine schedule with [5] warm-up epochs) and batch size [..]. After pretraining the
decoder was discarded and a linear classification head was added to the encoder.

## 3.6 Preprocessing and augmentation
Images were resized to [img_size] x [img_size] and normalised with the ImageNet mean/std for all models. Training
augmentation: random resized crop (scale 0.6-1.0, aspect ratio 0.9-1.1), horizontal and vertical flips, brightness/contrast jitter of 15 %.
Hue/saturation jitter, strong blur and erasing were deliberately excluded, because colour and lesion texture are the diagnostic signal.
Evaluation used deterministic resizing only. MAE pretraining used random resized crop (scale 0.4-1.0) and flips.

## 3.7 Training protocol
All classifiers were trained with AdamW (weight decay 0.05, no decay for biases/norms), cross-entropy loss, linear warm-up followed
by cosine decay, gradient clipping at 1.0, batch size [..], and a model-specific learning rate chosen on the validation set
([table of learning rates]). To avoid under-training at small budgets every run had at least [300] optimisation steps.
Validation macro-F1 was computed about [30] times per run; training stopped after [8] evaluations without improvement and the best-validation
checkpoint was restored. Each configuration was repeated with seeds {[0, 1, 2]}; the seed changes the labelled subset, initialisation of the head and data order.

## 3.8 Evaluation
On the held-out test set we report accuracy, macro precision, macro recall, macro-F1, per-class precision/recall/F1 and the
confusion matrix. Reliability of the predicted probabilities is measured by Expected Calibration Error (15 equal-width confidence bins),
the multi-class Brier score and reliability diagrams. High accuracy does not imply good calibration: a network can predict the right
class almost always and still be systematically over-confident. We additionally report ECE after temperature scaling, with the temperature fitted on the validation set only.
Training time, parameter count, float32 model size and peak memory are logged for every run; for E the MAE pretraining time is reported separately.
Results are reported as mean ± standard deviation over seeds.

## 3.9 Model selection
Best models were chosen on **validation** scores for four separate criteria: classification (validation macro-F1 at 100 % labels),
label efficiency (validation macro-F1 averaged over budgets), calibration (validation ECE averaged over budgets) and computational cost
(total training time and model size). Test scores are reported for these models but not used to select them.

## 3.10 Implementation and reproducibility
Python [version], PyTorch [version], timm [version]; hardware: [CPU/GPU, RAM]. Code, configuration files and the split file are
provided in [repository]. All random seeds are fixed; CUDA kernels can introduce small non-determinism.

## References to cite (verify details yourself)
- He et al., *Masked Autoencoders Are Scalable Vision Learners*, CVPR 2022.
- Dosovitskiy et al., *An Image is Worth 16x16 Words*, ICLR 2021.
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
- Hughes & Salathé, *An open access repository of images on plant health...* (PlantVillage), 2015.
- Papers on self-supervised / label-efficient plant disease recognition found in your own literature search.
