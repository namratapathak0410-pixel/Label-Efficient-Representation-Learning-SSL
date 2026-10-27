# Limitations, confounders and future work

## Limitations of this design
1. **Scaled-down models on CPU.** The CPU profile uses ResNet18, ViT-Tiny/16, 96 px images and <= 100 images per class.
   Conclusions apply to that regime and must not be presented as results for ResNet50 / ViT-B/16 at 224 px.
   Report which config produced each table.
2. **PlantVillage is a "lab" dataset**: leaves photographed one by one on a uniform background. High accuracy there
   does not imply field performance. Test the best models on a field dataset (e.g. PlantDoc) if possible.
3. **Near-duplicates are detected with a perceptual hash** (threshold 10/256 bits). It is not perfect: some
   visually different images of the same leaf (different angle) will not be grouped. Residual leakage is possible;
   state this honestly. Quantify with a sensitivity check using a stricter threshold.
4. **Pretraining budget.** MAE usually needs hundreds of epochs and large data to shine. With a few thousand
   images and ~100 epochs it may underperform; a negative result is a valid finding but cannot be generalised to "MAE does not help".
5. **ImageNet vs MAE are different data regimes** (external labelled data vs in-domain unlabelled data). C vs E
   differences mix algorithm, data source, label usage and compute.
6. **Single split.** Seeds vary the labelled subset and training randomness, not the train/val/test partition.
   A k-fold/repeated split would give a better variance estimate.
7. **Few seeds** (3 by default): standard deviations are rough; avoid significance claims without a proper test.
8. **ECE depends on binning** (15 equal-width bins) and on the number of test samples; Brier score is added for robustness.
   Temperature scaling is fitted on the validation set only.
9. **Fixed LR per model** with small/no tuning: a poorly tuned baseline makes MAE look better than it is.
   Give every model the same tuning effort and say what was tuned.
10. **MAE pretrained once** (one seed) and reused for all label budgets and seeds of model E: pretraining variance
    is not captured.

## Possible confounders to discuss
- Different learning rates, epochs and early-stopping behaviour across models.
- Different numbers of optimisation steps at small budgets (we enforce `min_updates` for all models alike).
- BatchNorm (CNN) vs LayerNorm (ViT) behaviour with very small labelled sets.
- Augmentation strength differs between pretraining and fine-tuning.
- Image resolution lower than the original ImageNet resolution for pretrained models (pos-embeddings are interpolated).
- Pretrained weights come from a specific timm checkpoint (training recipe differs between ResNet and ViT checkpoints).
- Class imbalance (some PlantVillage classes have many more images) -> accuracy vs macro-F1 can disagree.

## Future improvements
- Full-size run on GPU (`configs/gpu.yaml`) and more seeds.
- Second SSL method (SimCLR, DINO/DINOv2, iBOT) and a contrastive-vs-masked comparison.
- Semi-supervised baselines (FixMatch) that also use unlabelled data - the fair rival to SSL pretraining.
- Linear probe / k-NN evaluation of the representation, t-SNE/UMAP of embeddings.
- Domain-shift test on field images, robustness to corruptions.
- Other calibration methods (label smoothing, Mixup, focal loss) and calibration under domain shift.
- Statistical testing (paired bootstrap on test predictions, which `test_logits.npz` already supports).
