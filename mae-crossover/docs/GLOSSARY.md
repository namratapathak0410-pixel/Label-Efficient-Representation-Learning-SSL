# Glossary (plain English)

| Term | Meaning |
|---|---|
| **CNN / ResNet** | A network that slides small filters over the image. Built-in assumption that nearby pixels matter, which helps with little data. ResNet-18 = 18 layers |
| **ViT (Vision Transformer)** | Cuts the image into 16x16 patches (100 patches at 160 px) and treats them like words in a sentence; "attention" lets every patch look at every other patch. Few built-in assumptions, so it needs lots of data or pretraining |
| **ViT-Tiny / DeiT-Tiny** | A small ViT: 12 layers, 192 numbers per patch, ~5.5 M parameters. DeiT is a ViT recipe trained on ImageNet-1k only |
| **Pretraining** | Training a model on one task first so that it starts the real task with useful knowledge |
| **ImageNet** | 1.3 million labelled photos of 1,000 everyday object classes; the standard pretraining dataset |
| **ImageNetV2** | A newer, independent 10,000-image test set built like ImageNet; we use it as a public stand-in for ImageNet |
| **Self-supervised learning (SSL)** | Learning from images without labels by solving a made-up task |
| **MAE (Masked Autoencoder)** | The SSL task we use: hide 75 % of the patches, make the model redraw them. To do it, the model must learn what the images look like |
| **Encoder / decoder** | In MAE the encoder is the ViT we keep; the small decoder only redraws patches and is thrown away afterwards |
| **Fine-tuning** | Training the whole pretrained model on the labelled images of the real task |
| **Arm** | One of the four compared strategies (A, C, D, E) |
| **Label budget** | The share of training images that have labels (5 % ... 100 %) |
| **Crossover budget** | The label budget at which target-MAE (E) stops losing to ImageNet (C) and starts winning (or vice versa) |
| **Seed** | The number that fixes all randomness; repeating with 3 seeds shows how much results vary by chance |
| **Macro-F1** | F1 score computed per class, then averaged so that every class counts equally (fair under class imbalance) |
| **Calibration** | Whether the model's confidence is honest: of all predictions made with 80 % confidence, about 80 % should be correct |
| **ECE (Expected Calibration Error)** | Average gap between confidence and actual accuracy (15 bins); lower is better |
| **NLL / Brier score** | Two other measures of how good the predicted probabilities are; lower is better |
| **Temperature scaling** | One number that softens or sharpens the model's confidence, fitted on the validation set; fixes over-confidence without changing predictions |
| **FID** | Distance between two image collections, measured on features of a pretrained network; larger = more different from ImageNet |
| **OTDD** | Optimal-transport dataset distance: like FID but also uses the class labels |
| **Linear probe** | Freeze the ImageNet model, train only a simple linear classifier on its features; high accuracy = ImageNet features already fit this dataset |
| **dHash** | A 256-bit fingerprint of an image; near-identical images have near-identical fingerprints |
| **Data leakage** | Test information sneaking into training (e.g. the same leaf in both) - makes results look better than they are |
| **Pre-registration** | Writing down hypotheses and analysis before seeing results, with a timestamp |
| **Kendall's W** | 0-1 measure of how much the datasets agree on the ranking of the arms (1 = identical ranking everywhere) |
| **Layer-wise lr decay** | When fine-tuning a pretrained ViT, early layers get a smaller learning rate than later ones so their general knowledge is not destroyed |
