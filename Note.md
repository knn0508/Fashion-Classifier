# Multi-Modal Product Classifier Notes

## Architecture

- Text branch: BERT (`bert-base-uncased`) encodes cleaned product text. The pooled `[CLS]` representation is projected into a compact dense vector.
- Image branch: ResNet-18 encodes each product image. The final classification layer is removed and the visual representation is projected into the same embedding size as the text vector.
- Fusion branch: the default model concatenates the text and image vectors, then passes them through dense classification layers.
- Bonus fusion: an optional attention-based fusion layer learns a query over the image and text vectors before classification.

## Dataset Strategy

- Source: local Kaggle Fashion Product Images dataset in `fashion-dataset/`.
- Label target: top 10 `articleType` classes.
- Image input: `fashion-dataset/images/{id}.jpg`.
- Text input: cleaned rich product description from `fashion-dataset/styles/{id}.json` when available, plus product display name and safe metadata such as brand, gender, color, season, usage, master category, and subcategory.
- Leakage guard: `articleType` is used only as the target label, not as an added metadata field in the constructed text.

## Training Plan

- Start with frozen BERT and ResNet-18 backbones for practical CPU/GPU training.
- Train the fusion and classification heads first.
- Optionally unfreeze BERT and/or ResNet with `--unfreeze-text` and `--unfreeze-image` for stronger results when enough compute is available.

## Evaluation

- Main metric: macro-F1 across the 10 classes.
- Secondary metrics: overall accuracy, per-class precision/recall/F1, class-wise accuracy, and confusion matrix.
- Original target from project note: Macro-F1 around `0.81` on 10 product categories, with a multi-modal gain over single-modality baselines.
