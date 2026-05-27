# Multi-Modal Product Classifier

This project classifies fashion e-commerce products by combining product images and textual metadata. It uses the local Kaggle Fashion Product Images dataset in `fashion-dataset/`, with a ResNet-18 image encoder, a BERT text encoder, and a fused PyTorch classifier over the top 10 product `articleType` classes.

# Dataset
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset

## Project Structure

- `src/preprocess.py` prepares image paths, text fields, label mappings, and stratified splits.
- `src/model.py` defines the ResNet/BERT multi-input classifier with concat and attention fusion.
- `src/train.py` trains the model and writes checkpoints plus metrics.
- `src/evaluate.py` evaluates a checkpoint with macro-F1 and class-wise accuracy.
- `app.py` launches a Gradio image-plus-text prediction UI.
- `notebooks/multimodal_product_classifier.ipynb` walks through the full workflow.
- `scripts/smoke_test.py` validates preprocessing and, when dependencies are installed, model forward passes.

## Setup

Create and activate a Python environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file is configured to install PyTorch CUDA 12.4 (cu124) wheels.

## Prepare Data

```powershell
python -m src.preprocess --dataset-dir fashion-dataset --output-dir data/processed --top-k 10
```

This creates:

- `data/processed/metadata.csv`
- `data/processed/label_map.json`
- `data/processed/split_summary.json`

The pipeline drops rows without local images, keeps the top 10 `articleType` labels, uses JSON product descriptions where available, and falls back to product display names plus safe metadata.

## Train

Start with frozen BERT and ResNet backbones:

```powershell
python -m src.train --dataset-dir fashion-dataset --processed-dir data/processed --epochs 3 --batch-size 16 --fusion concat
```

For the bonus attention fusion:

```powershell
python -m src.train --dataset-dir fashion-dataset --processed-dir data/processed --epochs 3 --batch-size 16 --fusion attention
```

Useful quick-run options:

```powershell
python -m src.train --sample-per-class 50 --epochs 1 --batch-size 8
python -m src.train --unfreeze-text --unfreeze-image --epochs 2
```

Checkpoints and metrics are written under `outputs/`.

## Evaluate

```powershell
python -m src.evaluate --checkpoint outputs/best_model.pt --processed-dir data/processed --batch-size 16
```

Evaluation writes macro-F1, overall accuracy, class-wise accuracy, a classification report, and confusion matrix files into `outputs/evaluation/`.

## Gradio Demo

```powershell
python app.py --checkpoint outputs/best_model.pt
```

The app accepts an uploaded product image and product text/metadata, then returns top category predictions with probabilities.

## Smoke Tests

Preprocessing-only smoke test, useful before installing ML dependencies:

```powershell
python scripts/smoke_test.py --dataset-dir fashion-dataset --skip-model
```

Full smoke test after installing dependencies:

```powershell
python scripts/smoke_test.py --dataset-dir fashion-dataset
```
