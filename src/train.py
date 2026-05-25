from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .config import DEFAULT_DATASET_DIR, DEFAULT_MAX_TEXT_LENGTH, DEFAULT_OUTPUT_DIR, DEFAULT_PROCESSED_DIR, DEFAULT_SEED, DEFAULT_TEXT_MODEL
from .dataset import ProductMultimodalDataset, get_image_transform
from .metrics import classification_metrics, write_metrics
from .model import ModelConfig, MultimodalProductClassifier
from .preprocess import create_processed_dataset


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(
    model: MultimodalProductClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> tuple[float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, leave=False):
            batch = move_batch_to_device(batch, device)
            logits = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = criterion(logits, batch["labels"])

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * batch["labels"].size(0)
            predictions = logits.argmax(dim=1)
            y_true.extend(batch["labels"].detach().cpu().tolist())
            y_pred.extend(predictions.detach().cpu().tolist())

    average_loss = total_loss / max(1, len(loader.dataset))
    return average_loss, y_true, y_pred


def build_loaders(
    metadata_csv: Path,
    tokenizer: AutoTokenizer,
    batch_size: int,
    num_workers: int,
    max_length: int,
    sample_limit: int | None,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    frame = pd.read_csv(metadata_csv)
    if sample_limit:
        sampled_groups = []
        for _, group in frame.groupby(["split", "label"], sort=False):
            sampled_groups.append(group.sample(min(len(group), sample_limit), random_state=seed))
        frame = pd.concat(sampled_groups, ignore_index=True)

    datasets = {}
    for split in ["train", "val", "test"]:
        split_frame = frame[frame["split"] == split].reset_index(drop=True)
        datasets[split] = ProductMultimodalDataset(
            frame=split_frame,
            tokenizer=tokenizer,
            image_transform=get_image_transform(train=(split == "train")),
            max_length=max_length,
        )

    return (
        DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a BERT + ResNet multimodal product classifier.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text-model-name", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--fusion", choices=["concat", "attention"], default="concat")
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--fusion-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_TEXT_LENGTH)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional rows per split/class for fast experiments.")
    parser.add_argument("--sample-per-class", type=int, default=None, help="Create a smaller processed dataset before training.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--unfreeze-text", action="store_true")
    parser.add_argument("--unfreeze-image", action="store_true")
    parser.add_argument("--no-pretrained-image", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = args.processed_dir / "metadata.csv"
    label_map_path = args.processed_dir / "label_map.json"
    if not metadata_csv.exists() or not label_map_path.exists():
        create_processed_dataset(
            dataset_dir=args.dataset_dir,
            output_dir=args.processed_dir,
            sample_per_class=args.sample_per_class,
            seed=args.seed,
        )

    label_map = json.loads(label_map_path.read_text(encoding="utf-8"))
    id_to_label = label_map["id_to_label"]
    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)

    train_loader, val_loader, test_loader = build_loaders(
        metadata_csv=metadata_csv,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        sample_limit=args.sample_limit,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ModelConfig(
        num_classes=len(id_to_label),
        text_model_name=args.text_model_name,
        fusion=args.fusion,
        embedding_dim=args.embedding_dim,
        fusion_dim=args.fusion_dim,
        dropout=args.dropout,
        freeze_text=not args.unfreeze_text,
        freeze_image=not args.unfreeze_image,
        pretrained_image=not args.no_pretrained_image,
    )
    model = MultimodalProductClassifier(model_config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate)

    best_macro_f1 = -1.0
    history_rows = []
    best_checkpoint = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_true, train_pred = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_true, val_pred = run_epoch(model, val_loader, criterion, device)

        train_metrics = classification_metrics(train_true, train_pred, id_to_label)
        val_metrics = classification_metrics(val_true, val_pred, id_to_label)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_metrics["macro_f1"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_loss,
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history_rows.append(row)
        print(json.dumps(row, indent=2))

        if float(val_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(val_metrics["macro_f1"])
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(model_config),
                    "label_map": label_map,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_checkpoint,
            )

    history_path = args.output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_true, test_pred = run_epoch(model, test_loader, criterion, device)
    test_metrics = classification_metrics(test_true, test_pred, id_to_label)
    test_metrics["loss"] = test_loss
    write_metrics(test_metrics, args.output_dir / "test_metrics")
    print(f"Saved best checkpoint to {best_checkpoint}")
    print(json.dumps({"test_loss": test_loss, "test_macro_f1": test_metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
