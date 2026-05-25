from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import DEFAULT_MAX_TEXT_LENGTH, DEFAULT_OUTPUT_DIR, DEFAULT_PROCESSED_DIR
from .dataset import ProductMultimodalDataset, get_image_transform, load_split
from .metrics import classification_metrics, write_metrics
from .model import ModelConfig, MultimodalProductClassifier
from .train import move_batch_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a multimodal product classifier checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "evaluation")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_TEXT_LENGTH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    label_map = checkpoint["label_map"]
    id_to_label = label_map["id_to_label"]

    tokenizer = AutoTokenizer.from_pretrained(model_config.text_model_name)
    frame = load_split(args.processed_dir / "metadata.csv", args.split)
    dataset = ProductMultimodalDataset(
        frame=frame,
        tokenizer=tokenizer,
        image_transform=get_image_transform(train=False),
        max_length=args.max_length,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MultimodalProductClassifier(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            logits = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = criterion(logits, batch["labels"])
            total_loss += float(loss.item()) * batch["labels"].size(0)
            y_true.extend(batch["labels"].detach().cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, id_to_label)
    metrics["loss"] = total_loss / max(1, len(dataset))
    metrics["split"] = args.split
    write_metrics(metrics, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
