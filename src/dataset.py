from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import DEFAULT_IMAGE_SIZE, DEFAULT_MAX_TEXT_LENGTH


def get_image_transform(image_size: int = DEFAULT_IMAGE_SIZE, train: bool = False) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class ProductMultimodalDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        image_transform: Any,
        max_length: int = DEFAULT_MAX_TEXT_LENGTH,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        image = Image.open(Path(row["image_path"])).convert("RGB")
        pixel_values = self.image_transform(image)

        encoded = self.tokenizer(
            str(row["text"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(row["label_id"]), dtype=torch.long),
        }


def load_split(metadata_csv: Path, split: str, sample_limit: int | None = None, seed: int = 42) -> pd.DataFrame:
    frame = pd.read_csv(metadata_csv)
    split_frame = frame[frame["split"] == split].copy()
    if sample_limit and len(split_frame) > sample_limit:
        split_frame = split_frame.sample(sample_limit, random_state=seed).reset_index(drop=True)
    return split_frame.reset_index(drop=True)
