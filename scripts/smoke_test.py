from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import create_processed_dataset, validate_processed_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke tests for the multimodal product classifier project.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "fashion-dataset")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "smoke")
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def run_data_smoke(args: argparse.Namespace) -> dict[str, object]:
    paths = create_processed_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        sample_per_class=3,
        max_text_chars=400,
    )
    validation = validate_processed_dataset(paths["metadata"])
    return {"paths": {key: str(value) for key, value in paths.items()}, "validation": validation}


def run_model_smoke() -> dict[str, object]:
    import torch

    from src.model import ModelConfig, MultimodalProductClassifier

    results = {}
    for fusion in ["concat", "attention"]:
        config = ModelConfig(
            num_classes=10,
            fusion=fusion,
            embedding_dim=32,
            fusion_dim=64,
            freeze_text=True,
            freeze_image=True,
            pretrained_image=False,
        )
        model = MultimodalProductClassifier(config)
        model.eval()
        with torch.no_grad():
            logits = model(
                pixel_values=torch.randn(2, 3, 224, 224),
                input_ids=torch.ones(2, 8, dtype=torch.long),
                attention_mask=torch.ones(2, 8, dtype=torch.long),
            )
        assert tuple(logits.shape) == (2, 10), f"{fusion} logits shape mismatch: {tuple(logits.shape)}"
        results[fusion] = list(logits.shape)
    return results


def main() -> None:
    args = parse_args()
    report = {"data": run_data_smoke(args)}
    if not args.skip_model:
        report["model"] = run_model_smoke()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
