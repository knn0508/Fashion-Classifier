from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from transformers import AutoTokenizer

from src.config import DEFAULT_MAX_TEXT_LENGTH
from src.dataset import get_image_transform
from src.model import ModelConfig, MultimodalProductClassifier


def load_predictor(checkpoint_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = ModelConfig(**checkpoint["model_config"])
    label_map = checkpoint["label_map"]
    id_to_label = label_map["id_to_label"]

    tokenizer = AutoTokenizer.from_pretrained(model_config.text_model_name)
    transform = get_image_transform(train=False)
    model = MultimodalProductClassifier(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def predict(image: Image.Image, text: str, top_k: int = 5) -> dict[str, float]:
        if image is None:
            raise gr.Error("Please upload a product image.")
        if not text or not text.strip():
            raise gr.Error("Please enter product text or metadata.")

        pixel_values = transform(image.convert("RGB")).unsqueeze(0).to(device)
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=DEFAULT_MAX_TEXT_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()

        k = min(int(top_k), len(id_to_label))
        values, indices = torch.topk(probabilities, k=k)
        return {id_to_label[str(int(index))]: float(value) for value, index in zip(values, indices)}

    return predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Gradio demo for product classification.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict = load_predictor(args.checkpoint)
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="pil", label="Product image"),
            gr.Textbox(lines=6, label="Product text and metadata"),
            gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Top K"),
        ],
        outputs=gr.Label(num_top_classes=5, label="Predicted product category"),
        title="Multi-Modal Product Classifier",
        description="Classify fashion products from both image and text using BERT + ResNet fusion.",
    )
    demo.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
