from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModel

from .config import DEFAULT_TEXT_MODEL


@dataclass
class ModelConfig:
    num_classes: int
    text_model_name: str = DEFAULT_TEXT_MODEL
    fusion: str = "concat"
    embedding_dim: int = 256
    fusion_dim: int = 512
    dropout: float = 0.2
    freeze_text: bool = True
    freeze_image: bool = True
    pretrained_image: bool = True


class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float, freeze: bool = True, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values)
        return self.projection(features)


class TextEncoder(nn.Module):
    def __init__(self, model_name: str, embedding_dim: int, dropout: float, freeze: bool = True) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_dim = self.backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        if freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return self.projection(pooled)


class AttentionFusion(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float, num_heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([image_features, text_features], dim=1)
        query = self.query.expand(tokens.size(0), -1, -1)
        attended, _ = self.attention(query, tokens, tokens)
        fused = self.norm(attended.squeeze(1))
        return fused


class MultimodalProductClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.fusion not in {"concat", "attention"}:
            raise ValueError("fusion must be either 'concat' or 'attention'.")

        self.config = config
        self.image_encoder = ImageEncoder(
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            freeze=config.freeze_image,
            pretrained=config.pretrained_image,
        )
        self.text_encoder = TextEncoder(
            model_name=config.text_model_name,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            freeze=config.freeze_text,
        )

        self.fusion = config.fusion
        if self.fusion == "concat":
            self.fusion_layer = nn.Sequential(
                nn.Linear(config.embedding_dim * 2, config.fusion_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            )
            classifier_input = config.fusion_dim
        else:
            self.fusion_layer = AttentionFusion(config.embedding_dim, config.dropout)
            classifier_input = config.embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim, config.num_classes),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        image_features = self.image_encoder(pixel_values)
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.fusion == "concat":
            fused = self.fusion_layer(torch.cat([image_features, text_features], dim=1))
        else:
            fused = self.fusion_layer(image_features, text_features)

        logits = self.classifier(fused)
        if return_features:
            return {
                "logits": logits,
                "image_features": image_features,
                "text_features": text_features,
                "fused_features": fused,
            }
        return logits


def build_model(**kwargs: object) -> MultimodalProductClassifier:
    return MultimodalProductClassifier(ModelConfig(**kwargs))
