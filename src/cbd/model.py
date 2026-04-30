from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Small_Weights, convnext_small

from .common import DEFAULT_CLIP_LEN, resolve_input_size


@dataclass
class CBDModelOutput:
    pred_boxes: torch.Tensor
    type_logits: torch.Tensor | None = None
    center_cell_logits: torch.Tensor | None = None
    center_heatmap_logits: torch.Tensor | None = None
    attention_map: torch.Tensor | None = None
    grid_size: tuple[int, int] | None = None


def normalize_variant(model_config: dict) -> str:
    variant = str(model_config.get("variant", "v1_global_pool")).strip().lower()
    if variant in {"v1", "v1_global_pool"}:
        return "v1_global_pool"
    if variant in {"v2", "v2_spatiotemporal"}:
        return "v2_spatiotemporal"
    raise ValueError(f"Unsupported CBD model variant: {variant}")


def resolve_backbone_mode(model_config: dict, variant: str) -> str:
    if "unfreeze_backbone_mode" in model_config:
        return str(model_config["unfreeze_backbone_mode"]).strip().lower()
    if variant == "v2_spatiotemporal":
        return "last_stage"
    freeze_backbone = bool(model_config.get("freeze_backbone", True))
    return "freeze_all" if freeze_backbone else "full"


class CBDV1GlobalPool(nn.Module):
    def __init__(self, model_config: dict) -> None:
        super().__init__()
        self.clip_len = int(model_config.get("clip_len", DEFAULT_CLIP_LEN))
        self.mask_channels = int(model_config.get("mask_channels", 128))
        self.d_model = int(model_config.get("d_model", 256))
        self.num_layers = int(model_config.get("num_layers", 2))
        self.num_heads = int(model_config.get("num_heads", 8))
        self.dropout = float(model_config.get("dropout", 0.1))
        self.backbone_mode = resolve_backbone_mode(model_config, "v1_global_pool")
        pretrained = bool(model_config.get("pretrained", True))

        weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        self.rgb_backbone = convnext_small(weights=weights)
        self.rgb_backbone.classifier = nn.Identity()
        self.rgb_dim = 768
        self.rgb_norm = nn.LayerNorm(self.rgb_dim)

        if self.backbone_mode == "freeze_all":
            for parameter in self.rgb_backbone.parameters():
                parameter.requires_grad = False

        self.mask_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.mask_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(self.rgb_dim + self.mask_channels, self.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, self.clip_len, self.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 4),
        )

    def encode_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        features = self.rgb_backbone.features(rgb)
        features = features.mean(dim=(-2, -1))
        return self.rgb_norm(features)

    def encode_masks(self, masks: torch.Tensor) -> torch.Tensor:
        features = self.mask_encoder(masks)
        return features.flatten(1)

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.rgb_backbone.parameters() if parameter.requires_grad]

    def forward(self, rgb_clip: torch.Tensor, mask_clip: torch.Tensor) -> CBDModelOutput:
        batch_size, clip_len, _, _, _ = rgb_clip.shape
        rgb_flat = rgb_clip.reshape(batch_size * clip_len, *rgb_clip.shape[2:])
        mask_flat = mask_clip.reshape(batch_size * clip_len, *mask_clip.shape[2:])
        rgb_features = self.encode_rgb(rgb_flat)
        mask_features = self.encode_masks(mask_flat)
        fused = self.fusion(torch.cat([rgb_features, mask_features], dim=-1))
        fused = fused.view(batch_size, clip_len, self.d_model)
        fused = fused + self.position_embedding[:, :clip_len]
        temporal_features = self.temporal_transformer(fused)
        return CBDModelOutput(pred_boxes=self.head(temporal_features[:, -1]).sigmoid())


class CBDV2Spatiotemporal(nn.Module):
    def __init__(self, model_config: dict) -> None:
        super().__init__()
        self.clip_len = int(model_config.get("clip_len", DEFAULT_CLIP_LEN))
        self.mask_channels = int(model_config.get("mask_channels", 128))
        self.d_model = int(model_config.get("d_model", 256))
        self.num_layers = int(model_config.get("num_layers", 2))
        self.num_heads = int(model_config.get("num_heads", 8))
        self.dropout = float(model_config.get("dropout", 0.1))
        self.input_size = resolve_input_size(model_config)
        self.backbone_mode = resolve_backbone_mode(model_config, "v2_spatiotemporal")
        pretrained = bool(model_config.get("pretrained", True))

        weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        self.rgb_backbone = convnext_small(weights=weights)
        self.rgb_backbone.classifier = nn.Identity()
        self.rgb_dim = 768
        self.grid_size = max(1, self.input_size // 32)

        for parameter in self.rgb_backbone.parameters():
            parameter.requires_grad = False
        if self.backbone_mode == "full":
            for parameter in self.rgb_backbone.parameters():
                parameter.requires_grad = True
        elif self.backbone_mode == "last_stage":
            for module in self.rgb_backbone.features[6:]:
                for parameter in module.parameters():
                    parameter.requires_grad = True

        self.mask_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.mask_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fusion_projection = nn.Sequential(
            nn.Conv2d(self.rgb_dim + self.mask_channels, self.d_model, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout),
        )
        self.token_norm = nn.LayerNorm(self.d_model)
        self.temporal_position = nn.Parameter(torch.zeros(1, self.clip_len, 1, 1, self.d_model))
        self.row_position = nn.Parameter(torch.zeros(1, 1, self.grid_size, 1, self.d_model))
        self.col_position = nn.Parameter(torch.zeros(1, 1, 1, self.grid_size, self.d_model))
        self.clip_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.box_query = nn.Parameter(torch.zeros(1, 1, self.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.box_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.type_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 2),
        )
        self.type_conditioner = nn.Sequential(
            nn.Linear(2, self.d_model),
            nn.ReLU(inplace=True),
        )
        self.box_head = nn.Sequential(
            nn.LayerNorm(2 * self.d_model),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 4),
        )
        self.center_cell_head = nn.Linear(self.d_model, 1)
        self.center_heatmap_head = nn.Linear(self.d_model, 1)

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.rgb_backbone.parameters() if parameter.requires_grad]

    def forward(self, rgb_clip: torch.Tensor, mask_clip: torch.Tensor) -> CBDModelOutput:
        batch_size, clip_len, _, _, _ = rgb_clip.shape
        rgb_flat = rgb_clip.reshape(batch_size * clip_len, *rgb_clip.shape[2:])
        mask_flat = mask_clip.reshape(batch_size * clip_len, *mask_clip.shape[2:])

        rgb_features = self.rgb_backbone.features(rgb_flat)
        grid_h, grid_w = rgb_features.shape[-2:]
        mask_flat = F.interpolate(mask_flat, size=(grid_h, grid_w), mode="nearest")
        mask_features = self.mask_encoder(mask_flat)
        fused = self.fusion_projection(torch.cat([rgb_features, mask_features], dim=1))

        fused = fused.view(batch_size, clip_len, self.d_model, grid_h, grid_w).permute(0, 1, 3, 4, 2)
        fused = fused + self.temporal_position[:, :clip_len]
        fused = fused + self.row_position[:, :, :grid_h]
        fused = fused + self.col_position[:, :, :, :grid_w]
        fused = self.token_norm(fused)

        spatial_tokens = fused.reshape(batch_size, clip_len * grid_h * grid_w, self.d_model)
        clip_token = self.clip_token.expand(batch_size, -1, -1)
        tokens = torch.cat([clip_token, spatial_tokens], dim=1)
        tokens = self.temporal_transformer(tokens)

        clip_feature = tokens[:, 0]
        spatial_tokens = tokens[:, 1:].view(batch_size, clip_len, grid_h, grid_w, self.d_model)
        last_tokens = spatial_tokens[:, -1].reshape(batch_size, grid_h * grid_w, self.d_model)

        type_logits = self.type_head(clip_feature)
        type_probs = type_logits.softmax(dim=-1)
        type_feature = self.type_conditioner(type_probs)

        query = self.box_query.expand(batch_size, -1, -1)
        box_feature, attention_map = self.box_attention(query, last_tokens, last_tokens, need_weights=True)
        box_feature = box_feature.squeeze(1)
        conditioned_feature = torch.cat([box_feature, type_feature], dim=-1)
        pred_boxes = self.box_head(conditioned_feature).sigmoid()

        center_cell_logits = self.center_cell_head(last_tokens).squeeze(-1)
        center_heatmap_logits = self.center_heatmap_head(last_tokens).squeeze(-1).view(batch_size, grid_h, grid_w)

        return CBDModelOutput(
            pred_boxes=pred_boxes,
            type_logits=type_logits,
            center_cell_logits=center_cell_logits,
            center_heatmap_logits=center_heatmap_logits,
            attention_map=attention_map.squeeze(1).view(batch_size, grid_h, grid_w),
            grid_size=(grid_h, grid_w),
        )


class CBDBoxModel(nn.Module):
    def __init__(self, model_config: dict | None = None) -> None:
        super().__init__()
        self.model_config = model_config or {}
        self.variant = normalize_variant(self.model_config)
        if self.variant == "v2_spatiotemporal":
            self.impl = CBDV2Spatiotemporal(self.model_config)
        else:
            self.impl = CBDV1GlobalPool(self.model_config)

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return self.impl.backbone_trainable_parameters()

    def forward(self, rgb_clip: torch.Tensor, mask_clip: torch.Tensor) -> CBDModelOutput:
        return self.impl(rgb_clip, mask_clip)
