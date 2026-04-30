from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor

from .common import resolve_training_settings


class RTDETRv4Collator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        pixel_values = torch.stack([sample["image"] for sample in batch], dim=0)
        targets = [sample["target"] for sample in batch]
        return {
            "pixel_values": pixel_values,
            "targets": targets,
            "image_ids": [int(sample["image_id"]) for sample in batch],
            "annotation_ids": [int(sample["annotation_id"]) for sample in batch],
            "file_names": [sample["file_name"] for sample in batch],
            "image_paths": [sample["image_path"] for sample in batch],
            "original_sizes": torch.tensor([sample["original_size"] for sample in batch], dtype=torch.long),
            "transformed_sizes": torch.tensor([sample["transformed_size"] for sample in batch], dtype=torch.long),
            "target_boxes_xywh": torch.tensor([sample["target_bbox_xywh"] for sample in batch], dtype=torch.float32),
            "target_type_names": [sample["target_type_name"] for sample in batch],
            "records": [sample["record"] for sample in batch],
        }


def build_inference_transform(
    *,
    image_size: int,
) -> v2.Compose:
    return v2.Compose(
        [
            v2.Resize(size=(int(image_size), int(image_size)), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


def prepare_inference_image(
    image_path: str | Path,
    *,
    model_config: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    original_size = (int(image.height), int(image.width))
    recipe_settings = resolve_training_settings(model_config, training_config)
    transforms = build_inference_transform(image_size=int(recipe_settings["image_size"]))
    image_tensor = transforms(pil_to_tensor(image))
    return {
        "pixel_values": image_tensor.unsqueeze(0),
        "image_path": str(image_path),
        "file_name": image_path.name,
        "original_size": original_size,
        "score_threshold": float(recipe_settings.get("score_threshold", 0.0)),
    }
