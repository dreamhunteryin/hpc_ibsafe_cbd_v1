from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor

from data.dataset_bsafe import BsafeDataset

from .common import normalize_target_types, resolve_training_settings


@dataclass(frozen=True)
class CBDRTDetrV4Record:
    annotation_id: int
    image_id: int
    split: str
    file_name: str
    image_path: Path
    width: int
    height: int
    target_bbox_xywh: tuple[float, float, float, float]
    target_type_name: str


def build_bsafe_rtdetrv4_records(
    *,
    split: str,
    data_config: dict[str, Any] | None = None,
) -> tuple[BsafeDataset, list[CBDRTDetrV4Record]]:
    data_config = dict(data_config or {})
    dataset = BsafeDataset(
        root_dir=data_config.get("dataset_root"),
        dataset_name=data_config.get("dataset_name", "Bsafe"),
        split=str(split),
        annotation_file=data_config.get("annotation_file"),
    )

    allowed_target_types = set(normalize_target_types(data_config.get("target_type", "all")))
    records: list[CBDRTDetrV4Record] = []
    for annotation in dataset.context.coco["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id not in dataset.context.image_id_to_index:
            continue

        target_type_name = str(annotation.get("type", "")).strip().lower()
        if target_type_name not in allowed_target_types:
            continue

        metadata = dataset.context.coco["images"][dataset.context.image_id_to_index[image_id]]
        image_path = dataset.context.resolve_image_path(
            str(metadata["file_name"]),
            video_id=metadata.get("video_id"),
        )
        bbox = annotation["bbox"]
        records.append(
            CBDRTDetrV4Record(
                annotation_id=int(annotation["id"]),
                image_id=image_id,
                split=str(split),
                file_name=str(metadata["file_name"]),
                image_path=image_path,
                width=int(metadata["width"]),
                height=int(metadata["height"]),
                target_bbox_xywh=tuple(float(value) for value in bbox),
                target_type_name=target_type_name,
            )
        )

    records.sort(key=lambda record: record.annotation_id)
    return dataset, records


def _maybe_append(ops: list[Any], op: Any | None) -> None:
    if op is not None:
        ops.append(op)


def _build_photometric_transform(enabled: bool) -> Any | None:
    if not enabled:
        return None
    if hasattr(v2, "RandomPhotometricDistort"):
        return v2.RandomPhotometricDistort(p=0.5)
    return v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03)


def _build_zoom_out_transform(enabled: bool) -> Any | None:
    if not enabled or not hasattr(v2, "RandomZoomOut"):
        return None
    return v2.RandomZoomOut(fill=0)


def _build_iou_crop_transform(enabled: bool) -> Any | None:
    if not enabled or not hasattr(v2, "RandomIoUCrop"):
        return None
    return v2.RandomApply([v2.RandomIoUCrop()], p=0.8)


def _build_sanitize_boxes_transform() -> Any | None:
    sanitize_cls = getattr(v2, "SanitizeBoundingBoxes", None)
    if sanitize_cls is None:
        sanitize_cls = getattr(v2, "SanitizeBoundingBox", None)
    if sanitize_cls is None:
        return None
    try:
        return sanitize_cls(min_size=1, labels_getter=None)
    except TypeError:
        return sanitize_cls(min_size=1)


def build_sample_transforms(
    *,
    image_size: int,
    train: bool,
    augment: bool,
    recipe_settings: dict[str, Any],
) -> v2.Compose:
    ops: list[Any] = []
    use_spatial_augment = bool(recipe_settings.get("zoom_out")) or bool(recipe_settings.get("iou_crop"))
    if train and augment:
        _maybe_append(ops, _build_photometric_transform(bool(recipe_settings.get("photometric_distort"))))
        if bool(recipe_settings.get("color_jitter")):
            ops.append(v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.08, hue=0.02))
        _maybe_append(ops, _build_zoom_out_transform(bool(recipe_settings.get("zoom_out"))))
        _maybe_append(ops, _build_iou_crop_transform(bool(recipe_settings.get("iou_crop"))))
        if use_spatial_augment:
            _maybe_append(ops, _build_sanitize_boxes_transform())
        if bool(recipe_settings.get("horizontal_flip", True)):
            ops.append(v2.RandomHorizontalFlip(p=0.5))
    ops.append(v2.Resize(size=(int(image_size), int(image_size)), antialias=True))
    if train and augment and use_spatial_augment:
        _maybe_append(ops, _build_sanitize_boxes_transform())
    ops.append(v2.ToDtype(torch.float32, scale=True))
    return v2.Compose(ops)


def xyxy_to_normalized_cxcywh(box_xyxy: list[float], *, height: int, width: int) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in box_xyxy]
    box_w = max(0.0, x1 - x0)
    box_h = max(0.0, y1 - y0)
    cx = x0 + 0.5 * box_w
    cy = y0 + 0.5 * box_h
    return [
        cx / max(1.0, float(width)),
        cy / max(1.0, float(height)),
        box_w / max(1.0, float(width)),
        box_h / max(1.0, float(height)),
    ]


class CBDRTDetrV4Dataset(Dataset):
    def __init__(
        self,
        *,
        split: str,
        data_config: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        train: bool = False,
    ) -> None:
        self.dataset, self.records = build_bsafe_rtdetrv4_records(
            split=split,
            data_config=data_config,
        )
        self.split = str(split)
        self.train = bool(train)
        self.recipe_settings = resolve_training_settings(model_config, training_config)
        self.transforms = build_sample_transforms(
            image_size=int(self.recipe_settings["image_size"]),
            train=self.train,
            augment=self.train and bool(self.recipe_settings.get("augment", True)),
            recipe_settings=self.recipe_settings,
        )
        self.fallback_transforms = None
        if self.train and bool(self.recipe_settings.get("augment", True)) and (
            bool(self.recipe_settings.get("zoom_out")) or bool(self.recipe_settings.get("iou_crop"))
        ):
            fallback_settings = dict(self.recipe_settings)
            fallback_settings["zoom_out"] = False
            fallback_settings["iou_crop"] = False
            self.fallback_transforms = build_sample_transforms(
                image_size=int(self.recipe_settings["image_size"]),
                train=self.train,
                augment=True,
                recipe_settings=fallback_settings,
            )

    @classmethod
    def from_config(cls, config: dict, split: str, train: bool = False) -> CBDRTDetrV4Dataset:
        return cls(
            split=split,
            data_config=config.get("data", {}),
            model_config=config.get("model", {}),
            training_config=config.get("training", {}),
            train=train,
        )

    def __len__(self) -> int:
        return len(self.records)

    def get_record(self, idx: int) -> CBDRTDetrV4Record:
        return self.records[idx]

    def load_image(self, record: CBDRTDetrV4Record) -> Image.Image:
        return Image.open(record.image_path).convert("RGB")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        image = self.load_image(record)
        image_t = tv_tensors.Image(pil_to_tensor(image))

        x, y, w, h = record.target_bbox_xywh
        boxes = tv_tensors.BoundingBoxes(
            torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32),
            format="XYXY",
            canvas_size=(record.height, record.width),
        )

        original_image_t = image_t
        original_boxes = boxes
        image_t, boxes = self.transforms(image_t, boxes)
        if len(boxes) == 0 and self.fallback_transforms is not None:
            image_t, boxes = self.fallback_transforms(original_image_t, original_boxes)
        if len(boxes) == 0:
            raise RuntimeError(
                f"Augmentation pipeline removed all boxes for sample annotation_id={record.annotation_id}."
            )
        image_tensor = torch.as_tensor(image_t)
        image_h, image_w = [int(value) for value in image_tensor.shape[-2:]]

        box_xyxy = [float(value) for value in boxes[0].tolist()]
        box_xywh = [
            box_xyxy[0],
            box_xyxy[1],
            max(0.0, box_xyxy[2] - box_xyxy[0]),
            max(0.0, box_xyxy[3] - box_xyxy[1]),
        ]
        box_cxcywh = xyxy_to_normalized_cxcywh(box_xyxy, height=image_h, width=image_w)

        target = {
            "labels": torch.tensor([0], dtype=torch.long),
            "boxes": torch.tensor([box_cxcywh], dtype=torch.float32),
            "image_id": torch.tensor([record.image_id], dtype=torch.long),
            "orig_size": torch.tensor([record.width, record.height], dtype=torch.long),
            "size": torch.tensor([image_h, image_w], dtype=torch.long),
        }

        return {
            "image": image_tensor,
            "target": target,
            "image_id": record.image_id,
            "annotation_id": record.annotation_id,
            "file_name": record.file_name,
            "image_path": str(record.image_path),
            "original_size": (record.height, record.width),
            "transformed_size": (image_h, image_w),
            "target_bbox_xywh": [float(value) for value in record.target_bbox_xywh],
            "target_bbox_xywh_transformed": box_xywh,
            "target_type_name": record.target_type_name,
            "record": record,
        }
