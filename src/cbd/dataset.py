from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from sam3.image_utils import coco_bbox_to_normalized_cxcywh

from .common import (
    DEFAULT_CLIP_LEN,
    DEFAULT_IMAGE_SIZE,
    UNLABELED_TARGET_TYPE,
    apply_geometric_augmentation,
    load_effective_clip_frames,
    load_mask_cache_tensor,
    load_rgb_clip_tensor,
    normalize_target_types,
    resolve_input_size,
    target_type_to_label,
)
from .sources import (
    LEGACY_SOURCE_NAME,
    CBDSourceConfig,
    CBDSourceDataset,
    build_clip_dir,
    build_clip_key,
    build_mask_cache_dir,
    build_record_clip_id,
    build_source_dataset,
    normalize_requested_sources,
    resolve_phase_sources,
    resolve_source_configs,
    resolve_source_split,
    resolve_target_category_ids,
)


def build_clip_id(source_name: str, record_key: str) -> str:
    if source_name == LEGACY_SOURCE_NAME:
        return record_key
    return f"{source_name}:{record_key}"


@dataclass(frozen=True)
class CBDRecord:
    source_name: str
    source_kind: str
    source_config: CBDSourceConfig
    annotation_id: int
    image_id: int
    split: str
    source_split: str
    clip_key: str
    record_key: str
    clip_id: str
    metadata: dict[str, Any]
    clip_dir: Path
    mask_cache_dir: Path
    mask_cache_path: Path
    target_box: torch.Tensor
    target_type_name: str
    target_type_label: int


def _build_legacy_data_config(
    *,
    dataset_root: str | Path | None = None,
    clips_root: str | Path | None = None,
    mask_cache_root: str | Path | None = None,
    target_type: str | list[str] | tuple[str, ...] | None = "all",
) -> dict[str, Any]:
    data_config: dict[str, Any] = {"target_type": target_type}
    if dataset_root is not None:
        data_config["dataset_root"] = str(dataset_root)
    if clips_root is not None:
        data_config["clips_root"] = str(clips_root)
    if mask_cache_root is not None:
        data_config["easy_mask_cache_root"] = str(mask_cache_root)
    return data_config


def _build_records_for_source(
    source_name: str,
    source_config: CBDSourceConfig,
    dataset: CBDSourceDataset,
    *,
    split: str,
    target_type: str | list[str] | tuple[str, ...] | None = "all",
) -> list[CBDRecord]:
    valid_image_ids = set(dataset.context.image_id_to_index)
    duplicate_counts: Counter[str] = Counter()
    selected_annotations: list[dict[str, Any]] = []

    if source_config.kind == "bsafe":
        allowed_target_types = set(normalize_target_types(target_type))
        for annotation in dataset.context.coco["annotations"]:
            annotation_type = str(annotation.get("type", "")).strip().lower()
            if annotation_type not in allowed_target_types:
                continue
            image_id = int(annotation["image_id"])
            if image_id not in valid_image_ids:
                continue
            metadata = dataset.context.coco["images"][dataset.context.image_id_to_index[image_id]]
            duplicate_counts[build_clip_key(source_config, metadata)] += 1
            selected_annotations.append(annotation)
    else:
        target_category_ids = set(resolve_target_category_ids(source_config, dataset))
        for annotation in dataset.context.coco["annotations"]:
            if int(annotation["category_id"]) not in target_category_ids:
                continue
            image_id = int(annotation["image_id"])
            if image_id not in valid_image_ids:
                continue
            metadata = dataset.context.coco["images"][dataset.context.image_id_to_index[image_id]]
            duplicate_counts[build_clip_key(source_config, metadata)] += 1
            selected_annotations.append(annotation)

    records: list[CBDRecord] = []
    for annotation in selected_annotations:
        image_id = int(annotation["image_id"])
        metadata = dataset.context.coco["images"][dataset.context.image_id_to_index[image_id]]
        clip_key = build_clip_key(source_config, metadata)
        record_key = build_record_clip_id(
            clip_key,
            int(annotation["id"]),
            duplicate_counts[clip_key],
        )
        target_box = coco_bbox_to_normalized_cxcywh(
            annotation["bbox"],
            orig_w=int(metadata["width"]),
            orig_h=int(metadata["height"]),
            bbox_anchor=source_config.bbox_anchor,
        )
        if source_config.kind == "bsafe":
            target_type_name = str(annotation.get("type", "")).strip().lower()
            target_type_label = target_type_to_label(target_type_name)
        else:
            target_type_name = "cbd"
            target_type_label = UNLABELED_TARGET_TYPE

        records.append(
            CBDRecord(
                source_name=source_name,
                source_kind=source_config.kind,
                source_config=source_config,
                annotation_id=int(annotation["id"]),
                image_id=image_id,
                split=str(split),
                source_split=resolve_source_split(source_config, split),
                clip_key=clip_key,
                record_key=record_key,
                clip_id=build_clip_id(source_name, record_key),
                metadata=metadata,
                clip_dir=build_clip_dir(source_config.clips_root, clip_key),
                mask_cache_dir=build_mask_cache_dir(source_config.easy_mask_cache_root, split, clip_key),
                mask_cache_path=build_mask_cache_dir(source_config.easy_mask_cache_root, split, clip_key) / "masks.npz",
                target_box=target_box,
                target_type_name=target_type_name,
                target_type_label=target_type_label,
            )
        )

    return records


def build_cbd_records(
    *,
    split: str,
    data_config: dict[str, Any] | None = None,
    dataset_root: str | Path | None = None,
    clips_root: str | Path | None = None,
    mask_cache_root: str | Path | None = None,
    target_type: str | list[str] | tuple[str, ...] | None = "all",
    sources: str | list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, CBDSourceDataset], list[CBDRecord]]:
    resolved_data_config = dict(data_config or {})
    if not resolved_data_config:
        resolved_data_config = _build_legacy_data_config(
            dataset_root=dataset_root,
            clips_root=clips_root,
            mask_cache_root=mask_cache_root,
            target_type=target_type,
        )
    elif target_type is not None:
        resolved_data_config.setdefault("target_type", target_type)

    source_configs = resolve_source_configs(resolved_data_config)
    requested_sources = (
        normalize_requested_sources(sources, source_configs)
        if sources is not None
        else resolve_phase_sources(resolved_data_config, str(split), source_configs)
    )

    datasets_by_source: dict[str, CBDSourceDataset] = {}
    records: list[CBDRecord] = []
    for source_name in requested_sources:
        source_config = source_configs[source_name]
        dataset = build_source_dataset(source_config, split)
        datasets_by_source[source_name] = dataset
        records.extend(
            _build_records_for_source(
                source_name,
                source_config,
                dataset,
                split=split,
                target_type=resolved_data_config.get("target_type", target_type),
            )
        )

    records.sort(key=lambda record: (record.source_name, record.clip_key, record.annotation_id))
    return datasets_by_source, records


class CBDDataset(Dataset):
    def __init__(
        self,
        *,
        split: str,
        data_config: dict[str, Any] | None = None,
        dataset_root: str | Path | None = None,
        clips_root: str | Path | None = None,
        mask_cache_root: str | Path | None = None,
        target_type: str | list[str] | tuple[str, ...] | None = "all",
        sources: str | list[str] | tuple[str, ...] | None = None,
        clip_len: int = DEFAULT_CLIP_LEN,
        image_size: int = DEFAULT_IMAGE_SIZE,
        target_fps: int = 5,
        augmentation_level: int = 0,
    ) -> None:
        self.source_datasets, self.records = build_cbd_records(
            split=split,
            data_config=data_config,
            dataset_root=dataset_root,
            clips_root=clips_root,
            mask_cache_root=mask_cache_root,
            target_type=target_type,
            sources=sources,
        )
        self.context = (
            next(iter(self.source_datasets.values())).context if len(self.source_datasets) == 1 else None
        )
        self.split = str(split)
        self.target_type = normalize_target_types(target_type)
        self.clip_len = int(clip_len)
        self.image_size = int(image_size)
        self.target_fps = int(target_fps)
        self.augmentation_level = int(augmentation_level)

    @classmethod
    def from_config(cls, config: dict, split: str, augmentation_level: int = 0) -> CBDDataset:
        data_config = dict(config.get("data", {}))
        model_config = config.get("model", {})
        return cls(
            split=split,
            data_config=data_config,
            target_type=data_config.get("target_type", "all"),
            sources=resolve_phase_sources(data_config, split, resolve_source_configs(data_config)),
            clip_len=int(model_config.get("clip_len", DEFAULT_CLIP_LEN)),
            image_size=resolve_input_size(model_config),
            target_fps=int(data_config.get("clip_fps", 5)),
            augmentation_level=augmentation_level,
        )

    def __len__(self) -> int:
        return len(self.records)

    def get_record(self, idx: int) -> CBDRecord:
        return self.records[idx]

    def find_record_by_clip_id(self, clip_id: str) -> CBDRecord:
        normalized_clip_id = str(clip_id)
        matches = [
            record
            for record in self.records
            if record.clip_id == normalized_clip_id or record.record_key == normalized_clip_id
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(clip_id)
        raise KeyError(f"Clip id {clip_id!r} is ambiguous across the selected sources.")

    def load_frames(self, record: CBDRecord):
        dataset = self.source_datasets[record.source_name]
        return load_effective_clip_frames(
            dataset,
            record.source_config,
            record.metadata,
            record.clip_dir,
            clip_len=self.clip_len,
            target_fps=self.target_fps,
            target_box_cxcywh=record.target_box,
        )

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        frames = self.load_frames(record)
        rgb = load_rgb_clip_tensor(frames, image_size=self.image_size)
        masks, payload = load_mask_cache_tensor(
            record.mask_cache_path,
            image_size=self.image_size,
            clip_len=self.clip_len,
        )
        target_box = record.target_box.clone()

        if self.augmentation_level > 0:
            rgb, masks, target_box = apply_geometric_augmentation(
                rgb,
                masks,
                target_box,
                image_size=self.image_size,
                augmentation_level=self.augmentation_level,
            )

        return {
            "rgb": rgb,
            "masks": masks,
            "target_box": target_box,
            "target_type_label": torch.tensor(record.target_type_label, dtype=torch.long),
            "target_type_name": record.target_type_name,
            "clip_id": record.clip_id,
            "frame_names": list(frames.frame_names),
            "original_size": tuple(frames.original_size),
            "mask_prompts": [str(prompt) for prompt in payload["prompts"].tolist()],
            "record": record,
        }


def cbd_collate_fn(batch: list[dict]) -> dict:
    return {
        "rgb": torch.stack([sample["rgb"] for sample in batch], dim=0),
        "masks": torch.stack([sample["masks"] for sample in batch], dim=0),
        "target_box": torch.stack([sample["target_box"] for sample in batch], dim=0),
        "target_type_label": torch.stack([sample["target_type_label"] for sample in batch], dim=0),
        "target_type_name": [sample["target_type_name"] for sample in batch],
        "clip_id": [sample["clip_id"] for sample in batch],
        "frame_names": [sample["frame_names"] for sample in batch],
        "original_size": [sample["original_size"] for sample in batch],
        "mask_prompts": [sample["mask_prompts"] for sample in batch],
        "record": [sample["record"] for sample in batch],
    }
