import random
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image as PILImage, ImageFilter
from torch.utils.data import Dataset
from torchvision.transforms import v2

from sam3.image_utils import (
    coco_bbox_to_normalized_cxcywh,
    resize_image_to_square,
    resize_mask_to_square,
)
from data.dataset_camma import (
    CammaDataset,
    default_bbox_anchor_for_dataset,
    normalize_bbox_anchor,
)
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    InferenceMetadata,
    Object,
)


def normalize_category_name(name: str) -> str:
    return name.strip().lower()


def resolve_selected_categories(categories: dict[int, str], selected_names: Optional[Iterable[str]]):
    if isinstance(selected_names, str):
        selected_names = [part for part in selected_names.split(",") if part.strip()]
    elif selected_names is None:
        selected_names = []

    sorted_category_ids = sorted(categories)
    if not selected_names:
        return sorted_category_ids, [categories[cat_id] for cat_id in sorted_category_ids]

    name_to_id = {
        normalize_category_name(category_name): category_id
        for category_id, category_name in categories.items()
    }

    selected_category_ids = []
    canonical_names = []
    seen_category_ids = set()
    unknown_names = []
    for raw_name in selected_names:
        normalized_name = normalize_category_name(raw_name)
        category_id = name_to_id.get(normalized_name)
        if category_id is None:
            unknown_names.append(raw_name)
            continue
        if category_id in seen_category_ids:
            continue
        seen_category_ids.add(category_id)
        selected_category_ids.append(category_id)
        canonical_names.append(categories[category_id])

    if unknown_names:
        valid_names = ", ".join(categories[cat_id] for cat_id in sorted_category_ids)
        raise ValueError(f"Unknown class names: {unknown_names}. Valid class names: {valid_names}")

    return selected_category_ids, canonical_names


def normalize_split(split: str) -> str:
    split_normalized = split.strip().lower()
    if split_normalized in {"val", "valid", "validation"}:
        return "val"
    if split_normalized in {"train", "test"}:
        return split_normalized
    raise ValueError(f"Unsupported split: {split!r}. Expected one of train, val, test.")


def array_to_pil_rgb(image_array: np.ndarray) -> PILImage.Image:
    array = np.asarray(image_array)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Expected HWC image array, got shape {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        max_value = float(np.nanmax(array)) if array.size else 1.0
        array = np.nan_to_num(array, nan=0.0)
        if max_value <= 1.0:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    return PILImage.fromarray(array).convert("RGB")


def decode_segmentation_mask(segmentation, orig_h, orig_w):
    if not segmentation:
        return None
    if isinstance(segmentation, dict):
        return mask_utils.decode(segmentation)
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
        rle = mask_utils.merge(rles)
        return mask_utils.decode(rle)
    return None


class CammaSam3Dataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        dataset_name: str = "Endoscapes-Seg201-CBD",
        split: str = "train",
        annotation_file: str = "annotation_coco.json",
        selected_class_names: Optional[Iterable[str]] = None,
        include_negatives: bool = True,
        augment: bool = False,
        resolution: int = 1008,
        bbox_anchor: Optional[str] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.dataset_name = dataset_name
        self.split = normalize_split(split)
        self.annotation_file = annotation_file
        self.include_negatives = include_negatives
        self.augment = bool(augment and self.split == "train")
        self.resolution = int(resolution)
        if bbox_anchor is None:
            self.bbox_anchor = default_bbox_anchor_for_dataset(self.dataset_name)
        else:
            self.bbox_anchor = normalize_bbox_anchor(bbox_anchor)

        self.dataset = CammaDataset(
            root_dir=self.dataset_root,
            dataset_name=self.dataset_name,
            split=self.split,
            annotation_file=self.annotation_file,
        )
        self.categories = dict(self.dataset.context.category_id_to_name)
        self.selected_category_ids, self.selected_class_names = resolve_selected_categories(
            self.categories,
            selected_class_names,
        )
        self.indices = list(range(len(self.dataset)))
        self.image_ids = [int(self.dataset.context.coco["images"][index]["id"]) for index in self.indices]
        self.annotation_path = self.dataset.context.split_dir / self.annotation_file
        self.has_segmentation_masks = any(
            ann.get("segmentation") for ann in self.dataset.context.coco.get("annotations", [])
        )

        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.normalize = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.color_jitter = v2.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Datapoint:
        frame = self.dataset[self.indices[idx]]
        pil_image = array_to_pil_rgb(frame.pixel_array)
        orig_w, orig_h = pil_image.size

        if self.augment:
            if random.random() < 0.8:
                pil_image = self.color_jitter(pil_image)
            if random.random() < 0.2:
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.25)))

        pil_image = resize_image_to_square(pil_image, self.resolution)

        apply_horizontal_flip = self.augment and random.random() < 0.5
        if apply_horizontal_flip:
            pil_image = pil_image.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT)

        image_tensor = self.to_tensor(pil_image)
        if self.augment and random.random() < 0.25:
            image_tensor = torch.clamp(
                image_tensor + torch.randn_like(image_tensor) * random.uniform(0.0, 0.02),
                0.0,
                1.0,
            )
        image_tensor = self.normalize(image_tensor)

        objects = []
        category_to_object_ids = {cat_id: [] for cat_id in self.selected_category_ids}

        for object_index, ann in enumerate(frame.annotations):
            category_id = int(ann.get("category_id", -1))
            if category_id not in self.selected_category_ids:
                continue

            bbox_coco = ann.get("bbox")
            if bbox_coco is None:
                continue

            box_tensor = coco_bbox_to_normalized_cxcywh(
                bbox_coco,
                orig_w=orig_w,
                orig_h=orig_h,
                bbox_anchor=self.bbox_anchor,
            )
            if apply_horizontal_flip:
                box_tensor[0] = 1.0 - box_tensor[0]

            segment = None
            segmentation = ann.get("segmentation")
            if segmentation:
                mask_np = decode_segmentation_mask(segmentation, orig_h, orig_w)
                if mask_np is not None:
                    segment = resize_mask_to_square(
                        mask_np,
                        target_size=self.resolution,
                    )
                    if apply_horizontal_flip:
                        segment = torch.flip(segment, dims=[1])

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2] * box_tensor[3]).item(),
                object_id=len(objects),
                segment=segment,
                is_crowd=bool(ann.get("iscrowd", 0)),
            )
            objects.append(obj)
            category_to_object_ids[category_id].append(obj.object_id)

        image_obj = Image(data=image_tensor, objects=objects, size=(self.resolution, self.resolution))

        queries = []
        image_id = int(frame.metadata["id"])
        for category_id in self.selected_category_ids:
            object_ids = category_to_object_ids.get(category_id, [])
            if not object_ids and not self.include_negatives:
                continue
            queries.append(
                FindQueryLoaded(
                    query_text=self.categories[category_id],
                    image_id=0,
                    object_ids_output=object_ids,
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=image_id,
                        original_image_id=image_id,
                        original_category_id=category_id,
                        original_size=(orig_h, orig_w),
                        object_id=-1,
                        frame_index=-1,
                    ),
                )
            )

        return Datapoint(find_queries=queries, images=[image_obj], raw_images=[pil_image])


# Backward-compatible alias kept for existing imports and configs.
EndoscapesSam3Dataset = CammaSam3Dataset
