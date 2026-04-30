#!/usr/bin/env python3

import argparse
import contextlib
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
import torch
import yaml
from PIL import Image as PILImage
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
INFER = ROOT / "infer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(INFER) not in sys.path:
    sys.path.insert(0, str(INFER))

from camma_artifacts import build_camma_artifact_paths, read_json, write_json
from infer_lora import COLOR_CYCLE, render_overlay
from sam3 import build_sam3_image_model
from sam3.data import CammaSam3Dataset
from sam3.eval import CGF1Evaluator, COCOCustom
from sam3.image_utils import (
    anchored_bbox_to_original_xywh,
    compute_detection_scores,
    normalized_cxcywh_to_original_xywh,
    resize_mask_to_original,
)
from sam3.lora import LoRAConfig, apply_lora_to_model, count_parameters, load_lora_weights
from sam3.model.box_ops import box_cxcywh_to_xyxy, box_iou
from sam3.model.model_misc import SAM3Output
from sam3.perflib.nms import generic_nms
from sam3.train.data import collate_fn_api


def build_lora_config(lora_config: dict) -> LoRAConfig:
    return LoRAConfig(
        rank=int(lora_config.get("rank", 16)),
        alpha=int(lora_config.get("alpha", 32)),
        dropout=float(lora_config.get("dropout", 0.1)),
        target_modules=lora_config.get("target_modules"),
        target_module_match=str(lora_config.get("target_module_match", "exact")),
        apply_to_vision_encoder=bool(lora_config.get("apply_to_vision_encoder", False)),
        apply_to_text_encoder=bool(lora_config.get("apply_to_text_encoder", True)),
        apply_to_geometry_encoder=bool(lora_config.get("apply_to_geometry_encoder", False)),
        apply_to_detr_encoder=bool(lora_config.get("apply_to_detr_encoder", True)),
        apply_to_detr_decoder=bool(lora_config.get("apply_to_detr_decoder", True)),
        apply_to_decoder_text_attention=lora_config.get("apply_to_decoder_text_attention"),
        apply_to_mask_decoder=bool(lora_config.get("apply_to_mask_decoder", False)),
    )


def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            setattr(obj, field, move_to_device(getattr(obj, field), device))
        return obj
    return obj


def padded_mask_to_original(mask_tensor, orig_h, orig_w, target_size):
    del target_size
    return resize_mask_to_original(mask_tensor, orig_h, orig_w)


def normalize_iou_type(iou_type: str | None) -> str:
    if iou_type is None:
        return "segm"

    value = str(iou_type).strip().lower()
    if value in {"segm", "segmentation", "mask", "masks"}:
        return "segm"
    if value in {"bbox", "box", "boxes"}:
        return "bbox"
    raise ValueError(f"Unsupported iou_type={iou_type!r}. Expected one of: segm, bbox.")


def resolve_iou_type(evaluation_config: dict, requested_iou_type: str | None) -> str:
    if requested_iou_type is not None:
        return normalize_iou_type(requested_iou_type)

    configured_iou_type = evaluation_config.get("iou_type", evaluation_config.get("mode"))
    return normalize_iou_type(configured_iou_type)


def resolve_save_overlays(evaluation_config: dict, requested_save_overlays: bool | None) -> bool:
    if requested_save_overlays is not None:
        return bool(requested_save_overlays)
    return bool(evaluation_config.get("save_overlays", False))


def resolve_show_prediction_masks(
    evaluation_config: dict,
    requested_show_prediction_masks: bool | None,
) -> bool:
    if requested_show_prediction_masks is not None:
        return bool(requested_show_prediction_masks)
    return bool(evaluation_config.get("show_prediction_masks", True))


def append_stem_suffix(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def build_eval_paths(artifact_paths, iou_type: str) -> dict[str, Path]:
    suffix = "" if iou_type == "segm" else f"_{iou_type}"
    return {
        "predictions_coco_path": append_stem_suffix(artifact_paths.predictions_coco_path, suffix),
        "legacy_predictions_coco_path": append_stem_suffix(artifact_paths.legacy_predictions_coco_path, suffix),
        "cgf1_predictions_coco_path": append_stem_suffix(artifact_paths.cgf1_predictions_coco_path, suffix),
        "results_path": append_stem_suffix(artifact_paths.results_path, suffix),
        "legacy_results_path": append_stem_suffix(artifact_paths.legacy_results_path, suffix),
    }


def prediction_path_candidates_for_iou(eval_paths: dict[str, Path]) -> list[Path]:
    candidates = []
    seen = set()
    for path in (eval_paths["predictions_coco_path"], eval_paths["legacy_predictions_coco_path"]):
        if path in seen:
            continue
        seen.add(path)
        candidates.append(path)
    return candidates


def build_overlay_dir(eval_root: Path, iou_type: str) -> Path:
    return eval_root / ("overlays" if iou_type == "segm" else f"overlays_{iou_type}")


def nms_boxes(
    pred_probs: torch.Tensor,
    pred_boxes: torch.Tensor,
    prob_threshold: float,
    iou_threshold: float,
) -> torch.Tensor:
    is_valid = pred_probs > prob_threshold
    probs = pred_probs[is_valid]
    if probs.numel() == 0:
        return is_valid

    boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes[is_valid])
    ious, _ = box_iou(boxes_xyxy, boxes_xyxy)
    kept_inds = generic_nms(ious, probs, iou_threshold)

    valid_inds = torch.where(is_valid, is_valid.cumsum(dim=0) - 1, -1)
    keep = torch.isin(valid_inds, kept_inds)
    return keep


def decode_coco_segmentation(segmentation, height: int, width: int) -> np.ndarray | None:
    if segmentation is None:
        return None

    if isinstance(segmentation, np.ndarray):
        mask = np.asarray(segmentation)
    elif isinstance(segmentation, dict):
        mask = mask_utils.decode(segmentation)
    elif isinstance(segmentation, (list, tuple)):
        if len(segmentation) == 0:
            return np.zeros((height, width), dtype=np.uint8)
        polygons = segmentation
        if isinstance(polygons[0], (int, float)):
            polygons = [polygons]
        rles = mask_utils.frPyObjects(polygons, height, width)
        mask = mask_utils.decode(rles)
    else:
        raise TypeError(
            "Unsupported segmentation format for overlays. Expected an RLE dict, polygon list, or numpy mask."
        )

    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = np.any(mask, axis=-1)
    return mask.astype(np.uint8)


def convert_annotation_bbox_to_xywh(
    bbox: list[float] | tuple[float, float, float, float] | None,
    image_size: tuple[int, int],
    bbox_anchor: str = "topleft",
) -> list[float] | None:
    if bbox is None:
        return None

    width, height = image_size
    bbox_xywh = anchored_bbox_to_original_xywh(
        bbox,
        bbox_anchor=bbox_anchor,
        orig_w=width,
        orig_h=height,
    )
    if bbox_xywh[2] <= 0.0 or bbox_xywh[3] <= 0.0:
        return None
    return [float(value) for value in bbox_xywh]


def normalize_annotation_for_coco(
    annotation: dict,
    image_size: tuple[int, int],
    bbox_anchor: str = "topleft",
) -> dict:
    normalized = dict(annotation)
    bbox_xywh = convert_annotation_bbox_to_xywh(
        normalized.get("bbox"),
        image_size=image_size,
        bbox_anchor=bbox_anchor,
    )
    if bbox_xywh is not None:
        normalized["bbox"] = bbox_xywh
        if normalized.get("segmentation") is None:
            normalized["area"] = float(bbox_xywh[2] * bbox_xywh[3])
    return normalized


def build_image_size_lookup(coco_images: list[dict]) -> dict[int, tuple[int, int]]:
    return {
        int(image["id"]): (int(image["width"]), int(image["height"]))
        for image in coco_images
    }


def build_overlay_predictions(
    annotations: list[dict],
    category_ids: list[int],
    category_names: dict[int, str],
    image_size: tuple[int, int],
    include_masks: bool,
    bbox_anchor: str = "topleft",
) -> list[dict]:
    width, height = image_size
    detections_by_category = {int(category_id): [] for category_id in category_ids}

    for annotation in annotations:
        category_id = int(annotation["category_id"])
        if category_id not in detections_by_category:
            continue

        bbox_xywh = convert_annotation_bbox_to_xywh(
            annotation.get("bbox"),
            image_size=image_size,
            bbox_anchor=bbox_anchor,
        )
        mask = None
        if include_masks and annotation.get("segmentation") is not None:
            mask = decode_coco_segmentation(annotation["segmentation"], height=height, width=width)
            if mask is not None and not np.any(mask):
                mask = None

        if bbox_xywh is None and mask is None:
            continue

        detections_by_category[category_id].append(
            {
                "mask": mask,
                "bbox_xywh": bbox_xywh,
                "score": float(annotation.get("score", 1.0)),
            }
        )

    overlay_predictions = []
    for index, category_id in enumerate(category_ids):
        overlay_predictions.append(
            {
                "prompt": category_names[category_id],
                "color": COLOR_CYCLE[index % len(COLOR_CYCLE)],
                "detections": detections_by_category[category_id],
            }
        )
    return overlay_predictions


def save_eval_overlays(
    dataset: CammaSam3Dataset,
    image_ids: list[int],
    category_ids: list[int],
    coco_predictions: list[dict],
    overlay_dir: Path,
    include_gt_masks: bool,
    include_prediction_masks: bool,
) -> None:
    predictions_by_image: dict[int, list[dict]] = {}
    for prediction in coco_predictions:
        image_id = int(prediction["image_id"])
        predictions_by_image.setdefault(image_id, []).append(prediction)

    overlay_dir.mkdir(parents=True, exist_ok=True)
    for image_id in tqdm(image_ids, desc="Saving overlays"):
        frame = dataset.dataset.get_frame_by_id(int(image_id))
        image = PILImage.fromarray(np.asarray(frame.pixel_array)).convert("RGB")
        image_size = image.size
        gt_overlay = build_overlay_predictions(
            annotations=frame.annotations,
            category_ids=category_ids,
            category_names=dataset.categories,
            image_size=image_size,
            include_masks=include_gt_masks,
            bbox_anchor=dataset.bbox_anchor,
        )
        pred_overlay = build_overlay_predictions(
            annotations=predictions_by_image.get(int(image_id), []),
            category_ids=category_ids,
            category_names=dataset.categories,
            image_size=image_size,
            include_masks=include_prediction_masks,
            bbox_anchor="topleft",
        )

        output_file = overlay_dir / str(frame.metadata["file_name"])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        render_overlay(
            image=image,
            predictions=gt_overlay,
            output_path=output_file.with_stem(output_file.stem + "_gt").with_suffix(".png"),
            draw_masks=include_gt_masks,
        )
        render_overlay(
            image=image,
            predictions=pred_overlay,
            output_path=output_file.with_stem(output_file.stem + "_pred").with_suffix(".png"),
            draw_masks=include_prediction_masks,
        )


def apply_sam3_nms(
    pred_logits,
    pred_masks,
    pred_boxes,
    presence_logit=None,
    prob_threshold=0.3,
    nms_iou_threshold=0.7,
    max_detections=100,
    iou_type="segm",
):
    from sam3.perflib.nms import nms_masks

    if len(pred_logits) == 0:
        return pred_masks[:0], pred_logits[:0].squeeze(-1), pred_boxes[:0]

    pred_probs = compute_detection_scores(pred_logits, presence_logit)
    pred_masks_sigmoid = torch.sigmoid(pred_masks)
    normalized_iou_type = normalize_iou_type(iou_type)
    if normalized_iou_type == "bbox":
        keep_mask = nms_boxes(
            pred_probs=pred_probs,
            pred_boxes=pred_boxes,
            prob_threshold=prob_threshold,
            iou_threshold=nms_iou_threshold,
        )
    else:
        pred_masks_binary = pred_masks_sigmoid > 0.5
        keep_mask = nms_masks(
            pred_probs=pred_probs,
            pred_masks=pred_masks_binary.float(),
            prob_threshold=prob_threshold,
            iou_threshold=nms_iou_threshold,
        )

    filtered_masks = pred_masks_sigmoid[keep_mask]
    filtered_scores = pred_probs[keep_mask]
    filtered_boxes = pred_boxes[keep_mask]

    if max_detections > 0 and len(filtered_scores) > max_detections:
        top_scores, top_indices = torch.topk(filtered_scores, k=max_detections, largest=True)
        filtered_masks = filtered_masks[top_indices]
        filtered_scores = top_scores
        filtered_boxes = filtered_boxes[top_indices]

    return filtered_masks, filtered_scores, filtered_boxes


def resolve_eval_split(data_config: dict, split: str) -> str:
    split_name = split.strip().lower()
    if split_name in {"val", "valid", "validation"}:
        return str(data_config.get("val_split", "val"))
    if split_name == "test":
        return str(data_config.get("test_split", "test"))
    raise ValueError(f"Unsupported split: {split!r}. Expected one of: val or test.")


def build_filtered_coco_gt(dataset: CammaSam3Dataset):
    source_coco = dataset.dataset.context.coco
    selected_image_ids = set(dataset.image_ids)
    selected_category_ids = set(dataset.selected_category_ids)
    image_sizes = build_image_size_lookup(source_coco["images"])

    images = []
    for image in source_coco["images"]:
        image_id = int(image["id"])
        if image_id not in selected_image_ids:
            continue
        image_entry = dict(image)
        image_entry.setdefault("is_instance_exhaustive", True)
        images.append(image_entry)

    annotations = []
    for ann in source_coco["annotations"]:
        image_id = int(ann["image_id"])
        if image_id not in selected_image_ids or int(ann["category_id"]) not in selected_category_ids:
            continue
        annotation = normalize_annotation_for_coco(
            ann,
            image_size=image_sizes[image_id],
            bbox_anchor=dataset.bbox_anchor,
        )
        annotation["id"] = len(annotations) + 1
        annotations.append(annotation)
    categories = [
        dict(category)
        for category in source_coco["categories"]
        if int(category["id"]) in selected_category_ids
    ]
    return {
        "info": {"description": "Filtered CAMMA ground truth"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def build_query_level_cgf1_gt(dataset: CammaSam3Dataset):
    source_coco = dataset.dataset.context.coco
    selected_image_ids = set(dataset.image_ids)
    selected_category_ids = set(dataset.selected_category_ids)
    source_images = {
        int(image["id"]): image
        for image in source_coco["images"]
        if int(image["id"]) in selected_image_ids
    }
    image_sizes = {
        image_id: (int(image["width"]), int(image["height"]))
        for image_id, image in source_images.items()
    }

    query_image_id_map = {}
    query_images = []
    next_query_image_id = 1
    for image_id in dataset.image_ids:
        source_image = source_images[int(image_id)]
        for category_id in dataset.selected_category_ids:
            query_image_id = next_query_image_id
            next_query_image_id += 1
            query_image_id_map[(int(image_id), int(category_id))] = query_image_id
            query_images.append(
                {
                    "id": query_image_id,
                    "width": int(source_image["width"]),
                    "height": int(source_image["height"]),
                    "file_name": str(source_image.get("file_name", f"{image_id}.jpg")),
                    "original_image_id": int(image_id),
                    "query_category_id": int(category_id),
                    "is_instance_exhaustive": True,
                }
            )

    annotations = []
    for ann in source_coco["annotations"]:
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        if image_id not in selected_image_ids or category_id not in selected_category_ids:
            continue
        query_image_id = query_image_id_map[(image_id, category_id)]
        annotation = normalize_annotation_for_coco(
            ann,
            image_size=image_sizes[image_id],
            bbox_anchor=dataset.bbox_anchor,
        )
        annotation["id"] = len(annotations) + 1
        annotation["image_id"] = query_image_id
        annotations.append(annotation)

    categories = [
        dict(category)
        for category in source_coco["categories"]
        if int(category["id"]) in selected_category_ids
    ]
    return (
        {
            "info": {"description": "Query-level CAMMA ground truth for cgF1"},
            "images": query_images,
            "annotations": annotations,
            "categories": categories,
        },
        query_image_id_map,
    )


def remap_coco_image_ids_for_eval(
    coco_gt_dict: dict,
    coco_predictions: list[dict],
    eval_image_ids: list[int] | None = None,
) -> tuple[dict, list[dict], list[int]]:
    image_id_map = {
        int(image["id"]): index
        for index, image in enumerate(coco_gt_dict.get("images", []), start=1)
    }

    remapped_gt_images = []
    for image in coco_gt_dict.get("images", []):
        original_image_id = int(image["id"])
        remapped_image = dict(image)
        remapped_image["original_image_id"] = original_image_id
        remapped_image["id"] = image_id_map[original_image_id]
        remapped_gt_images.append(remapped_image)

    remapped_gt_annotations = []
    for index, annotation in enumerate(coco_gt_dict.get("annotations", []), start=1):
        original_image_id = int(annotation["image_id"])
        remapped_annotation = dict(annotation)
        remapped_annotation["id"] = index
        remapped_annotation["image_id"] = image_id_map[original_image_id]
        remapped_gt_annotations.append(remapped_annotation)

    remapped_predictions = []
    for prediction in coco_predictions:
        original_image_id = int(prediction["image_id"])
        mapped_image_id = image_id_map.get(original_image_id)
        if mapped_image_id is None:
            continue
        remapped_prediction = dict(prediction)
        remapped_prediction["original_image_id"] = original_image_id
        remapped_prediction["image_id"] = mapped_image_id
        remapped_predictions.append(remapped_prediction)

    remapped_eval_image_ids = (
        list(image_id_map.values())
        if eval_image_ids is None
        else [image_id_map[int(image_id)] for image_id in eval_image_ids if int(image_id) in image_id_map]
    )

    remapped_gt_dict = dict(coco_gt_dict)
    remapped_gt_dict["images"] = remapped_gt_images
    remapped_gt_dict["annotations"] = remapped_gt_annotations

    return remapped_gt_dict, remapped_predictions, remapped_eval_image_ids


def convert_predictions_to_coco_format(
    prediction_records,
    resolution=1008,
    prob_threshold=0.3,
    nms_iou_threshold=0.7,
    max_detections=100,
    query_image_id_map=None,
    iou_type="segm",
    include_segmentations=False,
):
    coco_predictions = []
    pred_counts = Counter()
    normalized_iou_type = normalize_iou_type(iou_type)

    for record in tqdm(prediction_records, desc="Converting predictions"):
        filtered_masks, filtered_scores, filtered_boxes = apply_sam3_nms(
            pred_logits=record["pred_logits"],
            pred_masks=record["pred_masks"],
            pred_boxes=record["pred_boxes"],
            presence_logit=record.get("presence_logit"),
            prob_threshold=prob_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            iou_type=normalized_iou_type,
        )

        orig_h, orig_w = record["original_size"]
        for mask_tensor, score, box in zip(filtered_masks, filtered_scores, filtered_boxes):
            bbox = normalized_cxcywh_to_original_xywh(box.tolist(), orig_w, orig_h)
            bbox = [float(v) for v in bbox]
            if bbox[2] <= 0.0 or bbox[3] <= 0.0:
                continue

            image_id = int(record["image_id"])
            category_id = int(record["category_id"])
            if query_image_id_map is not None:
                image_id = int(query_image_id_map[(image_id, category_id)])

            coco_prediction = {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "score": float(score.item()),
            }
            if normalized_iou_type == "segm" or include_segmentations:
                original_mask = padded_mask_to_original(mask_tensor, orig_h, orig_w, resolution) > 0.5
                if original_mask.sum().item() == 0:
                    if normalized_iou_type == "segm":
                        continue
                else:
                    mask_np = original_mask.cpu().numpy().astype(np.uint8)
                    rle = mask_utils.encode(np.asfortranarray(mask_np))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    coco_prediction["segmentation"] = rle

            coco_predictions.append(coco_prediction)
            pred_counts[category_id] += 1

    return coco_predictions, pred_counts


def filter_coco_predictions(coco_predictions, image_ids, category_ids):
    image_id_set = {int(image_id) for image_id in image_ids}
    category_id_set = {int(category_id) for category_id in category_ids}
    return [
        pred
        for pred in coco_predictions
        if int(pred["image_id"]) in image_id_set and int(pred["category_id"]) in category_id_set
    ]


def remap_predictions_to_query_image_ids(coco_predictions, query_image_id_map):
    remapped_predictions = []
    for pred in coco_predictions:
        image_id = int(pred["image_id"])
        category_id = int(pred["category_id"])
        query_image_id = query_image_id_map.get((image_id, category_id))
        if query_image_id is None:
            continue
        remapped_prediction = dict(pred)
        remapped_prediction["image_id"] = int(query_image_id)
        remapped_predictions.append(remapped_prediction)
    return remapped_predictions


def evaluate(
    config_path,
    weights_path=None,
    split="val",
    use_base_model=False,
    num_samples=None,
    prob_threshold=None,
    nms_iou=None,
    force_infer=False,
    iou_type=None,
    save_overlays=None,
    show_prediction_masks=None,
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", {})
    data_config = config.get("data", {})
    lora_config = config.get("lora", {})
    training_config = config.get("training", {})
    evaluation_config = config.get("evaluation", {})
    eval_iou_type = resolve_iou_type(evaluation_config, iou_type)
    should_save_overlays = resolve_save_overlays(evaluation_config, save_overlays)
    should_show_prediction_masks = resolve_show_prediction_masks(
        evaluation_config,
        show_prediction_masks,
    )
    hardware_config = config.get("hardware", {})
    dataset_name = data_config.get("dataset_name", "Endoscapes-Seg201-CBD")
    split_name = resolve_eval_split(data_config, split)
    artifact_paths = build_camma_artifact_paths(
        config=config,
        split_name=split_name,
        dataset_name=dataset_name,
    )
    eval_paths = build_eval_paths(artifact_paths, eval_iou_type)

    dataset = CammaSam3Dataset(
        dataset_root=data_config["dataset_root"],
        dataset_name=dataset_name,
        split=split_name,
        annotation_file=data_config.get("annotation_file", "annotation_coco.json"),
        selected_class_names=data_config.get("class_names"),
        include_negatives=bool(data_config.get("include_negatives", True)),
        augment=False,
        bbox_anchor=data_config.get("bbox_anchor"),
    )
    if num_samples is not None:
        dataset.indices = dataset.indices[:num_samples]
        dataset.image_ids = dataset.image_ids[:num_samples]

    unique_image_ids = list(dataset.image_ids)
    gt_dict = build_filtered_coco_gt(dataset)
    cgf1_gt_dict, query_image_id_map = build_query_level_cgf1_gt(dataset)
    gt_counts = Counter(int(ann["category_id"]) for ann in gt_dict["annotations"])
    write_json(artifact_paths.gt_coco_path, gt_dict)
    write_json(artifact_paths.cgf1_gt_coco_path, cgf1_gt_dict)

    is_full_split_eval = num_samples is None
    existing_prediction_path = None
    if not force_infer:
        for candidate_path in prediction_path_candidates_for_iou(eval_paths):
            if candidate_path.exists():
                existing_prediction_path = candidate_path
                break

    if existing_prediction_path is not None:
        print(f"Reusing {eval_iou_type} COCO predictions from {existing_prediction_path}")
        if use_base_model or weights_path is not None or nms_iou is not None:
            print(
                "[INFO] Stored predictions are being reused, so model/weights/NMS options will not regenerate detections. "
                "Pass --force-infer to recompute them."
            )
        coco_predictions = filter_coco_predictions(
            read_json(existing_prediction_path),
            image_ids=dataset.image_ids,
            category_ids=dataset.selected_category_ids,
        )
        pred_counts = Counter(int(pred["category_id"]) for pred in coco_predictions)
        if should_save_overlays and should_show_prediction_masks and not any(
            pred.get("segmentation") is not None for pred in coco_predictions
        ):
            print(
                "[WARN] Stored predictions do not include segmentation masks, so prediction-mask overlays cannot be drawn. "
                "Rerun with --force-infer and --show-prediction-masks to regenerate them."
            )
        if is_full_split_eval and existing_prediction_path != eval_paths["predictions_coco_path"]:
            write_json(eval_paths["predictions_coco_path"], coco_predictions)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and hardware_config.get("device", "cuda") == "cuda" else "cpu"
        )
        print(f"Using device: {device}")

        model = build_sam3_image_model(
            checkpoint_path=model_config.get("checkpoint_path"),
            device=device.type,
            eval_mode=False,
            compile=bool(hardware_config.get("use_compile", False)),
            bpe_path=model_config.get("bpe_path"),
            load_from_HF=bool(model_config.get("load_from_hf", True)),
        )

        if use_base_model:
            print("Using base SAM3 model without LoRA")
            batch_size = int(evaluation_config.get("batch_size", 1))
        else:
            lora_cfg = build_lora_config(lora_config)
            model = apply_lora_to_model(model, lora_cfg)
            if weights_path is None:
                raise ValueError(
                    "No reusable COCO predictions were found. Pass --weights or rerun after infer_camma.py has written them."
                )
            load_lora_weights(model, weights_path, expected_config=lora_cfg)
            batch_size = int(evaluation_config.get("batch_size", training_config.get("batch_size", 1)))

        stats = count_parameters(model)
        print(
            f"Trainable params: {stats['trainable_parameters']:,} "
            f"({stats['trainable_percentage']:.2f}%)"
        )

        def collate_fn(batch):
            return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=int(training_config.get("num_workers", 0)),
            pin_memory=bool(hardware_config.get("dataloader_pin_memory", True)),
        )

        model.to(device)
        model.eval()
        prediction_records = []

        use_amp = device.type == "cuda"
        amp_dtype = None
        precision = str(training_config.get("mixed_precision", "none")).lower()
        if precision == "bf16":
            amp_dtype = torch.bfloat16
        elif precision == "fp16":
            amp_dtype = torch.float16
        else:
            use_amp = False

        with torch.no_grad():
            for batch_dict in tqdm(loader, desc="Validation"):
                if batch_dict is None:
                    continue
                input_batch = move_to_device(batch_dict["input"], device)
                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        outputs_list = model(input_batch)
                else:
                    outputs_list = model(input_batch)

                with SAM3Output.iteration_mode(
                    outputs_list,
                    iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
                ) as outputs_iter:
                    final_stage = list(outputs_iter)[-1]
                    final_outputs = final_stage[-1]
                    metadata = input_batch.find_metadatas[0]
                    query_count = len(metadata.original_image_id)
                    for idx in range(query_count):
                        image_id = int(metadata.original_image_id[idx])
                        prediction_records.append(
                            {
                                "image_id": image_id,
                                "category_id": int(metadata.original_category_id[idx]),
                                "category_name": dataset.categories[int(metadata.original_category_id[idx])],
                                "original_size": tuple(int(v) for v in metadata.original_size[idx]),
                                "pred_logits": final_outputs["pred_logits"][idx].detach().cpu(),
                                "pred_boxes": final_outputs["pred_boxes"][idx].detach().cpu(),
                                "pred_masks": final_outputs["pred_masks"][idx].detach().cpu(),
                                "presence_logit": final_outputs["presence_logit_dec"][idx].detach().cpu()
                                if "presence_logit_dec" in final_outputs
                                else None,
                            }
                        )

        prob_threshold = float(
            prob_threshold if prob_threshold is not None else evaluation_config.get("prob_threshold", 0.3)
        )
        nms_iou_threshold = float(nms_iou if nms_iou is not None else evaluation_config.get("nms_iou", 0.7))
        max_detections = int(evaluation_config.get("max_detections", 100))
        coco_predictions, pred_counts = convert_predictions_to_coco_format(
            prediction_records,
            resolution=dataset.resolution,
            prob_threshold=prob_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            iou_type=eval_iou_type,
            include_segmentations=should_show_prediction_masks,
        )
        if is_full_split_eval:
            write_json(eval_paths["predictions_coco_path"], coco_predictions)
            write_json(eval_paths["legacy_predictions_coco_path"], coco_predictions)

    cgf1_predictions = remap_predictions_to_query_image_ids(coco_predictions, query_image_id_map)
    write_json(eval_paths["cgf1_predictions_coco_path"], cgf1_predictions)
    if should_save_overlays:
        save_eval_overlays(
            dataset=dataset,
            image_ids=dataset.image_ids,
            category_ids=dataset.selected_category_ids,
            coco_predictions=coco_predictions,
            overlay_dir=build_overlay_dir(artifact_paths.eval_root, eval_iou_type),
            include_gt_masks=eval_iou_type == "segm",
            include_prediction_masks=should_show_prediction_masks,
        )

    overall_metrics = {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0}
    cgf1 = 0.0
    cgf1_50 = 0.0
    cgf1_75 = 0.0
    per_class_metrics = []

    prob_threshold = float(prob_threshold if prob_threshold is not None else evaluation_config.get("prob_threshold", 0.3))

    if coco_predictions:
        coco_gt_eval_dict, coco_predictions_for_eval, coco_eval_image_ids = remap_coco_image_ids_for_eval(
            gt_dict,
            coco_predictions,
            eval_image_ids=unique_image_ids,
        )
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_gt = COCOCustom()
                coco_gt.dataset = coco_gt_eval_dict
                coco_gt.createIndex()
                coco_dt = coco_gt.loadRes(coco_predictions_for_eval)
                coco_eval = COCOeval(coco_gt, coco_dt, eval_iou_type)
                coco_eval.params.useCats = True
                coco_eval.params.catIds = dataset.selected_category_ids
                coco_eval.params.imgIds = coco_eval_image_ids
                coco_eval.evaluate()
                coco_eval.accumulate()

        coco_eval.summarize()
        overall_metrics = {
            "mAP": float(coco_eval.stats[0]),
            "mAP50": float(coco_eval.stats[1]),
            "mAP75": float(coco_eval.stats[2]),
        }

        for category_id in dataset.selected_category_ids:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    per_class_eval = COCOeval(coco_gt, coco_dt, eval_iou_type)
                    per_class_eval.params.useCats = True
                    per_class_eval.params.catIds = [category_id]
                    per_class_eval.params.imgIds = coco_eval_image_ids
                    per_class_eval.evaluate()
                    per_class_eval.accumulate()
                    per_class_eval.summarize()
            per_class_metrics.append(
                {
                    "category_id": category_id,
                    "category_name": dataset.categories[category_id],
                    "mAP": float(per_class_eval.stats[0]),
                    "mAP50": float(per_class_eval.stats[1]),
                    "mAP75": float(per_class_eval.stats[2]),
                    "gt_count": gt_counts[category_id],
                    "pred_count": pred_counts[category_id],
                }
            )
    else:
        print("[WARN] No predictions available for evaluation.")
        for category_id in dataset.selected_category_ids:
            per_class_metrics.append(
                {
                    "category_id": category_id,
                    "category_name": dataset.categories[category_id],
                    "mAP": 0.0,
                    "mAP50": 0.0,
                    "mAP75": 0.0,
                    "gt_count": gt_counts[category_id],
                    "pred_count": 0,
                }
            )

    cgf1_evaluator = CGF1Evaluator(
        gt_path=str(artifact_paths.cgf1_gt_coco_path),
        iou_type=eval_iou_type,
        threshold=prob_threshold,
        verbose=True,
    )
    cgf1_results = cgf1_evaluator.evaluate(str(eval_paths["cgf1_predictions_coco_path"]))
    cgf1_key_prefix = f"cgF1_eval_{eval_iou_type}"
    cgf1 = cgf1_results.get(f"{cgf1_key_prefix}_cgF1", 0.0)
    cgf1_50 = cgf1_results.get(f"{cgf1_key_prefix}_cgF1@0.5", 0.0)
    cgf1_75 = cgf1_results.get(f"{cgf1_key_prefix}_cgF1@0.75", 0.0)

    metric_label = "Segmentation" if eval_iou_type == "segm" else "BBox"
    print(f"\nPer-class COCO {metric_label} Metrics")
    for metrics in per_class_metrics:
        print(
            f"{metrics['category_name']}: "
            f"GT={metrics['gt_count']}, Pred={metrics['pred_count']}, "
            f"mAP={metrics['mAP']:.4f}, mAP@50={metrics['mAP50']:.4f}, mAP@75={metrics['mAP75']:.4f}"
        )

    print("\nFINAL RESULTS")
    print(f"Selected classes: {dataset.selected_class_names}")
    print(f"Evaluation mode: {eval_iou_type}")
    print(f"mAP (IoU 0.50:0.95): {overall_metrics['mAP']:.4f}")
    print(f"mAP@50: {overall_metrics['mAP50']:.4f}")
    print(f"mAP@75: {overall_metrics['mAP75']:.4f}")
    print(f"cgF1 (IoU 0.50:0.95): {cgf1:.4f}")
    print(f"cgF1@50: {cgf1_50:.4f}")
    print(f"cgF1@75: {cgf1_75:.4f}")

    if evaluation_config.get("save_predictions", True):
        results_payload = {
            "iou_type": eval_iou_type,
            "overall": overall_metrics,
            "cgf1": {"cgF1": cgf1, "cgF1@50": cgf1_50, "cgF1@75": cgf1_75},
            "per_class": per_class_metrics,
        }
        write_json(eval_paths["results_path"], results_payload)
        write_json(eval_paths["legacy_results_path"], results_payload)


def main():
    parser = argparse.ArgumentParser(description="Minimal SAM3 LoRA evaluation for CAMMA datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--weights", type=str, default=None, help="Path to LoRA weights")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--use-base-model", action="store_true", help="Evaluate the unfinetuned SAM3 base model")
    parser.add_argument("--num-samples", type=int, default=None, help="Limit evaluation to N images for debugging")
    parser.add_argument("--prob-threshold", type=float, default=None, help="Override probability threshold")
    parser.add_argument("--nms-iou", type=float, default=None, help="Override NMS IoU threshold")
    parser.add_argument(
        "--iou-type",
        "--mode",
        dest="iou_type",
        type=str,
        default=None,
        help="Evaluation mode: segm/masks or bbox/boxes. Defaults to evaluation.iou_type or segm.",
    )
    parser.add_argument(
        "--save-overlays",
        dest="save_overlays",
        action="store_true",
        help="Save GT and prediction overlays alongside the evaluation artifacts.",
    )
    parser.add_argument(
        "--no-save-overlays",
        dest="save_overlays",
        action="store_false",
        help="Do not save evaluation overlays, even if the config enables them.",
    )
    parser.add_argument(
        "--show-prediction-masks",
        "--show-predictions-masks",
        dest="show_prediction_masks",
        action="store_true",
        help="Render predicted masks in saved prediction overlays. Boxes are always shown.",
    )
    parser.add_argument(
        "--hide-prediction-masks",
        "--hide-predictions-masks",
        dest="show_prediction_masks",
        action="store_false",
        help="Hide predicted masks in saved prediction overlays while keeping boxes visible.",
    )
    parser.add_argument(
        "--force-infer",
        action="store_true",
        help="Run the model even if reusable COCO predictions already exist in the standard output directory.",
    )
    parser.set_defaults(save_overlays=None, show_prediction_masks=None)
    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        weights_path=args.weights,
        split=args.split,
        use_base_model=args.use_base_model,
        num_samples=args.num_samples,
        prob_threshold=args.prob_threshold,
        nms_iou=args.nms_iou,
        force_infer=args.force_infer,
        iou_type=args.iou_type,
        save_overlays=args.save_overlays,
        show_prediction_masks=args.show_prediction_masks,
    )


if __name__ == "__main__":
    main()
