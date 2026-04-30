from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .common import CBD_CATEGORY_ID, CBD_CLASS_NAME, DEFAULT_TARGET_TYPE_ORDER


IOU_THRESHOLDS = [round(0.50 + 0.05 * index, 2) for index in range(10)]


def xyxy_to_xywh(box_xyxy: list[float]) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in box_xyxy]
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def xywh_to_xyxy(box_xywh: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(value) for value in box_xywh]
    return x, y, x + w, y + h


def bbox_iou_xywh(box_a: list[float] | None, box_b: list[float] | None) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(box_a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def compute_top_box_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "mean_iou": 0.0,
            "precision_50": 0.0,
            "recall_50": 0.0,
            "mean_precision_50_95": 0.0,
            "mean_recall_50_95": 0.0,
        }

    ious = [float(record.get("iou") or 0.0) for record in records]
    num_predictions = sum(record.get("pred_bbox_xywh") is not None for record in records)
    true_positives_by_threshold = [sum(iou >= threshold for iou in ious) for threshold in IOU_THRESHOLDS]
    precision_by_threshold = [
        true_positives / num_predictions if num_predictions else 0.0
        for true_positives in true_positives_by_threshold
    ]
    recall_by_threshold = [
        true_positives / len(records)
        for true_positives in true_positives_by_threshold
    ]
    return {
        "mean_iou": sum(ious) / len(ious),
        "precision_50": precision_by_threshold[0],
        "recall_50": recall_by_threshold[0],
        "mean_precision_50_95": sum(precision_by_threshold) / len(precision_by_threshold),
        "mean_recall_50_95": sum(recall_by_threshold) / len(recall_by_threshold),
    }


def compute_all_box_pr_at_iou(records: list[dict[str, Any]], threshold: float = 0.50) -> dict[str, float]:
    if not records:
        return {"all_box_precision_50": 0.0, "all_box_recall_50": 0.0}

    true_positive = 0
    false_positive = 0
    for record in records:
        target_box = record.get("target_bbox_xywh")
        matched_target = False
        detections = sorted(
            record.get("detections", []),
            key=lambda detection: float(detection.get("score", 0.0)),
            reverse=True,
        )
        for detection in detections:
            iou = bbox_iou_xywh(detection.get("bbox_xywh"), target_box)
            if iou >= threshold and not matched_target:
                true_positive += 1
                matched_target = True
            else:
                false_positive += 1

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / len(records)
    return {"all_box_precision_50": precision, "all_box_recall_50": recall}


def build_coco_ground_truth(records: list[dict[str, Any]]) -> COCO:
    coco = COCO()
    coco.dataset = {
        "images": [
            {
                "id": int(record["image_id"]),
                "width": int(record["original_size"][1]),
                "height": int(record["original_size"][0]),
                "file_name": record["file_name"],
            }
            for record in records
        ],
        "annotations": [
            {
                "id": index + 1,
                "image_id": int(record["image_id"]),
                "category_id": CBD_CATEGORY_ID,
                "bbox": [float(value) for value in record["target_bbox_xywh"]],
                "area": float(record["target_bbox_xywh"][2] * record["target_bbox_xywh"][3]),
                "iscrowd": 0,
            }
            for index, record in enumerate(records)
        ],
        "categories": [{"id": CBD_CATEGORY_ID, "name": CBD_CLASS_NAME}],
    }
    coco.createIndex()
    return coco


def build_coco_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for record in records:
        for detection in record["detections"]:
            predictions.append(
                {
                    "image_id": int(record["image_id"]),
                    "category_id": CBD_CATEGORY_ID,
                    "bbox": [float(value) for value in detection["bbox_xywh"]],
                    "score": float(detection["score"]),
                }
            )
    return predictions


def compute_coco_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {"ap50": 0.0, "map50_95": 0.0}
    predictions = build_coco_predictions(records)
    if not predictions:
        return {"ap50": 0.0, "map50_95": 0.0}

    coco_gt = build_coco_ground_truth(records)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {"map50_95": float(evaluator.stats[0]), "ap50": float(evaluator.stats[1])}


def compute_metric_subset(records: list[dict[str, Any]]) -> dict[str, float]:
    standard = compute_coco_metrics(records)
    cbd_style = compute_top_box_metrics(records)
    all_box_pr50 = compute_all_box_pr_at_iou(records, threshold=0.50)
    return {
        "count": float(len(records)),
        "ap50": standard["ap50"],
        "map50_95": standard["map50_95"],
        "precision_50": cbd_style["precision_50"],
        "mean_precision_50_95": cbd_style["mean_precision_50_95"],
        "top_box_precision_50": cbd_style["precision_50"],
        "top_box_mean_precision_50_95": cbd_style["mean_precision_50_95"],
        "mean_iou": cbd_style["mean_iou"],
        "recall_50": all_box_pr50["all_box_recall_50"],
        "all_box_precision_50": all_box_pr50["all_box_precision_50"],
        "all_box_recall_50": all_box_pr50["all_box_recall_50"],
        "top_box_recall_50": cbd_style["recall_50"],
        "mean_recall_50_95": cbd_style["mean_recall_50_95"],
    }


def build_metrics_payload(
    records: list[dict[str, Any]],
    *,
    split: str,
    loss: float | None = None,
) -> dict[str, Any]:
    overall = compute_metric_subset(records)
    by_target_type = {
        target_type: compute_metric_subset(
            [record for record in records if record["target_type_name"] == target_type]
        )
        for target_type in DEFAULT_TARGET_TYPE_ORDER
    }
    payload: dict[str, Any] = {
        "split": split,
        "num_samples": len(records),
        "overall": overall,
        "by_target_type": by_target_type,
    }
    if loss is not None:
        payload["loss"] = float(loss)
    return payload


def flatten_metrics_payload(payload: dict[str, Any]) -> dict[str, float]:
    flat = {
        "loss": float(payload.get("loss", 0.0)),
        "num_samples": float(payload["overall"]["count"]),
        "ap50": float(payload["overall"]["ap50"]),
        "map50_95": float(payload["overall"]["map50_95"]),
        "precision_50": float(payload["overall"]["precision_50"]),
        "mean_precision_50_95": float(payload["overall"]["mean_precision_50_95"]),
        "top_box_precision_50": float(payload["overall"]["top_box_precision_50"]),
        "top_box_mean_precision_50_95": float(payload["overall"]["top_box_mean_precision_50_95"]),
        "mean_iou": float(payload["overall"]["mean_iou"]),
        "recall_50": float(payload["overall"]["recall_50"]),
        "all_box_precision_50": float(payload["overall"]["all_box_precision_50"]),
        "all_box_recall_50": float(payload["overall"]["all_box_recall_50"]),
        "top_box_recall_50": float(payload["overall"]["top_box_recall_50"]),
        "mean_recall_50_95": float(payload["overall"]["mean_recall_50_95"]),
    }
    for target_type in DEFAULT_TARGET_TYPE_ORDER:
        subset = payload["by_target_type"][target_type]
        flat[f"{target_type}_count"] = float(subset["count"])
        flat[f"{target_type}_ap50"] = float(subset["ap50"])
        flat[f"{target_type}_map50_95"] = float(subset["map50_95"])
        flat[f"{target_type}_precision_50"] = float(subset["precision_50"])
        flat[f"{target_type}_mean_precision_50_95"] = float(subset["mean_precision_50_95"])
        flat[f"{target_type}_top_box_precision_50"] = float(subset["top_box_precision_50"])
        flat[f"{target_type}_top_box_mean_precision_50_95"] = float(
            subset["top_box_mean_precision_50_95"]
        )
        flat[f"{target_type}_mean_iou"] = float(subset["mean_iou"])
        flat[f"{target_type}_recall_50"] = float(subset["recall_50"])
        flat[f"{target_type}_all_box_precision_50"] = float(subset["all_box_precision_50"])
        flat[f"{target_type}_all_box_recall_50"] = float(subset["all_box_recall_50"])
        flat[f"{target_type}_top_box_recall_50"] = float(subset["top_box_recall_50"])
        flat[f"{target_type}_mean_recall_50_95"] = float(subset["mean_recall_50_95"])
    return flat


def write_metrics_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_predictions_jsonl(path: str | Path, records: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path
