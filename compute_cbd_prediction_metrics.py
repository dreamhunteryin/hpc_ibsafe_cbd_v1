#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


IOU_THRESHOLDS = [round(0.50 + index * 0.05, 2) for index in range(10)]
TARGET_TYPE_ORDER = ("soft", "hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute detection and type-classification metrics from cached CBD prediction JSON files."
    )
    parser.add_argument(
        "predictions_dir",
        type=Path,
        help="Directory containing *_overlay.json files produced by infer_cbd.py or cached inference.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to save the computed metrics as JSON.",
    )
    return parser.parse_args()


def discover_prediction_files(predictions_dir: Path) -> list[Path]:
    files = sorted(predictions_dir.glob("*_overlay.json"))
    if not files:
        raise FileNotFoundError(f"No *_overlay.json files found in {predictions_dir}")
    return files


def load_records(predictions_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in discover_prediction_files(predictions_dir):
        payload = json.loads(path.read_text())
        payload["_path"] = str(path)
        payload["iou"] = compute_record_iou(payload)
        payload["pred_type_name"] = normalize_label(payload.get("pred_type_name"))
        payload["target_type"] = normalize_label(payload.get("target_type"))
        records.append(payload)
    return records


def normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


def compute_record_iou(record: dict[str, Any]) -> float:
    value = record.get("iou")
    if value is not None:
        return float(value)

    pred_box = record.get("pred_bbox_xywh")
    target_box = record.get("target_bbox_xywh")
    if pred_box is None or target_box is None:
        raise ValueError(f"Missing IoU and bbox data for {record.get('_path', '<unknown file>')}")
    return bbox_iou_xywh(pred_box, target_box)


def bbox_iou_xywh(box_a: list[float], box_b: list[float]) -> float:
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


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(value) for value in box]
    return x, y, x + w, y + h


def compute_average_precision(scores: list[float], threshold: float) -> float:
    if not scores:
        return 0.0

    ranked = sorted(((score, score >= threshold) for score in scores), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    recalls: list[float] = []
    precisions: list[float] = []
    gt_count = len(ranked)

    for _, is_match in ranked:
        if is_match:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / gt_count)
        precisions.append(tp / (tp + fp))

    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])

    ap = 0.0
    for recall_step in range(101):
        recall_threshold = recall_step / 100.0
        precision_value = 0.0
        for recall, precision in zip(recalls, precisions):
            if recall >= recall_threshold:
                precision_value = precision
                break
        ap += precision_value
    return ap / 101.0


def compute_detection_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        return {
            "num_samples": 0,
            "mean_iou": 0.0,
            "mAP": 0.0,
            "mAP50": 0.0,
            "recall": 0.0,
            "mean_recall_50_95": 0.0,
        }

    scores = [float(record["iou"]) for record in records]
    ap_by_threshold = {threshold: compute_average_precision(scores, threshold) for threshold in IOU_THRESHOLDS}
    recall_by_threshold = {
        threshold: sum(score >= threshold for score in scores) / len(scores) for threshold in IOU_THRESHOLDS
    }
    return {
        "num_samples": len(scores),
        "mean_iou": sum(scores) / len(scores),
        "mAP": sum(ap_by_threshold.values()) / len(ap_by_threshold),
        "mAP50": ap_by_threshold[0.50],
        "recall": recall_by_threshold[0.50],
        "mean_recall_50_95": sum(recall_by_threshold.values()) / len(recall_by_threshold),
    }


def average_metric_rows(rows: list[dict[str, float | int]]) -> dict[str, float]:
    if not rows:
        return {
            "mean_iou": 0.0,
            "mAP": 0.0,
            "mAP50": 0.0,
            "recall": 0.0,
            "mean_recall_50_95": 0.0,
        }
    keys = ("mean_iou", "mAP", "mAP50", "recall", "mean_recall_50_95")
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def compute_binary_classification_metrics(
    records: list[dict[str, Any]],
    labels: tuple[str, ...] = TARGET_TYPE_ORDER,
) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "num_samples": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion_matrix": {},
        }

    correct = sum(record["pred_type_name"] == record["target_type"] for record in records)
    per_class: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []

    for label in labels:
        true_positive = sum(
            record["pred_type_name"] == label and record["target_type"] == label for record in records
        )
        false_positive = sum(
            record["pred_type_name"] == label and record["target_type"] != label for record in records
        )
        false_negative = sum(
            record["pred_type_name"] != label and record["target_type"] == label for record in records
        )
        support = sum(record["target_type"] == label for record in records)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_score = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    confusion_matrix: dict[str, dict[str, int]] = {}
    for target_label in labels:
        confusion_matrix[target_label] = {}
        for predicted_label in labels:
            confusion_matrix[target_label][predicted_label] = sum(
                record["target_type"] == target_label and record["pred_type_name"] == predicted_label
                for record in records
            )

    return {
        "num_samples": total,
        "accuracy": correct / total,
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


def build_results_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    detection_by_target: dict[str, dict[str, float | int]] = {}
    detection_rows: list[dict[str, float | int]] = []

    for target_type in TARGET_TYPE_ORDER:
        subset = [record for record in records if record["target_type"] == target_type]
        metrics = compute_detection_metrics(subset)
        detection_by_target[target_type] = metrics
        detection_rows.append(metrics)

    return {
        "num_samples": len(records),
        "counts_by_target_type": {
            target_type: sum(record["target_type"] == target_type for record in records) for target_type in TARGET_TYPE_ORDER
        },
        "detection": {
            "by_target_type": detection_by_target,
            "average_of_soft_and_hard": average_metric_rows(detection_rows),
            "overall": compute_detection_metrics(records),
            "score_definition": "prediction score = IoU(record.pred_bbox_xywh, record.target_bbox_xywh)",
            "recall_definition": "share of samples with IoU >= 0.50",
        },
        "classification": compute_binary_classification_metrics(records),
    }


def print_detection_table(results: dict[str, Any]) -> None:
    print("Detection metrics (score = IoU, recall = IoU >= 0.50)")
    print(
        f"{'subset':<10} {'n':>5} {'mean_iou':>10} {'mAP':>10} {'mAP50':>10} {'recall':>10} {'mr_50_95':>10}"
    )

    for subset_name in TARGET_TYPE_ORDER:
        metrics = results["detection"]["by_target_type"][subset_name]
        print(
            f"{subset_name:<10} {int(metrics['num_samples']):>5} "
            f"{float(metrics['mean_iou']):>10.4f} {float(metrics['mAP']):>10.4f} "
            f"{float(metrics['mAP50']):>10.4f} {float(metrics['recall']):>10.4f} "
            f"{float(metrics['mean_recall_50_95']):>10.4f}"
        )

    averaged = results["detection"]["average_of_soft_and_hard"]
    print(
        f"{'average':<10} {'-':>5} "
        f"{float(averaged['mean_iou']):>10.4f} {float(averaged['mAP']):>10.4f} "
        f"{float(averaged['mAP50']):>10.4f} {float(averaged['recall']):>10.4f} "
        f"{float(averaged['mean_recall_50_95']):>10.4f}"
    )


def print_classification_summary(results: dict[str, Any]) -> None:
    classification = results["classification"]
    print("\nType classification metrics (pred_type_name vs target_type)")
    print(f"accuracy        {float(classification['accuracy']):.4f}")
    print(f"macro_precision {float(classification['macro_precision']):.4f}")
    print(f"macro_recall    {float(classification['macro_recall']):.4f}")
    print(f"macro_f1        {float(classification['macro_f1']):.4f}")

    print("\nPer-class metrics")
    print(f"{'label':<10} {'support':>8} {'precision':>10} {'recall':>10} {'f1':>10}")
    for label in TARGET_TYPE_ORDER:
        metrics = classification["per_class"][label]
        print(
            f"{label:<10} {int(metrics['support']):>8} "
            f"{float(metrics['precision']):>10.4f} {float(metrics['recall']):>10.4f} {float(metrics['f1']):>10.4f}"
        )

    print("\nConfusion matrix (rows = target, cols = prediction)")
    header = " " * 12 + " ".join(f"{label:>10}" for label in TARGET_TYPE_ORDER)
    print(header)
    for target_label in TARGET_TYPE_ORDER:
        counts = classification["confusion_matrix"][target_label]
        row = f"{target_label:<12}" + " ".join(f"{int(counts[predicted_label]):>10}" for predicted_label in TARGET_TYPE_ORDER)
        print(row)


def main() -> None:
    args = parse_args()
    records = load_records(args.predictions_dir)
    results = build_results_payload(records)

    print(f"Loaded {len(records)} prediction JSON files from {args.predictions_dir}")
    print_detection_table(results)
    print_classification_summary(results)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved JSON metrics to {args.json_output}")


if __name__ == "__main__":
    main()
