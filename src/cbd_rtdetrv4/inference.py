from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def build_json_payload(prediction: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "mode": prediction.get("mode", "unknown"),
        "file_name": prediction.get("file_name"),
        "image_path": prediction.get("image_path"),
        "original_size": prediction.get("original_size"),
        "pred_bbox_xywh": prediction.get("pred_bbox_xywh"),
        "pred_score": prediction.get("pred_score"),
        "detections": prediction.get("detections", []),
    }
    if prediction.get("image_id") is not None:
        payload["image_id"] = int(prediction["image_id"])
    if prediction.get("annotation_id") is not None:
        payload["annotation_id"] = int(prediction["annotation_id"])
    if prediction.get("target_bbox_xywh") is not None:
        payload["target_bbox_xywh"] = prediction["target_bbox_xywh"]
    if prediction.get("target_type_name") is not None:
        payload["target_type_name"] = prediction["target_type_name"]
    if prediction.get("iou") is not None:
        payload["iou"] = float(prediction["iou"])
    return payload


def _xywh_to_xyxy(box_xywh: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(value) for value in box_xywh]
    return x, y, x + w, y + h


def render_overlay(
    prediction: dict[str, Any],
    *,
    show_gt: bool = False,
) -> Image.Image:
    image = Image.open(prediction["image_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font(max(12, int(round(min(image.size) * 0.025))))
    pred_box = prediction.get("pred_bbox_xywh")
    gt_box = prediction.get("target_bbox_xywh")

    if pred_box is not None:
        draw.rectangle(_xywh_to_xyxy(pred_box), outline=(60, 220, 120), width=4)
        score = prediction.get("pred_score")
        label = "CBD"
        if score is not None:
            label = f"{label} {float(score):.2f}"
        draw.text((12, 12), label, fill=(60, 220, 120), font=font)
    else:
        draw.text((12, 12), "CBD no detections", fill=(255, 90, 90), font=font)

    if show_gt and gt_box is not None:
        draw.rectangle(_xywh_to_xyxy(gt_box), outline=(255, 170, 40), width=3)
        target_type_name = prediction.get("target_type_name")
        if target_type_name:
            draw.text((12, 40), f"gt {target_type_name}", fill=(255, 170, 40), font=font)
    return image


def write_prediction_artifacts(
    prediction: dict[str, Any],
    output_path: str | Path,
    *,
    show_gt: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = render_overlay(prediction, show_gt=show_gt)
    overlay.save(output_path)

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as handle:
        json.dump(build_json_payload(prediction), handle, indent=2)

    return {
        "mode": prediction.get("mode", "unknown"),
        "output_path": str(output_path),
        "json_path": str(json_path),
    }
