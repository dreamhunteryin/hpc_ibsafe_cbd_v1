#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
INFER = ROOT / "infer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(INFER) not in sys.path:
    sys.path.insert(0, str(INFER))

from cbd.engine import CBDTrainer, load_config
from infer_lora import COLOR_CYCLE, render_overlay
from sam3.model.box_ops import box_cxcywh_to_xyxy, fast_diag_box_iou


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render stage-2 CBD predictions from cached clips.")
    parser.add_argument("--config", required=True, help="Path to the CBD YAML config.")
    parser.add_argument("--clip-id", help="Clip id in the form videoId_frameId. If omitted, run on the whole split.")
    parser.add_argument("--split", default="test", help="Dataset split containing the clip.")
    parser.add_argument("--weights", help="Optional checkpoint path. Defaults to output_dir/best_cbd.pt.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG path for a single clip, or output directory/base path for whole-split inference.",
    )
    parser.add_argument("--show-gt", action="store_true", help="Overlay the ground-truth box as well.")
    return parser


def add_heatmap_overlay(
    image: PILImage.Image,
    heatmap: np.ndarray | None,
    *,
    color: tuple[int, int, int] = (255, 170, 0),
    max_alpha: int = 144,
) -> PILImage.Image:
    if heatmap is None:
        return image
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.size == 0:
        return image
    heatmap = heatmap - heatmap.min()
    if float(heatmap.max()) > 0.0:
        heatmap = heatmap / float(heatmap.max())
    alpha = np.clip(heatmap * float(max_alpha), 0.0, float(max_alpha)).astype(np.uint8)
    alpha_image = PILImage.fromarray(alpha, mode="L").resize(image.size, resample=PILImage.BILINEAR)
    overlay = PILImage.new("RGBA", image.size, color + (0,))
    overlay.putalpha(alpha_image)
    return PILImage.alpha_composite(image.convert("RGBA"), overlay)


def normalize_heatmap(heatmap: np.ndarray | None) -> np.ndarray | None:
    if heatmap is None:
        return None
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.size == 0:
        return None
    heatmap = heatmap - heatmap.min()
    max_value = float(heatmap.max())
    if max_value > 0.0:
        heatmap = heatmap / max_value
    return heatmap


def colorize_heatmap(heatmap: np.ndarray | None) -> PILImage.Image | None:
    heatmap = normalize_heatmap(heatmap)
    if heatmap is None:
        return None
    anchors = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    colors = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [44.0, 0.0, 255.0],
            [0.0, 220.0, 255.0],
            [255.0, 230.0, 0.0],
            [255.0, 32.0, 0.0],
        ],
        dtype=np.float32,
    )
    flat = heatmap.reshape(-1)
    rgb = np.empty((flat.size, 3), dtype=np.uint8)
    for channel in range(3):
        rgb[:, channel] = np.interp(flat, anchors, colors[:, channel]).astype(np.uint8)
    return PILImage.fromarray(rgb.reshape(heatmap.shape + (3,)), mode="RGB")


def heatmap_tensor_to_numpy(heatmap) -> np.ndarray | None:
    if heatmap is None:
        return None
    if hasattr(heatmap, "detach"):
        heatmap = heatmap.detach()
    if hasattr(heatmap, "cpu"):
        heatmap = heatmap.cpu()
    if hasattr(heatmap, "float"):
        heatmap = heatmap.float()
    if hasattr(heatmap, "numpy"):
        return heatmap.numpy()
    return np.asarray(heatmap, dtype=np.float32)


def add_title(image: PILImage.Image, text: str) -> PILImage.Image:
    composed = image.convert("RGBA")
    draw = ImageDraw.Draw(composed)
    min_dim = min(composed.size)
    font_size = max(14, int(round(min_dim * 0.02)))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    margin = max(8, int(round(min_dim * 0.02)))
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width = right - left
    text_x = max(margin, int(round((composed.width - text_width) / 2.0 - left)))
    text_y = margin
    box_padding_x = max(6, int(round(font_size * 0.35)))
    box_padding_y = max(4, int(round(font_size * 0.25)))
    draw.rectangle(
        (
            text_x + left - box_padding_x,
            text_y + top - box_padding_y,
            text_x + right + box_padding_x,
            text_y + bottom + box_padding_y,
        ),
        fill=(0, 0, 0, 180),
    )
    draw.text((text_x, text_y), text, fill=(255, 255, 255, 255), font=font)
    return composed


def add_corner_label(image: PILImage.Image, text: str) -> PILImage.Image:
    composed = image.convert("RGBA")
    draw = ImageDraw.Draw(composed)
    min_dim = min(composed.size)
    font_size = max(12, int(round(min_dim * 0.017)))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    margin = max(8, int(round(min_dim * 0.02)))
    left, top, right, bottom = draw.textbbox((margin, margin), text, font=font)
    box_padding_x = max(5, int(round(font_size * 0.35)))
    box_padding_y = max(3, int(round(font_size * 0.2)))
    draw.rectangle(
        (left - box_padding_x, top - box_padding_y, right + box_padding_x, bottom + box_padding_y),
        fill=(0, 0, 0, 180),
    )
    draw.text((margin, margin), text, fill=(255, 255, 255, 255), font=font)
    return composed


def resolve_weights_path(config: dict, weights_path: str | None) -> Path:
    if weights_path is not None:
        return Path(weights_path)
    return Path(config.get("output", {}).get("output_dir", "outputs/bsafe_cbd")) / "best_cbd.pt"


def compute_prediction_iou(prediction: dict) -> float | None:
    pred_box = prediction.get("pred_box_norm_cxcywh")
    target_box = prediction.get("target_box_norm_cxcywh")
    if pred_box is None or target_box is None:
        return None
    pred_tensor = torch.as_tensor(pred_box, dtype=torch.float32).view(1, 4)
    target_tensor = torch.as_tensor(target_box, dtype=torch.float32).view(1, 4)
    iou = fast_diag_box_iou(box_cxcywh_to_xyxy(pred_tensor), box_cxcywh_to_xyxy(target_tensor))
    return float(iou[0].item())


def compute_pred_type_confidence(prediction: dict) -> float | None:
    pred_type_name = prediction.get("pred_type_name")
    pred_type_probs = prediction.get("pred_type_probs") or {}
    if pred_type_name is None:
        return None
    if pred_type_name not in pred_type_probs:
        return None
    return float(pred_type_probs[pred_type_name])


def build_json_payload(prediction: dict) -> dict:
    payload = {
        "clip_id": prediction["clip_id"],
        "split": prediction["split"],
        "pred_box_norm_cxcywh": prediction["pred_box_norm_cxcywh"],
        "pred_bbox_xywh": prediction["pred_bbox_xywh"],
        "pred_type_name": prediction.get("pred_type_name"),
        "pred_type_probs": prediction.get("pred_type_probs"),
    }
    pred_type_confidence = prediction.get("pred_type_confidence")
    if pred_type_confidence is None:
        pred_type_confidence = compute_pred_type_confidence(prediction)
    if pred_type_confidence is not None:
        payload["pred_type_confidence"] = float(pred_type_confidence)
    if prediction.get("pred_center_cell_confidence") is not None:
        payload["pred_center_cell_confidence"] = float(prediction["pred_center_cell_confidence"])

    record = prediction.get("record")
    if record is not None:
        annotation_id = getattr(record, "annotation_id", None)
        image_id = getattr(record, "image_id", None)
        if annotation_id is not None:
            payload["annotation_id"] = int(annotation_id)
        if image_id is not None:
            payload["image_id"] = int(image_id)

    if prediction.get("target_box_norm_cxcywh") is not None:
        payload["target_box_norm_cxcywh"] = prediction["target_box_norm_cxcywh"]
    if prediction.get("target_bbox_xywh") is not None:
        payload["target_bbox_xywh"] = prediction["target_bbox_xywh"]
    if prediction.get("target_type") is not None:
        payload["target_type"] = prediction["target_type"]

    iou = prediction.get("iou")
    if iou is None:
        iou = compute_prediction_iou(prediction)
    if iou is not None:
        payload["iou"] = float(iou)
    return payload


def build_title(prediction: dict, *, show_gt: bool) -> str:
    pred_type_name = prediction.get("pred_type_name", "unknown")
    title = f"{prediction['clip_id']} | pred={pred_type_name}"
    if show_gt and prediction.get("target_type") is not None:
        title = f"{title} | gt={prediction['target_type']}"
    iou = prediction.get("iou")
    if iou is None:
        iou = compute_prediction_iou(prediction)
    if iou is not None:
        title = f"{title} | IoU={iou:.3f}"
    return title


def render_heatmap_panel(image: PILImage.Image, heatmap: np.ndarray | None) -> PILImage.Image:
    base = image.convert("RGB")
    heatmap = normalize_heatmap(heatmap)
    if heatmap is None:
        panel = base.convert("L").convert("RGB")
        return add_corner_label(panel, "heatmap unavailable")

    heatmap_rgb = colorize_heatmap(heatmap)
    assert heatmap_rgb is not None
    heatmap_rgb = heatmap_rgb.resize(base.size, resample=PILImage.NEAREST)
    heatmap_alpha = (np.clip(heatmap, 0.0, 1.0) ** 0.55 * 255.0).astype(np.uint8)
    heatmap_alpha = PILImage.fromarray(heatmap_alpha, mode="L").resize(base.size, resample=PILImage.NEAREST)

    panel = base.convert("L").convert("RGBA")
    overlay = heatmap_rgb.convert("RGBA")
    overlay.putalpha(heatmap_alpha)
    panel = PILImage.alpha_composite(panel, overlay)
    return add_corner_label(panel, "center heatmap")


def compose_side_by_side(left: PILImage.Image, right: PILImage.Image) -> PILImage.Image:
    left = left.convert("RGB")
    right = right.convert("RGB")
    gap = max(8, int(round(min(left.width, left.height, right.width, right.height) * 0.02)))
    height = max(left.height, right.height)
    canvas = PILImage.new("RGB", (left.width + gap + right.width, height), color=(0, 0, 0))
    canvas.paste(left, (0, (height - left.height) // 2))
    canvas.paste(right, (left.width + gap, (height - right.height) // 2))
    return canvas


def write_prediction_artifacts(prediction: dict, output_path: str | Path, *, show_gt: bool) -> dict:
    dataset = prediction["dataset"]
    record = prediction["record"]
    frames = dataset.load_frames(record)

    pred_type_name = prediction.get("pred_type_name", "unknown")
    pred_type_probs = prediction.get("pred_type_probs", {}) or {}
    pred_type_score = float(pred_type_probs.get(pred_type_name, 0.0))
    render_items = [
        {
            "prompt": f"cbd_pred {pred_type_name} {pred_type_score:.2f}",
            "color": COLOR_CYCLE[0],
            "detections": [{"bbox_xywh": prediction["pred_bbox_xywh"], "mask": None, "score": 1.0}],
        }
    ]
    if show_gt and prediction.get("target_bbox_xywh") is not None:
        render_items.append(
            {
                "prompt": f"cbd_gt {prediction.get('target_type', 'unknown')}",
                "color": COLOR_CYCLE[1],
                "detections": [{"bbox_xywh": prediction["target_bbox_xywh"], "mask": None, "score": 1.0}],
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlay = render_overlay(frames.images[-1], render_items, draw_masks=False, save=False)
    overlay = add_corner_label(overlay, "overlay")
    heatmap_panel = render_heatmap_panel(frames.images[-1], heatmap_tensor_to_numpy(prediction.get("pred_center_heatmap")))
    composed = compose_side_by_side(overlay, heatmap_panel)
    composed = add_title(composed, build_title(prediction, show_gt=show_gt))
    composed.save(output_path)

    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as handle:
        json.dump(build_json_payload(prediction), handle, indent=2)

    return {
        "clip_id": prediction["clip_id"],
        "output_path": str(output_path),
        "json_path": str(json_path),
    }


def resolve_split_output_dir(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    if output_path.suffix:
        return output_path.parent / output_path.stem
    return output_path


def build_split_output_name(record, sample_index: int, clip_id_counts: Counter[str]) -> str:
    clip_id = str(getattr(record, "clip_id"))
    if clip_id_counts[clip_id] <= 1:
        return f"{clip_id}_overlay.png"
    annotation_id = getattr(record, "annotation_id", None)
    if annotation_id is not None:
        return f"{clip_id}_ann{annotation_id}_overlay.png"
    return f"{clip_id}_idx{sample_index:05d}_overlay.png"


def predict_dataset_index(trainer, dataset, sample_index: int, split: str) -> dict:
    if hasattr(trainer, "predict_dataset_index"):
        return trainer.predict_dataset_index(dataset, sample_index)
    record = dataset.get_record(sample_index)
    return trainer.predict_record(split, record.clip_id)


def run_inference(args, trainer_cls=CBDTrainer) -> dict:
    config = load_config(args.config)
    trainer = trainer_cls(config)
    weights_path = resolve_weights_path(config, args.weights)
    trainer.load_checkpoint(weights_path)

    if args.clip_id:
        item = write_prediction_artifacts(
            trainer.predict_record(args.split, args.clip_id),
            args.output,
            show_gt=args.show_gt,
        )
        return {"mode": "single", "count": 1, "items": [item]}

    dataset = trainer.build_dataset(args.split, apply_augmentation=False)
    if len(dataset) == 0:
        raise ValueError(f"No CBD records found for split {args.split!r}.")

    output_dir = resolve_split_output_dir(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_id_counts = Counter(record.clip_id for record in dataset.records)
    items = []
    for sample_index, record in enumerate(dataset.records):
        output_path = output_dir / build_split_output_name(record, sample_index, clip_id_counts)
        prediction = predict_dataset_index(trainer, dataset, sample_index, args.split)
        items.append(write_prediction_artifacts(prediction, output_path, show_gt=args.show_gt))
    return {"mode": "split", "count": len(items), "output_dir": str(output_dir), "items": items}


def main() -> None:
    args = build_argparser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
