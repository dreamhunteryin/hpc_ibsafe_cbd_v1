#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
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


RGB_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
RGB_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
COLOR_CYCLE = [
    (230, 57, 70),
    (29, 78, 216),
    (22, 163, 74),
    (217, 119, 6),
    (8, 145, 178),
    (168, 85, 247),
]


@dataclass(frozen=True)
class ClipFrames:
    images: list[PILImage.Image]
    frame_names: list[str]
    original_size: tuple[int, int]


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def load_stage1_config(config_path: str | Path) -> dict:
    return load_config(config_path)


def parse_frame_number_from_path(path: str | Path) -> int:
    stem = Path(path).stem
    if "_" in stem:
        return int(stem.rsplit("_", 1)[-1])
    return int(stem)


def pil_to_normalized_tensor(image: PILImage.Image, image_size: int) -> torch.Tensor:
    resized = image.resize((image_size, image_size), PILImage.BILINEAR)
    image_np = np.asarray(resized, dtype=np.float32) / 255.0
    image_t = torch.from_numpy(image_np).permute(2, 0, 1)
    return (image_t - RGB_MEAN) / RGB_STD


def load_rgb_clip_tensor(frames: ClipFrames, image_size: int) -> torch.Tensor:
    return torch.stack([pil_to_normalized_tensor(image, image_size) for image in frames.images], dim=0)


def resize_mask_sequence(mask_array: np.ndarray, image_size: int) -> torch.Tensor:
    masks = torch.from_numpy(mask_array).float()
    return F.interpolate(masks, size=(image_size, image_size), mode="nearest")


def coco_bbox_to_normalized_cxcywh(
    bbox_xywh: list[float] | tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
) -> torch.Tensor:
    x, y, w, h = [float(value) for value in bbox_xywh]
    return torch.tensor(
        [(x + w / 2.0) / orig_w, (y + h / 2.0) / orig_h, w / orig_w, h / orig_h],
        dtype=torch.float32,
    )


def normalized_cxcywh_to_original_xywh(
    box: list[float] | tuple[float, float, float, float],
    orig_w: int,
    orig_h: int,
) -> tuple[float, float, float, float]:
    cx, cy, w_norm, h_norm = box
    x1 = (cx - w_norm / 2.0) * orig_w
    y1 = (cy - h_norm / 2.0) * orig_h
    x2 = (cx + w_norm / 2.0) * orig_w
    y2 = (cy + h_norm / 2.0) * orig_h
    x1 = max(0.0, min(float(orig_w), x1))
    y1 = max(0.0, min(float(orig_h), y1))
    x2 = max(0.0, min(float(orig_w), x2))
    y2 = max(0.0, min(float(orig_h), y2))
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def fast_diag_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.minimum(boxes1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def render_overlay(
    image: PILImage.Image,
    predictions: list[dict],
    *,
    alpha: int = 96,
    draw_masks: bool = True,
) -> PILImage.Image:
    base = image.convert("RGBA")
    overlay = PILImage.new("RGBA", base.size, (0, 0, 0, 0))

    if draw_masks:
        for result in predictions:
            color = result["color"]
            for detection in result["detections"]:
                mask = detection.get("mask")
                if mask is None:
                    continue
                mask_alpha = np.asarray(mask, dtype=np.uint8) * alpha
                colored_mask = PILImage.new("RGBA", base.size, color + (0,))
                colored_mask.putalpha(PILImage.fromarray(mask_alpha, mode="L"))
                overlay = PILImage.alpha_composite(overlay, colored_mask)

    composed = PILImage.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(composed)
    min_dim = min(composed.size)
    box_width = max(2, int(round(min_dim * 0.004)))

    for result in predictions:
        color = result["color"]
        for detection in result["detections"]:
            bbox_xywh = detection.get("bbox_xywh")
            if bbox_xywh is None:
                continue
            x, y, w, h = bbox_xywh
            if w <= 0 or h <= 0:
                continue
            draw.rectangle((x, y, x + w, y + h), outline=color + (255,), width=box_width)

    font_size = max(14, int(round(min_dim * 0.018)))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    x = max(12, int(round(min_dim * 0.03)))
    y = x
    swatch_size = max(12, int(round(font_size * 0.8)))
    line_gap = max(10, int(round(font_size * 0.5)))
    for result in predictions:
        label = f"{result['prompt']} ({len(result['detections'])})"
        color = result["color"]
        text_x = x + swatch_size + 8
        left, top, right, bottom = draw.textbbox((text_x, y), label, font=font)
        draw.rectangle((x, y + 2, x + swatch_size, y + 2 + swatch_size), fill=color + (255,))
        draw.rectangle((left - 6, top - 3, right + 6, bottom + 3), fill=(0, 0, 0, 170))
        draw.text((text_x, y), label, fill=(255, 255, 255, 255), font=font)
        y = bottom + line_gap
    return composed.convert("RGB")


def resolve_relative(base_path: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    candidates = [(base_path.parent / path).resolve(), (ROOT / path).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def resolve_cbd_weights_path(config_path: str | Path, config: dict, weights_path: str | None) -> Path:
    if weights_path is not None:
        return Path(weights_path).resolve()

    config_path = Path(config_path).resolve()
    output_dir = config.get("output", {}).get("output_dir")
    candidates = [config_path.parent / "best_cbd.pt"]
    if output_dir is not None:
        output_dir_path = Path(output_dir)
        candidates.append(output_dir_path / "best_cbd.pt")
        if not output_dir_path.is_absolute():
            candidates.append((ROOT / output_dir_path).resolve() / "best_cbd.pt")
    candidates.append(config_path.parent / "last_cbd.pt")
    if output_dir is not None:
        output_dir_path = Path(output_dir)
        candidates.append(output_dir_path / "last_cbd.pt")
        if not output_dir_path.is_absolute():
            candidates.append((ROOT / output_dir_path).resolve() / "last_cbd.pt")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def sample_tail_indices(num_frames: int, clip_len: int, step: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError("At least one frame is required for CBD inference.")
    step = max(1, int(step))
    last_index = num_frames - 1
    return [max(0, last_index - (clip_len - 1 - index) * step) for index in range(clip_len)]


def add_title(image: PILImage.Image, text: str) -> PILImage.Image:
    composed = image.convert("RGBA")
    draw = ImageDraw.Draw(composed)
    min_dim = min(composed.size)
    font_size = max(14, int(round(min_dim * 0.02)))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((12, 12), text, font=font)
    draw.rectangle((8, 8, right + 10, bottom + 8), fill=(0, 0, 0, 180))
    draw.text((12, 12), text, fill=(255, 255, 255, 255), font=font)
    return composed


def _load_directory_frames(directory: Path) -> tuple[list[PILImage.Image], list[str]]:
    frame_paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {directory}")
    try:
        frame_paths.sort(key=parse_frame_number_from_path)
    except ValueError:
        frame_paths.sort(key=lambda path: path.name)

    images = []
    for path in frame_paths:
        with PILImage.open(path) as image:
            images.append(image.convert("RGB"))
    return images, [path.name for path in frame_paths]


def _load_video_frames(video_path: Path) -> tuple[list[PILImage.Image], list[str], float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for video clip inference from MP4 files.") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        fps = 0.0

    images = []
    frame_names = []
    frame_index = 0
    while True:
        success, frame_bgr = capture.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        images.append(PILImage.fromarray(frame_rgb))
        frame_names.append(f"frame_{frame_index:06d}.png")
        frame_index += 1

    capture.release()
    if not images:
        raise RuntimeError(f"No frames could be read from {video_path}")
    return images, frame_names, fps


def load_inference_clip_frames(
    clip_path: str | Path,
    *,
    clip_len: int,
    target_fps: int,
) -> ClipFrames:
    resource = Path(clip_path)
    if resource.is_dir():
        images, frame_names = _load_directory_frames(resource)
        step = 1
    elif resource.suffix.lower() in VIDEO_EXTS:
        images, frame_names, fps = _load_video_frames(resource)
        step = max(1, int(round(fps / float(max(1, target_fps))))) if fps > 0.0 else 1
    elif resource.suffix.lower() in IMAGE_EXTS:
        with PILImage.open(resource) as image:
            images = [image.convert("RGB")]
        frame_names = [resource.name]
        step = 1
    else:
        raise ValueError(f"Unsupported clip input: {clip_path!r}. Expected a frame directory, image, or video file.")

    indices = sample_tail_indices(len(images), int(clip_len), step)
    selected_images = [images[index].copy() for index in indices]
    selected_names = [frame_names[index] for index in indices]
    width, height = selected_images[-1].size
    return ClipFrames(images=selected_images, frame_names=selected_names, original_size=(height, width))


def infer_clip_id(clip_path: str | Path, clip_id: str | None) -> str | None:
    if clip_id:
        return str(clip_id)
    resource = Path(clip_path)
    candidate = resource.name if resource.is_dir() else resource.stem
    return candidate or None


def compute_box_iou(pred_box: torch.Tensor, target_box: torch.Tensor) -> float:
    pred_xyxy = box_cxcywh_to_xyxy(pred_box.view(1, 4))
    target_xyxy = box_cxcywh_to_xyxy(target_box.view(1, 4))
    return float(fast_diag_box_iou(pred_xyxy, target_xyxy)[0].item())


def resolve_ground_truth(args, trainer, clip_id: str | None, original_size: tuple[int, int]) -> dict | None:
    orig_h, orig_w = original_size
    if args.gt_bbox is not None:
        gt_bbox_xywh = [float(value) for value in args.gt_bbox]
        gt_box_norm = coco_bbox_to_normalized_cxcywh(gt_bbox_xywh, orig_w=orig_w, orig_h=orig_h)
        return {
            "source": "cli_bbox",
            "type_name": args.gt_type,
            "box_norm_cxcywh": [float(value) for value in gt_box_norm.tolist()],
            "bbox_xywh": gt_bbox_xywh,
        }

    if args.split is None:
        return None
    if clip_id is None:
        raise ValueError("Could not infer a clip id for dataset GT lookup. Pass --clip-id or use --gt-bbox instead.")

    dataset = trainer.build_dataset(args.split, apply_augmentation=False)
    record = dataset.find_record_by_clip_id(clip_id)
    gt_box_norm = record.target_box.detach().cpu()
    return {
        "source": f"dataset:{args.split}",
        "type_name": record.target_type_name,
        "box_norm_cxcywh": [float(value) for value in gt_box_norm.tolist()],
        "bbox_xywh": list(normalized_cxcywh_to_original_xywh(gt_box_norm.tolist(), orig_w, orig_h)),
    }


def build_title(
    clip_label: str,
    pred_type_name: str | None,
    pred_type_score: float | None,
    gt_payload: dict | None,
    iou: float | None,
) -> str:
    parts = [clip_label]
    if pred_type_name is not None:
        pred_text = f"pred={pred_type_name}"
        if pred_type_score is not None:
            pred_text = f"{pred_text} {pred_type_score:.2f}"
        parts.append(pred_text)
    if gt_payload is not None and gt_payload.get("type_name"):
        parts.append(f"gt={gt_payload['type_name']}")
    if iou is not None:
        parts.append(f"IoU={iou:.3f}")
    return " | ".join(parts)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stage-2 CBD inference on a single clip resource.")
    parser.add_argument("--config", required=True, help="Path to the CBD YAML config.")
    parser.add_argument("--clip", required=True, help="Clip resource: frame directory, image, or video file.")
    parser.add_argument("--output", required=True, help="Output overlay PNG path.")
    parser.add_argument("--weights", help="Optional CBD checkpoint path. Defaults to best_cbd.pt near the config/run.")
    parser.add_argument("--clip-id", help="Optional clip id override, used for titles and dataset GT lookup.")
    parser.add_argument("--split", help="Optional dataset split used to fetch the GT box when the clip id matches a dataset record.")
    parser.add_argument("--gt-bbox", nargs=4, type=float, metavar=("X", "Y", "W", "H"), help="Optional GT CBD box in original-frame xywh pixels.")
    parser.add_argument("--gt-type", help="Optional GT CBD type label when --gt-bbox is provided.")
    return parser


def run_clip_inference(
    args,
    *,
    trainer_cls=None,
    mask_builder_cls=None,
) -> dict:
    if trainer_cls is None:
        from cbd.engine import CBDTrainer

        trainer_cls = CBDTrainer
    if mask_builder_cls is None:
        from cbd.cache import EasyMaskCacheBuilder

        mask_builder_cls = EasyMaskCacheBuilder

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    weights_path = resolve_cbd_weights_path(config_path, config, args.weights)

    trainer = trainer_cls(config)
    trainer.load_checkpoint(weights_path)

    clip_len = int(config.get("model", {}).get("clip_len", 25))
    target_fps = int(config.get("data", {}).get("clip_fps", 5))
    clip_frames = load_inference_clip_frames(args.clip, clip_len=clip_len, target_fps=target_fps)
    clip_id = infer_clip_id(args.clip, args.clip_id)

    stage1_cfg = config.get("stage1_sam3", {})
    stage1_config_path = resolve_relative(config_path, stage1_cfg.get("config_path"))
    if stage1_config_path is None:
        raise ValueError("stage1_sam3.config_path must be set in the CBD config.")
    stage1_weights_path = resolve_relative(config_path, stage1_cfg.get("weights_path"))
    tracker_checkpoint = resolve_relative(config_path, stage1_cfg.get("tracker_checkpoint"))
    easy_prompts = [str(prompt).strip() for prompt in stage1_cfg.get("easy_prompts", []) if str(prompt).strip()]
    if len(easy_prompts) != 2:
        raise ValueError("stage1_sam3.easy_prompts must contain exactly two prompts.")

    mask_builder = mask_builder_cls(
        stage1_config=load_stage1_config(stage1_config_path),
        stage1_weights_path=stage1_weights_path,
        tracker_checkpoint=tracker_checkpoint,
        device=str(stage1_cfg.get("device", config.get("hardware", {}).get("device", "cuda"))),
        resolution=int(stage1_cfg.get("resolution", 1008)),
    )
    mask_prediction = mask_builder.predict_masks_for_frames(
        clip_frames=clip_frames,
        prompts=easy_prompts,
        strategy=str(stage1_cfg.get("strategy", "adaptive")),
        stride=int(stage1_cfg.get("stride", 8)),
        adaptive_health_threshold=float(stage1_cfg.get("adaptive_health_threshold", 0.5)),
        adaptive_detector_threshold=float(stage1_cfg.get("adaptive_detector_threshold", 0.5)),
        adaptive_min_gap=int(stage1_cfg.get("adaptive_min_gap", 8)),
    )

    rgb = load_rgb_clip_tensor(clip_frames, image_size=trainer.input_size).unsqueeze(0)
    masks = resize_mask_sequence(mask_prediction["masks"], image_size=trainer.input_size).unsqueeze(0)
    model_output = trainer.predict_batch({"rgb": rgb, "masks": masks})

    pred_box = model_output.pred_boxes.detach().cpu()[0]
    orig_h, orig_w = clip_frames.original_size
    pred_bbox_xywh = list(normalized_cxcywh_to_original_xywh(pred_box.tolist(), orig_w, orig_h))

    pred_type_name = None
    pred_type_probs = None
    pred_type_score = None
    if getattr(model_output, "type_logits", None) is not None:
        type_probs_tensor = model_output.type_logits.detach().cpu().softmax(dim=-1)[0]
        pred_label = int(type_probs_tensor.argmax().item())
        pred_type_name = ("soft", "hard")[pred_label]
        pred_type_probs = {
            name: float(prob.item()) for name, prob in zip(("soft", "hard"), type_probs_tensor)
        }
        pred_type_score = pred_type_probs[pred_type_name]

    gt_payload = resolve_ground_truth(args, trainer, clip_id, clip_frames.original_size)
    iou = None
    if gt_payload is not None:
        gt_box = torch.tensor(gt_payload["box_norm_cxcywh"], dtype=torch.float32)
        iou = compute_box_iou(pred_box, gt_box)

    render_items = []
    for index, prompt in enumerate(easy_prompts):
        mask = np.asarray(mask_prediction["masks"][-1, index], dtype=np.uint8)
        detections = [{"mask": mask, "bbox_xywh": None, "score": 1.0}] if mask.any() else []
        render_items.append(
            {
                "prompt": prompt,
                "color": COLOR_CYCLE[(index + 1) % len(COLOR_CYCLE)],
                "detections": detections,
            }
        )
    cbd_prompt = "cbd_pred" if pred_type_name is None else f"cbd_pred {pred_type_name}"
    render_items.append(
        {
            "prompt": cbd_prompt,
            "color": COLOR_CYCLE[0],
            "detections": [{"bbox_xywh": pred_bbox_xywh, "mask": None, "score": 1.0}],
        }
    )
    if gt_payload is not None:
        gt_prompt = "cbd_gt" if not gt_payload.get("type_name") else f"cbd_gt {gt_payload['type_name']}"
        render_items.append(
            {
                "prompt": gt_prompt,
                "color": COLOR_CYCLE[3],
                "detections": [{"bbox_xywh": gt_payload["bbox_xywh"], "mask": None, "score": 1.0}],
            }
        )

    clip_label = clip_id or Path(args.clip).stem
    title = build_title(clip_label, pred_type_name, pred_type_score, gt_payload, iou)
    overlay = add_title(render_overlay(clip_frames.images[-1], render_items), title)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)

    payload = {
        "clip_path": str(Path(args.clip).resolve()),
        "clip_id": clip_id,
        "frame_names": list(clip_frames.frame_names),
        "weights_path": str(weights_path),
        "mask_prompts": list(easy_prompts),
        "pred_box_norm_cxcywh": [float(value) for value in pred_box.tolist()],
        "pred_bbox_xywh": pred_bbox_xywh,
        "pred_type_name": pred_type_name,
        "pred_type_probs": pred_type_probs,
        "gt": gt_payload,
        "iou": iou,
        "title": title,
        "num_prompt_events": len(mask_prediction["prompt_events"]),
        "num_tracks": len(mask_prediction["tracks"]),
    }
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2)

    return {
        "output_path": str(output_path),
        "json_path": str(json_path),
        "weights_path": str(weights_path),
        "clip_id": clip_id,
        "title": title,
        "iou": iou,
    }


def main() -> None:
    args = build_argparser().parse_args()
    summary = run_clip_inference(args)
    print(f"Saved overlay to {summary['output_path']}")
    print(f"Saved details to {summary['json_path']}")
    print(f"Used weights {summary['weights_path']}")
    if summary["iou"] is not None:
        print(f"Overlay title: {summary['title']}")


if __name__ == "__main__":
    main()
