#!/usr/bin/env python3

import argparse
import contextlib
import sys
from pathlib import Path

import numpy as np
import torch
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

from sam3 import build_sam3_image_model
from sam3.image_utils import (
    compute_detection_scores,
    normalized_cxcywh_to_original_xywh,
    resize_image_to_square,
    resize_mask_to_original,
)
from sam3.lora import LoRAConfig, apply_lora_to_model, load_lora_weights
from sam3.model.model_misc import SAM3Output
from sam3.perflib.nms import nms_masks
from sam3.train.data import Datapoint, FindQueryLoaded, Image, InferenceMetadata, collate_fn_api


COLOR_CYCLE = [
    (230, 57, 70),
    (29, 78, 216),
    (22, 163, 74),
    (217, 119, 6),
    (8, 145, 178),
    (168, 85, 247),
]


def build_lora_config(lora_config: dict) -> LoRAConfig:
    return LoRAConfig(
        rank=int(lora_config.get("rank", 16)),
        alpha=int(lora_config.get("alpha", 32)),
        dropout=float(lora_config.get("dropout", 0.0)),
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
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            setattr(obj, field, move_to_device(getattr(obj, field), device))
        return obj
    return obj


def flatten_prompts(prompt_groups: list[list[str]]) -> list[str]:
    prompts = [prompt.strip() for group in prompt_groups for prompt in group if prompt.strip()]
    if not prompts:
        raise ValueError("At least one non-empty prompt is required.")
    return prompts


def preprocess_image(pil_image: PILImage.Image, target_size: int) -> torch.Tensor:
    resized = resize_image_to_square(pil_image, target_size)
    image_np = np.asarray(resized, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    return (image_tensor - 0.5) / 0.5


def resized_mask_to_original(mask_tensor: torch.Tensor, orig_h: int, orig_w: int) -> torch.Tensor:
    return resize_mask_to_original(mask_tensor, orig_h, orig_w)


def padded_mask_to_original(mask_tensor: torch.Tensor, orig_h: int, orig_w: int, target_size: int) -> torch.Tensor:
    del target_size
    return resized_mask_to_original(mask_tensor, orig_h, orig_w)


def build_inference_datapoint(
    image_tensor: torch.Tensor,
    prompts: list[str],
    original_size: tuple[int, int],
    resolution: int,
) -> Datapoint:
    orig_h, orig_w = original_size
    queries = []
    for index, prompt in enumerate(prompts):
        queries.append(
            FindQueryLoaded(
                query_text=prompt,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=0,
                    original_image_id=0,
                    original_category_id=index,
                    original_size=(orig_h, orig_w),
                    object_id=-1,
                    frame_index=0,
                ),
            )
        )
    return Datapoint(
        find_queries=queries,
        images=[Image(data=image_tensor, objects=[], size=(resolution, resolution))],
    )


def filter_predictions(
    pred_logits: torch.Tensor,
    pred_masks: torch.Tensor,
    pred_boxes: torch.Tensor,
    presence_logit: torch.Tensor | None,
    prob_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
):
    if pred_logits.numel() == 0:
        empty = pred_logits[:0].squeeze(-1)
        return pred_masks[:0], empty, pred_boxes[:0]

    scores = compute_detection_scores(pred_logits, presence_logit)
    masks = pred_masks.sigmoid()
    keep = nms_masks(
        pred_probs=scores,
        pred_masks=(masks > 0.5).float(),
        prob_threshold=prob_threshold,
        iou_threshold=nms_iou_threshold,
    )

    masks = masks[keep]
    scores = scores[keep]
    boxes = pred_boxes[keep]

    if max_detections > 0 and len(scores) > max_detections:
        scores, top_indices = torch.topk(scores, k=max_detections, largest=True)
        masks = masks[top_indices]
        boxes = boxes[top_indices]

    return masks, scores, boxes


def build_detection(
    mask_tensor: torch.Tensor,
    score: torch.Tensor,
    box_tensor: torch.Tensor,
    orig_h: int,
    orig_w: int,
) -> dict | None:
    original_mask = resized_mask_to_original(mask_tensor, orig_h, orig_w) > 0.5
    mask_is_valid = bool(original_mask.sum().item() > 0)
    bbox_xywh = normalized_cxcywh_to_original_xywh(box_tensor.tolist(), orig_w, orig_h)
    bbox_is_valid = bool(bbox_xywh[2] > 0.0 and bbox_xywh[3] > 0.0)
    if not mask_is_valid and not bbox_is_valid:
        return None

    return {
        "mask": original_mask.detach().cpu().numpy() if mask_is_valid else None,
        "score": float(score.item()),
        "box_norm_cxcywh": [float(value) for value in box_tensor.tolist()],
        "bbox_xywh": [float(value) for value in bbox_xywh],
    }


def render_overlay(
    image: PILImage.Image,
    predictions: list[dict],
    output_path: Path | None = None,
    alpha: int = 96,
    draw_masks: bool = True,
    save: bool = True,
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
                mask = mask.astype(np.uint8) * alpha
                colored_mask = PILImage.new("RGBA", base.size, color + (0,))
                colored_mask.putalpha(PILImage.fromarray(mask, mode="L"))
                overlay = PILImage.alpha_composite(overlay, colored_mask)

    composed = PILImage.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(composed)
    width, height = composed.size
    min_dim = min(width, height)
    box_outline_width = max(2, int(round(min_dim * 0.004)))

    for result in predictions:
        color = result["color"]
        for detection in result["detections"]:
            bbox_xywh = detection.get("bbox_xywh")
            if bbox_xywh is None:
                continue
            x, y, w, h = bbox_xywh
            if w <= 0 or h <= 0:
                continue
            draw.rectangle(
                (x, y, x + w, y + h),
                outline=color + (255,),
                width=box_outline_width,
            )

    font_size = max(14, int(round(min_dim * 0.018)))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    margin = max(12, int(round(min_dim * 0.030)))
    text_gap = max(8, int(round(font_size * 0.45)))
    swatch_size = max(12, int(round(font_size * 0.8)))
    swatch_offset = max(2, int(round(font_size * 0.15)))
    box_padding_x = max(6, int(round(font_size * 0.35)))
    box_padding_y = max(3, int(round(font_size * 0.2)))
    line_gap = max(10, int(round(font_size * 0.5)))

    x = margin
    y = margin
    for result in predictions:
        color = result["color"]
        label = f"{result['prompt']} ({len(result['detections'])})"
        text_x = x + swatch_size + text_gap
        text_left, text_top, text_right, text_bottom = draw.textbbox((text_x, y), label, font=font)

        swatch_y = y + swatch_offset
        draw.rectangle(
            (x, swatch_y, x + swatch_size, swatch_y + swatch_size),
            fill=color + (255,),
        )
        draw.rectangle(
            (
                text_left - box_padding_x,
                text_top - box_padding_y,
                text_right + box_padding_x,
                text_bottom + box_padding_y,
            ),
            fill=(0, 0, 0, 170),
        )
        draw.text((text_x, y), label, fill=(255, 255, 255, 255), font=font)
        y = text_bottom + box_padding_y + line_gap
    composed = composed.convert("RGB")
    if save and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composed.save(output_path)
    return composed


def resolve_weights_path(config: dict, weights_path: str | None) -> Path:
    if weights_path is not None:
        return Path(weights_path)
    output_dir = Path(config.get("output", {}).get("output_dir", "outputs/endoscapes_lora"))
    return output_dir / "best_lora_weights.pt"


def resolve_device(config: dict, requested_device: str | None) -> torch.device:
    if requested_device is not None:
        return torch.device(requested_device)
    preferred = str(config.get("hardware", {}).get("device", "cuda"))
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_context(device: torch.device, training_config: dict):
    precision = str(training_config.get("mixed_precision", "none")).lower()
    if device.type != "cuda":
        return contextlib.nullcontext()
    if precision == "bf16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def run_inference(args) -> list[dict]:
    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    model_config = config.get("model", {})
    evaluation_config = config.get("evaluation", {})
    training_config = config.get("training", {})
    lora_config = config.get("lora", {})
    prompts = flatten_prompts(args.prompt)
    weights_path = resolve_weights_path(config, args.weights)
    device = resolve_device(config, args.device)

    if not weights_path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {weights_path}")
    if model_config.get("checkpoint_path") is None and not bool(model_config.get("load_from_hf", True)):
        raise ValueError("Config must set model.checkpoint_path or model.load_from_hf=true for inference.")

    model = build_sam3_image_model(
        checkpoint_path=model_config.get("checkpoint_path"),
        device=device.type,
        eval_mode=True,
        compile=bool(config.get("hardware", {}).get("use_compile", False)),
        bpe_path=model_config.get("bpe_path"),
        load_from_HF=bool(model_config.get("load_from_hf", True)),
    )
    lora_cfg = build_lora_config(lora_config)
    model = apply_lora_to_model(model, lora_cfg)
    load_lora_weights(model, str(weights_path), expected_config=lora_cfg)
    model.to(device)
    model.eval()

    image_path = Path(args.image)
    image = PILImage.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    image_tensor = preprocess_image(image, args.resolution)
    datapoint = build_inference_datapoint(image_tensor, prompts, (orig_h, orig_w), args.resolution)
    batch = collate_fn_api([datapoint], dict_key="input", with_seg_masks=False)
    input_batch = move_to_device(batch["input"], device)

    with torch.no_grad():
        with autocast_context(device, training_config):
            outputs_list = model(input_batch)

    with SAM3Output.iteration_mode(
        outputs_list,
        iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
    ) as outputs_iter:
        final_outputs = list(outputs_iter)[-1][-1]

    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(evaluation_config.get("prob_threshold", 0.5))
    )
    nms_iou = (
        float(args.nms_iou)
        if args.nms_iou is not None
        else float(evaluation_config.get("nms_iou", 0.5))
    )
    max_detections = int(
        args.max_detections
        if args.max_detections is not None
        else evaluation_config.get("max_detections", 100)
    )

    predictions = []
    for index, prompt in enumerate(prompts):
        presence_logit = None
        if "presence_logit_dec" in final_outputs:
            presence_logit = final_outputs["presence_logit_dec"][index]
        masks, scores, boxes = filter_predictions(
            pred_logits=final_outputs["pred_logits"][index],
            pred_masks=final_outputs["pred_masks"][index],
            pred_boxes=final_outputs["pred_boxes"][index],
            presence_logit=presence_logit,
            prob_threshold=threshold,
            nms_iou_threshold=nms_iou,
            max_detections=max_detections,
        )

        detections = []
        for mask_tensor, score, box_tensor in zip(masks, scores, boxes):
            detection = build_detection(mask_tensor, score, box_tensor, orig_h=orig_h, orig_w=orig_w)
            if detection is None:
                continue
            detections.append(detection)

        predictions.append(
            {
                "prompt": prompt,
                "color": COLOR_CYCLE[index % len(COLOR_CYCLE)],
                "detections": detections,
            }
        )

    render_overlay(image, predictions, Path(args.output), draw_masks=args.show_masks)
    if args.save_predictions:
        np.savez_compressed(Path(args.output).with_suffix(".npz"), predictions=predictions)
    return predictions


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAM3 + LoRA inference on one image and save a mask overlay.",
    )
    parser.add_argument("--config", required=True, help="Path to the training config YAML.")
    parser.add_argument("--weights", help="Path to the LoRA weights. Defaults to output_dir/best_lora_weights.pt.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--prompt",
        required=True,
        action="append",
        nargs="+",
        help='One or more text prompts, e.g. --prompt "tool" or --prompt "tool" "gallbladder".',
    )
    parser.add_argument("--output", required=True, help="Where to save the overlay image.")
    parser.add_argument("--resolution", type=int, default=1008, help="Square SAM3 input resolution.")
    parser.add_argument("--threshold", type=float, help="Override the config probability threshold.")
    parser.add_argument("--nms-iou", type=float, help="Override the config mask-NMS IoU threshold.")
    parser.add_argument("--max-detections", type=int, help="Maximum detections to keep per prompt.")
    parser.add_argument("--device", help='Override the config device, e.g. "cuda" or "cpu".')
    parser.add_argument('--save-predictions', action='store_true', help="Whether to save raw predictions as a .npz file alongside the overlay.")
    parser.add_argument(
        "--show-masks",
        dest="show_masks",
        action="store_true",
        help="Render predicted masks in the overlay. Boxes are always shown.",
    )
    parser.add_argument(
        "--hide-masks",
        dest="show_masks",
        action="store_false",
        help="Hide predicted masks in the overlay while keeping boxes visible.",
    )
    parser.set_defaults(show_masks=True)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    predictions = run_inference(args)
    total = sum(len(result["detections"]) for result in predictions)
    print(f"Saved overlay to {args.output}")
    for result in predictions:
        print(f"  {result['prompt']}: {len(result['detections'])} detections")
    print(f"  total: {total}")


if __name__ == "__main__":
    main()
