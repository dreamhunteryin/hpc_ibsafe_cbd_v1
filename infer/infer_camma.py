from pathlib import Path
import sys

import numpy as np
import pycocotools.mask as mask_utils
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
INFER = ROOT / "infer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(INFER) not in sys.path:
    sys.path.insert(0, str(INFER))

from camma_artifacts import build_camma_artifact_paths, resolve_camma_split, write_json
from data.dataset_camma import CammaDataset
from data.viz import show_camma_frame_annotations
from infer_lora import *


def _normalize_prompt_name(name: str) -> str:
    return name.strip().lower()


def _deduplicate_prompts(prompts: list[str]) -> list[str]:
    deduplicated = []
    seen = set()
    for prompt in prompts:
        normalized = _normalize_prompt_name(prompt)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(prompt.strip())
    return deduplicated


def resolve_prompts(
    prompt_groups: list[list[str]] | None,
    data_config: dict,
    dataset: CammaDataset,
) -> list[str]:
    if prompt_groups:
        prompts = flatten_prompts(prompt_groups)
    else:
        configured_prompts = data_config.get("class_names")
        if configured_prompts:
            prompts = [str(prompt).strip() for prompt in configured_prompts if str(prompt).strip()]
        else:
            prompts = [
                dataset.context.category_id_to_name[category_id]
                for category_id in sorted(dataset.context.category_id_to_name)
            ]

    prompts = _deduplicate_prompts(prompts)
    if not prompts:
        raise ValueError("No prompts were provided or found in data.class_names.")
    return prompts


def resolve_prompt_specs(dataset: CammaDataset, prompts: list[str]) -> tuple[list[dict], list[int] | None, list[str]]:
    name_to_category_id = {
        _normalize_prompt_name(category_name): category_id
        for category_id, category_name in dataset.context.category_id_to_name.items()
    }
    prompt_specs = []
    category_ids = []
    matched_category_ids = set()
    unmatched_prompts = []
    for prompt in prompts:
        category_id = name_to_category_id.get(_normalize_prompt_name(prompt))
        prompt_specs.append({"prompt": prompt, "category_id": category_id})
        if category_id is None:
            unmatched_prompts.append(prompt)
            continue
        if category_id in matched_category_ids:
            continue
        matched_category_ids.add(category_id)
        category_ids.append(category_id)
    return prompt_specs, (category_ids or None), unmatched_prompts


def build_coco_prediction(mask: np.ndarray, image_id: int, category_id: int, score: float) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    bbox = mask_utils.toBbox(rle).tolist()
    return {
        "image_id": int(image_id),
        "category_id": int(category_id),
        "segmentation": rle,
        "bbox": [float(value) for value in bbox],
        "score": float(score),
    }


def run_inference(args) -> dict:
    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    model_config = config.get("model", {})
    data_config = config.get("data", {})
    evaluation_config = config.get("evaluation", {})
    training_config = config.get("training", {})
    lora_config = config.get("lora", {})
    weights_path = resolve_weights_path(config, args.weights)
    device = resolve_device(config, args.device)
    dataset_name = data_config.get("dataset_name", "Endoscapes-Seg201-CBD")
    split_name = resolve_camma_split(data_config, args.split, default_key="test_split")
    artifact_paths = build_camma_artifact_paths(
        config=config,
        split_name=split_name,
        dataset_name=dataset_name,
        overlay_root=args.output,
    )

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

    dataset = CammaDataset(
        root_dir=data_config.get("dataset_root", "/home2020/home/miv/vedrenne/data/camma"),
        dataset_name=dataset_name,
        split=split_name,
        annotation_file=data_config.get("annotation_file", "annotation_coco.json"),
    )
    prompts = resolve_prompts(args.prompt, data_config, dataset)
    prompt_specs, gt_category_ids, unmatched_prompts = resolve_prompt_specs(dataset, prompts)
    artifact_paths.overlays_dir.mkdir(parents=True, exist_ok=True)
    if unmatched_prompts:
        print(
            "[WARN] These prompts do not match dataset categories and will be skipped in the COCO export: "
            + ", ".join(unmatched_prompts)
        )

    coco_predictions = []
    pbar = tqdm(dataset, desc="Inference")
    for frame in pbar:
        fig, ax = show_camma_frame_annotations(
            frame=frame,
            annotation_types=('mask',),
            category_ids=gt_category_ids,
            category_id_to_name=dataset.context.category_id_to_name,
        )
        output_file = artifact_paths.overlays_dir / f"{frame.metadata['file_name']}"
        output_file = output_file.with_stem(output_file.stem + "_gt")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file.with_suffix(".png"))
        fig.clf()
        plt.close()
        image_path = dataset.context.resolve_image_path(frame.metadata['file_name'])
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
        image_id = int(frame.metadata["id"])
        for index, prompt_spec in enumerate(prompt_specs):
            prompt = prompt_spec["prompt"]
            category_id = prompt_spec["category_id"]
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
                mask_np = detection["mask"]
                if category_id is not None and mask_np is not None:
                    coco_predictions.append(
                        build_coco_prediction(
                            mask=mask_np,
                            image_id=image_id,
                            category_id=category_id,
                            score=detection["score"],
                        )
                    )

            predictions.append(
                {
                    "prompt": prompt,
                    "color": COLOR_CYCLE[index % len(COLOR_CYCLE)],
                    "detections": detections,
                }
            )
        output_file = artifact_paths.overlays_dir / f"{frame.metadata['file_name']}"
        output_file = output_file.with_stem(output_file.stem + "_pred")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        render_overlay(image, predictions, output_file.with_suffix(".png"))
    pbar.close()

    write_json(artifact_paths.predictions_coco_path, coco_predictions)
    write_json(artifact_paths.legacy_predictions_coco_path, coco_predictions)
    return {
        "dataset_name": dataset_name,
        "split": split_name,
        "overlay_dir": artifact_paths.overlays_dir,
        "predictions_path": artifact_paths.predictions_coco_path,
        "num_predictions": len(coco_predictions),
        "num_images": len(dataset),
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAM3 + LoRA inference on a CAMMA dataset and save overlays plus COCO predictions.",
    )
    parser.add_argument("--config", required=True, help="Path to the training config YAML.")
    parser.add_argument("--weights", help="Path to the LoRA weights. Defaults to output_dir/best_lora_weights.pt.")
    parser.add_argument(
        "--prompt",
        action="append",
        nargs="+",
        help='Optional prompt override, e.g. --prompt "tool" or --prompt "tool" "gallbladder". Defaults to data.class_names.',
    )
    parser.add_argument(
        "--split",
        help='Dataset split to run inference on. Defaults to data.test_split from the config.',
    )
    parser.add_argument(
        "--output",
        help="Optional directory for overlay images. Defaults to <output_dir>/inference/<split>/overlays.",
    )
    parser.add_argument("--resolution", type=int, default=1008, help="Square SAM3 input resolution.")
    parser.add_argument("--threshold", type=float, help="Override the config probability threshold.")
    parser.add_argument("--nms-iou", type=float, help="Override the config mask-NMS IoU threshold.")
    parser.add_argument("--max-detections", type=int, help="Maximum detections to keep per prompt.")
    parser.add_argument("--device", help='Override the config device, e.g. "cuda" or "cpu".')
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    result = run_inference(args)
    print(f"Saved overlays to {result['overlay_dir']}")
    print(f"Saved {result['num_predictions']} COCO predictions to {result['predictions_path']}")
    print(f"Processed {result['num_images']} images from {result['dataset_name']} ({result['split']})")


if __name__ == "__main__":
    main()
