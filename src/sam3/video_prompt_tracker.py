from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pycocotools.mask as mask_utils
import torch
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[2]
INFER = ROOT / "infer"
if str(INFER) not in sys.path:
    sys.path.insert(0, str(INFER))

from infer_lora import (
    COLOR_CYCLE,
    autocast_context,
    build_detection,
    build_inference_datapoint,
    build_lora_config,
    filter_predictions,
    move_to_device,
    preprocess_image,
    render_overlay,
)
from sam3.lora import apply_lora_to_model, load_lora_weights
from sam3.model.io_utils import IMAGE_EXTS, VIDEO_EXTS, load_resource_as_video_frames
from sam3.model.model_misc import SAM3Output
from sam3.train.data import collate_fn_api


MIN_PROMPT_MASK_AREA = 16
DEFAULT_DIRECTORY_FPS = 10.0


def thin_frame_indices(frame_indices: list[int], min_gap: int) -> list[int]:
    if min_gap <= 0:
        return sorted(set(frame_indices))

    selected = []
    for frame_idx in sorted(set(frame_indices)):
        if not selected or frame_idx - selected[-1] >= min_gap:
            selected.append(frame_idx)
    return selected


def select_prompt_frame_indices(num_frames: int, strategy: str, stride: int | None = None) -> list[int]:
    if num_frames <= 0:
        return []

    normalized = str(strategy).strip().lower()
    if normalized == "first":
        return [0]
    if normalized == "stride":
        if stride is None or int(stride) <= 0:
            raise ValueError("stride strategy requires --stride > 0")
        return list(range(0, num_frames, int(stride)))
    if normalized == "adaptive":
        return [0]
    raise ValueError(f"Unsupported strategy: {strategy!r}")


def encode_binary_mask(mask: np.ndarray) -> dict | None:
    mask_uint8 = np.asarray(mask, dtype=np.uint8)
    if mask_uint8.ndim != 2 or not np.any(mask_uint8):
        return None
    rle = mask_utils.encode(np.asfortranarray(mask_uint8))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def decode_binary_mask(mask_rle: dict | None) -> np.ndarray | None:
    if mask_rle is None:
        return None
    mask = mask_utils.decode(mask_rle)
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint8)


def bbox_xywh_to_normalized_xyxy(
    bbox_xywh: list[float] | tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> list[float] | None:
    if bbox_xywh is None or width <= 0 or height <= 0:
        return None
    x, y, w, h = [float(value) for value in bbox_xywh]
    if w <= 0.0 or h <= 0.0:
        return None
    x1 = x / float(width)
    y1 = y / float(height)
    x2 = (x + w) / float(width)
    y2 = (y + h) / float(height)
    return [x1, y1, x2, y2]


def choose_prompt_type(detection: dict | None, min_mask_area: int = MIN_PROMPT_MASK_AREA) -> str | None:
    if detection is None:
        return None
    mask = detection.get("mask")
    if mask is not None and int(np.asarray(mask, dtype=np.uint8).sum()) >= int(min_mask_area):
        return "mask"
    if detection.get("bbox_xywh") is not None:
        return "box"
    return None


def compute_tracker_health(mask_logits: torch.Tensor, object_score_logits: torch.Tensor) -> float:
    probs = torch.sigmoid(mask_logits.detach().float())
    foreground = probs > 0.5
    if not torch.any(foreground):
        return 0.0
    foreground_mean = float(probs[foreground].mean().item())
    object_score = float(torch.sigmoid(object_score_logits.detach().float()).mean().item())
    return foreground_mean * object_score


def build_prompt_event(
    *,
    frame_index: int,
    prompt: str,
    object_id: int,
    reason: str,
    detection: dict | None,
    image_size: tuple[int, int],
    min_mask_area: int = MIN_PROMPT_MASK_AREA,
) -> dict | None:
    if detection is None:
        return None

    width, height = image_size
    prompt_type = choose_prompt_type(detection, min_mask_area=min_mask_area)
    if prompt_type is None:
        return None

    mask = detection.get("mask")
    mask_rle = encode_binary_mask(mask) if mask is not None else None
    bbox_xywh = detection.get("bbox_xywh")
    return {
        "frame_index": int(frame_index),
        "prompt": str(prompt),
        "object_id": int(object_id),
        "reason": str(reason),
        "prompt_type": prompt_type,
        "detector_score": float(detection["score"]),
        "bbox_xywh": [float(value) for value in bbox_xywh] if bbox_xywh is not None else None,
        "box_xyxy_norm": bbox_xywh_to_normalized_xyxy(bbox_xywh, width=width, height=height),
        "mask_rle": mask_rle,
        "mask_area": int(np.asarray(mask, dtype=np.uint8).sum()) if mask is not None else 0,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


class PromptedVideoTracker:
    def __init__(
        self,
        *,
        config: dict,
        weights_path: str | Path | None = None,
        tracker_checkpoint: str | Path | None = None,
        device: str = "cuda",
        resolution: int = 1008,
    ) -> None:
        self.config = config
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        self.evaluation_config = config.get("evaluation", {})
        self.hardware_config = config.get("hardware", {})
        self.lora_config = config.get("lora", {})

        self.device = torch.device(device)
        self.resolution = int(resolution)
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.tracker_checkpoint = (
            Path(tracker_checkpoint) if tracker_checkpoint is not None else None
        )

        self.detector = None
        self.tracker = None

    def _ensure_models(self) -> None:
        if self.device.type != "cuda":
            raise ValueError("Video tracking currently requires a CUDA device.")

        if self.detector is None:
            from sam3 import build_sam3_image_model

            if self.weights_path is None:
                raise ValueError("LoRA weights path is required for video prompting.")
            self.detector = build_sam3_image_model(
                checkpoint_path=self.model_config.get("checkpoint_path"),
                device=self.device.type,
                eval_mode=True,
                compile=bool(self.hardware_config.get("use_compile", False)),
                bpe_path=self.model_config.get("bpe_path"),
                load_from_HF=bool(self.model_config.get("load_from_hf", True)),
            )
            lora_cfg = build_lora_config(self.lora_config)
            self.detector = apply_lora_to_model(self.detector, lora_cfg)
            load_lora_weights(self.detector, str(self.weights_path), expected_config=lora_cfg)
            self.detector.to(self.device)
            self.detector.eval()

        if self.tracker is None:
            from sam3 import build_sam3_tracker_model

            self.tracker = build_sam3_tracker_model(
                checkpoint_path=str(self.tracker_checkpoint) if self.tracker_checkpoint else None,
                device=self.device.type,
                eval_mode=True,
                compile=False,
                load_from_HF=self.tracker_checkpoint is None,
                with_backbone=True,
                apply_temporal_disambiguation=True,
            )
            self.tracker.to(self.device)
            self.tracker.eval()

    def _load_frames(self, video_path: str | Path) -> tuple[list[PILImage.Image], float]:
        resource = Path(video_path)
        if resource.is_dir():
            return self._load_frames_from_directory(resource), DEFAULT_DIRECTORY_FPS

        suffix = resource.suffix.lower()
        if suffix in VIDEO_EXTS:
            return self._load_frames_from_video_file(resource)

        raise ValueError(
            f"Unsupported video input: {video_path!r}. Expected an MP4/video file or a frame directory."
        )

    def _load_frames_from_directory(self, directory: Path) -> list[PILImage.Image]:
        frame_paths = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
        if not frame_paths:
            raise FileNotFoundError(f"No supported image frames found in {directory}")
        try:
            frame_paths.sort(key=lambda path: int(path.stem))
        except ValueError:
            frame_paths.sort(key=lambda path: path.name)
        return [PILImage.open(path).convert("RGB") for path in frame_paths]

    def _load_frames_from_video_file(self, video_path: Path) -> tuple[list[PILImage.Image], float]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for MP4 video input support.") from exc

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0:
            fps = DEFAULT_DIRECTORY_FPS

        frames = []
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(PILImage.fromarray(frame_rgb))

        capture.release()
        if not frames:
            raise RuntimeError(f"No frames could be read from {video_path}")
        return frames, fps

    def _build_tracker_state(self, frames: list[PILImage.Image]) -> dict:
        if self.tracker is None:
            raise RuntimeError("Tracker model has not been initialized.")
        first_width, first_height = frames[0].size
        inference_state = self.tracker.init_state(
            video_height=first_height,
            video_width=first_width,
            num_frames=len(frames),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
        )
        images, _, _ = load_resource_as_video_frames(
            resource_path=frames,
            image_size=self.tracker.image_size,
            offload_video_to_cpu=True,
            img_mean=(0.5, 0.5, 0.5),
            img_std=(0.5, 0.5, 0.5),
        )
        inference_state["images"] = images
        return inference_state

    def _detect_frame(
        self,
        image: PILImage.Image,
        prompts: list[str],
        *,
        threshold_override: float | None = None,
    ) -> dict[str, dict | None]:
        if self.detector is None:
            raise RuntimeError("Detector model has not been initialized.")

        orig_w, orig_h = image.size
        image_tensor = preprocess_image(image, self.resolution)
        datapoint = build_inference_datapoint(image_tensor, prompts, (orig_h, orig_w), self.resolution)
        batch = collate_fn_api([datapoint], dict_key="input", with_seg_masks=False)
        input_batch = move_to_device(batch["input"], self.device)

        with torch.no_grad():
            with autocast_context(self.device, self.training_config):
                outputs_list = self.detector(input_batch)

        with SAM3Output.iteration_mode(
            outputs_list,
            iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
        ) as outputs_iter:
            final_outputs = list(outputs_iter)[-1][-1]

        threshold = (
            float(threshold_override)
            if threshold_override is not None
            else float(self.evaluation_config.get("prob_threshold", 0.5))
        )
        nms_iou = float(self.evaluation_config.get("nms_iou", 0.5))
        max_detections = int(self.evaluation_config.get("max_detections", 100))

        detections_by_prompt: dict[str, dict | None] = {}
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

            best_detection = None
            for mask_tensor, score, box_tensor in zip(masks, scores, boxes):
                detection = build_detection(mask_tensor, score, box_tensor, orig_h=orig_h, orig_w=orig_w)
                if detection is None:
                    continue
                if best_detection is None or detection["score"] > best_detection["score"]:
                    best_detection = detection
            detections_by_prompt[prompt] = best_detection

        return detections_by_prompt

    def _detect_prompt_events(
        self,
        frames: list[PILImage.Image],
        prompts: list[str],
        frame_indices: list[int],
        *,
        reason: str,
        threshold_override: float | None = None,
    ) -> list[dict]:
        events = []
        for frame_index in sorted(set(frame_indices)):
            detections = self._detect_frame(
                frames[frame_index],
                prompts,
                threshold_override=threshold_override,
            )
            image_size = frames[frame_index].size
            for object_id, prompt in enumerate(prompts, start=1):
                event = build_prompt_event(
                    frame_index=frame_index,
                    prompt=prompt,
                    object_id=object_id,
                    reason=reason,
                    detection=detections.get(prompt),
                    image_size=image_size,
                )
                if event is not None:
                    events.append(event)
        return events

    def _confirm_adaptive_prompt_events(
        self,
        frames: list[PILImage.Image],
        prompts: list[str],
        candidate_frames_by_object: dict[int, list[int]],
        *,
        detector_threshold: float,
    ) -> list[dict]:
        if not candidate_frames_by_object:
            return []

        union_frames = sorted(
            {
                frame_index
                for frame_indices in candidate_frames_by_object.values()
                for frame_index in frame_indices
            }
        )
        accepted = []
        for frame_index in union_frames:
            detections = self._detect_frame(
                frames[frame_index],
                prompts,
                threshold_override=detector_threshold,
            )
            image_size = frames[frame_index].size
            for object_id, prompt in enumerate(prompts, start=1):
                if frame_index not in set(candidate_frames_by_object.get(object_id, [])):
                    continue
                detection = detections.get(prompt)
                if detection is None or float(detection["score"]) < float(detector_threshold):
                    continue
                event = build_prompt_event(
                    frame_index=frame_index,
                    prompt=prompt,
                    object_id=object_id,
                    reason="adaptive_low_health",
                    detection=detection,
                    image_size=image_size,
                )
                if event is not None:
                    accepted.append(event)
        return accepted

    def _apply_prompt_event(self, inference_state: dict, event: dict) -> None:
        if self.tracker is None:
            raise RuntimeError("Tracker model has not been initialized.")

        frame_index = int(event["frame_index"])
        object_id = int(event["object_id"])
        prompt_type = event["prompt_type"]
        if prompt_type == "mask" and event.get("mask_rle") is not None:
            mask = decode_binary_mask(event["mask_rle"])
            if mask is None:
                return
            mask_tensor = torch.from_numpy(mask.astype(np.uint8))
            self.tracker.add_new_mask(
                inference_state,
                frame_idx=frame_index,
                obj_id=object_id,
                mask=mask_tensor,
            )
            return

        box_xyxy_norm = event.get("box_xyxy_norm")
        if box_xyxy_norm is None:
            return
        box_tensor = torch.tensor(box_xyxy_norm, dtype=torch.float32)
        self.tracker.add_new_points_or_box(
            inference_state,
            frame_idx=frame_index,
            obj_id=object_id,
            box=box_tensor,
            clear_old_points=True,
            rel_coordinates=True,
        )

    def _run_tracking_pass(
        self,
        frames: list[PILImage.Image],
        prompts: list[str],
        prompt_events: list[dict],
    ) -> list[dict]:
        if self.tracker is None:
            raise RuntimeError("Tracker model has not been initialized.")
        if not prompt_events:
            return []

        inference_state = self._build_tracker_state(frames)
        for event in sorted(prompt_events, key=lambda item: (item["frame_index"], item["object_id"])):
            self._apply_prompt_event(inference_state, event)

        first_prompt_frame_by_object = {
            int(object_id): min(event["frame_index"] for event in prompt_events if event["object_id"] == object_id)
            for object_id in {event["object_id"] for event in prompt_events}
        }
        prompt_lookup = {
            (int(event["frame_index"]), int(event["object_id"])): event
            for event in prompt_events
        }
        prompt_by_object_id = {index: prompt for index, prompt in enumerate(prompts, start=1)}

        self.tracker.propagate_in_video_preflight(inference_state)

        tracks = []
        for frame_index, obj_ids, _, video_res_masks, obj_scores in self.tracker.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=None,
            reverse=False,
            tqdm_disable=False,
        ):
            for obj_position, obj_id in enumerate(obj_ids):
                obj_id = int(obj_id)
                if frame_index < first_prompt_frame_by_object.get(obj_id, float("inf")):
                    continue

                mask_logits = video_res_masks[obj_position : obj_position + 1]
                object_score_logits = obj_scores[obj_position : obj_position + 1]
                tracker_health = compute_tracker_health(mask_logits, object_score_logits)
                prob_mask = torch.sigmoid(mask_logits[0, 0].detach().float()).cpu().numpy()
                binary_mask = (prob_mask > 0.5).astype(np.uint8)
                mask_rle = encode_binary_mask(binary_mask)
                bbox_xywh = None
                if mask_rle is not None:
                    bbox_xywh = [float(value) for value in mask_utils.toBbox(mask_rle).tolist()]

                prompt_event = prompt_lookup.get((int(frame_index), obj_id))
                tracks.append(
                    {
                        "frame_index": int(frame_index),
                        "prompt": prompt_by_object_id[obj_id],
                        "object_id": obj_id,
                        "bbox_xywh": bbox_xywh,
                        "mask_rle": mask_rle,
                        "tracker_health": float(tracker_health),
                        "object_score": float(torch.sigmoid(object_score_logits.detach().float()).mean().item()),
                        "detector_prompted": prompt_event is not None,
                    }
                )

        return tracks

    def _collect_adaptive_candidate_frames(
        self,
        tracks: list[dict],
        prompt_events: list[dict],
        *,
        health_threshold: float,
        min_gap: int,
    ) -> dict[int, list[int]]:
        prompted_frames_by_object: dict[int, set[int]] = defaultdict(set)
        for event in prompt_events:
            prompted_frames_by_object[int(event["object_id"])].add(int(event["frame_index"]))

        candidate_frames_by_object: dict[int, list[int]] = {}
        for object_id in sorted({int(track["object_id"]) for track in tracks}):
            raw_candidates = []
            for track in tracks:
                if int(track["object_id"]) != object_id:
                    continue
                frame_index = int(track["frame_index"])
                if frame_index in prompted_frames_by_object[object_id]:
                    continue
                if track["mask_rle"] is None or float(track["tracker_health"]) < float(health_threshold):
                    raw_candidates.append(frame_index)
            candidate_frames_by_object[object_id] = thin_frame_indices(raw_candidates, min_gap)
        return candidate_frames_by_object

    def _build_overlay_predictions(
        self,
        frame_tracks: list[dict],
        prompts: list[str],
    ) -> list[dict]:
        detections_by_prompt = {prompt: [] for prompt in prompts}
        for track in frame_tracks:
            detection = {
                "mask": decode_binary_mask(track["mask_rle"]),
                "bbox_xywh": track["bbox_xywh"],
                "score": track["object_score"],
            }
            detections_by_prompt[track["prompt"]].append(detection)

        predictions = []
        for index, prompt in enumerate(prompts):
            predictions.append(
                {
                    "prompt": prompt,
                    "color": COLOR_CYCLE[index % len(COLOR_CYCLE)],
                    "detections": detections_by_prompt[prompt],
                }
            )
        return predictions

    def _write_overlay_video(self, overlays_dir: Path, output_path: Path, fps: float) -> bool:
        try:
            import cv2
        except ImportError:
            return False

        frame_paths = sorted(overlays_dir.glob("*.png"))
        if not frame_paths:
            return False

        first_frame = cv2.imread(str(frame_paths[0]))
        if first_frame is None:
            return False
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0.0 else DEFAULT_DIRECTORY_FPS,
            (width, height),
        )
        if not writer.isOpened():
            return False

        try:
            for frame_path in frame_paths:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                writer.write(frame)
        finally:
            writer.release()
        return True

    def _write_outputs(
        self,
        *,
        frames: list[PILImage.Image],
        prompts: list[str],
        tracks: list[dict],
        prompt_events: list[dict],
        output_dir: Path,
        fps: float,
        strategy: str,
    ) -> dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        overlays_dir = output_dir / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)

        tracks_payload = {
            "strategy": strategy,
            "num_frames": len(frames),
            "prompts": prompts,
            "tracks": tracks,
        }
        prompt_payload = {
            "strategy": strategy,
            "prompt_events": prompt_events,
        }
        _write_json(output_dir / "tracks.json", tracks_payload)
        _write_json(output_dir / "prompt_events.json", prompt_payload)

        tracks_by_frame: dict[int, list[dict]] = defaultdict(list)
        for track in tracks:
            tracks_by_frame[int(track["frame_index"])].append(track)

        for frame_index, image in enumerate(frames):
            predictions = self._build_overlay_predictions(
                tracks_by_frame.get(frame_index, []),
                prompts,
            )
            render_overlay(
                image=image,
                predictions=predictions,
                output_path=overlays_dir / f"{frame_index:05d}.png",
                draw_masks=True,
            )

        overlay_video_path = output_dir / "overlay.mp4"
        overlay_video_written = self._write_overlay_video(overlays_dir, overlay_video_path, fps)

        return {
            "output_dir": str(output_dir),
            "tracks_path": str(output_dir / "tracks.json"),
            "prompt_events_path": str(output_dir / "prompt_events.json"),
            "overlays_dir": str(overlays_dir),
            "overlay_video_path": str(overlay_video_path) if overlay_video_written else None,
            "overlay_video_written": overlay_video_written,
            "num_tracks": len(tracks),
            "num_prompt_events": len(prompt_events),
            "num_frames": len(frames),
        }

    def run(
        self,
        *,
        video_path: str | Path,
        output_dir: str | Path,
        prompts: list[str],
        strategy: str = "first",
        stride: int = 16,
        adaptive_health_threshold: float = 0.5,
        adaptive_detector_threshold: float = 0.5,
        adaptive_min_gap: int = 8,
    ) -> dict[str, Any]:
        self._ensure_models()

        prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        if not prompts:
            raise ValueError("At least one prompt is required for video tracking.")

        frames, fps = self._load_frames(video_path)
        output_dir = Path(output_dir)
        initial_frame_indices = select_prompt_frame_indices(
            num_frames=len(frames),
            strategy=strategy,
            stride=stride,
        )

        if strategy == "first":
            prompt_events = self._detect_prompt_events(
                frames,
                prompts,
                initial_frame_indices,
                reason="first",
            )
            tracks = self._run_tracking_pass(frames, prompts, prompt_events)
            return self._write_outputs(
                frames=frames,
                prompts=prompts,
                tracks=tracks,
                prompt_events=prompt_events,
                output_dir=output_dir,
                fps=fps,
                strategy=strategy,
            )

        if strategy == "stride":
            prompt_events = self._detect_prompt_events(
                frames,
                prompts,
                initial_frame_indices,
                reason="stride",
            )
            tracks = self._run_tracking_pass(frames, prompts, prompt_events)
            return self._write_outputs(
                frames=frames,
                prompts=prompts,
                tracks=tracks,
                prompt_events=prompt_events,
                output_dir=output_dir,
                fps=fps,
                strategy=strategy,
            )

        if strategy != "adaptive":
            raise ValueError(f"Unsupported strategy: {strategy!r}")

        initial_events = self._detect_prompt_events(frames, prompts, [0], reason="first")
        pass1_tracks = self._run_tracking_pass(frames, prompts, initial_events)
        candidate_frames_by_object = self._collect_adaptive_candidate_frames(
            pass1_tracks,
            initial_events,
            health_threshold=adaptive_health_threshold,
            min_gap=adaptive_min_gap,
        )
        adaptive_events = self._confirm_adaptive_prompt_events(
            frames,
            prompts,
            candidate_frames_by_object,
            detector_threshold=adaptive_detector_threshold,
        )

        existing_keys = {
            (int(event["frame_index"]), int(event["object_id"]))
            for event in initial_events
        }
        merged_events = list(initial_events)
        for event in adaptive_events:
            key = (int(event["frame_index"]), int(event["object_id"]))
            if key in existing_keys:
                continue
            existing_keys.add(key)
            merged_events.append(event)

        final_tracks = self._run_tracking_pass(frames, prompts, merged_events)
        return self._write_outputs(
            frames=frames,
            prompts=prompts,
            tracks=final_tracks,
            prompt_events=merged_events,
            output_dir=output_dir,
            fps=fps,
            strategy=strategy,
        )
