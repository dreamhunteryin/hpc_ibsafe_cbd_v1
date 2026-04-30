from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from sam3.video_prompt_tracker import PromptedVideoTracker, decode_binary_mask, select_prompt_frame_indices

from .dataset import CBDDataset


def load_stage1_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


class EasyMaskCacheBuilder:
    def __init__(
        self,
        *,
        stage1_config: dict,
        stage1_weights_path: str | Path | None = None,
        tracker_checkpoint: str | Path | None = None,
        device: str = "cuda",
        resolution: int = 1008,
        tracker_factory=PromptedVideoTracker,
    ) -> None:
        self.stage1_config = stage1_config
        self.stage1_weights_path = stage1_weights_path
        self.tracker_checkpoint = tracker_checkpoint
        self.device = device
        self.resolution = int(resolution)
        self.tracker_factory = tracker_factory
        self._tracker = None

    @property
    def tracker(self):
        if self._tracker is None:
            self._tracker = self.tracker_factory(
                config=self.stage1_config,
                weights_path=self.stage1_weights_path,
                tracker_checkpoint=self.tracker_checkpoint,
                device=self.device,
                resolution=self.resolution,
            )
            self._tracker._ensure_models()
        return self._tracker

    def _run_tracker(
        self,
        frames,
        prompts: list[str],
        *,
        strategy: str,
        stride: int,
        adaptive_health_threshold: float,
        adaptive_detector_threshold: float,
        adaptive_min_gap: int,
    ) -> tuple[list[dict], list[dict]]:
        tracker = self.tracker
        if strategy == "adaptive":
            initial_events = tracker._detect_prompt_events(frames, prompts, [0], reason="first")
            pass1_tracks = tracker._run_tracking_pass(frames, prompts, initial_events)
            candidate_frames_by_object = tracker._collect_adaptive_candidate_frames(
                pass1_tracks,
                initial_events,
                health_threshold=adaptive_health_threshold,
                min_gap=adaptive_min_gap,
            )
            adaptive_events = tracker._confirm_adaptive_prompt_events(
                frames,
                prompts,
                candidate_frames_by_object,
                detector_threshold=adaptive_detector_threshold,
            )
            existing_keys = {(int(event["frame_index"]), int(event["object_id"])) for event in initial_events}
            merged_events = list(initial_events)
            for event in adaptive_events:
                key = (int(event["frame_index"]), int(event["object_id"]))
                if key in existing_keys:
                    continue
                merged_events.append(event)
                existing_keys.add(key)
            tracks = tracker._run_tracking_pass(frames, prompts, merged_events)
            return tracks, merged_events

        frame_indices = select_prompt_frame_indices(len(frames), strategy=strategy, stride=stride)
        reason = "first" if strategy == "first" else "stride"
        prompt_events = tracker._detect_prompt_events(frames, prompts, frame_indices, reason=reason)
        tracks = tracker._run_tracking_pass(frames, prompts, prompt_events)
        return tracks, prompt_events

    def _pack_masks(self, tracks: list[dict], prompts: list[str], num_frames: int, original_size: tuple[int, int]) -> np.ndarray:
        height, width = original_size
        masks = np.zeros((num_frames, len(prompts), height, width), dtype=np.uint8)
        prompt_to_index = {prompt: index for index, prompt in enumerate(prompts)}
        for track in tracks:
            prompt_index = prompt_to_index[track["prompt"]]
            frame_index = int(track["frame_index"])
            mask = decode_binary_mask(track.get("mask_rle"))
            if mask is None:
                continue
            masks[frame_index, prompt_index] = mask.astype(np.uint8)
        return masks

    def predict_masks_for_frames(
        self,
        *,
        clip_frames,
        prompts: list[str],
        strategy: str = "adaptive",
        stride: int = 8,
        adaptive_health_threshold: float = 0.5,
        adaptive_detector_threshold: float = 0.5,
        adaptive_min_gap: int = 8,
    ) -> dict:
        tracks, prompt_events = self._run_tracker(
            clip_frames.images,
            prompts,
            strategy=strategy,
            stride=stride,
            adaptive_health_threshold=adaptive_health_threshold,
            adaptive_detector_threshold=adaptive_detector_threshold,
            adaptive_min_gap=adaptive_min_gap,
        )
        masks = self._pack_masks(
            tracks,
            prompts,
            num_frames=len(clip_frames.images),
            original_size=clip_frames.original_size,
        )
        return {
            "masks": masks,
            "tracks": tracks,
            "prompt_events": prompt_events,
            "prompts": list(prompts),
            "frame_names": list(clip_frames.frame_names),
            "original_size": tuple(clip_frames.original_size),
        }

    def build_cache(
        self,
        *,
        dataset: CBDDataset,
        record,
        prompts: list[str],
        strategy: str = "adaptive",
        stride: int = 8,
        adaptive_health_threshold: float = 0.5,
        adaptive_detector_threshold: float = 0.5,
        adaptive_min_gap: int = 8,
        keep_debug_artifacts: bool = False,
        target_fps: int = 5,
    ) -> Path:
        clip_frames = dataset.load_frames(record)
        tracks, prompt_events = self._run_tracker(
            clip_frames.images,
            prompts,
            strategy=strategy,
            stride=stride,
            adaptive_health_threshold=adaptive_health_threshold,
            adaptive_detector_threshold=adaptive_detector_threshold,
            adaptive_min_gap=adaptive_min_gap,
        )

        cache_dir = Path(record.mask_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        masks = self._pack_masks(tracks, prompts, num_frames=len(clip_frames.images), original_size=clip_frames.original_size)
        np.savez_compressed(
            cache_dir / "masks.npz",
            masks=masks,
            prompts=np.asarray(prompts),
            frame_names=np.asarray(clip_frames.frame_names),
            original_size=np.asarray(clip_frames.original_size, dtype=np.int64),
            strategy=np.asarray(strategy),
        )

        if keep_debug_artifacts:
            with open(cache_dir / "tracks.json", "w") as handle:
                json.dump({"tracks": tracks, "prompts": prompts, "strategy": strategy}, handle, indent=2)
            with open(cache_dir / "prompt_events.json", "w") as handle:
                json.dump({"prompt_events": prompt_events, "strategy": strategy}, handle, indent=2)

        return cache_dir / "masks.npz"


def build_cache_records_from_config(
    config: dict,
    split: str,
    *,
    sources: str | list[str] | tuple[str, ...] | None = None,
):
    dataset = CBDDataset(
        split=split,
        data_config=config.get("data", {}),
        target_type=config.get("data", {}).get("target_type", "hard"),
        sources=sources,
        clip_len=int(config.get("model", {}).get("clip_len", 25)),
        image_size=int(config.get("model", {}).get("input_size", config.get("model", {}).get("image_size", 384))),
        target_fps=int(config.get("data", {}).get("clip_fps", 5)),
    )
    return dataset, dataset.records
