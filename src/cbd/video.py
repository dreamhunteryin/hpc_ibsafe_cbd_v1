from __future__ import annotations

from dataclasses import dataclass
import json
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from .sources import CBDSourceConfig


REFERENCE_IMAGE_SIZE = (320, 180)
FRAME_RESOLUTION_CACHE_NAME = "frame_resolution_cache.json"


@dataclass(frozen=True)
class ResolvedVideoFrame:
    video_path: Path
    frame_index: int


def _resolve_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(
            f"Could not find {name!r} on PATH. Please activate an environment that provides it."
        )
    return path


def resolve_ffmpeg_binary() -> str:
    return _resolve_binary("ffmpeg")


def resolve_ffprobe_binary() -> str:
    return _resolve_binary("ffprobe")


def parse_rate(value: str) -> float:
    value = str(value).strip()
    if not value:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video_fps(video_path: str | Path) -> float:
    ffprobe = resolve_ffprobe_binary()
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rates = [parse_rate(line) for line in result.stdout.splitlines()]
    fps = max(rates) if rates else 0.0
    if fps <= 0.0:
        raise RuntimeError(f"Could not recover FPS for video {video_path}.")
    return fps


def _load_cache(cache_path: Path) -> dict[str, dict[str, str | int]]:
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path, "r") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _save_cache(cache_path: Path, payload: dict[str, dict[str, str | int]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def cache_path_for_source(source_config: CBDSourceConfig) -> Path:
    return source_config.clips_root.parent / FRAME_RESOLUTION_CACHE_NAME


def load_frame_resolution_from_cache(
    source_config: CBDSourceConfig,
    file_name: str,
) -> ResolvedVideoFrame | None:
    payload = _load_cache(cache_path_for_source(source_config))
    cached = payload.get(str(file_name))
    if not isinstance(cached, dict):
        return None
    video_path = cached.get("video_path")
    frame_index = cached.get("frame_index")
    if video_path is None or frame_index is None:
        return None
    return ResolvedVideoFrame(video_path=Path(str(video_path)), frame_index=int(frame_index))


def save_frame_resolution_to_cache(
    source_config: CBDSourceConfig,
    file_name: str,
    resolved: ResolvedVideoFrame,
) -> None:
    cache_path = cache_path_for_source(source_config)
    payload = _load_cache(cache_path)
    payload[str(file_name)] = {
        "video_path": str(resolved.video_path),
        "frame_index": int(resolved.frame_index),
    }
    try:
        _save_cache(cache_path, payload)
    except OSError:
        return


def reference_frames_dir(source_config: CBDSourceConfig, video_id: int) -> Path | None:
    if source_config.reference_frames_root is None:
        return None
    path = source_config.reference_frames_root / str(video_id)
    if not path.exists():
        return None
    return path


def _rank_map(values: list[float], reverse: bool = False) -> dict[int, int]:
    order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=reverse)
    return {idx: rank for rank, idx in enumerate(order)}


def _load_reference_candidates(
    source_config: CBDSourceConfig,
    video_id: int,
) -> list[dict]:
    frames_dir = reference_frames_dir(source_config, video_id)
    if frames_dir is None:
        return []

    annotation_path = frames_dir / "annotations_file.json"
    candidates: list[dict] = []
    if annotation_path.exists():
        with open(annotation_path, "r") as handle:
            coco = json.load(handle)
        image_by_id = {int(image["id"]): image for image in coco.get("images", [])}
        annotations_by_image_id: dict[int, list[dict]] = {}
        for annotation in coco.get("annotations", []):
            annotations_by_image_id.setdefault(int(annotation["image_id"]), []).append(annotation)
        for image in coco.get("images", []):
            file_path = frames_dir / Path(str(image["file_name"])).name
            if not file_path.exists():
                continue
            frame_index = int(Path(file_path).stem.split("_")[-1])
            candidates.append(
                {
                    "frame_index": frame_index,
                    "path": file_path,
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "annotations": annotations_by_image_id.get(int(image["id"]), []),
                }
            )
        if candidates:
            return candidates

    for file_path in sorted(frames_dir.glob("*.jpg")):
        frame_index = int(file_path.stem.split("_")[-1])
        candidates.append(
            {
                "frame_index": frame_index,
                "path": file_path,
                "width": 1280,
                "height": 720,
                "annotations": [],
            }
        )
    return candidates


def _normalize_image_for_scoring(image: PILImage.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB").resize(REFERENCE_IMAGE_SIZE, resample=PILImage.BILINEAR), dtype=np.float32)


def _candidate_box_distance(
    candidate: dict,
    target_box_cxcywh: tuple[float, float, float, float] | np.ndarray | None,
) -> float:
    if target_box_cxcywh is None:
        return 0.0

    target = np.asarray(target_box_cxcywh, dtype=np.float32)
    candidate_boxes = []
    for annotation in candidate.get("annotations", []):
        x, y, w, h = [float(value) for value in annotation["bbox"]]
        candidate_boxes.append(
            np.asarray(
                (
                    (x + w / 2.0) / float(candidate["width"]),
                    (y + h / 2.0) / float(candidate["height"]),
                    w / float(candidate["width"]),
                    h / float(candidate["height"]),
                ),
                dtype=np.float32,
            )
        )
    if not candidate_boxes:
        return 1.0
    return min(float(np.square(target - box).sum()) for box in candidate_boxes)


def resolve_icg_reference_frame_index(
    source_config: CBDSourceConfig,
    *,
    file_name: str,
    image_path: Path,
    video_id: int,
    target_box_cxcywh: tuple[float, float, float, float] | np.ndarray | None = None,
    fallback_frame_token: int | None = None,
) -> int:
    candidates = _load_reference_candidates(source_config, video_id)
    if not candidates:
        if fallback_frame_token is not None:
            return int(fallback_frame_token)
        raise FileNotFoundError(
            f"Could not find reference stills for video_id={video_id} under {source_config.reference_frames_root}."
        )

    with PILImage.open(image_path) as target_image:
        target_repr = _normalize_image_for_scoring(target_image)

    image_scores: list[float] = []
    box_scores: list[float] = []
    for candidate in candidates:
        with PILImage.open(candidate["path"]) as reference_image:
            reference_repr = _normalize_image_for_scoring(reference_image)
        image_scores.append(float(np.abs(target_repr - reference_repr).mean()))
        box_scores.append(_candidate_box_distance(candidate, target_box_cxcywh))

    image_ranks = _rank_map(image_scores)
    box_ranks = _rank_map(box_scores)
    scored = sorted(
        (
            image_ranks[index] + 2 * box_ranks[index],
            box_scores[index],
            image_scores[index],
            candidates[index]["frame_index"],
        )
        for index in range(len(candidates))
    )
    return int(scored[0][-1])


def _candidate_video_paths(source_config: CBDSourceConfig, metadata: dict) -> list[Path]:
    if source_config.videos_root is None:
        raise ValueError(f"Source {source_config.name!r} does not define a videos_root.")

    file_name = str(metadata.get("file_name", ""))
    stem = Path(file_name).stem
    family = str(metadata.get("file_family", "")).strip().lower()
    video_id = metadata.get("video_id")
    if video_id is None:
        raise ValueError(f"Could not recover a video_id for {file_name!r}.")
    video_id = int(video_id)

    dated_video_candidates = sorted(source_config.videos_root.glob(f"*Video{video_id}_output.mp4"))
    candidate_names: list[str] = []
    if family == "video_dated":
        video_key = str(metadata.get("video_key", ""))
        date_token = video_key.rsplit("_Video", 1)[0].replace("_", "-")
        candidate_names.extend(
            [
                f"{date_token}_Video{video_id}_output.mp4",
                *[path.name for path in dated_video_candidates],
                f"CMRP_{video_id}.mp4",
                f"DL_{video_id}.mp4",
            ]
        )
    elif family == "photo":
        candidate_names.extend(
            [
                *[path.name for path in dated_video_candidates],
                f"CMRP_{video_id}.mp4",
                f"DL_{video_id}.mp4",
            ]
        )
    elif family == "dl_cmrp":
        candidate_names.extend([f"DL_{video_id}.mp4", f"CMRP_{video_id}.mp4"])
    elif family == "vid_mp4":
        if stem.startswith("2024-"):
            candidate_names.extend([f"DL_{video_id}.mp4", f"CMRP_{video_id}.mp4"])
        else:
            candidate_names.extend([f"CMRP_{video_id}.mp4", f"DL_{video_id}.mp4"])
    else:
        candidate_names.extend([f"CMRP_{video_id}.mp4", f"DL_{video_id}.mp4", *[path.name for path in dated_video_candidates]])

    resolved: list[Path] = []
    seen: set[Path] = set()
    for name in candidate_names:
        path = source_config.videos_root / name
        if path.exists() and path not in seen:
            resolved.append(path)
            seen.add(path)
    return resolved


def resolve_video_frame_location(
    dataset,
    source_config: CBDSourceConfig,
    metadata: dict,
    *,
    target_box_cxcywh: tuple[float, float, float, float] | np.ndarray | None = None,
) -> ResolvedVideoFrame:
    file_name = str(metadata.get("file_name", ""))
    cached = load_frame_resolution_from_cache(source_config, file_name)
    if cached is not None and cached.video_path.exists():
        return cached

    candidate_video_paths = _candidate_video_paths(source_config, metadata)
    if not candidate_video_paths:
        raise FileNotFoundError(
            f"Could not find any source video for {file_name!r} under {source_config.videos_root}."
        )

    frame_index = metadata.get("frame_id")
    if frame_index is not None:
        resolved = ResolvedVideoFrame(video_path=candidate_video_paths[0], frame_index=int(frame_index))
        save_frame_resolution_to_cache(source_config, file_name, resolved)
        return resolved

    image_path = dataset.context.resolve_image_path(file_name)
    frame_index = resolve_icg_reference_frame_index(
        source_config,
        file_name=file_name,
        image_path=image_path,
        video_id=int(metadata["video_id"]),
        target_box_cxcywh=target_box_cxcywh,
        fallback_frame_token=metadata.get("frame_token"),
    )
    resolved = ResolvedVideoFrame(video_path=candidate_video_paths[0], frame_index=int(frame_index))
    save_frame_resolution_to_cache(source_config, file_name, resolved)
    return resolved


def _select_filter_expression(frame_indices: list[int]) -> str:
    unique_indices = sorted({int(frame_index) for frame_index in frame_indices if int(frame_index) >= 0})
    if not unique_indices:
        raise ValueError("Expected at least one non-negative frame index.")
    return "select=" + "+".join(f"eq(n\\,{frame_index})" for frame_index in unique_indices)


def load_frames_from_video(
    video_path: str | Path,
    frame_indices: list[int],
) -> list[PILImage.Image]:
    unique_indices = sorted({int(frame_index) for frame_index in frame_indices if int(frame_index) >= 0})
    if not unique_indices:
        return []

    ffmpeg = resolve_ffmpeg_binary()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        output_pattern = tmpdir_path / "frame_%06d.png"
        subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-vf",
                _select_filter_expression(unique_indices),
                "-vsync",
                "0",
                str(output_pattern),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        frame_paths = sorted(tmpdir_path.glob("frame_*.png"))
        if len(frame_paths) != len(unique_indices):
            raise RuntimeError(
                f"Expected {len(unique_indices)} extracted frames from {video_path}, got {len(frame_paths)}."
            )
        images: list[PILImage.Image] = []
        for path in frame_paths:
            with PILImage.open(path) as image:
                images.append(image.convert("RGB").copy())
        return images


def export_frames_to_dir(
    video_path: str | Path,
    frame_indices: list[int],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in output_dir.glob("frame_*.png"):
            path.unlink()

    images = load_frames_from_video(video_path, frame_indices)
    written_paths: list[Path] = []
    for frame_index, image in zip(sorted({int(frame_index) for frame_index in frame_indices if int(frame_index) >= 0}), images):
        output_path = output_dir / f"frame_{frame_index}.png"
        image.save(output_path)
        written_paths.append(output_path)
    return written_paths


def encode_video_from_frames(
    frame_paths: list[Path],
    output_path: str | Path,
    *,
    frame_rate: int,
) -> Path:
    output_path = Path(output_path)
    ffmpeg = resolve_ffmpeg_binary()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        for index, frame_path in enumerate(frame_paths, start=1):
            shutil.copy2(frame_path, tmpdir_path / f"{index:06d}.png")
        subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(int(frame_rate)),
                "-i",
                str(tmpdir_path / "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    return output_path
