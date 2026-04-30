from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.dataset_bsafe import (
    BsafeDataset,
    DEFAULT_ROOT as BSAFE_DEFAULT_ROOT,
    parse_frame_id as parse_bsafe_frame_id,
)
from data.dataset_camma import (
    CammaDataset,
    DEFAULT_ROOT as CAMMA_DEFAULT_ROOT,
    default_bbox_anchor_for_dataset,
)


LEGACY_SOURCE_NAME = "bsafe"
ICG_SOURCE_NAME = "icglceaes"


CBDSourceDataset = BsafeDataset | CammaDataset


@dataclass(frozen=True)
class CBDSourceConfig:
    name: str
    kind: str
    dataset_root: Path
    dataset_name: str
    annotation_file: str | Path | None
    bbox_anchor: str
    clips_root: Path
    easy_mask_cache_root: Path
    split_map: dict[str, str]
    target_category_name: str | None = None
    target_category_id: int | None = None
    videos_root: Path | None = None
    reference_frames_root: Path | None = None


def normalize_source_key(value: str) -> str:
    normalized = "".join(character for character in str(value).strip().lower() if character.isalnum())
    if normalized == "icglceaes":
        return ICG_SOURCE_NAME
    return normalized


def normalize_requested_sources(
    value: str | list[str] | tuple[str, ...] | None,
    available_sources: dict[str, CBDSourceConfig],
) -> tuple[str, ...]:
    available_names = tuple(available_sources)
    if value is None:
        return available_names

    if isinstance(value, str):
        raw_parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw_parts = [str(part).strip() for part in value if str(part).strip()]

    normalized_parts: list[str] = []
    for part in raw_parts:
        if part.lower() in {"all", "both", "*"}:
            return available_names
        normalized = normalize_source_key(part)
        if normalized not in available_sources:
            raise ValueError(
                f"Unsupported source={part!r}. Expected one of: {tuple(available_sources)} or 'both'."
            )
        if normalized not in normalized_parts:
            normalized_parts.append(normalized)

    if not normalized_parts:
        return available_names
    return tuple(normalized_parts)


def _resolve_split_map(global_data_config: dict[str, Any], source_data_config: dict[str, Any]) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        split_map[split_name] = str(
            source_data_config.get(
                f"{split_name}_split",
                global_data_config.get(f"{split_name}_split", split_name),
            )
        )
    return split_map


def _infer_source_kind(source_name: str, source_data_config: dict[str, Any]) -> str:
    if source_name == LEGACY_SOURCE_NAME:
        return "bsafe"
    dataset_name = str(source_data_config.get("dataset_name", "")).strip()
    if dataset_name and dataset_name.lower() != "bsafe":
        return "camma"
    return "camma"


def resolve_source_configs(data_config: dict[str, Any]) -> dict[str, CBDSourceConfig]:
    datasets_config = data_config.get("datasets")
    if not datasets_config:
        dataset_root = Path(data_config.get("dataset_root", BSAFE_DEFAULT_ROOT))
        return {
            LEGACY_SOURCE_NAME: CBDSourceConfig(
                name=LEGACY_SOURCE_NAME,
                kind="bsafe",
                dataset_root=dataset_root,
                dataset_name="Bsafe",
                annotation_file=data_config.get("annotation_file"),
                bbox_anchor="topleft",
                clips_root=Path(data_config.get("clips_root", dataset_root / "clips")),
                easy_mask_cache_root=Path(
                    data_config.get("easy_mask_cache_root", dataset_root / "cbd_easy_masks")
                ),
                split_map=_resolve_split_map(data_config, {}),
                videos_root=Path(data_config.get("videos_root", dataset_root / "videos")),
                reference_frames_root=Path(
                    data_config.get("reference_frames_root", dataset_root)
                ),
            )
        }

    resolved: dict[str, CBDSourceConfig] = {}
    for raw_name, raw_source_config in datasets_config.items():
        source_config = dict(raw_source_config or {})
        source_name = normalize_source_key(raw_name)
        if not source_name:
            raise ValueError(f"Invalid CBD source name: {raw_name!r}")
        kind = str(source_config.get("kind", _infer_source_kind(source_name, source_config))).strip().lower()

        if kind == "bsafe":
            dataset_root = Path(source_config.get("dataset_root", data_config.get("dataset_root", BSAFE_DEFAULT_ROOT)))
            clips_root = Path(source_config.get("clips_root", data_config.get("clips_root", dataset_root / "clips")))
            easy_mask_cache_root = Path(
                source_config.get(
                    "easy_mask_cache_root",
                    data_config.get("easy_mask_cache_root", dataset_root / "cbd_easy_masks"),
                )
            )
            resolved[source_name] = CBDSourceConfig(
                name=source_name,
                kind="bsafe",
                dataset_root=dataset_root,
                dataset_name=str(source_config.get("dataset_name", "Bsafe")),
                annotation_file=source_config.get("annotation_file", data_config.get("annotation_file")),
                bbox_anchor="topleft",
                clips_root=clips_root,
                easy_mask_cache_root=easy_mask_cache_root,
                split_map=_resolve_split_map(data_config, source_config),
                videos_root=Path(source_config.get("videos_root", data_config.get("videos_root", dataset_root / "videos"))),
                reference_frames_root=Path(
                    source_config.get(
                        "reference_frames_root",
                        data_config.get("reference_frames_root", dataset_root),
                    )
                ),
            )
            continue

        if kind != "camma":
            raise ValueError(f"Unsupported CBD source kind={kind!r} for source {raw_name!r}.")

        dataset_name = str(source_config.get("dataset_name", raw_name))
        dataset_root = Path(source_config.get("dataset_root", CAMMA_DEFAULT_ROOT))
        dataset_dir = dataset_root / dataset_name
        default_clips_root = dataset_dir / "cbd_v2" / "clips"
        default_easy_mask_cache_root = dataset_dir / "cbd_v2" / "easy_masks"
        videos_root_value = source_config.get("videos_root", data_config.get("videos_root"))
        reference_frames_root_value = source_config.get(
            "reference_frames_root",
            data_config.get(
                "reference_frames_root",
                None if videos_root_value is None else Path(videos_root_value).parent,
            ),
        )
        resolved[source_name] = CBDSourceConfig(
            name=source_name,
            kind="camma",
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            annotation_file=source_config.get("annotation_file", data_config.get("annotation_file", "annotation_coco.json")),
            bbox_anchor=str(
                source_config.get("bbox_anchor", default_bbox_anchor_for_dataset(dataset_name))
            ),
            clips_root=Path(source_config.get("clips_root", default_clips_root)),
            easy_mask_cache_root=Path(
                source_config.get("easy_mask_cache_root", default_easy_mask_cache_root)
            ),
            split_map=_resolve_split_map(data_config, source_config),
            target_category_name=source_config.get("target_category", source_config.get("target_category_name", "CBD")),
            target_category_id=(
                None
                if source_config.get("target_category_id") is None
                else int(source_config["target_category_id"])
            ),
            videos_root=None if videos_root_value is None else Path(videos_root_value),
            reference_frames_root=(
                None if reference_frames_root_value is None else Path(reference_frames_root_value)
            ),
        )

    return resolved


def resolve_phase_sources(
    data_config: dict[str, Any],
    phase: str,
    available_sources: dict[str, CBDSourceConfig],
) -> tuple[str, ...]:
    for key in (f"{phase}_sources", f"{phase}_datasets", f"{phase}_dataset"):
        if key in data_config:
            return normalize_requested_sources(data_config.get(key), available_sources)
    return tuple(available_sources)


def resolve_source_split(source_config: CBDSourceConfig, split: str) -> str:
    return str(source_config.split_map.get(str(split), str(split)))


def build_source_dataset(source_config: CBDSourceConfig, split: str) -> CBDSourceDataset:
    resolved_split = resolve_source_split(source_config, split)
    if source_config.kind == "bsafe":
        return BsafeDataset(
            root_dir=source_config.dataset_root,
            dataset_name=source_config.dataset_name,
            split=resolved_split,
            annotation_file=source_config.annotation_file,
        )
    return CammaDataset(
        root_dir=source_config.dataset_root,
        dataset_name=source_config.dataset_name,
        split=resolved_split,
        annotation_file=str(source_config.annotation_file or "annotation_coco.json"),
    )


def metadata_video_key(source_config: CBDSourceConfig, metadata: dict[str, Any]) -> str:
    if source_config.kind == "camma":
        video_key = str(metadata.get("video_key", "")).strip()
        if video_key:
            return video_key

    video_id = metadata.get("video_id")
    if video_id is not None:
        return str(int(video_id))

    stem = Path(str(metadata.get("file_name", ""))).stem
    prefix, separator, _ = stem.rpartition("_")
    if separator:
        return prefix
    return stem


def metadata_frame_index(source_config: CBDSourceConfig, metadata: dict[str, Any]) -> int | None:
    frame_id = metadata.get("frame_id")
    if frame_id is not None:
        return int(frame_id)

    file_name = str(metadata.get("file_name", ""))
    if source_config.kind == "bsafe":
        return parse_bsafe_frame_id(file_name)

    if str(source_config.dataset_name).strip() == "ICG-LC-EAES":
        return None

    stem = Path(file_name).stem
    _, separator, trailing = stem.rpartition("_")
    if not separator:
        trailing = stem
    digits = "".join(character for character in trailing if character.isdigit())
    if not digits:
        return None
    return int(digits)


def build_clip_key(source_config: CBDSourceConfig, metadata: dict[str, Any]) -> str:
    if source_config.kind == "bsafe":
        video_id = metadata.get("video_id")
        frame_id = metadata_frame_index(source_config, metadata)
        if video_id is not None and frame_id is not None:
            return f"{int(video_id)}_{int(frame_id)}"
    return Path(str(metadata["file_name"])).stem


def build_record_clip_id(clip_key: str, annotation_id: int, duplicate_count: int) -> str:
    if duplicate_count <= 1:
        return clip_key
    return f"{clip_key}_ann{int(annotation_id)}"


def build_clip_dir(clips_root: str | Path, clip_key: str) -> Path:
    return Path(clips_root) / clip_key


def build_mask_cache_dir(mask_cache_root: str | Path, split: str, clip_key: str) -> Path:
    return Path(mask_cache_root) / str(split) / clip_key


def resolve_target_category_ids(source_config: CBDSourceConfig, dataset: CBDSourceDataset) -> tuple[int, ...]:
    if source_config.kind == "bsafe":
        return ()
    if source_config.target_category_id is not None:
        return (int(source_config.target_category_id),)

    target_name = str(source_config.target_category_name or "").strip().lower()
    category_ids = tuple(
        sorted(
            category_id
            for category_id, category_name in dataset.context.category_id_to_name.items()
            if str(category_name).strip().lower() == target_name
        )
    )
    if not category_ids:
        raise ValueError(
            f"Could not find target category {source_config.target_category_name!r} "
            f"in dataset {source_config.dataset_name!r}."
        )
    return category_ids
