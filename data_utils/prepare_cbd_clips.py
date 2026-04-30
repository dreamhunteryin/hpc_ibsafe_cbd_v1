#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cbd.common import DEFAULT_CLIP_LEN, sample_clip_frame_indices
from cbd.dataset import CBDRecord, build_cbd_records
from cbd.sources import normalize_requested_sources, resolve_phase_sources, resolve_source_configs
from cbd.video import (
    encode_video_from_frames,
    export_frames_to_dir,
    probe_video_fps,
    resolve_video_frame_location,
)


def load_yaml_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def deduplicate_records(records: list[CBDRecord]) -> list[CBDRecord]:
    unique_by_clip: dict[tuple[str, str], CBDRecord] = {}
    for record in records:
        unique_by_clip.setdefault((record.source_name, record.clip_key), record)
    return list(unique_by_clip.values())


def resolve_requested_sources(
    *,
    data_config: dict,
    split: str,
    raw_sources: list[str] | None,
) -> tuple[str, ...]:
    source_configs = resolve_source_configs(data_config)
    if raw_sources:
        return normalize_requested_sources(raw_sources, source_configs)
    return resolve_phase_sources(data_config, split, source_configs)


def export_split_clips(
    *,
    config: dict,
    split: str,
    target_type: str,
    clip_len: int,
    target_fps: int,
    limit: int | None,
    overwrite: bool,
    requested_sources: list[str] | None,
) -> int:
    data_config = dict(config.get("data", {}))
    sources = resolve_requested_sources(
        data_config=data_config,
        split=split,
        raw_sources=requested_sources,
    )
    datasets_by_source, records = build_cbd_records(
        split=split,
        data_config=data_config,
        target_type=target_type,
        sources=sources,
    )
    records = deduplicate_records(records)
    if limit is not None:
        records = records[:limit]

    exported = 0
    for record in tqdm(records, desc=f"Exporting {split} clips"):
        output_dir = record.clip_dir
        existing_frames = list(output_dir.glob("frame_*.png"))
        if existing_frames and not overwrite:
            continue

        source_dataset = datasets_by_source[record.source_name]
        resolved = resolve_video_frame_location(
            source_dataset,
            record.source_config,
            record.metadata,
            target_box_cxcywh=record.target_box,
        )
        fps = max(1, int(round(probe_video_fps(resolved.video_path))))
        frame_indices = sample_clip_frame_indices(
            target_frame_id=resolved.frame_index,
            fps=fps,
            clip_len=clip_len,
            target_fps=target_fps,
        )
        valid_indices = [frame_index for frame_index in frame_indices if frame_index >= 0]
        if not valid_indices:
            continue

        if overwrite:
            for path in output_dir.glob("frame_*.png"):
                path.unlink()
            clip_path = output_dir / "clip.mp4"
            if clip_path.exists():
                clip_path.unlink()

        written_frames = export_frames_to_dir(
            resolved.video_path,
            valid_indices,
            output_dir,
            overwrite=False,
        )
        if written_frames:
            encode_video_from_frames(
                written_frames,
                output_dir / "clip.mp4",
                frame_rate=target_fps,
            )
            exported += 1

    return exported


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export CBD clips for the requested dataset split(s) and source annotation set(s)."
    )
    parser.add_argument("--config", help="Optional stage-2 YAML config for defaults.")
    parser.add_argument("--split", action="append", help="Split to export. Can be repeated. Defaults to train,val,test.")
    parser.add_argument(
        "--source",
        action="append",
        help="Dataset source(s) to export, for example: --source bsafe --source icglceaes. Defaults to the config selection for each split.",
    )
    parser.add_argument("--target-type", default=None, help="BSAFE annotation type to export. Defaults to config target_type.")
    parser.add_argument("--clip-len", type=int, default=None, help="Number of frames per clip. Defaults to the model clip length.")
    parser.add_argument("--target-fps", type=int, default=None, help="Output clip FPS. Defaults to data.clip_fps.")
    parser.add_argument("--limit", type=int, help="Optional max number of unique clips per split.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite clip folders if they already exist.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = load_yaml_config(args.config)
    data_config = config.get("data", {})
    model_config = config.get("model", {})

    splits = args.split or [
        data_config.get("train_split", "train"),
        data_config.get("val_split", "val"),
        data_config.get("test_split", "test"),
    ]
    target_type = args.target_type or data_config.get("target_type", "hard")
    clip_len = args.clip_len or int(model_config.get("clip_len", DEFAULT_CLIP_LEN))
    target_fps = args.target_fps or int(data_config.get("clip_fps", 5))

    for split in splits:
        exported = export_split_clips(
            config=config,
            split=split,
            target_type=target_type,
            clip_len=clip_len,
            target_fps=target_fps,
            limit=args.limit,
            overwrite=args.overwrite,
            requested_sources=args.source,
        )
        print(f"{split}: exported {exported} clips")


if __name__ == "__main__":
    main()
