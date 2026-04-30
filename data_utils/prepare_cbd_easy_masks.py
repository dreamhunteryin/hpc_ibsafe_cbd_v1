#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import time
import sys
from pathlib import Path

from tqdm.auto import tqdm

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'codex')}")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
INFER = ROOT / "infer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(INFER) not in sys.path:
    sys.path.insert(0, str(INFER))

from cbd.cache import EasyMaskCacheBuilder, build_cache_records_from_config, load_stage1_config
from cbd.engine import load_config


def resolve_relative(base_path: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_path.parent / path).resolve())


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache SAM3 easy-class masks for the CBD stage-2 dataset.")
    parser.add_argument("--config", required=True, help="Path to the stage-2 CBD YAML config.")
    parser.add_argument("--split", required=True, help="Single dataset split to cache.")
    parser.add_argument(
        "--source",
        action="append",
        help="Dataset source(s) to cache, for example: --source bsafe --source icglceaes. Defaults to the config selection for the split.",
    )
    parser.add_argument("--clip-start", type=int, default=0, help="Inclusive clip index to start from.")
    parser.add_argument("--clip-end", type=int, help="Exclusive clip index to stop at. Defaults to the end of the split.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing cache files.")
    parser.add_argument("--keep-debug-artifacts", action="store_true", help="Keep tracker JSON artifacts next to masks.npz.")
    parser.add_argument("--device", help="Override the stage-1 tracker device.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    data_config = config["data"]
    stage1_cfg = config["stage1_sam3"]

    stage1_config_path = resolve_relative(config_path, stage1_cfg.get("config_path"))
    if stage1_config_path is None:
        raise ValueError("stage1_sam3.config_path must be set in the CBD config.")
    stage1_weights_path = resolve_relative(config_path, stage1_cfg.get("weights_path"))
    tracker_checkpoint = resolve_relative(config_path, stage1_cfg.get("tracker_checkpoint"))
    prompts = [str(prompt).strip() for prompt in stage1_cfg.get("easy_prompts", []) if str(prompt).strip()]
    if len(prompts) != 2:
        raise ValueError("stage1_sam3.easy_prompts must contain exactly two prompts.")

    builder = EasyMaskCacheBuilder(
        stage1_config=load_stage1_config(stage1_config_path),
        stage1_weights_path=stage1_weights_path,
        tracker_checkpoint=tracker_checkpoint,
        device=args.device or stage1_cfg.get("device", config.get("hardware", {}).get("device", "cuda")),
        resolution=int(stage1_cfg.get("resolution", 1008)),
    )

    split = str(args.split)
    dataset, records = build_cache_records_from_config(config, split, sources=args.source)
    total_records = len(records)
    clip_start = int(args.clip_start)
    clip_end = total_records if args.clip_end is None else int(args.clip_end)
    if clip_start < 0:
        raise ValueError("--clip-start must be non-negative.")
    if clip_start > total_records:
        raise ValueError(f"--clip-start ({clip_start}) exceeds the number of clips in split {split!r} ({total_records}).")
    if clip_end < clip_start:
        raise ValueError("--clip-end must be greater than or equal to --clip-start.")
    if clip_end > total_records:
        raise ValueError(f"--clip-end ({clip_end}) exceeds the number of clips in split {split!r} ({total_records}).")
    records = records[clip_start:clip_end]

    split_start = time.time()
    created = 0
    skipped = 0
    remaining = sum(
        1
        for record in records
        if args.overwrite or not record.mask_cache_path.exists()
    )
    pbar = tqdm(records, desc=f"Caching {split}[{clip_start}:{clip_end}] masks")
    for record in pbar:
        if record.mask_cache_path.exists() and not args.overwrite:
            skipped += 1
            elapsed = time.time() - split_start
            avg = elapsed / max(1, created)
            eta = avg * remaining if created > 0 else 0.0
            pbar.set_postfix(
                created=created,
                skipped=skipped,
                remaining=remaining,
                eta=f"{eta / 60.0:.1f}m",
            )
            continue
        builder.build_cache(
            dataset=dataset,
            record=record,
            prompts=prompts,
            strategy=str(stage1_cfg.get("strategy", "adaptive")),
            stride=int(stage1_cfg.get("stride", 8)),
            adaptive_health_threshold=float(stage1_cfg.get("adaptive_health_threshold", 0.5)),
            adaptive_detector_threshold=float(stage1_cfg.get("adaptive_detector_threshold", 0.5)),
            adaptive_min_gap=int(stage1_cfg.get("adaptive_min_gap", 8)),
            keep_debug_artifacts=bool(args.keep_debug_artifacts),
            target_fps=int(data_config.get("clip_fps", 5)),
        )
        created += 1
        remaining -= 1
        elapsed = time.time() - split_start
        avg = elapsed / max(1, created)
        eta = avg * remaining
        pbar.set_postfix(
            created=created,
            skipped=skipped,
            remaining=remaining,
            eta=f"{eta / 60.0:.1f}m",
        )
    pbar.close()
    elapsed = time.time() - split_start
    print(
        f"{split}[{clip_start}:{clip_end}] / {total_records}: cached {created} mask files, "
        f"skipped {skipped}, elapsed {elapsed / 60.0:.1f}m"
    )


if __name__ == "__main__":
    main()
