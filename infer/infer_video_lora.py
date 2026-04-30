#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
INFER = ROOT / "infer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(INFER) not in sys.path:
    sys.path.insert(0, str(INFER))

from infer_lora import flatten_prompts, resolve_device, resolve_weights_path
from sam3.video_prompt_tracker import PromptedVideoTracker


def resolve_video_prompts(prompt_groups: list[list[str]] | None, data_config: dict) -> list[str]:
    if prompt_groups:
        prompts = flatten_prompts(prompt_groups)
    else:
        prompts = [str(prompt).strip() for prompt in data_config.get("class_names", []) if str(prompt).strip()]
    deduplicated = []
    seen = set()
    for prompt in prompts:
        normalized = prompt.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduplicated.append(prompt.strip())
    if not deduplicated:
        raise ValueError("No prompts were provided and no data.class_names were found in the config.")
    return deduplicated


def run_video_inference(args, tracker_factory=PromptedVideoTracker) -> dict:
    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)

    prompts = resolve_video_prompts(args.prompt, config.get("data", {}))
    weights_path = resolve_weights_path(config, args.weights)
    device = resolve_device(config, args.device)
    tracker_runner = tracker_factory(
        config=config,
        weights_path=weights_path,
        tracker_checkpoint=args.tracker_checkpoint,
        device=device.type,
        resolution=args.resolution,
    )
    return tracker_runner.run(
        video_path=args.video,
        output_dir=args.output_dir,
        prompts=prompts,
        strategy=args.strategy,
        stride=args.stride,
        adaptive_health_threshold=args.adaptive_health_threshold,
        adaptive_detector_threshold=args.adaptive_detector_threshold,
        adaptive_min_gap=args.adaptive_min_gap,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run mask-prompted SAM3 video tracking using the LoRA image detector as an external prompter.",
    )
    parser.add_argument("--config", required=True, help="Path to the training config YAML.")
    parser.add_argument("--video", required=True, help="Path to an MP4/video file or a directory of ordered frames.")
    parser.add_argument("--output-dir", required=True, help="Directory where tracking outputs will be written.")
    parser.add_argument("--weights", help="Path to the LoRA weights. Defaults to output_dir/best_lora_weights.pt.")
    parser.add_argument(
        "--tracker-checkpoint",
        help="Optional path to the official SAM3 checkpoint used to initialize the tracker. Defaults to facebook/sam3.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        nargs="+",
        help='Optional prompt override, e.g. --prompt "gallbladder" --prompt "liver". Defaults to data.class_names.',
    )
    parser.add_argument("--device", help='Override the config device, e.g. "cuda".')
    parser.add_argument("--resolution", type=int, default=1008, help="Square input resolution for the LoRA image detector.")
    parser.add_argument(
        "--strategy",
        choices=("first", "stride", "adaptive"),
        default="first",
        help="Prompt-frame strategy to use.",
    )
    parser.add_argument("--stride", type=int, default=16, help="Frame stride for the stride strategy.")
    parser.add_argument(
        "--adaptive-health-threshold",
        type=float,
        default=0.5,
        help="Tracker-health threshold below which adaptive re-prompting is triggered.",
    )
    parser.add_argument(
        "--adaptive-detector-threshold",
        type=float,
        default=0.5,
        help="Minimum detector score required to accept an adaptive re-prompt.",
    )
    parser.add_argument(
        "--adaptive-min-gap",
        type=int,
        default=8,
        help="Minimum frame gap between adaptive re-prompt events for the same object.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    summary = run_video_inference(args)
    print(f"Saved tracks to {summary['tracks_path']}")
    print(f"Saved prompt events to {summary['prompt_events_path']}")
    print(f"Saved overlays to {summary['overlays_dir']}")
    if summary["overlay_video_written"]:
        print(f"Saved overlay video to {summary['overlay_video_path']}")
    else:
        print("Overlay video was not written; per-frame overlays are still available.")


if __name__ == "__main__":
    main()
