#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
INFER = ROOT / "infer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(INFER) not in sys.path:
    sys.path.insert(0, str(INFER))

from cbd_rtdetrv4.engine import CBDRTDetrV4Trainer, load_config
from cbd_rtdetrv4.inference import write_prediction_artifacts


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lightweight RT-DETRv4 CBD visual inference.")
    parser.add_argument("--config", required=True, help="Path to the RT-DETRv4 CBD YAML config.")
    parser.add_argument("--weights", help="Optional checkpoint path. Defaults to output_dir/best_cbd_rtdetrv4.pt.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--split", default="test", help="Dataset split for --sample-index mode.")
    parser.add_argument("--sample-index", type=int, help="Dataset sample index for BSAFE visual inspection.")
    parser.add_argument("--image-path", help="Arbitrary image path for single-image inference.")
    parser.add_argument("--score-threshold", type=float, default=None, help="Optional score threshold override.")
    parser.add_argument("--show-gt", action="store_true", help="Overlay the ground-truth box in sample mode.")
    return parser


def resolve_weights_path(config: dict, weights_path: str | None) -> Path:
    if weights_path is not None:
        return Path(weights_path)
    return Path(config.get("output", {}).get("output_dir", "outputs/bsafe_cbd_rtdetrv4")) / "best_cbd_rtdetrv4.pt"


def run_inference(args, trainer_cls=CBDRTDetrV4Trainer) -> dict:
    if (args.sample_index is None) == (args.image_path is None):
        raise ValueError("Pass exactly one of `--sample-index` or `--image-path`.")

    config = load_config(args.config)
    trainer = trainer_cls(config)
    trainer.load_checkpoint(resolve_weights_path(config, args.weights))

    if args.sample_index is not None:
        dataset = trainer.build_dataset(args.split, train=False)
        prediction = trainer.predict_dataset_index(
            dataset,
            args.sample_index,
            score_threshold=args.score_threshold,
        )
    else:
        prediction = trainer.predict_image_path(
            args.image_path,
            score_threshold=args.score_threshold,
        )

    item = write_prediction_artifacts(prediction, args.output, show_gt=bool(args.show_gt))
    return {"mode": prediction["mode"], "count": 1, "items": [item]}


def main() -> None:
    args = build_argparser().parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
