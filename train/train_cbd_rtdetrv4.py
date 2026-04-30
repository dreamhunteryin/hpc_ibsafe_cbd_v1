#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cbd_rtdetrv4.engine import CBDRTDetrV4Trainer, cleanup_distributed, is_main_process, load_config


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate the RT-DETRv4 CBD detector.")
    parser.add_argument("--config", required=True, help="Path to the RT-DETRv4 CBD YAML config.")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only.")
    parser.add_argument("--eval-split", default="val", help="Split to evaluate when --eval-only is used.")
    parser.add_argument(
        "--weights",
        help="Optional checkpoint path. Defaults to output_dir/best_cbd_rtdetrv4.pt.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args.config)
    trainer = CBDRTDetrV4Trainer(config)
    default_weights = Path(config.get("output", {}).get("output_dir", "outputs/bsafe_cbd_rtdetrv4")) / "best_cbd_rtdetrv4.pt"

    try:
        if args.eval_only:
            stats = trainer.evaluate(args.eval_split, args.weights or default_weights)
            if is_main_process():
                print(json.dumps({"split": args.eval_split, **stats}, indent=2))
            return

        stats = trainer.train()
        if is_main_process():
            print(json.dumps(stats, indent=2))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
