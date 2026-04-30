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

from cbd.engine import CBDTrainer, load_config


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or evaluate the stage-2 CBD clip box model.")
    parser.add_argument("--config", required=True, help="Path to the CBD YAML config.")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only.")
    parser.add_argument("--eval-split", default="val", help="Split to evaluate when --eval-only is used.")
    parser.add_argument("--weights", help="Optional checkpoint path. Defaults to output_dir/best_cbd.pt.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = load_config(args.config)
    trainer = CBDTrainer(config)
    default_weights = Path(config.get("output", {}).get("output_dir", "outputs/bsafe_cbd")) / "best_cbd.pt"

    if args.eval_only:
        stats = trainer.evaluate(args.eval_split, args.weights or default_weights)
        print(json.dumps({"split": args.eval_split, **stats}, indent=2))
        return

    best_stats = trainer.train()
    print(json.dumps(best_stats, indent=2))


if __name__ == "__main__":
    main()
