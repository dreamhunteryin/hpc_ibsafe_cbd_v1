#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate RT-DETRv4 CBD checkpoints from runs/cbd_rtdetrv4_ourdino on the BSAFE test set."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/cbd_rtdetrv4_ourdino"),
        help="Root containing timestamp/run_* RT-DETRv4 training runs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Specific run directory to evaluate. Can be passed more than once.",
    )
    parser.add_argument("--config-name", default="bsafe_cbd_rtdetrv4_base.yaml")
    parser.add_argument("--weights-name", default="best_cbd_rtdetrv4.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/cbd_rtdetrv4_ourdino_test_inference"),
        help="Directory where per-run test metrics and predictions are written.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Optional eval batch-size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional dataloader worker override.")
    parser.add_argument(
        "--mixed-precision",
        choices=("none", "fp16", "bf16"),
        default=None,
        help="Optional mixed-precision override for evaluation.",
    )
    parser.add_argument("--device", default=None, help="Optional hardware.device override, e.g. cuda or cpu.")
    return parser


def run_label(run_dir: Path) -> str:
    return f"{run_dir.parent.name}_{run_dir.name}"


def discover_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        candidates = [Path(path) for path in args.run_dir]
    else:
        candidates = sorted(Path(args.runs_root).glob("*/run_*"))

    run_dirs: list[Path] = []
    for run_dir in candidates:
        config_path = run_dir / args.config_name
        weights_path = run_dir / args.weights_name
        if config_path.exists() and weights_path.exists():
            run_dirs.append(run_dir)
    if not run_dirs:
        raise FileNotFoundError(
            f"No runs with {args.config_name!r} and {args.weights_name!r} found under {args.runs_root}."
        )
    return run_dirs


def apply_eval_overrides(config: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    config = dict(config)
    config.setdefault("output", {})["output_dir"] = str(output_dir)

    training = config.setdefault("training", {})
    if args.batch_size is not None:
        training["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        training["num_workers"] = int(args.num_workers)
    if args.mixed_precision is not None:
        training["mixed_precision"] = str(args.mixed_precision)

    if args.device is not None:
        config.setdefault("hardware", {})["device"] = str(args.device)

    return config


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from cbd_rtdetrv4.engine import CBDRTDetrV4Trainer, load_config

    label = run_label(run_dir)
    output_dir = Path(args.output_root) / label
    output_dir.mkdir(parents=True, exist_ok=True)

    source_config_path = run_dir / args.config_name
    weights_path = run_dir / args.weights_name
    config = apply_eval_overrides(load_config(source_config_path), args, output_dir)

    eval_config_path = output_dir / source_config_path.name
    eval_config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    trainer = CBDRTDetrV4Trainer(config)
    stats = trainer.evaluate(args.split, weights_path)

    result = {
        "run_dir": str(run_dir),
        "config": str(eval_config_path),
        "weights": str(weights_path),
        "output_dir": str(output_dir),
        "split": str(args.split),
        "stats": stats,
    }

    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = build_argparser().parse_args()
    results: list[dict[str, Any]] = []
    try:
        for run_dir in discover_run_dirs(args):
            result = evaluate_run(run_dir, args)
            results.append(result)
            print(json.dumps(result, indent=2))

        summary_path = Path(args.output_root) / f"summary_{args.split}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(results, indent=2))
        print(f"Wrote summary to {summary_path}")
    finally:
        from cbd_rtdetrv4.engine import cleanup_distributed

        cleanup_distributed()


if __name__ == "__main__":
    main()
