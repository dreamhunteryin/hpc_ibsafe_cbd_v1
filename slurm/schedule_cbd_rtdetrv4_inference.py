from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
import os
import shlex
from pathlib import Path


DEFAULT_EXCLUDED_NODES = ("hpc-n968",)


def shell_join(parts: list[str | Path | int | float]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def run_label(run_dir: Path) -> str:
    return f"{run_dir.parent.name}_{run_dir.name}"


def discover_run_dirs(args: Namespace) -> list[Path]:
    if args.run_dir:
        candidates = [Path(path) for path in args.run_dir]
    else:
        candidates = sorted(Path(args.runs_root).glob("*/run_*"))

    run_dirs = [
        run_dir
        for run_dir in candidates
        if (run_dir / args.config_name).exists() and (run_dir / args.weights_name).exists()
    ]
    if not run_dirs:
        raise FileNotFoundError(
            f"No runs with {args.config_name!r} and {args.weights_name!r} found under {args.runs_root}."
        )
    return run_dirs


def build_command(run_dir: Path, output_root: Path, args: Namespace) -> str:
    command: list[str | Path | int | float] = [
        "python",
        "infer/inference_cbd_rtdetrv4_ourdino_test.py",
        "--run-dir",
        run_dir,
        "--output-root",
        output_root,
        "--split",
        args.split,
        "--config-name",
        args.config_name,
        "--weights-name",
        args.weights_name,
    ]
    if args.batch_size is not None:
        command.extend(["--batch-size", args.batch_size])
    if args.dataloader_workers is not None:
        command.extend(["--num-workers", args.dataloader_workers])
    if args.mixed_precision is not None:
        command.extend(["--mixed-precision", args.mixed_precision])
    if args.device is not None:
        command.extend(["--device", args.device])
    return shell_join(command)


def get_script(script_path: Path, run_dir: Path, output_root: Path, args: Namespace) -> str:
    working_dir = Path(__file__).parent.parent.resolve()
    output = script_path.with_suffix(".out")
    if args.gpu is not None:
        gpu = "|".join(f"gpu{x}" for x in args.gpu)
    else:
        gpu = "gpuh200|gpuh100|gpua100hgx|gpua100|gpua40|gpul40s|gpuv100"
    exclude_nodes = ",".join(DEFAULT_EXCLUDED_NODES)
    command = build_command(run_dir, output_root, args)

    return f"""#! /bin/bash

#SBATCH -p pri2021gpu -A qoscammagpu2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --tasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o {output}
#SBATCH --constraint="{gpu}"
#SBATCH --exclude={exclude_nodes}

module load cuda/cuda-11.8 gcc/gcc-12 matlab/r2020b
source /home2020/home/miv/vedrenne/mambaforge/etc/profile.d/conda.sh
source /home2020/home/miv/vedrenne/mambaforge/etc/profile.d/mamba.sh
mamba deactivate
mamba activate py311cu118
cd {working_dir}

srun {command}

"""


def schedule(args: Namespace) -> None:
    experiment = args.experiment or "cbd_rtdetrv4_ourdino_test_inference"
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_dir = Path(args.runs_dir) / experiment / now
    jobs_dir = save_dir / "jobs"
    output_root = save_dir / "outputs"
    jobs_dir.mkdir(exist_ok=True, parents=True)
    output_root.mkdir(exist_ok=True, parents=True)

    run_dirs = discover_run_dirs(args)
    print(f":: Scheduling {len(run_dirs)} RT-DETRv4 CBD test inference job(s)")
    for run_dir in run_dirs:
        label = run_label(run_dir)
        script_path = jobs_dir / f"infer_cbd_rtdetrv4_{label}.job"
        script = get_script(script_path, run_dir, output_root, args)
        script_path.write_text(script)
        print(f":: {label} -> {output_root / label}")
        if args.no_submit:
            continue
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")


def parse_args(args=None):
    parser = ArgumentParser(description="Schedule RT-DETRv4 CBD test-set inference/evaluation jobs.")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--runs-dir", type=str, default="./runs/")
    parser.add_argument("--runs-root", type=Path, default=Path("runs/cbd_rtdetrv4_ourdino"))
    parser.add_argument("--run-dir", type=Path, action="append", default=None)
    parser.add_argument("--config-name", default="bsafe_cbd_rtdetrv4_base.yaml")
    parser.add_argument("--weights-name", default="best_cbd_rtdetrv4.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dataloader-workers", type=int, default=None)
    parser.add_argument("--mixed-precision", choices=("none", "fp16", "bf16"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--gpu",
        type=str,
        nargs="+",
        default=("h200", "h100", "a100hgx", "a100", "l40s"),
    )
    parser.add_argument("--no-submit", action="store_true")
    return parser.parse_args(args)


if __name__ == "__main__":
    schedule(parse_args())
