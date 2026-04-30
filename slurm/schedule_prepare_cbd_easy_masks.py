from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
import os
import shlex
import shutil
import sys
from pathlib import Path

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

from cbd.cache import build_cache_records_from_config
from cbd.engine import load_config


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_command(config_path: Path, args: Namespace, clip_start: int, clip_end: int) -> str:
    command = [
        "python",
        "data_utils/prepare_cbd_easy_masks.py",
        "--config",
        str(config_path),
        "--split",
        str(args.split),
        "--clip-start",
        str(clip_start),
        "--clip-end",
        str(clip_end),
    ]
    for source in args.source or ():
        command.extend(["--source", str(source)])
    if args.overwrite:
        command.append("--overwrite")
    if args.keep_debug_artifacts:
        command.append("--keep-debug-artifacts")
    if args.device is not None:
        command.extend(["--device", str(args.device)])
    return shell_join(command)


def build_clip_ranges(total_records: int, num_workers: int) -> list[tuple[int, int]]:
    if num_workers < 1:
        raise ValueError("--num-workers must be at least 1.")
    if total_records < 0:
        raise ValueError("total_records must be non-negative.")
    base, remainder = divmod(total_records, num_workers)
    ranges: list[tuple[int, int]] = []
    start = 0
    for worker_index in range(num_workers):
        size = base + (1 if worker_index < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def get_script(script_path: Path, config_path: Path, args: Namespace, clip_start: int, clip_end: int) -> str:
    working_dir = ROOT.resolve()
    output = script_path.with_suffix(".out")
    if args.gpu is not None:
        gpu = "|".join(f"gpu{x}" for x in args.gpu)
    else:
        gpu = "gpuh200|gpuh100|gpua100hgx|gpua100|gpua40|gpul40s|gpuv100"
    command = build_command(config_path, args, clip_start, clip_end)

    return f"""#! /bin/bash

# +------------------------------------------------------------------------------------+ #
# |                                  SLURM PARAMETERS                                  | #
# +------------------------------------------------------------------------------------+ #

#SBATCH -p pri2021gpu -A qoscammagpu2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --tasks-per-node 1
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o {output}
#SBATCH --constraint="{gpu}"


# +------------------------------------------------------------------------------------+ #
# |                                ENVIRONNEMENT SET UP                                | #
# +------------------------------------------------------------------------------------+ #

module load cuda/cuda-11.8 gcc/gcc-11 matlab/r2020b
source /home2020/home/miv/vedrenne/mambaforge/etc/profile.d/conda.sh
source /home2020/home/miv/vedrenne/mambaforge/etc/profile.d/mamba.sh
mamba deactivate
mamba activate py311cu118
cd {working_dir}

hostname=$(hostname)
python_version=$(python --version)
working_directory=$(pwd)
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader --id=0)
vram_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader --id=0)
vram_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader --id=0)
vram_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader --id=0)

vram_free_value=$(echo $vram_free | head -n1 | cut -d " " -f1)
vram_used_value=$(echo $vram_used | head -n1 | cut -d " " -f1)
vram_total_value=$(echo $vram_total | head -n1 | cut -d " " -f1)

free_percent=$(echo "scale=2; 100*$vram_free_value/$vram_total_value" | bc)
used_percent=$(echo "scale=2; 100*$vram_used_value/$vram_total_value" | bc)

printf "
Hostname.........: $hostname
Python version...: $python_version
GPU name.........: $gpu_name
GPU memory total.: $vram_total
GPU memory free..: $vram_free ($free_percent %%)
GPU memory used..: $vram_used ($used_percent %%)
Working Directory: $working_directory

\n"

# +------------------------------------------------------------------------------------+ #
# |                                 RUN PYTHON SCRIPT                                  | #
# +------------------------------------------------------------------------------------+ #

srun {command}

"""


def schedule(args: Namespace) -> None:
    print(":: Scheduling CBD easy-mask caching with SLURM")
    experiment = args.experiment or "cbd_easy_masks"
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_dir = Path(args.runs_dir) / experiment / now
    save_dir.mkdir(exist_ok=True, parents=True)

    config = load_config(Path(args.config).resolve())
    _, records = build_cache_records_from_config(config, args.split, sources=args.source)
    total_records = len(records)
    if total_records == 0:
        print(f":: Split {args.split} has 0 clips to cache. Nothing to schedule.")
        return
    if args.num_workers > total_records:
        raise ValueError(
            f"--num-workers ({args.num_workers}) exceeds the number of clips in split {args.split!r} ({total_records})."
        )
    clip_ranges = build_clip_ranges(total_records, args.num_workers)
    print(
        f":: Split {args.split} has {total_records} clips; scheduling {args.num_workers} jobs "
        f"covering {[f'[{start}:{end})' for start, end in clip_ranges]}"
    )

    config_path = save_dir / Path(args.config).name
    shutil.copyfile(args.config, config_path)
    for worker_index, (clip_start, clip_end) in enumerate(clip_ranges, start=1):
        script_path = save_dir / f"prepare_cbd_easy_masks_worker_{worker_index:02d}.job"
        script = get_script(script_path, config_path, args, clip_start, clip_end)
        with open(script_path, "w") as handle:
            handle.write(script)
        if args.no_submit:
            continue
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")


def parse_args(args=None):
    parser = ArgumentParser(description="Schedule CBD easy-mask cache preparation with SLURM.")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--runs-dir", type=str, default="./runs/")
    parser.add_argument(
        "--gpu",
        type=str,
        nargs="+",
        default=("h200", "h100", "a100hgx", "a100", "l40s"),
        help="Types of GPU ids to request.",
    )
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--config", type=str, required=True, help="Path to the CBD config file.")
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Single dataset split to cache, for example: --split train",
    )
    parser.add_argument(
        "--source",
        type=str,
        nargs="+",
        default=None,
        help="Optional source selection, for example: --source bsafe icglceaes",
    )
    parser.add_argument("--num-workers", type=int, required=True, help="Number of SLURM jobs to shard the split across.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite existing cache files.")
    parser.add_argument(
        "--keep-debug-artifacts",
        action="store_true",
        help="Keep tracker JSON artifacts next to masks.npz.",
    )
    parser.add_argument("--device", help="Override the stage-1 tracker device.")
    return parser.parse_args(args)


if __name__ == "__main__":
    schedule(parse_args())
