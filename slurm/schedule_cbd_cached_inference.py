from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
import os
import shlex
import shutil
from pathlib import Path


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def infer_output_name(args: Namespace) -> str:
    if args.output_name is not None:
        return str(args.output_name)
    if args.clip_id is not None:
        return f"{args.clip_id}_overlay.png"
    split_label = str(args.split or "split")
    return f"{split_label}_predictions"


def build_command(config_path: Path, output_path: Path, args: Namespace) -> str:
    command = [
        "python",
        "infer/infer_cbd.py",
        "--config",
        str(config_path),
        "--output",
        str(output_path),
    ]
    if args.clip_id is not None:
        command.extend(["--clip-id", str(args.clip_id)])
    if args.split is not None:
        command.extend(["--split", str(args.split)])
    if args.weights is not None:
        command.extend(["--weights", str(args.weights)])
    if args.show_gt:
        command.append("--show-gt")
    return shell_join(command)


def get_script(script_path: Path, config_path: Path, output_path: Path, args: Namespace) -> str:
    working_dir = Path(__file__).parent.parent.resolve()
    output = script_path.with_suffix(".out")
    if args.gpu is not None:
        gpu = "|".join(f"gpu{x}" for x in args.gpu)
    else:
        gpu = "gpuh200|gpuh100|gpua100hgx|gpua100|gpua40|gpul40s|gpuv100"
    command = build_command(config_path, output_path, args)
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
    print(":: Scheduling cached CBD inference with SLURM")
    experiment = args.experiment or "cbd_cached_inference"
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_dir = Path(args.runs_dir) / experiment / now
    save_dir.mkdir(exist_ok=True, parents=True)

    config_path = save_dir / Path(args.config).name
    output_path = save_dir / infer_output_name(args)
    script_path = save_dir / "infer_cbd.job"
    shutil.copyfile(args.config, config_path)

    script = get_script(script_path, config_path, output_path, args)
    with open(script_path, "w") as handle:
        handle.write(script)

    if args.no_submit:
        return
    os.system(f"chmod +x {script_path}")
    os.system(f"sbatch {script_path}")


def parse_args(args=None):
    parser = ArgumentParser(description="Schedule cached stage-2 CBD inference with SLURM.")
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
        "--clip-id",
        type=str,
        help="Optional clip id passed to infer_cbd.py. If omitted, the scheduled job runs on the whole split.",
    )
    parser.add_argument("--split", type=str, help="Optional dataset split override.")
    parser.add_argument("--weights", type=str, help="Optional CBD checkpoint override.")
    parser.add_argument("--show-gt", action="store_true", help="Render the ground-truth CBD box as well.")
    parser.add_argument(
        "--output-name",
        type=str,
        help=(
            "Optional output path inside the scheduled run directory. "
            "Use a PNG filename for single-clip mode or a directory name for whole-split mode."
        ),
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    schedule(parse_args())
