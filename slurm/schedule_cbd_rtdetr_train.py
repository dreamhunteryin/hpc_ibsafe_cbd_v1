from __future__ import annotations

from argparse import ArgumentParser, Namespace
from datetime import datetime
import os
import shutil
from pathlib import Path

import yaml


def get_script(script_path: Path, config_path: Path, args: Namespace) -> str:
    working_dir = Path(__file__).parent.parent.resolve()
    output = script_path.with_suffix(".out")
    if args.gpu is not None:
        gpu = "|".join(f"gpu{x}" for x in args.gpu)
    else:
        gpu = "gpuh200|gpuh100|gpua100hgx|gpua100|gpua40|gpul40s|gpuv100"
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

module load cuda/cuda-11.8 gcc/gcc-12 matlab/r2020b
source /home2020/home/miv/vedrenne/mambaforge/etc/profile.d/conda.sh
source /home2020/home/miv/vedrenne/mambaforge/etc/profile.d/mamba.sh
mamba deactivate
mamba activate py311cu118
cd {working_dir}

srun python train/train_cbd_rtdetr.py --config {config_path}

"""


def schedule(args: Namespace) -> None:
    experiment = args.experiment or Path(args.config).stem
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    runs_dir = Path(args.runs_dir) / experiment / now
    runs_dir.mkdir(exist_ok=True, parents=True)
    for index in range(args.num_runs):
        run_name = f"run_{index + 1}"
        save_dir = runs_dir / run_name
        save_dir.mkdir(exist_ok=True, parents=True)
        config_path = save_dir / Path(args.config).name
        script_path = save_dir / f"train_cbd_rtdetr_{run_name}.job"
        shutil.copyfile(args.config, config_path)
        with open(config_path, "r") as handle:
            config = yaml.safe_load(handle)
        config.setdefault("output", {})["output_dir"] = str(save_dir)
        with open(config_path, "w") as handle:
            yaml.safe_dump(config, handle)
        script = get_script(script_path, config_path, args)
        with open(script_path, "w") as handle:
            handle.write(script)
        if args.no_submit:
            continue
        os.system(f"chmod +x {script_path}")
        os.system(f"sbatch {script_path}")


def parse_args(args=None):
    parser = ArgumentParser(description="Schedule RT-DETR CBD training runs with SLURM.")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--num-runs", type=int, required=True)
    parser.add_argument("--runs-dir", type=str, default="./runs/")
    parser.add_argument(
        "--gpu",
        type=str,
        nargs="+",
        default=("h200", "h100", "a100hgx", "a100", "l40s"),
    )
    parser.add_argument("--no-submit", action="store_true")
    parser.add_argument("--config", type=str, required=True, help="Path to the RT-DETR CBD config file.")
    return parser.parse_args(args)


if __name__ == "__main__":
    schedule(parse_args())
