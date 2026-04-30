from __future__ import annotations
from argparse import ArgumentParser, Namespace
from datetime import datetime
import os
import shutil
from functools import wraps
from pathlib import Path
import yaml


def get_script(script_path: Path, config_path: Path, args: Namespace) -> str:
    working_dir = Path(__file__).parent.parent.resolve()
    output = script_path.with_suffix('.out')
    if args.gpu is not None:
        gpu = '|'.join(f'gpu{x}' for x in args.gpu)
    else:
        gpu = 'gpuh200|gpuh100|gpua100hgx|gpua100|gpua40|gpul40s|gpuv100'
    return f"""#! /bin/bash

# +------------------------------------------------------------------------------------+ #
# |                                  SLURM PARAMETERS                                  | #
# +------------------------------------------------------------------------------------+ #

#SBATCH -p pri2021gpu -A qoscammagpu2
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --tasks-per-node 2
#SBATCH --gres=gpu:2
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

vram_free_value=$(echo $vram_free | head -n1 | cut -d \" \" -f1)
vram_used_value=$(echo $vram_used | head -n1 | cut -d \" \" -f1)
vram_total_value=$(echo $vram_total | head -n1 | cut -d \" \" -f1)

free_percent=$(echo \"scale=2; 100*$vram_free_value/$vram_total_value\" | bc)
used_percent=$(echo \"scale=2; 100*$vram_used_value/$vram_total_value\" | bc)

printf \"
Hostname.........: $hostname
Python version...: $python_version
GPU name.........: $gpu_name
GPU memory total.: $vram_total
GPU memory free..: $vram_free ($free_percent %%)
GPU memory used..: $vram_used ($used_percent %%)
Working Directory: $working_directory

\n\"

# +------------------------------------------------------------------------------------+ #
# |                                 RUN PYTHON SCRIPT                                  | #
# +------------------------------------------------------------------------------------+ #

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_IB_HCA=hfi1_0
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))

srun python train/train_lora.py --config {config_path}

"""


def schedule(args: Namespace) -> None:
    print(':: Scheduling runs with SLURM')
    now = datetime.now().strftime("%Y-%m-%d-%H-%M")
    runs_dir = Path(args.runs_dir) / args.experiment / now
    runs_dir.mkdir(exist_ok=True, parents=True)
    for i in range(args.num_runs):
        run = f'run_{i + 1}'
        save_dir = runs_dir / run
        script_path = save_dir / f'train_{run}.job'
        save_dir.mkdir(exist_ok=True, parents=True)
        # 1. copy config
        config_path = save_dir / Path(args.config).name
        shutil.copyfile(args.config, config_path)
        # code_dir = save_dir / 'code'
        # code_dir.mkdir(exist_ok=True)
        # for file in Path('src').glob('*.py'):
        #     shutil.copyfile(file, code_dir / file.name)
        # args.save_dir = str(save_dir)
        # 2. patch config output path
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        config['output']['output_dir'] = str(save_dir)
        with open(config_path, 'w') as file:
            yaml.safe_dump(config, file)
        # 3. create script
        script = get_script(script_path, config_path, args)
        with open(script_path, 'w') as file:
            file.write(script)
        if args.no_submit:
            continue
        os.system(f'chmod +x {script_path}')
        os.system(f'sbatch {script_path}')


def schedulable(function):
    @wraps(function)
    def wrapper(args: Namespace):
        if args.num_runs is not None:
            schedule(args)
        else:
            return function(args)
    return wrapper


def parse_args(args=None):
    parser = ArgumentParser(description="SAM3 Fine-tuning")
    # Seed
    # parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    # parser.add_argument('--no-seed', action='store_true', help='If set, do not set any seed.')
    # SLURM
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--num-runs', type=int, default=None)
    parser.add_argument('--runs-dir', type=str, default='./runs/')
    parser.add_argument('--gpu', type=str, nargs='+',
                        default=('h200', 'h100', 'a100hgx', 'a100', 'l40s'),
                        help="Types of GPU ids to request.")
    
    # Debug
    parser.add_argument('--no-submit', action='store_true')
    # Actual args
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    schedule(args)
