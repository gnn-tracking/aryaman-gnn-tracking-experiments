#!/usr/bin/env bash

#SBATCH --job-name=pixel-oc
#SBATCH --output=slurm_logs/pixel-nc-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task 6
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=25G
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=f20190767@hyderabad.bits-pilani.ac.in
#SBATCH --signal=SIGUSR1@1800
#SBATCH --constraint=gpu80

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
srun python3 run_nc.py fit $@