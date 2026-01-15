#!/bin/bash
#SBATCH --job-name=prepare_dataset
#SBATCH --output=logs/prepare_dataset_%j.out
#SBATCH --error=logs/prepare_dataset_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=nodes

module purge
module load gcc/14.2.0

VENV_DIR="/users/sgzwa126/data/venv"
source $VENV_DIR/bin/activate

python step1_prepare_dataset.py