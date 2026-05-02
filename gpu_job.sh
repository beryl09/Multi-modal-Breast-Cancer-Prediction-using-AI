#!/bin/bash
#SBATCH --job-name=my_gpu_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=18:00:00
#SBATCH --output=output.log

module load cuda/11.7.1  

source ~/miniconda3/etc/profile.d/conda.sh
conda activate my_env
python main.py
