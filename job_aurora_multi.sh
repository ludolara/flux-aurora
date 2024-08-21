#!/bin/bash
#SBATCH --job-name=flux
#SBATCH --output=train_log/job_error.txt
#SBATCH --error=train_log/job_output.txt
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=120:00:00

source $SCRATCH/env/flux/bin/activate
module load python/3.10
module load libffi
module load cudatoolkit/11.7
accelerate launch train_flux_deepspeed_controlnet.py \
 --config "train_configs/test_aurora_controlnet.yaml" 
