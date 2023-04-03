#!/usr/bin/env bash
#SBATCH --job-name="AGE1829"
#SBATCH --output=outputs/AGE1829_train.out
#SBATCH --error=outputs/AGE1829_train.err
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --account=ds--6013
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32GB

# this is for when you are using singularity
singularity run --nv ./tft_pytorch.sif python train_age_group.py