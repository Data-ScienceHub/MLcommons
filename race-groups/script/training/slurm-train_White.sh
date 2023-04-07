#!/usr/bin/env bash
#SBATCH --job-name="White"
#SBATCH --output=outputs/White_train.out
#SBATCH --error=outputs/White_train.err
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --account=ds--6013
#SBATCH --gres=gpu
#SBATCH --mem=32GB

# this is for when you are using singularity
singularity run --nv ./tft_pytorch.sif python train_race_group_White.py