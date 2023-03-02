#!/usr/bin/env bash
#SBATCH --job-name="AGE4049"
#SBATCH --output=outputs/AGE4049_sensitivity.out
#SBATCH --error=outputs/AGE4049_sensitivity_error.err
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --account=ds--6013
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB

# this is for when you are using singularity
singularity run --nv ./tft_pytorch.sif python sensitivity_analysis.py
