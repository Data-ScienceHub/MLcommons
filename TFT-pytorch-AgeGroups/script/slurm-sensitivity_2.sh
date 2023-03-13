#!/usr/bin/env bash
#SBATCH --job-name="AGE517"
#SBATCH --output=outputs/AGE517_sensitivity_mod_all.out
#SBATCH --error=outputs/AGE517_sensitivity_mod_all_error.err
#SBATCH --partition=gpu
#SBATCH --time=40:00:00
#SBATCH --account=ds--6013
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB

# this is for when you are using singularity
singularity run --nv ./tft_pytorch.sif python sensitivity_analysis_2.py
