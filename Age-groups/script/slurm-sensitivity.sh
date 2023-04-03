#!/usr/bin/env bash
#SBATCH --job-name="UNDER5"
#SBATCH --output=outputs/UNDER5_sensitivity.out
#SBATCH --error=outputs/UNDER5_sensitivity.err
#SBATCH --partition=gpu
#SBATCH --time=20:00:00
#SBATCH --account=ds--6013
#SBATCH --gres=gpu
#SBATCH --mem=32GB

# this is for when you are using singularity
singularity run --nv ./tft_pytorch.sif python sensitivity_analysis_UNDER5.py
