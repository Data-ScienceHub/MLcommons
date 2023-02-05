#!/usr/bin/env bash

#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32GB
#SBATCH --account=ds--6013

#SBATCH --job-name="MLcommons_job"
#SBATCH --output=output.out
#SBATCH --error=error.err

# this is for when you are using singularity
singularity run --nv /home/xje4cy/tft_pytorch.sif python sensitivity_analysis.py
