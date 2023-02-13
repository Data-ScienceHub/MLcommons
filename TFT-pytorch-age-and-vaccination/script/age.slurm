#!/usr/bin/env bash

#SBATCH --time=7:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32GB
#SBATCH --account=ds--6013

#SBATCH --job-name="age"
#SBATCH --output=output_age.out
#SBATCH --error=error_age.err

# this is for when you are using singularity
singularity run --nv /home/xje4cy/tft_pytorch.sif python Train_019.py
singularity run --nv /home/xje4cy/tft_pytorch.sif python Sensitivity_Analysis_019.py

singularity run --nv /home/xje4cy/tft_pytorch.sif python Train_2029.py
singularity run --nv /home/xje4cy/tft_pytorch.sif python Sensitivity_Analysis_2029.py

singularity run --nv /home/xje4cy/tft_pytorch.sif python Train_3039.py
singularity run --nv /home/xje4cy/tft_pytorch.sif python Sensitivity_Analysis_3039.py

singularity run --nv /home/xje4cy/tft_pytorch.sif python Train_4049.py
singularity run --nv /home/xje4cy/tft_pytorch.sif python Sensitivity_Analysis_4049.py

singularity run --nv /home/xje4cy/tft_pytorch.sif python Train_5064.py
singularity run --nv /home/xje4cy/tft_pytorch.sif python Sensitivity_Analysis_5064.py

singularity run --nv /home/xje4cy/tft_pytorch.sif python Train_6579.py
singularity run --nv /home/xje4cy/tft_pytorch.sif python Sensitivity_Analysis_6579.py

singularity run --nv /home/xje4cy/tft_pytorch.sif python Train_80PLUS.py
singularity run --nv /home/xje4cy/tft_pytorch.sif python Sensitivity_Analysis_80PLUS.py