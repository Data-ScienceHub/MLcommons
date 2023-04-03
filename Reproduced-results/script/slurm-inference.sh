#!/usr/bin/env bash
#SBATCH --job-name="inference"
#SBATCH --output=outputs/inference.out
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --account=ds--6013
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

# this is for when you are using singularity
module load cuda cudnn singularity
singularity run --nv ../tft_pytorch.sif python inference.py

# # this is for when you have a working virtual env
# module load cuda cudnn anaconda

# conda deactivate
# conda activate ml

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mi3se/.conda/envs/ml/lib
# python inference.py