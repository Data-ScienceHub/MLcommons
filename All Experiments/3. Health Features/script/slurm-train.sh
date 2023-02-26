#!/usr/bin/env bash
#SBATCH --job-name="Adult_obesity"
#SBATCH --output=Adult_obesity.out
#SBATCH --error=Adult_obesity.err
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --account=ds--6013
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32GB

#source /etc/profile.d/modules.sh
#source ~/.bashrc

# this is for when you are using singularity
#module load cuda cudnn singularity
singularity run --nv /home/xje4cy/tft_pytorch.sif python train_health_group.py

# # this is for when you have a working virtual env
# module load cuda cudnn anaconda

# conda deactivate
# conda activate ml

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mi3se/.conda/envs/ml/lib
# python train.py