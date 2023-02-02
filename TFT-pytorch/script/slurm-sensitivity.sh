#!/usr/bin/env bash
#SBATCH --job-name="sensitivity"
#SBATCH --output=outputs/sensitivity_version_1.out
#SBATCH --partition=gpu
#---SBATCH --time=1:00:00
#SBATCH --account=ds6013
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

# this is for when you have a working virtual env
module load cuda-toolkit cudnn anaconda3
conda deactivate
conda activate ml
python sensitivity_analysis.py

# this is for when you are using singularity
# module load cuda cudnn singularity
# singularity run --nv $CONTAINERDIR/pytorch-1.10.0.sif sensitivity_analysis.py