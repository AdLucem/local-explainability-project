#!/bin/bash

#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1 
#SBATCH --time=8:00:00
#SBATCH --mem=80GB
#SBATCH --mail-user=u1419632@umail.utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o localexpl

source ~/miniconda3/etc/profile.d/conda.sh
conda activate condenv

# rm -rf /scratch/general/vast/u1419632/huggingface_cache
mkdir -p /scratch/general/vast/u1419632/huggingface_cache
export TRANSFORMER_CACHE=/scratch/general/vast/u1419632/huggingface_cache
export HF_DATASETS_CACHE=/scratch/general/vast/u1419632/huggingface_cache

python run_tcav.py 
