#!/bin/bash --login
#SBATCH --job-name=evolve_DEAd
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-199
#SBATCH --time=3:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --ntasks=1

# -----------------------------
# Initialize Conda
source ~/miniforge3/etc/profile.d/conda.sh
conda activate DEAd
export PATH=~/miniforge3/envs/DEAd/bin:$PATH


# -----------------------------
# Compute replicate and class
REPLICATE=$(( SLURM_ARRAY_TASK_ID / 10 ))
CLASS=$(( SLURM_ARRAY_TASK_ID % 10 ))

# -----------------------------
# Run simulation
python main.py \
  --clusters 1 \
  --nodes_per_cluster 30 \
  --dataset_name mnist \
  --model_name SVM \
  --replicate $REPLICATE \
  --target_class $CLASS