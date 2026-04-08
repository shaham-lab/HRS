#!/bin/bash
# SLURM calibration job: runs temperature scaling on the dev split.
# See Architecture §14 (Scripts) and Detailed Design §5 (calibrate.py).
#
# Usage:
#   sbatch src/reward_model/calibrate_job.sh
#
# Run only after a successful training job has written best_model.pt.
# Typically submitted as an afterok dependency on reward_job.sh via
# submit_reward.sh — see Architecture §14 (Job Chain).
#
# SLURM resource sizing (Architecture §14, Memory Requirements §10):
#   GPUs : 1  (single GPU; model weights ~1.4 GB)
#   RAM  : 32G (dev split loaded lazily via ParquetDataset; well within 32G)
#   CPUs : 4  (DataLoader prefetch workers)
#   Time : 1h (calibration completes in < 5 min; conservative upper bound)
#
# Customise --partition, --account, and --constraint for your cluster.
#SBATCH --job-name=calib_reward
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/reward_calib_%j.out
#SBATCH --error=logs/reward_calib_%j.err

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Start: $(date)"

nvidia-smi

set -euo pipefail

cd ~/Python/HRS
mkdir -p logs

# Load cluster modules — adjust module names for your cluster.
# module load cuda/12.x python/3.11

# Activate conda environment.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hrs

# Run temperature scaling calibration on the dev split.
python src/reward_model/calibrate.py --config config/reward_model.yaml

echo "End: $(date)"