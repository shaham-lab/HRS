#!/bin/bash
# SLURM training job: launches torchrun with NUM_GPUS workers.
# See Architecture §14 (Scripts) and Detailed Design §6.1 (Process Launch).
#
# Usage:
#   sbatch src/reward_model/reward_job.sh          # fresh run
#   sbatch src/reward_model/reward_job.sh --resume  # resume from checkpoint
#
# SLURM resource sizing (Architecture §14, Memory Requirements §10):
#   GPUs : 2  (NUM_GPUS default; each holds a full model copy under DDP)
#   RAM  : 64G (dataset loaded lazily via ParquetDataset; ~355 MB dataset RAM)
#   CPUs : 8  (DataLoader prefetch workers; 4 per GPU)
#   Time : 48h (conservative upper bound; adversarial batches cost 2x per batch)
#
# Customise --partition, --account, and --constraint for your cluster.
#SBATCH --job-name=reward_train
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/reward_train_%j.out
#SBATCH --error=logs/reward_train_%j.err

set -euo pipefail

cd ~/Python/HRS
mkdir -p logs

# Load cluster modules — adjust module names for your cluster.
# module load cuda/12.x python/3.11

# Activate conda environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate HRS

# Launch DDP training.
# --nproc_per_node must equal the number of GPUs requested above (--gres=gpu:N).
# --rdzv_backend=c10d uses the built-in rendezvous; no external store required.
# "$@" forwards any arguments passed to sbatch (e.g. --resume).
torchrun \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    src/reward_model/train.py \
    --config config/reward_model.yaml \
    "$@"
