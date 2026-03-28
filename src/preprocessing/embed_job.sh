#!/bin/bash
#SBATCH --job-name=hrs_embed
#SBATCH --output=logs/hrs_embed_%j.out
#SBATCH --error=logs/hrs_embed_%j.err
#SBATCH --partition=B200-4h
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --mail-user=eli.kazum@biu.ac.il
#SBATCH --mail-type=END,FAIL

# Usage: sbatch embed_job.sh <slice_index>
# Example: sbatch embed_job.sh 3
SLICE_INDEX=${1:-0}

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Slice index: $SLICE_INDEX"
echo "GPUs: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Start: $(date)"

nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hrs

cd ~/Python/HRS/
mkdir -p logs

python ./src/preprocessing/embed_features.py \
    --config config/preprocessing.yaml \
    --slice-index "$SLICE_INDEX"

echo "End: $(date)"
