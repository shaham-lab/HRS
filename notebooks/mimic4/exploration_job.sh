#!/bin/bash
#SBATCH --job-name=hrs_explore
#SBATCH --output=logs/hrs_explore_%j.out
#SBATCH --error=logs/hrs_explore_%j.err
#SBATCH --partition=B200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# SLURM job that executes a single exploration notebook on one A100 GPU.
# Usage:
#   sbatch notebooks/exploration_job.sh notebooks/mimic4_data_exploration.ipynb

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch notebooks/mimic4/exploration_job.sh <notebook_path>" >&2
    exit 1
fi

NOTEBOOK_PATH="$1"
if [[ ! -f "$NOTEBOOK_PATH" ]]; then
    echo "Notebook not found: $NOTEBOOK_PATH" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

NOTEBOOK_ABS="$(realpath "$NOTEBOOK_PATH")"
NOTEBOOK_NAME="$(basename "${NOTEBOOK_ABS}" .ipynb)"
OUTPUT_DIR="$(dirname "${NOTEBOOK_ABS}")"

cd "${REPO_ROOT}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hrs

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-<none>}"
echo "Notebook: ${NOTEBOOK_ABS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Start: $(date)"

jupyter nbconvert \
    --to notebook \
    --execute "${NOTEBOOK_ABS}" \
    --inplace \
    --ExecutePreprocessor.timeout=-1 \
    --ClearOutputPreprocessor.enabled=True

echo "End: $(date)"
