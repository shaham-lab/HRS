#!/bin/bash
#SBATCH --job-name=hrs_explore
#SBATCH --partition=A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/explore_%x_%j.out
#SBATCH --error=logs/explore_%x_%j.err

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

cd "${REPO_ROOT}"
mkdir -p logs notebooks/mimic4/executed

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hrs

OUTPUT_DIR="notebooks/mimic4/executed"
OUTPUT_NAME="${NOTEBOOK_NAME}_executed.ipynb"

echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-<none>}"
echo "Notebook: ${NOTEBOOK_ABS}"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Start: $(date)"

jupyter nbconvert \
    --to notebook \
    --execute "${NOTEBOOK_ABS}" \
    --output "${OUTPUT_NAME}" \
    --output-dir "${OUTPUT_DIR}" \
    --ExecutePreprocessor.timeout=-1

echo "End: $(date)"
