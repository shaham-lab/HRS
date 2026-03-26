#!/bin/bash
#SBATCH --job-name=hrs_explore
#SBATCH --output=logs/explore_%x_%j.out
#SBATCH --error=logs/explore_%x_%j.err
#SBATCH --partition=A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
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

# Robustly source conda (supports cluster installs where /usr/etc/profile.d/conda.sh is missing).
if [[ -z "${CONDA_EXE:-}" ]]; then
    # Try standard etc/profile.d location first.
    if [[ -f "/etc/profile.d/conda.sh" ]]; then
        source "/etc/profile.d/conda.sh"
    else
        # Fallback to conda info --base, suppressing errors.
        if BASE_PATH="$(conda info --base 2>/dev/null)"; then
            if [[ -f "${BASE_PATH}/etc/profile.d/conda.sh" ]]; then
                source "${BASE_PATH}/etc/profile.d/conda.sh"
            fi
        fi
    fi
fi

# Only attempt activation if conda was successfully sourced.
if command -v conda >/dev/null 2>&1; then
    conda activate hrs
else
    echo "conda not found; ensure conda is available on compute node" >&2
    exit 1
fi

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
