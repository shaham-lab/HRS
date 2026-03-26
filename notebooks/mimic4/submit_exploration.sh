#!/bin/bash
# Submit exploration notebooks to SLURM (one A100 GPU per notebook).
# Inspired by preprocessing/reward_model submit scripts.
#
# Usage:
#   bash notebooks/submit_exploration.sh --all
#   bash notebooks/submit_exploration.sh notebooks/mimic4_data_exploration.ipynb mimic4_chartevents_exploration.ipynb
#
# Notes:
#   - Requests 1× A100 GPU per notebook job.
#   - Job name is derived from the notebook file name.

set -euo pipefail

usage() {
    cat <<'EOF'
Submit exploration notebooks to SLURM (one job per notebook, 1x A100 GPU).

Usage:
  bash notebooks/submit_exploration.sh --all
  bash notebooks/submit_exploration.sh <nb1.ipynb> [<nb2.ipynb> ...]

Arguments:
  --all           Submit all notebooks matching notebooks/*exploration*.ipynb
  <notebook>      Specific notebooks to run (path or basename). Multiple allowed.

Examples:
  bash notebooks/submit_exploration.sh --all
  bash notebooks/submit_exploration.sh mimic4_data_exploration.ipynb mimic4_chartevents_exploration.ipynb
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

cd ~/Python/HRS
mkdir -p logs

declare -a REQUESTED

if [[ "$1" == "--all" ]]; then
    mapfile -t REQUESTED < <(ls notebooks/*exploration*.ipynb)
    shift
else
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                usage
                exit 0
                ;;
            *)
                REQUESTED+=("$1")
                shift
                ;;
        esac
    done
fi

if [[ ${#REQUESTED[@]} -eq 0 ]]; then
    echo "No notebooks specified." >&2
    exit 1
fi

resolve_notebook() {
    local nb="$1"
    if [[ -f "$nb" ]]; then
        echo "$nb"
    elif [[ -f "notebooks/$nb" ]]; then
        echo "notebooks/$nb"
    elif [[ -f "notebooks/${nb}.ipynb" ]]; then
        echo "notebooks/${nb}.ipynb"
    else
        return 1
    fi
}

echo "Submitting exploration notebooks..."

for nb in "${REQUESTED[@]}"; do
    if ! resolved=$(resolve_notebook "$nb"); then
        echo "Skip (not found): $nb" >&2
        continue
    fi
    base="$(basename "$resolved" .ipynb)"
    job_name="explore_${base}"
    job_id=$(sbatch --parsable --job-name="$job_name" notebooks/exploration_job.sh "$resolved")
    echo "  [${job_id}] $resolved (job-name: $job_name)"
done

echo "Done. Monitor with: squeue -u $USER"
