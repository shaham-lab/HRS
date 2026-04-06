#!/bin/bash
# Submit exploration notebooks to SLURM (one A100 GPU per notebook).
# Inspired by preprocessing/reward_model submit scripts.
#
# Usage:
#   bash notebooks/mimic4/submit_exploration.sh --all
#   bash notebooks/mimic4/submit_exploration.sh notebooks/mimic4/mimic4_data_exploration.ipynb mimic4_chartevents_exploration.ipynb
#
# Notes:
#   - Requests 1× A100 GPU per notebook job.
#   - Job name is derived from the notebook file name.

set -euo pipefail

usage() {
    cat <<'EOF'
Submit exploration notebooks to SLURM (one job per notebook, 1x A100 GPU).

Usage:
  bash notebooks/mimic4/submit_exploration.sh --all
  bash notebooks/mimic4/submit_exploration.sh <nb1.ipynb> [<nb2.ipynb> ...]

Arguments:
  --all           Submit all notebooks matching notebooks/mimic4/*exploration*.ipynb
  <notebook>      Specific notebooks to run. If a path is provided, it is used as-is.
                  If only a file name is provided, it is resolved in the current
                  working directory. Multiple notebooks allowed.

Examples:
  bash notebooks/mimic4/submit_exploration.sh --all
  bash notebooks/mimic4/submit_exploration.sh notebooks/mimic4/mimic4_data_exploration.ipynb mimic4_chartevents_exploration.ipynb
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CALLER_PWD="$(pwd)"

declare -a REQUESTED

if [[ "$1" == "--all" ]]; then
    mapfile -t REQUESTED < <(cd "${REPO_ROOT}" && ls notebooks/mimic4/*exploration*.ipynb)
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
    local candidate="$nb"

    # Resolve relative paths (including basenames) against the caller's cwd.
    if [[ "$nb" != /* ]]; then
        candidate="${CALLER_PWD}/${nb}"
    fi

    # Allow missing .ipynb extension when a basename is provided.
    if [[ ! -f "$candidate" && "$candidate" != *.ipynb ]]; then
        candidate="${candidate}.ipynb"
    fi

    if [[ -f "$candidate" ]]; then
        realpath "$candidate"
        return 0
    fi

    return 1
}

echo "Submitting exploration notebooks..."

cd "${REPO_ROOT}"

for nb in "${REQUESTED[@]}"; do
    if ! resolved=$(resolve_notebook "$nb"); then
        echo "Skip (not found): $nb" >&2
        continue
    fi
    base="$(basename "$resolved" .ipynb)"
    job_id=$(sbatch --parsable "${SCRIPT_DIR}/exploration_job.sh" "$resolved")
    echo "  [${job_id}] $resolved"
done

echo "Done. Monitor with: squeue -u $USER"
