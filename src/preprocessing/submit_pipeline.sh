#!/bin/bash
# Submit the CDSS preprocessing pipeline via Snakemake.
#
# Usage:
#   bash src/preprocessing/submit_pipeline.sh            # SLURM
#   bash src/preprocessing/submit_pipeline.sh --local    # local, 4 cores
#   bash src/preprocessing/submit_pipeline.sh --local 8  # local, 8 cores
#   bash src/preprocessing/submit_pipeline.sh --dry-run  # preview
#   bash src/preprocessing/submit_pipeline.sh \
#     --forcerun extract_demographics                    # force one rule

set -euo pipefail
cd ~/Python/HRS
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate hrs

if [[ "${1:-}" == "--local" ]]; then
    shift
    CORES="${1:-4}"
    shift 2>/dev/null || true
    snakemake \
        --snakefile src/preprocessing/Snakefile \
        --cores "$CORES" \
        "$@"
else
    snakemake \
        --snakefile src/preprocessing/Snakefile \
        --profile config/snakemake/slurm \
        --cores 1 \
        "$@"
fi
