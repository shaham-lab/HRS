#!/bin/bash
# Submit the full CDSS preprocessing pipeline with automatic job sequencing.
#
# Usage:
#   bash submit_all.sh              # full pipeline
#   bash submit_all.sh --embed-only # resubmit embedding + combine only
#
# Jobs run in sequence via --dependency=afterok.
# If embedding is interrupted, re-run this script — completed features are
# skipped automatically by the resume logic in embed_features.py.

set -euo pipefail

cd ~/Python/HRS/
mkdir -p logs

MODE="${1:-}"

if [[ "$MODE" == "--embed-only" ]]; then
    echo "Submitting embedding + combine only..."

    EMBED_JOB=$(sbatch --parsable embed_job.sh)
    echo "  Embedding job : $EMBED_JOB  (embed_job.sh — 2× L4 GPU)"

    COMBINE_JOB=$(sbatch --parsable \
        --dependency=afterok:$EMBED_JOB \
        combine_job.sh)
    echo "  Combine job   : $COMBINE_JOB  (depends on $EMBED_JOB)"

else
    echo "Submitting full pipeline..."

    PREPROCESS_JOB=$(sbatch --parsable pipeline_job.sh)
    echo "  Preprocessing job : $PREPROCESS_JOB  (pipeline_job.sh — steps 0-8)"

    EMBED_JOB=$(sbatch --parsable \
        --dependency=afterok:$PREPROCESS_JOB \
        embed_job.sh)
    echo "  Embedding job     : $EMBED_JOB  (embed_job.sh — 2× L4 GPU, depends on $PREPROCESS_JOB)"

    COMBINE_JOB=$(sbatch --parsable \
        --dependency=afterok:$EMBED_JOB \
        combine_job.sh)
    echo "  Combine job       : $COMBINE_JOB  (depends on $EMBED_JOB)"
fi

echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     ls logs/"
echo ""
echo "To cancel all jobs:"
if [[ "$MODE" == "--embed-only" ]]; then
    echo "  scancel $EMBED_JOB $COMBINE_JOB"
else
    echo "  scancel $PREPROCESS_JOB $EMBED_JOB $COMBINE_JOB"
fi
