#!/bin/bash
# Submit the full CDSS preprocessing pipeline with automatic state detection.
#
# Usage:
#   bash submit_all.sh
#
# Behaviour:
#   - Checks preprocessing and embedding output state automatically.
#   - If nothing exists: submits pipeline → embed → combine in sequence.
#   - If preprocessing done but embedding incomplete: submits embed → combine.
#   - If all embedding complete: submits combine only.
#   - If everything complete: prints status and exits without submitting.
#
# Re-run this script at any time — it always picks up from where it left off.

set -euo pipefail

cd ~/Python/HRS/
mkdir -p logs

CONFIG="config/preprocessing.yaml"

echo "============================================================"
echo "  CDSS Preprocessing Pipeline — Auto-Submit"
echo "  $(date)"
echo "============================================================"
echo ""

# ------------------------------------------------------------------ #
# Check current state using Python helper
# ------------------------------------------------------------------ #
echo "Checking pipeline state..."
echo ""

set +e
EMBED_STATUS_OUTPUT=$(python src/preprocessing/check_embed_status.py \
    --config "$CONFIG" 2>&1)
EMBED_STATUS_CODE=$?
set -e



echo "$EMBED_STATUS_OUTPUT"
echo ""

# ------------------------------------------------------------------ #
# Decide which jobs to submit based on exit code
# ------------------------------------------------------------------ #

if [[ $EMBED_STATUS_CODE -eq 2 ]]; then
    # Preprocessing incomplete → full pipeline
    echo "Decision: submitting full pipeline (preprocessing + embed + combine)."
    echo ""

    PREPROCESS_JOB=$(sbatch --parsable pipeline_job.sh)
    echo "  [1/3] Preprocessing : job $PREPROCESS_JOB (pipeline_job.sh)"

    EMBED_JOB=$(sbatch --parsable \
        --dependency=afterok:$PREPROCESS_JOB \
        embed_job.sh)
    echo "  [2/3] Embedding     : job $EMBED_JOB (embed_job.sh, depends on $PREPROCESS_JOB)"

    COMBINE_JOB=$(sbatch --parsable \
        --dependency=afterok:$EMBED_JOB \
        combine_job.sh)
    echo "  [3/3] Combine       : job $COMBINE_JOB (combine_job.sh, depends on $EMBED_JOB)"

    echo ""
    echo "To cancel all: scancel $PREPROCESS_JOB $EMBED_JOB $COMBINE_JOB"

elif [[ $EMBED_STATUS_CODE -eq 1 ]]; then
    # Preprocessing done, embedding incomplete → embed + combine
    echo "Decision: submitting embed + combine (preprocessing already complete)."
    echo ""

    EMBED_JOB=$(sbatch --parsable embed_job.sh)
    echo "  [1/2] Embedding : job $EMBED_JOB (embed_job.sh)"

    COMBINE_JOB=$(sbatch --parsable \
        --dependency=afterok:$EMBED_JOB \
        combine_job.sh)
    echo "  [2/2] Combine   : job $COMBINE_JOB (combine_job.sh, depends on $EMBED_JOB)"

    echo ""
    echo "To cancel all: scancel $EMBED_JOB $COMBINE_JOB"

elif [[ $EMBED_STATUS_CODE -eq 0 ]]; then
    # Embedding complete → combine only
    echo "Decision: submitting combine only (all embeddings already complete)."
    echo ""

    COMBINE_JOB=$(sbatch --parsable combine_job.sh)
    echo "  [1/1] Combine : job $COMBINE_JOB (combine_job.sh)"

    echo ""
    echo "To cancel: scancel $COMBINE_JOB"

else
    echo "ERROR: check_embed_status.py returned unexpected exit code $EMBED_STATUS_CODE"
    exit 1
fi

echo ""
echo "Monitor : squeue -u \$USER"
echo "Logs    : ls logs/"
