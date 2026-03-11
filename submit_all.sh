#!/bin/bash
# Submit the full CDSS preprocessing pipeline with automatic state detection.
#
# Usage:
#   bash submit_all.sh
#
# Behaviour:
#   - Checks preprocessing and embedding output state automatically.
#   - If nothing exists: submits pipeline → 14 embed slices (chained) → combine.
#   - If preprocessing done but embedding incomplete: submits remaining slices → combine.
#   - If all embedding complete: submits combine only.
#   - If everything complete: prints status and exits without submitting.
#
# Re-run this script at any time — it always picks up from where it left off.
# Each embed slice job detects its already-completed rows and skips them, so
# a re-submitted slice only processes what remains.
#
# IMPORTANT: Embed slices are chained sequentially (--dependency=afterok) to
# prevent concurrent fastparquet append conflicts.

set -euo pipefail

cd ~/Python/HRS/
mkdir -p logs

CONFIG="config/preprocessing.yaml"

# Total number of embed slices: ceil(546028 / (20000 * 2)) = 14
N_SLICES=14
LAST_SLICE=$(( N_SLICES - 1 ))   # 13

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

CURRENT_DEP=""   # will hold the last submitted job ID for chaining

if [[ $EMBED_STATUS_CODE -eq 2 ]]; then
    # Preprocessing incomplete → submit pipeline job first
    echo "Decision: submitting full pipeline (preprocessing + embed slices + combine)."
    echo ""

    PREPROCESS_JOB=$(sbatch --parsable pipeline_job.sh)
    echo "  [1/3] Preprocessing : job $PREPROCESS_JOB (pipeline_job.sh)"
    CURRENT_DEP="$PREPROCESS_JOB"
fi

if [[ $EMBED_STATUS_CODE -eq 1 || $EMBED_STATUS_CODE -eq 2 ]]; then
    # Embedding incomplete → submit all 14 slice jobs chained sequentially
    echo "  Submitting $N_SLICES embed slice jobs (chained sequentially):"
    for i in $(seq 0 $LAST_SLICE); do
        if [[ -n "$CURRENT_DEP" ]]; then
            EMBED_JOB=$(sbatch --parsable \
                --dependency=afterok:"$CURRENT_DEP" \
                embed_job.sh "$i")
        else
            EMBED_JOB=$(sbatch --parsable embed_job.sh "$i")
        fi
        echo "    embed slice $i : job $EMBED_JOB (embed_job.sh $i${CURRENT_DEP:+, depends on $CURRENT_DEP})"
        CURRENT_DEP="$EMBED_JOB"
    done
fi

# Always submit combine (with dependency on last embed slice if one was submitted)
if [[ -n "$CURRENT_DEP" ]]; then
    COMBINE_JOB=$(sbatch --parsable \
        --dependency=afterok:"$CURRENT_DEP" \
        combine_job.sh)
    echo "  Combine : job $COMBINE_JOB (combine_job.sh, depends on $CURRENT_DEP)"
else
    COMBINE_JOB=$(sbatch --parsable combine_job.sh)
    echo "  Combine : job $COMBINE_JOB (combine_job.sh)"
fi

echo ""
if [[ $EMBED_STATUS_CODE -eq 0 ]]; then
    echo "All embeddings already complete — submitted combine only."
    echo "To cancel: scancel $COMBINE_JOB"
else
    echo "Monitor : squeue -u \$USER"
    echo "Logs    : ls logs/"
fi
