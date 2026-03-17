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

# Compute the number of embed slice jobs dynamically from config and
# data_splits.parquet.  Falls back to config-only estimate if the parquet
# does not yet exist (status code 2 path — pipeline hasn't run yet).
N_SLICES=$(python3 - <<'PYEOF'
import math, sys, os
import yaml

with open("config/preprocessing.yaml") as f:
    cfg = yaml.safe_load(f)

slice_size = int(cfg.get("BERT_SLICE_SIZE_PER_GPU", 20000))
n_gpus     = int(cfg.get("BERT_MAX_GPUS", 2) or 2)

splits_path = os.path.join(str(cfg.get("PREPROCESSING_DIR", "data/preprocessing")),
                           "data_splits.parquet")
if os.path.exists(splits_path):
    import pandas as pd
    total = pd.read_parquet(splits_path, columns=["hadm_id"])["hadm_id"].nunique()
else:
    # Pipeline hasn't run yet — use known dataset size
    total = 546028

print(math.ceil(total / (slice_size * n_gpus)))
PYEOF
)

if [ -z "$N_SLICES" ] || [ "$N_SLICES" -lt 1 ]; then
    echo "ERROR: could not compute N_SLICES" >&2
    exit 1
fi

LAST_SLICE=$(( N_SLICES - 1 ))

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

    PREPROCESS_JOB=$(sbatch --parsable src/preprocessing/pipeline_job.sh)
    echo "  [1/3] Preprocessing : job $PREPROCESS_JOB (src/preprocessing/pipeline_job.sh)"
    CURRENT_DEP="$PREPROCESS_JOB"
fi

if [[ $EMBED_STATUS_CODE -eq 1 || $EMBED_STATUS_CODE -eq 2 ]]; then
    # Embedding incomplete → submit all 14 slice jobs chained sequentially
    echo "  Submitting $N_SLICES embed slice jobs (chained sequentially):"
    for i in $(seq 0 $LAST_SLICE); do
        if [[ -n "$CURRENT_DEP" ]]; then
            EMBED_JOB=$(sbatch --parsable \
                --dependency=afterok:"$CURRENT_DEP" \
                src/preprocessing/embed_job.sh "$i")
            echo "    embed slice $i : job $EMBED_JOB (src/preprocessing/embed_job.sh $i, depends on $CURRENT_DEP)"
        else
            EMBED_JOB=$(sbatch --parsable src/preprocessing/embed_job.sh "$i")
            echo "    embed slice $i : job $EMBED_JOB (src/preprocessing/embed_job.sh $i)"
        fi
        CURRENT_DEP="$EMBED_JOB"
    done
fi

# Always submit combine (with dependency on last embed slice if one was submitted)
if [[ -n "$CURRENT_DEP" ]]; then
    COMBINE_JOB=$(sbatch --parsable \
        --dependency=afterok:"$CURRENT_DEP" \
        src/preprocessing/combine_job.sh)
    echo "  Combine : job $COMBINE_JOB (src/preprocessing/combine_job.sh, depends on $CURRENT_DEP)"
else
    COMBINE_JOB=$(sbatch --parsable src/preprocessing/combine_job.sh)
    echo "  Combine : job $COMBINE_JOB (src/preprocessing/combine_job.sh)"
fi

echo ""
if [[ $EMBED_STATUS_CODE -eq 0 ]]; then
    echo "All embeddings already complete — submitted combine only."
    echo "To cancel: scancel $COMBINE_JOB"
else
    echo "Monitor : squeue -u \$USER"
    echo "Logs    : ls logs/"
fi
