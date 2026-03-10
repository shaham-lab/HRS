# GitHub Copilot Prompt — Automatic embedding state detection in submit_all.sh

## Context

`submit_all.sh` currently requires a `--embed-only` flag to resume after an interrupted
embedding run. This prompt makes resumption fully automatic: the script detects which
embedding outputs already exist and are valid, then decides on its own whether to submit
the full pipeline, embedding only, or just combine.

The detection is done by a small Python helper script `check_embed_status.py` that
reuses the same validity logic as `embed_features.py`. `submit_all.sh` calls it and
reads its exit code to decide what to submit.

---

## File 1 — `src/preprocessing/check_embed_status.py` (new file)

Create this file. It checks all 18 expected embedding output parquets and exits with:
- **exit code 0** — all 18 complete and valid → embedding can be skipped
- **exit code 1** — some or all missing/invalid → embedding must run
- **exit code 2** — preprocessing outputs missing → full pipeline must run

```python
"""
check_embed_status.py — check which embedding outputs exist and are valid.

Used by submit_all.sh to decide which jobs to submit without requiring
a manual --embed-only flag.

Exit codes:
  0 — all 18 embedding parquets complete and valid  → skip embedding
  1 — embedding incomplete (some features missing)  → run embed only
  2 — preprocessing outputs missing                 → run full pipeline

Prints a human-readable status summary to stdout.
"""

import os
import sys
import argparse
import yaml
import pandas as pd


def _output_is_valid(path: str, expected_rows: int, embedding_col: str) -> bool:
    """Mirror of embed_features._output_is_valid — kept in sync manually."""
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_parquet(path)
    except Exception:
        return False
    if len(df) != expected_rows:
        return False
    if embedding_col not in df.columns:
        return False
    if df[embedding_col].isnull().any():
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/preprocessing.yaml",
        help="Path to preprocessing.yaml",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    preprocessing_dir   = str(config["PREPROCESSING_DIR"])
    features_dir        = str(config["FEATURES_DIR"])
    embeddings_dir      = str(config["EMBEDDINGS_DIR"])
    classifications_dir = str(config["CLASSIFICATIONS_DIR"])

    # ------------------------------------------------------------------ #
    # Check preprocessing outputs exist (data_splits, feature parquets)
    # ------------------------------------------------------------------ #
    required_preprocess_outputs = [
        os.path.join(preprocessing_dir, "data_splits.parquet"),
        os.path.join(features_dir, "diag_history_features.parquet"),
        os.path.join(features_dir, "discharge_history_features.parquet"),
        os.path.join(features_dir, "triage_features.parquet"),
        os.path.join(features_dir, "chief_complaint_features.parquet"),
        os.path.join(features_dir, "radiology_features.parquet"),
        os.path.join(features_dir, "labs_features.parquet"),
        os.path.join(classifications_dir, "y_labels.parquet"),
        os.path.join(classifications_dir, "lab_panel_config.yaml"),
    ]
    missing_preprocess = [p for p in required_preprocess_outputs
                          if not os.path.exists(p)]
    if missing_preprocess:
        print("STATUS: PREPROCESSING INCOMPLETE")
        print(f"  Missing {len(missing_preprocess)} preprocessing output(s):")
        for p in missing_preprocess:
            print(f"    MISSING  {os.path.basename(p)}")
        print("  → Will run full pipeline.")
        sys.exit(2)

    # ------------------------------------------------------------------ #
    # Load expected row count for each embedding feature
    # ------------------------------------------------------------------ #
    splits_df = pd.read_parquet(
        os.path.join(preprocessing_dir, "data_splits.parquet")
    )[["subject_id", "hadm_id"]].drop_duplicates()
    n_admissions = len(splits_df)

    # Non-lab text features: one row per admission in splits
    text_features = [
        ("diag_history_embeddings.parquet",      "diag_history_embedding"),
        ("discharge_history_embeddings.parquet",  "discharge_history_embedding"),
        ("triage_embeddings.parquet",             "triage_embedding"),
        ("chief_complaint_embeddings.parquet",    "chief_complaint_embedding"),
        ("radiology_embeddings.parquet",          "radiology_embedding"),
    ]

    # Lab group features: one row per admission in splits (zero vector if no events)
    lab_panel_config_path = os.path.join(classifications_dir, "lab_panel_config.yaml")
    with open(lab_panel_config_path, encoding="utf-8") as fh:
        lab_panel_config: dict = yaml.safe_load(fh)

    lab_features = [
        (f"lab_{group_name}_embeddings.parquet", f"lab_{group_name}_embedding")
        for group_name in lab_panel_config
    ]

    all_features = text_features + lab_features   # 18 total

    # ------------------------------------------------------------------ #
    # Check each embedding output
    # ------------------------------------------------------------------ #
    complete = []
    incomplete = []

    for filename, embedding_col in all_features:
        path = os.path.join(embeddings_dir, filename)
        if _output_is_valid(path, expected_rows=n_admissions, embedding_col=embedding_col):
            complete.append(filename)
        else:
            reason = "missing" if not os.path.exists(path) else "invalid/incomplete"
            incomplete.append((filename, reason))

    # ------------------------------------------------------------------ #
    # Print summary and exit
    # ------------------------------------------------------------------ #
    print(f"Embedding status: {len(complete)}/{len(all_features)} complete")
    print("")

    if complete:
        print(f"  Complete ({len(complete)}):")
        for f in complete:
            print(f"    OK       {f}")

    if incomplete:
        print(f"  Incomplete ({len(incomplete)}):")
        for f, reason in incomplete:
            print(f"    {reason.upper():8s} {f}")

    print("")

    if not incomplete:
        print("STATUS: EMBEDDING COMPLETE — combine_dataset only needed.")
        sys.exit(0)
    elif len(incomplete) == len(all_features):
        print("STATUS: EMBEDDING NOT STARTED — will run embed + combine.")
        sys.exit(1)
    else:
        print(
            f"STATUS: EMBEDDING PARTIAL "
            f"({len(complete)} done, {len(incomplete)} remaining) "
            f"— will resume embed + combine."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

## File 2 — `submit_all.sh` (replace existing)

Replace the entire contents of `submit_all.sh` with this. The `--embed-only` flag is
removed — state is detected automatically:

```bash
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

EMBED_STATUS_OUTPUT=$(python src/preprocessing/check_embed_status.py \
    --config "$CONFIG" 2>&1)
EMBED_STATUS_CODE=$?

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
```

---

## How it works end to end

```
bash submit_all.sh
  └── calls check_embed_status.py
      ├── exit 2 → preprocessing missing  → submits pipeline + embed + combine
      ├── exit 1 → embedding incomplete   → submits embed + combine
      └── exit 0 → embedding complete     → submits combine only

# After embedding is interrupted mid-run:
bash submit_all.sh
  └── check_embed_status.py sees 7/18 complete
      └── exit 1 → submits embed + combine
          └── embed_features.py skips the 7 completed features automatically
```

Every re-run of `bash submit_all.sh` is safe — it always picks up
exactly where the pipeline left off with no manual flags required.
