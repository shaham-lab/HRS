"""
check_embed_status.py — check which embedding outputs exist and are valid.

Used by submit_all.sh to decide which jobs to submit without requiring
a manual --embed-only flag.

Exit codes:
  0 — all embedding parquets complete and valid  → skip embedding
  1 — embedding incomplete (some features missing)  → run embed only
  2 — preprocessing outputs missing                 → run full pipeline

Prints a human-readable status summary to stdout.
"""

import os
import sys
import argparse
import yaml
import pandas as pd

from preprocessing_utils import _output_is_valid


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
        os.path.join(classifications_dir, "micro_panel_config.yaml"),
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

    # Micro panel features: one row per admission in splits (zero vector if no events)
    micro_panel_config_path = os.path.join(classifications_dir, "micro_panel_config.yaml")
    micro_features = []
    if os.path.exists(micro_panel_config_path):
        with open(micro_panel_config_path, encoding="utf-8") as fh:
            micro_panel_config: dict = yaml.safe_load(fh)
        micro_features = [
            (f"micro_{panel_name}_embeddings.parquet", f"micro_{panel_name}_embedding")
            for panel_name in micro_panel_config.get("panels", {})
        ]
    else:
        print("  WARNING: micro_panel_config.yaml not found — micro embeddings not checked.")

    all_features = text_features + lab_features + micro_features  # 18 + 37 = 55 total

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
