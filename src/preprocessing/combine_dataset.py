"""
combine_dataset.py – Merge all features into the final CDSS dataset.

Reads:
  • data/preprocessing/embeddings/   – all embedding parquets
  • data/preprocessing/features/     – demographics_features.parquet
  • data/preprocessing/              – data_splits.parquet
  • data/preprocessing/classifications/ – y_labels.parquet

Performs a left join on (subject_id, hadm_id), starting from the admissions
universe defined by data_splits.parquet. Raw text parquets from features/ are
excluded. Missing feature values appear as nulls.

Expected config keys:
    FEATURES_DIR          – directory containing raw feature parquets
    EMBEDDINGS_DIR        – directory containing embedding parquets
    CLASSIFICATIONS_DIR   – directory containing labels
    PREPROCESSING_DIR     – directory containing data_splits.parquet
"""

import logging
import os

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Feature parquets from features/ that are included in the final flat dataset.
# NOTE: labs_features.parquet is intentionally excluded — lab events are stored
# in long format (one row per event) and are joined dynamically at training time
# when the MDP agent selects a subset of tests. Including them here would require
# pivoting to wide format which reintroduces sparsity and loses temporal structure.
_FEATURES_TO_INCLUDE = [
    "demographics_features.parquet",
]

# All embedding parquets (discovered dynamically from embeddings_dir)
# Text parquets are intentionally excluded from the final dataset.


def run(config: dict) -> None:
    """Combine all feature and label parquets into the final dataset."""
    required_keys = [
        "FEATURES_DIR",
        "EMBEDDINGS_DIR",
        "CLASSIFICATIONS_DIR",
        "PREPROCESSING_DIR",
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    features_dir = str(config["FEATURES_DIR"])
    embeddings_dir = str(config["EMBEDDINGS_DIR"])
    classifications_dir = str(config["CLASSIFICATIONS_DIR"])
    preprocessing_dir = str(config["PREPROCESSING_DIR"])

    # ------------------------------------------------------------------ #
    # Start from splits (defines the admission universe)
    # ------------------------------------------------------------------ #
    splits_path = os.path.join(preprocessing_dir, "data_splits.parquet")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"data_splits.parquet not found at {splits_path}. "
            "Run create_splits.py first."
        )
    logger.info("Loading splits from %s…", splits_path)
    base = pd.read_parquet(splits_path)  # subject_id, hadm_id, split

    # ------------------------------------------------------------------ #
    # Merge labels
    # ------------------------------------------------------------------ #
    labels_path = os.path.join(classifications_dir, "y_labels.parquet")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"y_labels.parquet not found at {labels_path}. "
            "Run extract_y_data.py first."
        )
    logger.info("Merging y_labels…")
    labels = pd.read_parquet(labels_path)
    base = base.merge(labels, on=["subject_id", "hadm_id"], how="left")

    # ------------------------------------------------------------------ #
    # Merge selected feature parquets
    # ------------------------------------------------------------------ #
    for filename in tqdm(_FEATURES_TO_INCLUDE, desc="Merging features", unit="file"):
        path = str(os.path.join(features_dir, filename))
        if not os.path.exists(path):
            logger.warning("Feature file not found, skipping: %s", path)
            continue
        logger.info("Merging feature file: %s", filename)
        feat_df = pd.read_parquet(path)
        base = base.merge(feat_df, on=["subject_id", "hadm_id"], how="left")

    # ------------------------------------------------------------------ #
    # Merge embedding parquets (all *.parquet files in embeddings_dir)
    # ------------------------------------------------------------------ #
    if os.path.isdir(embeddings_dir):
        embedding_files = sorted(
            f for f in os.listdir(embeddings_dir) if f.endswith(".parquet")
        )
        for filename in tqdm(embedding_files, desc="Merging embeddings", unit="file"):
            path = str(os.path.join(embeddings_dir, filename))
            logger.info("Merging embedding file: %s", filename)
            emb_df = pd.read_parquet(path)
            base = base.merge(emb_df, on=["subject_id", "hadm_id"], how="left")
    else:
        logger.warning(
            "Embeddings directory not found (%s) – no embeddings merged.",
            embeddings_dir,
        )

    # ------------------------------------------------------------------ #
    # Verify split column is present
    # ------------------------------------------------------------------ #
    if "split" not in base.columns:
        raise RuntimeError(
            "The 'split' column is missing from the final dataset. "
            "Ensure data_splits.parquet contains a 'split' column."
        )

    # ------------------------------------------------------------------ #
    # Save final dataset inside the classifications directory
    # ------------------------------------------------------------------ #
    output_path = str(os.path.join(classifications_dir, "final_cdss_dataset.parquet"))
    base.to_parquet(output_path, index=False)
    logger.info(
        "Saved final CDSS dataset to %s  (shape=%s)", output_path, base.shape
    )
