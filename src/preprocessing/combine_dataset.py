"""
combine_dataset.py – Merge all features into the final CDSS dataset.

Reads:
  • data/preprocessing/embeddings/   – all embedding parquets
  • data/preprocessing/features/     – demographics_features.parquet
  • data/preprocessing/              – data_splits.parquet
  • data/preprocessing/classifications/ – y_labels.parquet

Performs a left join on (subject_id, hadm_id), starting from the admissions
universe defined by data_splits.parquet. Raw text parquets from features/ are
excluded. Non-lab feature values are null for admissions where that feature
is absent. Lab group embedding columns are always populated — admissions with
no events in a given lab group receive a zero vector from embed_features.py.

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

from preprocessing_utils import _check_required_keys

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


def _merge_labels(base: pd.DataFrame, classifications_dir: str) -> pd.DataFrame:
    """Merge y_labels.parquet into *base* on (subject_id, hadm_id)."""
    labels_path = os.path.join(classifications_dir, "y_labels.parquet")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(
            f"y_labels.parquet not found at {labels_path}. "
            "Run extract_y_data.py first."
        )
    logger.info("Merging y_labels…")
    labels = pd.read_parquet(labels_path)
    base = base.merge(labels, on=["subject_id", "hadm_id"], how="left")
    n_missing_y1 = int(base["y1_mortality"].isna().sum())
    if n_missing_y1:
        logger.warning("  %d admissions missing Y1 after label merge", n_missing_y1)
    return base


def _merge_feature_parquets(base: pd.DataFrame, features_dir: str) -> pd.DataFrame:
    """Merge the selected feature parquets from *features_dir* into *base*."""
    for filename in tqdm(_FEATURES_TO_INCLUDE, desc="Merging features", unit="file"):
        path = str(os.path.join(features_dir, filename))
        if not os.path.exists(path):
            logger.warning("Feature file not found, skipping: %s", path)
            continue
        logger.info("Merging feature file: %s", filename)
        feat_df = pd.read_parquet(path)
        base = base.merge(feat_df, on=["subject_id", "hadm_id"], how="left")
    return base


def _merge_embedding_parquets(base: pd.DataFrame, embeddings_dir: str) -> pd.DataFrame:
    """Merge all embedding parquets from *embeddings_dir* into *base*."""
    if not os.path.isdir(embeddings_dir):
        logger.warning(
            "Embeddings directory not found (%s) – no embeddings merged.",
            embeddings_dir,
        )
        return base
    embedding_files = sorted(
        f for f in os.listdir(embeddings_dir) if f.endswith(".parquet")
    )
    for filename in tqdm(embedding_files, desc="Merging embeddings", unit="file"):
        path = str(os.path.join(embeddings_dir, filename))
        logger.info("Merging embedding file: %s", filename)
        emb_df = pd.read_parquet(path)
        base = base.merge(emb_df, on=["subject_id", "hadm_id"], how="left")
    logger.info("  Merged %d embedding files", len(embedding_files))
    return base


def _build_canonical_columns(config: dict) -> list[str]:
    """Build the canonical column order from config files.

    Order is:
      1. Fixed metadata: subject_id, hadm_id, split
      2. Fixed labels: y1_mortality, y2_readmission
      3. Fixed structured vector: demographic_vec
      4. Fixed history/triage embeddings (F2-F5):
           diag_history_embedding, discharge_history_embedding,
           triage_embedding, chief_complaint_embedding
      5. Lab group embeddings — derived from lab_panel_config.yaml,
         keys in insertion order (same order as build_lab_panel_config.py
         produces them): f"lab_{panel_name}_embedding" for each panel
      6. Fixed radiology: radiology_embedding
      7. Microbiology panel embeddings — derived from
         micro_panel_config.yaml panels keys in insertion order:
           f"micro_{panel_name}_embedding" for each panel
    """
    import yaml

    # Load lab panel config
    lab_config_path = config.get(
        "LAB_PANEL_CONFIG_PATH",
        os.path.join(config["CLASSIFICATIONS_DIR"], "lab_panel_config.yaml"),
    )
    with open(lab_config_path) as f:
        lab_cfg = yaml.safe_load(f)
    lab_panel_names = list(lab_cfg.keys())  # insertion order

    # Load micro panel config
    micro_config_path = config["MICRO_PANEL_CONFIG_PATH"]
    with open(micro_config_path) as f:
        micro_cfg = yaml.safe_load(f)
    micro_panel_names = list(micro_cfg["panels"].keys())  # insertion order

    return (
        ["subject_id", "hadm_id", "split",
         "y1_mortality", "y2_readmission",
         "demographic_vec",
         "diag_history_embedding",
         "discharge_history_embedding",
         "triage_embedding",
         "chief_complaint_embedding"]
        + [f"lab_{name}_embedding" for name in lab_panel_names]
        + ["radiology_embedding"]
        + [f"micro_{name}_embedding" for name in micro_panel_names]
    )


def run(config: dict) -> None:
    """Combine all feature and label parquets into the final dataset."""
    _check_required_keys(config, [
        "FEATURES_DIR",
        "EMBEDDINGS_DIR",
        "CLASSIFICATIONS_DIR",
        "PREPROCESSING_DIR",
    ])

    features_dir = str(config["FEATURES_DIR"])
    embeddings_dir = str(config["EMBEDDINGS_DIR"])
    classifications_dir = str(config["CLASSIFICATIONS_DIR"])
    preprocessing_dir = str(config["PREPROCESSING_DIR"])

    steps = [
        "Load data_splits.parquet",
        "Merge y_labels",
        "Merge feature parquets",
        "Merge embedding parquets",
        "Validate and save final_cdss_dataset.parquet",
    ]
    with tqdm(total=len(steps), desc="combine_dataset", unit="step", dynamic_ncols=True) as pbar:
        # ------------------------------------------------------------------ #
        # Start from splits (defines the admission universe)
        # ------------------------------------------------------------------ #
        pbar.set_description("combine_dataset — loading data_splits.parquet")
        splits_path = os.path.join(preprocessing_dir, "data_splits.parquet")
        if not os.path.exists(splits_path):
            raise FileNotFoundError(
                f"data_splits.parquet not found at {splits_path}. "
                "Run create_splits.py first."
            )
        logger.info("Loading splits from %s…", splits_path)
        base = pd.read_parquet(splits_path)  # subject_id, hadm_id, split
        logger.info("  Splits: %d admissions (%d train  %d dev  %d test)",
                    len(base),
                    int((base["split"] == "train").sum()),
                    int((base["split"] == "dev").sum()),
                    int((base["split"] == "test").sum()))
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Merge labels
        # ------------------------------------------------------------------ #
        pbar.set_description("combine_dataset — merging y_labels")
        base = _merge_labels(base, classifications_dir)
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Merge selected feature parquets
        # ------------------------------------------------------------------ #
        pbar.set_description("combine_dataset — merging feature parquets")
        base = _merge_feature_parquets(base, features_dir)
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Merge embedding parquets (all *.parquet files in embeddings_dir)
        # ------------------------------------------------------------------ #
        pbar.set_description("combine_dataset — merging embedding parquets")
        base = _merge_embedding_parquets(base, embeddings_dir)
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Verify split column is present
        # ------------------------------------------------------------------ #
        pbar.set_description("combine_dataset — saving final_cdss_dataset.parquet")
        if "split" not in base.columns:
            raise RuntimeError(
                "The 'split' column is missing from the final dataset. "
                "Ensure data_splits.parquet contains a 'split' column."
            )

        # ------------------------------------------------------------------ #
        # Enforce canonical column order derived from config files
        # ------------------------------------------------------------------ #
        canonical_cols = _build_canonical_columns(config)
        missing = [c for c in canonical_cols if c not in base.columns]
        extra = [c for c in base.columns if c not in canonical_cols]
        if missing:
            raise ValueError(
                f"combine_dataset: {len(missing)} expected columns "
                f"missing from final DataFrame: {missing}. "
                f"Ensure all extraction and embedding steps completed."
            )
        if extra:
            logger.warning(
                "combine_dataset: %d unexpected columns will be dropped: %s",
                len(extra), extra,
            )
        base = base[canonical_cols]

        # ------------------------------------------------------------------ #
        # Save final dataset inside the classifications directory
        # ------------------------------------------------------------------ #
        output_path = str(os.path.join(classifications_dir, "final_cdss_dataset.parquet"))
        base.to_parquet(output_path, index=False)
        logger.info("  Final dataset: %d rows × %d columns", base.shape[0], base.shape[1])
        logger.info("  Columns: %s", list(base.columns))
        pbar.update(1)
