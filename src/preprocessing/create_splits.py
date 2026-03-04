"""
create_splits.py – Step 1 of the CDSS preprocessing pipeline.

Groups admissions by subject_id and performs a stratified 3-way split
(train / dev / test) based on patient-level hospital_expire_flag rate.

Expected config keys:
    MIMIC_DATA_DIR        – root directory containing MIMIC-IV tables
    SPLIT_TRAIN           – fraction for training set
    SPLIT_DEV             – fraction for dev set
    SPLIT_TEST            – fraction for test set
    CLASSIFICATIONS_DIR   – output directory for splits parquet
"""

import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing_utils import _gz_or_csv, _record_hashes, _sources_unchanged

logger = logging.getLogger(__name__)


def run(config: dict) -> None:
    """Create patient-level stratified train/dev/test splits."""
    # ------------------------------------------------------------------ #
    # Validate configuration
    # ------------------------------------------------------------------ #
    required_keys = [
        "MIMIC_DATA_DIR",
        "SPLIT_TRAIN",
        "SPLIT_DEV",
        "SPLIT_TEST",
        "CLASSIFICATIONS_DIR",
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    mimic_dir = config["MIMIC_DATA_DIR"]
    split_train = float(config["SPLIT_TRAIN"])
    split_dev = float(config["SPLIT_DEV"])
    split_test = float(config["SPLIT_TEST"])
    classifications_dir = config["CLASSIFICATIONS_DIR"]
    registry_path = config.get("HASH_REGISTRY_PATH", "")

    if abs(split_train + split_dev + split_test - 1.0) > 1e-6:
        raise ValueError(
            f"Split fractions must sum to 1.0, got "
            f"{split_train + split_dev + split_test:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Hash-based skip check
    # ------------------------------------------------------------------ #
    source_paths = [p for p in [
        _gz_or_csv(mimic_dir, "hosp", "admissions"),
    ] if os.path.exists(p)]
    output_paths = [os.path.join(classifications_dir, "data_splits.parquet")]

    if registry_path and not config.get("FORCE_RERUN", False):
        if _sources_unchanged("create_splits", source_paths,
                               output_paths, registry_path, logger):
            return

    # ------------------------------------------------------------------ #
    # Load admissions
    # ------------------------------------------------------------------ #
    admissions_path = os.path.join(mimic_dir, "hosp", "admissions.csv.gz")
    if not os.path.exists(admissions_path):
        admissions_path = os.path.join(mimic_dir, "hosp", "admissions.csv")
    if not os.path.exists(admissions_path):
        raise FileNotFoundError(
            f"admissions table not found in {os.path.join(mimic_dir, 'hosp')}"
        )

    logger.info("Loading admissions from %s", admissions_path)
    admissions = pd.read_csv(
        admissions_path,
        usecols=["subject_id", "hadm_id", "hospital_expire_flag"],
        dtype={"subject_id": int, "hadm_id": int, "hospital_expire_flag": int},
    )
    logger.info("Loaded %d admissions for %d patients",
                len(admissions), admissions["subject_id"].nunique())

    # ------------------------------------------------------------------ #
    # Build patient-level outcome rate for stratification
    # ------------------------------------------------------------------ #
    patient_stats = (
        admissions.groupby("subject_id")["hospital_expire_flag"]
        .mean()
        .reset_index()
        .rename(columns={"hospital_expire_flag": "outcome_rate"})
    )
    # Binary stratification label: 1 if any admission ended in death
    patient_stats["strat_label"] = (patient_stats["outcome_rate"] > 0).astype(int)

    # ------------------------------------------------------------------ #
    # Stratified split: first split off test, then split remainder into train/dev
    # ------------------------------------------------------------------ #
    dev_test_fraction = split_dev + split_test
    train_patients, devtest_patients = train_test_split(
        patient_stats,
        test_size=dev_test_fraction,
        stratify=patient_stats["strat_label"],
        random_state=42,
    )

    relative_test_size = split_test / dev_test_fraction
    dev_patients, test_patients = train_test_split(
        devtest_patients,
        test_size=relative_test_size,
        stratify=devtest_patients["strat_label"],
        random_state=42,
    )

    logger.info(
        "Patient split – train: %d, dev: %d, test: %d",
        len(train_patients), len(dev_patients), len(test_patients),
    )

    # ------------------------------------------------------------------ #
    # Map patients to split labels and join back to admissions
    # ------------------------------------------------------------------ #
    train_patients = train_patients[["subject_id"]].copy()
    train_patients["split"] = "train"
    dev_patients = dev_patients[["subject_id"]].copy()
    dev_patients["split"] = "dev"
    test_patients = test_patients[["subject_id"]].copy()
    test_patients["split"] = "test"

    patient_split = pd.concat([train_patients, dev_patients, test_patients],
                               ignore_index=True)

    splits_df = admissions[["subject_id", "hadm_id"]].merge(
        patient_split, on="subject_id", how="left"
    )

    # ------------------------------------------------------------------ #
    # Save output
    # ------------------------------------------------------------------ #
    os.makedirs(classifications_dir, exist_ok=True)
    output_path = os.path.join(classifications_dir, "data_splits.parquet")
    splits_df.to_parquet(output_path, index=False)
    logger.info("Saved splits to %s  (shape=%s)", output_path, splits_df.shape)

    if registry_path:
        _record_hashes("create_splits", source_paths, registry_path)
