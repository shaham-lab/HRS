"""
create_splits.py – Step 1 of the CDSS preprocessing pipeline.

Groups admissions by subject_id and performs a stratified 3-way split
(train / dev / test) based on patient-level hospital_expire_flag rate.

Expected config keys:
    MIMIC_DATA_DIR        – root directory containing MIMIC-IV tables
    SPLIT_TRAIN           – fraction for training set
    SPLIT_DEV             – fraction for dev set
    SPLIT_TEST            – fraction for test set
    PREPROCESSING_DIR     – output directory for the splits parquet
"""

import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from preprocessing_utils import _check_required_keys, _gz_or_csv, _record_hashes, _setup_logging

logger = logging.getLogger(__name__)


def _compute_patient_stats(admissions: pd.DataFrame) -> pd.DataFrame:
    """Build patient-level outcome rate for stratification.

    Returns a DataFrame with subject_id, outcome_rate, strat_label.
    """
    patient_stats = (
        admissions.groupby("subject_id")["hospital_expire_flag"]
        .mean()
        .reset_index()
        .rename(columns={"hospital_expire_flag": "outcome_rate"})
    )
    # Binary stratification label: 1 if any admission ended in death
    patient_stats["strat_label"] = (patient_stats["outcome_rate"] > 0).astype(int)
    return patient_stats


def _stratified_patient_split(
    patient_stats: pd.DataFrame,
    split_train: float,
    split_dev: float,
    split_test: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified 3-way split of *patient_stats*.

    Returns (train_patients, dev_patients, test_patients) DataFrames
    each containing subject_id and split columns.
    """
    dev_test_fraction = split_dev + split_test
    train_df, devtest_df = train_test_split(
        patient_stats,
        test_size=dev_test_fraction,
        stratify=patient_stats["strat_label"],
        random_state=42,
    )

    relative_test_size = split_test / dev_test_fraction
    dev_df, test_df = train_test_split(
        devtest_df,
        test_size=relative_test_size,
        stratify=devtest_df["strat_label"],
        random_state=42,
    )
    logger.info("  Patient counts — train: %d  dev: %d  test: %d",
                len(train_df), len(dev_df), len(test_df))
    return train_df, dev_df, test_df


def _build_splits_df(
    admissions: pd.DataFrame,
    train_patients: pd.DataFrame,
    dev_patients: pd.DataFrame,
    test_patients: pd.DataFrame,
) -> pd.DataFrame:
    """Map patients to split labels and join to admission rows.

    Returns a DataFrame with subject_id, hadm_id, split columns.
    """
    train_df = train_patients[["subject_id"]].copy()
    train_df["split"] = "train"
    dev_df = dev_patients[["subject_id"]].copy()
    dev_df["split"] = "dev"
    test_df = test_patients[["subject_id"]].copy()
    test_df["split"] = "test"

    patient_split = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    return admissions[["subject_id", "hadm_id"]].merge(
        patient_split, on="subject_id", how="left"
    )


def run(config: dict) -> None:
    """Create patient-level stratified train/dev/test splits."""
    # ------------------------------------------------------------------ #
    # Validate configuration
    # ------------------------------------------------------------------ #
    _check_required_keys(config, [
        "MIMIC_DATA_DIR",
        "SPLIT_TRAIN",
        "SPLIT_DEV",
        "SPLIT_TEST",
        "PREPROCESSING_DIR",
    ])

    mimic_dir = config["MIMIC_DATA_DIR"]
    split_train = float(config["SPLIT_TRAIN"])
    split_dev = float(config["SPLIT_DEV"])
    split_test = float(config["SPLIT_TEST"])
    preprocessing_dir = config["PREPROCESSING_DIR"]
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
    output_paths = [os.path.join(preprocessing_dir, "data_splits.parquet")]

    # Resolve admissions path before starting progress bar
    admissions_path = os.path.join(mimic_dir, "hosp", "admissions.csv.gz")
    if not os.path.exists(admissions_path):
        admissions_path = os.path.join(mimic_dir, "hosp", "admissions.csv")
    if not os.path.exists(admissions_path):
        raise FileNotFoundError(
            f"admissions table not found in {os.path.join(mimic_dir, 'hosp')}"
        )

    steps = [
        "Load admissions",
        "Build patient stats",
        "Split patients",
        "Assign splits",
        "Save output",
    ]
    with tqdm(total=len(steps), desc="create_splits", unit="step", dynamic_ncols=True) as pbar:
        # ------------------------------------------------------------------ #
        # Load admissions
        # ------------------------------------------------------------------ #
        pbar.set_description("create_splits — loading admissions")
        logger.info("Loading admissions from %s", admissions_path)
        admissions = pd.read_csv(
            admissions_path,
            usecols=["subject_id", "hadm_id", "hospital_expire_flag"],
            dtype={"subject_id": int, "hadm_id": int, "hospital_expire_flag": int},
        )
        logger.info("  Loaded %d admissions for %d unique patients",
                    len(admissions), admissions["subject_id"].nunique())
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Build patient-level outcome rate for stratification
        # ------------------------------------------------------------------ #
        pbar.set_description("create_splits — computing patient outcome rates")
        patient_stats = _compute_patient_stats(admissions)
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Stratified split
        # ------------------------------------------------------------------ #
        pbar.set_description("create_splits — stratified patient split")
        train_patients, dev_patients, test_patients = _stratified_patient_split(
            patient_stats, split_train, split_dev, split_test
        )
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Map patients to split labels and join back to admissions
        # ------------------------------------------------------------------ #
        pbar.set_description("create_splits — assigning split labels to admissions")
        splits_df = _build_splits_df(admissions, train_patients, dev_patients, test_patients)
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Save output
        # ------------------------------------------------------------------ #
        pbar.set_description("create_splits — saving data_splits.parquet")
        os.makedirs(preprocessing_dir, exist_ok=True)
        output_path = os.path.join(preprocessing_dir, "data_splits.parquet")
        splits_df.to_parquet(output_path, index=False)
        logger.info("  Saved %d rows (%d admissions, %d patients) to %s",
                    len(splits_df), splits_df["hadm_id"].nunique(),
                    splits_df["subject_id"].nunique(), output_path)
        pbar.update(1)

    if registry_path:
        _record_hashes("create_splits", source_paths, registry_path)


if __name__ == "__main__":
    import argparse
    from preprocessing_utils import _load_config
    _setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/preprocessing.yaml")
    args = parser.parse_args()
    run(_load_config(args.config))

elif "snakemake" in dir():
    from preprocessing_utils import _normalize_config
    _setup_logging()
    run(_normalize_config(dict(snakemake.config)))
