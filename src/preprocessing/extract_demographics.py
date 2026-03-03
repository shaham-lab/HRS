"""
extract_demographics.py – Extract age, gender, height, weight, BMI.

Feature vector (demographic_vec): 8 floats
    [Age, Gender, Height, Weight, BMI,
     height_missing, weight_missing, bmi_missing]

Missing height / weight are imputed by sampling from
N(mean, std) derived from the train split only, stratified by
(age_bin × gender). Statistics are persisted to imputation_stats.json.
BMI is derived from height/weight when absent, never imputed independently.
No normalisation is applied.

Expected config keys:
    MIMIC_DATA_DIR       – root directory containing MIMIC-IV tables
    FEATURES_DIR         – output directory for feature parquets
    CLASSIFICATIONS_DIR  – directory containing data_splits.parquet
"""

import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Chartevents item IDs commonly used for height / weight / BMI
_CHART_HEIGHT_ITEMIDS = [226730, 1394]      # Height (Inches), Height
_CHART_WEIGHT_ITEMIDS = [224639, 763]       # Weight (kg), Weight
_CHART_BMI_ITEMIDS = [227457]               # BMI

# Age bins boundaries (left-closed)
_AGE_BINS = [18, 30, 45, 65, 75, 200]
_AGE_LABELS = ["18-29", "30-44", "45-64", "65-74", "75+"]


def _age_bin(age: float) -> str:
    for i, (lo, hi) in enumerate(zip(_AGE_BINS[:-1], _AGE_BINS[1:])):
        if lo <= age < hi:
            return _AGE_LABELS[i]
    return _AGE_LABELS[-1]


def _load_admissions(mimic_dir: str) -> pd.DataFrame:
    path = os.path.join(mimic_dir, "hosp", "admissions.csv.gz")
    if not os.path.exists(path):
        path = os.path.join(mimic_dir, "hosp", "admissions.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"admissions table not found under {mimic_dir}")
    return pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "admittime"],
        parse_dates=["admittime"],
    )


def _load_patients(mimic_dir: str) -> pd.DataFrame:
    path = os.path.join(mimic_dir, "hosp", "patients.csv.gz")
    if not os.path.exists(path):
        path = os.path.join(mimic_dir, "hosp", "patients.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"patients table not found under {mimic_dir}")
    return pd.read_csv(
        path,
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
    )


def _load_omr(mimic_dir: str) -> pd.DataFrame:
    path = os.path.join(mimic_dir, "hosp", "omr.csv.gz")
    if not os.path.exists(path):
        path = os.path.join(mimic_dir, "hosp", "omr.csv")
    if not os.path.exists(path):
        logger.warning("OMR table not found – will rely on chartevents only")
        return pd.DataFrame(
            columns=["subject_id", "chartdate", "result_name", "result_value"]
        )
    return pd.read_csv(
        path,
        usecols=["subject_id", "chartdate", "result_name", "result_value"],
        parse_dates=["chartdate"],
    )


def _load_chartevents(mimic_dir: str, item_ids: list) -> pd.DataFrame:
    path = os.path.join(mimic_dir, "icu", "chartevents.csv.gz")
    if not os.path.exists(path):
        path = os.path.join(mimic_dir, "icu", "chartevents.csv")
    if not os.path.exists(path):
        logger.warning("chartevents table not found – no fallback for vitals")
        return pd.DataFrame(
            columns=["subject_id", "hadm_id", "itemid", "valuenum", "charttime"]
        )
    logger.info("Loading chartevents (may be large)…")
    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "itemid", "valuenum", "charttime"],
        dtype={"subject_id": int, "hadm_id": float, "itemid": int},
        parse_dates=["charttime"],
        chunksize=500_000,
    ):
        chunk = chunk[chunk["itemid"].isin(item_ids)]
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame(
            columns=["subject_id", "hadm_id", "itemid", "valuenum", "charttime"]
        )
    return pd.concat(chunks, ignore_index=True)


def _extract_omr_vitals(omr: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """Return per-admission latest height, weight, BMI from OMR."""
    omr_height = omr[omr["result_name"].str.lower().str.contains("height", na=False)].copy()
    omr_weight = omr[omr["result_name"].str.lower().str.contains("weight", na=False)].copy()
    omr_bmi = omr[omr["result_name"].str.lower().str.contains("bmi", na=False)].copy()

    def _latest_before_admit(vital_df: pd.DataFrame, col: str) -> pd.DataFrame:
        # Join on subject_id, keep rows where chartdate <= admittime
        merged = admissions.merge(vital_df, on="subject_id", how="left")
        merged["result_value"] = pd.to_numeric(
            merged["result_value"], errors="coerce"
        )
        merged = merged[
            merged["chartdate"].isna()
            | (merged["chartdate"] <= merged["admittime"])
        ]
        # Take the latest
        latest = (
            merged.sort_values("chartdate")
            .groupby(["subject_id", "hadm_id"])["result_value"]
            .last()
            .reset_index()
            .rename(columns={"result_value": col})
        )
        return latest

    h = _latest_before_admit(omr_height, "omr_height")
    w = _latest_before_admit(omr_weight, "omr_weight")
    b = _latest_before_admit(omr_bmi, "omr_bmi")

    result = admissions[["subject_id", "hadm_id"]].copy()
    for df in [h, w, b]:
        result = result.merge(df, on=["subject_id", "hadm_id"], how="left")
    return result


def _extract_chart_vitals(
    mimic_dir: str, admissions: pd.DataFrame
) -> pd.DataFrame:
    all_item_ids = (
        _CHART_HEIGHT_ITEMIDS + _CHART_WEIGHT_ITEMIDS + _CHART_BMI_ITEMIDS
    )
    chart = _load_chartevents(mimic_dir, all_item_ids)
    if chart.empty:
        result = admissions[["subject_id", "hadm_id"]].copy()
        result["chart_height"] = np.nan
        result["chart_weight"] = np.nan
        result["chart_bmi"] = np.nan
        return result

    chart["hadm_id"] = chart["hadm_id"].astype("Int64")
    adm = admissions[["subject_id", "hadm_id"]].copy()

    def _agg_itemids(item_ids: list, col: str) -> pd.DataFrame:
        sub = chart[chart["itemid"].isin(item_ids)].copy()
        if sub.empty:
            return adm.copy().assign(**{col: np.nan})
        # Keep first recorded value per admission
        first = (
            sub.sort_values("charttime")
            .groupby(["subject_id", "hadm_id"])["valuenum"]
            .first()
            .reset_index()
            .rename(columns={"valuenum": col})
        )
        return adm.merge(first, on=["subject_id", "hadm_id"], how="left")

    h = _agg_itemids(_CHART_HEIGHT_ITEMIDS, "chart_height")
    w = _agg_itemids(_CHART_WEIGHT_ITEMIDS, "chart_weight")
    b = _agg_itemids(_CHART_BMI_ITEMIDS, "chart_bmi")

    result = adm.copy()
    for df in [h, w, b]:
        result = result.merge(df, on=["subject_id", "hadm_id"], how="left")
    return result


def _compute_age(patients: pd.DataFrame, admissions: pd.DataFrame) -> pd.DataFrame:
    """Approximate age at admission using anchor_age and anchor_year."""
    merged = admissions.merge(
        patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
        on="subject_id",
        how="left",
    )
    merged["admit_year"] = merged["admittime"].dt.year
    merged["age"] = merged["anchor_age"] + (
        merged["admit_year"] - merged["anchor_year"]
    )
    merged["gender_numeric"] = (merged["gender"] == "M").astype(float)
    return merged[["subject_id", "hadm_id", "age", "gender_numeric"]]


def _compute_imputation_stats(
    df: pd.DataFrame, splits: pd.DataFrame, classifications_dir: str
) -> dict:
    """Compute per-stratum height/weight stats from train split only."""
    train_mask = splits["split"] == "train"
    train_ids = splits.loc[train_mask, "hadm_id"]
    train_df = df[df["hadm_id"].isin(train_ids)].copy()

    train_df["age_bin"] = train_df["age"].apply(_age_bin)
    train_df["stratum"] = (
        train_df["age_bin"].astype(str) + "_" + train_df["gender_numeric"].astype(str)
    )

    stats: dict = {}
    for stratum, grp in train_df.groupby("stratum"):
        stats[stratum] = {
            "height_mean": float(grp["height"].mean()),
            "height_std": float(grp["height"].std()),
            "weight_mean": float(grp["weight"].mean()),
            "weight_std": float(grp["weight"].std()),
        }

    global_stats = {
        "height_mean": float(train_df["height"].mean()),
        "height_std": float(train_df["height"].std()),
        "weight_mean": float(train_df["weight"].mean()),
        "weight_std": float(train_df["weight"].std()),
    }
    stats["__global__"] = global_stats

    os.makedirs(classifications_dir, exist_ok=True)
    stats_path = os.path.join(classifications_dir, "imputation_stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Saved imputation statistics to %s", stats_path)
    return stats


def _impute_value(row, col: str, stats: dict, rng: np.random.Generator) -> float:
    stratum = f"{row['age_bin']}_{row['gender_numeric']}"
    s: dict = stats.get(stratum, stats["__global__"])
    mean = s[f"{col}_mean"]
    std = s[f"{col}_std"]
    if pd.isna(mean):
        return np.nan
    if pd.isna(std) or std == 0:
        return float(mean)
    return float(rng.normal(mean, std))


def run(config: dict) -> None:
    """Extract and save demographics feature vectors."""
    required_keys = [
        "MIMIC_DATA_DIR",
        "FEATURES_DIR",
        "CLASSIFICATIONS_DIR",
    ]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    mimic_dir = config["MIMIC_DATA_DIR"]
    features_dir = config["FEATURES_DIR"]
    classifications_dir = config["CLASSIFICATIONS_DIR"]

    # ------------------------------------------------------------------ #
    # Load splits
    # ------------------------------------------------------------------ #
    splits_path = os.path.join(classifications_dir, "data_splits.parquet")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(
            f"data_splits.parquet not found at {splits_path}. "
            "Run create_splits.py first."
        )
    splits = pd.read_parquet(splits_path)

    # ------------------------------------------------------------------ #
    # Load source tables
    # ------------------------------------------------------------------ #
    logger.info("Loading patients and admissions…")
    admissions = _load_admissions(mimic_dir)
    patients = _load_patients(mimic_dir)
    omr = _load_omr(mimic_dir)

    # ------------------------------------------------------------------ #
    # Age + gender
    # ------------------------------------------------------------------ #
    logger.info("Computing age and gender…")
    age_gender = _compute_age(patients, admissions)

    # ------------------------------------------------------------------ #
    # Height / Weight / BMI from OMR (preferred)
    # ------------------------------------------------------------------ #
    logger.info("Extracting vitals from OMR…")
    omr_vitals = _extract_omr_vitals(omr, admissions)

    # ------------------------------------------------------------------ #
    # Fallback via chartevents
    # ------------------------------------------------------------------ #
    logger.info("Extracting vitals from chartevents (fallback)…")
    chart_vitals = _extract_chart_vitals(mimic_dir, admissions)

    # ------------------------------------------------------------------ #
    # Merge and combine sources
    # ------------------------------------------------------------------ #
    df = age_gender.merge(omr_vitals, on=["subject_id", "hadm_id"], how="left")
    df = df.merge(chart_vitals, on=["subject_id", "hadm_id"], how="left")

    df["height"] = df["omr_height"].combine_first(df["chart_height"])
    df["weight"] = df["omr_weight"].combine_first(df["chart_weight"])
    df["bmi"] = df["omr_bmi"].combine_first(df["chart_bmi"])

    # ------------------------------------------------------------------ #
    # Missingness indicators (before imputation)
    # ------------------------------------------------------------------ #
    df["height_missing"] = df["height"].isna().astype(float)
    df["weight_missing"] = df["weight"].isna().astype(float)
    df["bmi_missing"] = df["bmi"].isna().astype(float)

    # ------------------------------------------------------------------ #
    # Compute imputation statistics from train split, then impute
    # ------------------------------------------------------------------ #
    logger.info("Computing imputation statistics from train split…")
    stats = _compute_imputation_stats(df, splits, classifications_dir)

    df["age_bin"] = df["age"].apply(_age_bin)
    rng = np.random.default_rng(seed=42)

    logger.info("Imputing missing height and weight…")
    missing_height_mask = df["height"].isna()
    missing_weight_mask = df["weight"].isna()

    if missing_height_mask.any():
        df.loc[missing_height_mask, "height"] = df[missing_height_mask].apply(
            lambda r: _impute_value(r, "height", stats, rng), axis=1
        )
    if missing_weight_mask.any():
        df.loc[missing_weight_mask, "weight"] = df[missing_weight_mask].apply(
            lambda r: _impute_value(r, "weight", stats, rng), axis=1
        )

    # BMI: derive from height/weight if still missing.
    # NOTE: OMR height is typically recorded in inches; chartevents height
    # (item IDs 226730, 1394) is also in inches. The factor 0.0254 converts
    # inches → metres for the BMI formula (kg / m²).
    still_missing_bmi = df["bmi"].isna()
    if still_missing_bmi.any():
        height_m = df.loc[still_missing_bmi, "height"] * 0.0254  # inches → metres
        weight_kg = df.loc[still_missing_bmi, "weight"]
        df.loc[still_missing_bmi, "bmi"] = weight_kg / (height_m ** 2)

    # ------------------------------------------------------------------ #
    # Assemble feature vector
    # ------------------------------------------------------------------ #
    logger.info("Assembling demographic_vec…")
    feature_cols = [
        "age", "gender_numeric", "height", "weight", "bmi",
        "height_missing", "weight_missing", "bmi_missing",
    ]
    df["demographic_vec"] = df[feature_cols].values.tolist()

    out_df = df[["subject_id", "hadm_id", "demographic_vec"]].copy()

    # ------------------------------------------------------------------ #
    # Save output
    # ------------------------------------------------------------------ #
    os.makedirs(features_dir, exist_ok=True)
    output_path = os.path.join(features_dir, "demographics_features.parquet")
    out_df.to_parquet(output_path, index=False)
    logger.info("Saved demographics features to %s  (shape=%s)",
                output_path, out_df.shape)
