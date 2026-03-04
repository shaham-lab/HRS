"""
extract_demographics.py – Extract age, gender, height, weight, BMI.

Feature vector (demographic_vec): 8 floats
    [Age, Gender, Height (cm), Weight (kg), BMI,
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

from preprocessing_utils import _load_csv

logger = logging.getLogger(__name__)

# Fix 1: MIMIC-IV only chartevents item IDs for height and weight
# {itemid: unit}
_CHART_HEIGHT_ITEMS: dict[int, str] = {
    226707: "inch",   # Height          → convert to cm (* 2.54)
    226730: "cm",     # Height (cm)     → use as-is
}

# List of (itemid, unit) in descending priority
_CHART_WEIGHT_ITEMS: list[tuple[int, str]] = [
    (226512, "kg"),   # Admission Weight (Kg)   → highest priority
    (224639, "kg"),   # Daily Weight
    (226531, "lbs"),  # Admission Weight (lbs.) → convert to kg (* 0.453592)
    (226846, "kg"),   # Feeding Weight          → lowest priority
]
_CHART_WEIGHT_ITEMIDS: list[int] = [i for i, _ in _CHART_WEIGHT_ITEMS]
_CHART_WEIGHT_UNIT: dict[int, str] = {i: u for i, u in _CHART_WEIGHT_ITEMS}

# Age bins boundaries (left-closed)
_AGE_BINS = [18, 30, 45, 65, 75, 200]
_AGE_LABELS = ["18-29", "30-44", "45-64", "65-74", "75+"]


def _age_bin(age: float) -> str:
    for i, (lo, hi) in enumerate(zip(_AGE_BINS[:-1], _AGE_BINS[1:])):
        if lo <= age < hi:
            return _AGE_LABELS[i]
    return _AGE_LABELS[-1]


def _load_admissions(mimic_dir: str) -> pd.DataFrame:
    hosp = os.path.join(mimic_dir, "hosp")
    return _load_csv(
        os.path.join(hosp, "admissions.csv.gz"),
        os.path.join(hosp, "admissions.csv"),
        usecols=["subject_id", "hadm_id", "admittime"],
        parse_dates=["admittime"],
        dtype={"subject_id": int, "hadm_id": int},
    )


def _load_patients(mimic_dir: str) -> pd.DataFrame:
    hosp = os.path.join(mimic_dir, "hosp")
    return _load_csv(
        os.path.join(hosp, "patients.csv.gz"),
        os.path.join(hosp, "patients.csv"),
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        dtype={"subject_id": int},
    )


def _load_omr(mimic_dir: str) -> pd.DataFrame:
    hosp = os.path.join(mimic_dir, "hosp")
    gz  = os.path.join(hosp, "omr.csv.gz")
    csv = os.path.join(hosp, "omr.csv")
    if not os.path.exists(gz) and not os.path.exists(csv):
        logger.warning("OMR table not found – will rely on chartevents only.")
        empty = pd.DataFrame(
            columns=["subject_id", "chartdate", "result_name", "result_value"]
        )
        empty["chartdate"] = pd.to_datetime(empty["chartdate"])
        return empty
    return _load_csv(
        gz, csv,
        usecols=["subject_id", "chartdate", "result_name", "result_value"],
        parse_dates=["chartdate"],
        dtype={"subject_id": int},
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
    """Return per-admission latest height (cm), weight (kg), BMI from OMR."""
    # Fix 3: Deduplicate OMR to remove seq_num duplicates
    omr = omr.drop_duplicates(subset=["subject_id", "chartdate", "result_name"]).copy()
    # Fix 3: Cast chartdate to datetime explicitly for safe comparison with admittime
    omr["chartdate"] = pd.to_datetime(omr["chartdate"], errors="coerce")

    omr_height = omr[omr["result_name"].str.lower().str.contains("height", na=False)].copy()
    omr_weight = omr[omr["result_name"].str.lower().str.contains("weight", na=False)].copy()
    omr_bmi = omr[omr["result_name"].str.lower().str.contains("bmi", na=False)].copy()

    # Fix 3: Normalise height to cm
    omr_height["result_value"] = pd.to_numeric(omr_height["result_value"], errors="coerce").astype(float)
    inches_mask = omr_height["result_name"].str.lower().str.contains("inches", na=False)
    omr_height.loc[inches_mask, "result_value"] = (
        omr_height.loc[inches_mask, "result_value"] * 2.54
    )

    # Fix 3: Normalise weight to kg
    omr_weight["result_value"] = pd.to_numeric(omr_weight["result_value"], errors="coerce").astype(float)
    lbs_mask = omr_weight["result_name"].str.lower().str.contains("lbs", na=False)
    omr_weight.loc[lbs_mask, "result_value"] = (
        omr_weight.loc[lbs_mask, "result_value"] * 0.453592
    )

    omr_bmi["result_value"] = pd.to_numeric(omr_bmi["result_value"], errors="coerce").astype(float)

    def _latest_before_admit(vital_df: pd.DataFrame, col: str) -> pd.DataFrame:
        # Join on subject_id, keep rows where chartdate <= admittime
        merged = admissions.merge(vital_df, on="subject_id", how="left")
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

    h = _latest_before_admit(omr_height, "omr_height_cm")
    w = _latest_before_admit(omr_weight, "omr_weight_kg")
    b = _latest_before_admit(omr_bmi, "omr_bmi")

    result = admissions[["subject_id", "hadm_id"]].copy()
    for df in [h, w, b]:
        result = result.merge(df, on=["subject_id", "hadm_id"], how="left")
    return result


def _extract_chart_vitals(
    mimic_dir: str, admissions: pd.DataFrame
) -> pd.DataFrame:
    # Fix 2: Use new itemid collections (no BMI from chartevents)
    all_item_ids = list(_CHART_HEIGHT_ITEMS.keys()) + _CHART_WEIGHT_ITEMIDS
    weight_priority = {item_id: rank for rank, (item_id, _) in enumerate(_CHART_WEIGHT_ITEMS)}

    icu = os.path.join(mimic_dir, "icu")
    gz  = os.path.join(icu, "chartevents.csv.gz")
    csv = os.path.join(icu, "chartevents.csv")
    if not os.path.exists(gz) and not os.path.exists(csv):
        logger.warning("chartevents table not found – no fallback for vitals")
        result = admissions[["subject_id", "hadm_id"]].copy()
        result["chart_height_cm"] = np.nan
        result["chart_weight_kg"] = np.nan
        return result
    path = gz if os.path.exists(gz) else csv

    logger.info("Loading chartevents (may be large)…")
    height_chunks: list = []
    weight_chunks: list = []

    # Fix 2 CRITICAL: apply unit conversion and range filtering within each chunk
    for chunk in pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "itemid", "valuenum", "charttime"],
        dtype={"subject_id": int, "hadm_id": float, "itemid": int},
        parse_dates=["charttime"],
        chunksize=500_000,
    ):
        chunk = chunk[chunk["itemid"].isin(all_item_ids)].copy()
        if chunk.empty:
            continue

        # Height rows: convert inches→cm, filter implausible values
        h_chunk = chunk[chunk["itemid"].isin(list(_CHART_HEIGHT_ITEMS.keys()))].copy()
        if not h_chunk.empty:
            inch_mask = h_chunk["itemid"] == 226707
            h_chunk.loc[inch_mask, "valuenum"] = h_chunk.loc[inch_mask, "valuenum"] * 2.54
            h_chunk = h_chunk[(h_chunk["valuenum"] >= 50) & (h_chunk["valuenum"] <= 250)]
            height_chunks.append(h_chunk)

        # Weight rows: convert lbs→kg, filter implausible values
        w_chunk = chunk[chunk["itemid"].isin(_CHART_WEIGHT_ITEMIDS)].copy()
        if not w_chunk.empty:
            lbs_mask = w_chunk["itemid"] == 226531
            w_chunk.loc[lbs_mask, "valuenum"] = w_chunk.loc[lbs_mask, "valuenum"] * 0.453592
            w_chunk = w_chunk[(w_chunk["valuenum"] >= 20) & (w_chunk["valuenum"] <= 400)]
            weight_chunks.append(w_chunk)

    adm = admissions[["subject_id", "hadm_id"]].copy()

    if height_chunks:
        h_all = pd.concat(height_chunks, ignore_index=True)
        h_all["hadm_id"] = h_all["hadm_id"].astype("Int64")
        first_h = (
            h_all.sort_values("charttime")
            .groupby(["subject_id", "hadm_id"])["valuenum"]
            .first()
            .reset_index()
            .rename(columns={"valuenum": "chart_height_cm"})
        )
        chart_height = adm.merge(first_h, on=["subject_id", "hadm_id"], how="left")
    else:
        chart_height = adm.copy().assign(chart_height_cm=np.nan)

    if weight_chunks:
        w_all = pd.concat(weight_chunks, ignore_index=True)
        w_all["hadm_id"] = w_all["hadm_id"].astype("Int64")
        w_all["priority"] = w_all["itemid"].map(weight_priority)
        first_w = (
            w_all.sort_values(["priority", "charttime"])
            .groupby(["subject_id", "hadm_id"])["valuenum"]
            .first()
            .reset_index()
            .rename(columns={"valuenum": "chart_weight_kg"})
        )
        chart_weight = adm.merge(first_w, on=["subject_id", "hadm_id"], how="left")
    else:
        chart_weight = adm.copy().assign(chart_weight_kg=np.nan)

    result = adm.copy()
    result = result.merge(chart_height, on=["subject_id", "hadm_id"], how="left")
    result = result.merge(chart_weight, on=["subject_id", "hadm_id"], how="left")
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
    # Fix 7: Use .map() so unknown gender becomes NaN instead of False
    merged["gender_numeric"] = merged["gender"].map({"M": 1.0, "F": 0.0})
    return merged[["subject_id", "hadm_id", "age", "gender_numeric"]]


def _compute_imputation_stats(
    df: pd.DataFrame, splits: pd.DataFrame, classifications_dir: str
) -> dict:
    """Compute per-stratum height/weight stats from train split only."""
    train_mask = splits["split"] == "train"
    train_ids = splits.loc[train_mask, "hadm_id"]
    train_df = df[df["hadm_id"].isin(train_ids)].copy()

    train_df["age_bin"] = pd.cut(
        train_df["age"], bins=_AGE_BINS, labels=_AGE_LABELS, right=False
    ).astype(str)
    train_df["stratum"] = (
        train_df["age_bin"].astype(str) + "_" + train_df["gender_numeric"].astype(str)
    )

    stats: dict = {}
    for stratum, grp in train_df.groupby("stratum"):
        stats[stratum] = {
            "height_cm_mean": float(grp["height_cm"].mean()),
            "height_cm_std":  float(grp["height_cm"].std()),
            "weight_kg_mean": float(grp["weight_kg"].mean()),
            "weight_kg_std":  float(grp["weight_kg"].std()),
        }

    global_stats = {
        "height_cm_mean": float(train_df["height_cm"].mean()),
        "height_cm_std":  float(train_df["height_cm"].std()),
        "weight_kg_mean": float(train_df["weight_kg"].mean()),
        "weight_kg_std":  float(train_df["weight_kg"].std()),
    }
    stats["__global__"] = global_stats

    os.makedirs(classifications_dir, exist_ok=True)
    stats_path = os.path.join(classifications_dir, "imputation_stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    logger.info("Saved imputation statistics to %s", stats_path)
    return stats


def _impute_vectorised(
    df: pd.DataFrame,
    col: str,
    stats: dict,
    rng: np.random.Generator,
) -> pd.Series:
    """Vectorised imputation: sample from N(mean, std) per stratum group.

    Falls back to global stats when a stratum has no entry.
    """
    result = df[col].copy()
    missing_mask = result.isna()
    if not missing_mask.any():
        return result

    missing_df = df[missing_mask].copy()
    for stratum, grp_idx in missing_df.groupby("stratum").groups.items():
        _s = stats.get(str(stratum))
        s: dict = _s if _s is not None else stats["__global__"]
        mean = s[f"{col}_mean"]
        std = s[f"{col}_std"]
        n = len(grp_idx)
        if pd.isna(mean):
            values = np.full(n, np.nan)
        elif pd.isna(std) or std == 0:
            values = np.full(n, float(mean))
        else:
            values = rng.normal(mean, std, size=n)
        result.loc[grp_idx] = values
    return result


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

    # Fix 4: Merge canonical columns
    df["height_cm"] = df["omr_height_cm"].combine_first(df["chart_height_cm"])
    df["weight_kg"] = df["omr_weight_kg"].combine_first(df["chart_weight_kg"])
    df["bmi"]       = df["omr_bmi"]  # BMI only from OMR

    # Log coverage percentages after merging sources
    n_total = len(df)
    pct_h = 100.0 * df["height_cm"].notna().sum() / n_total
    pct_w = 100.0 * df["weight_kg"].notna().sum() / n_total
    pct_b = 100.0 * df["bmi"].notna().sum() / n_total
    logger.info(
        "Coverage after merge — height: %.1f%%, weight: %.1f%%, BMI: %.1f%%",
        pct_h, pct_w, pct_b,
    )

    # ------------------------------------------------------------------ #
    # Missingness indicators (before imputation)
    # ------------------------------------------------------------------ #
    # Fix 4: Use canonical column names
    df["height_missing"] = df["height_cm"].isna().astype(float)
    df["weight_missing"] = df["weight_kg"].isna().astype(float)
    df["bmi_missing"]    = df["bmi"].isna().astype(float)

    # ------------------------------------------------------------------ #
    # Compute imputation statistics from train split, then impute
    # ------------------------------------------------------------------ #
    logger.info("Computing imputation statistics from train split…")
    stats = _compute_imputation_stats(df, splits, classifications_dir)

    df["age_bin"] = pd.cut(
        df["age"], bins=_AGE_BINS, labels=_AGE_LABELS, right=False
    ).astype(str)
    df["stratum"] = df["age_bin"].astype(str) + "_" + df["gender_numeric"].astype(str)
    rng = np.random.default_rng(seed=42)

    logger.info("Imputing missing height and weight…")
    # Fix 6: Vectorised imputation
    df["height_cm"] = _impute_vectorised(df, "height_cm", stats, rng)
    df["weight_kg"] = _impute_vectorised(df, "weight_kg", stats, rng)

    # BMI: derive from height/weight if still missing.
    # Fix 5: height is now in cm; divide by 100 to get metres
    still_missing_bmi = df["bmi"].isna()
    if still_missing_bmi.any():
        height_m = df.loc[still_missing_bmi, "height_cm"] / 100.0
        weight_kg = df.loc[still_missing_bmi, "weight_kg"]
        df.loc[still_missing_bmi, "bmi"] = weight_kg / (height_m ** 2)

    # ------------------------------------------------------------------ #
    # Assemble feature vector
    # ------------------------------------------------------------------ #
    logger.info("Assembling demographic_vec…")
    # Fix 8: Use canonical column names
    feature_cols = [
        "age", "gender_numeric", "height_cm", "weight_kg", "bmi",
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
