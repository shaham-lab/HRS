"""
inspect_data.py – Standalone diagnostic utility for MIMIC-IV source files.

Loads each source file used by the preprocessing pipeline and prints a short
diagnostic snapshot and statistics for each. No output files are written.

Usage:
    python src/preprocessing/inspect_data.py
    python src/preprocessing/inspect_data.py --config /path/to/preprocessing.yaml
"""

import argparse
import os
import sys
from typing import Any, cast

import pandas as pd
import yaml

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.2f}".format)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "preprocessing.yaml")
_PATH_KEYS = {"MIMIC_DATA_DIR", "FEATURES_DIR", "EMBEDDINGS_DIR", "CLASSIFICATIONS_DIR"}


# ---------------------------------------------------------------------------
# Config loading (same pattern as run_pipeline.py)
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Configuration file {config_path} must contain a YAML mapping."
        )
    for key in _PATH_KEYS:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = os.path.expanduser(cfg[key])
    return cast(dict, cfg)


# ---------------------------------------------------------------------------
# Shared helpers (imported lazily to avoid circular import with preprocessing_utils)
# ---------------------------------------------------------------------------

def _load_csv(path_gz: str, path_csv: str, **kwargs: Any) -> pd.DataFrame:
    if os.path.exists(path_gz):
        return cast(pd.DataFrame, pd.read_csv(path_gz, **kwargs))
    if os.path.exists(path_csv):
        return cast(pd.DataFrame, pd.read_csv(path_csv, **kwargs))
    raise FileNotFoundError(f"Neither {path_gz} nor {path_csv} found.")


def _print_header(label: str, path: str) -> None:
    print("\n" + "=" * 70)
    print(f"SOURCE: {label}  ({path})")
    print("=" * 70)


def _print_snapshot(df: pd.DataFrame) -> None:
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nDtypes:\n{df.dtypes.to_string()}")
    print(f"\nFirst 5 rows:\n{df.head().to_string(index=False)}")
    print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        print(f"\nNumeric summary:\n{df[num_cols].describe().to_string()}")


def _print_value_counts(
    df: pd.DataFrame, col: str, top_n: int = 20, label: str = ""
) -> None:
    title = label or col
    print(f"\n{title} — top {top_n} value counts:")
    print(df[col].value_counts().head(top_n).to_string())


def _print_category_stats(df: pd.DataFrame, col: str) -> None:
    vc = df[col].value_counts()
    print(f"\n{col} distribution ({len(vc)} unique values):")
    print(vc.to_string())
    print(f"  Missing: {df[col].isna().sum():,} ({100 * df[col].isna().mean():.1f}%)")


# ---------------------------------------------------------------------------
# Per-file inspection functions
# ---------------------------------------------------------------------------

def _inspect_admissions(mimic_dir: str) -> None:
    gz = os.path.join(mimic_dir, "hosp", "admissions.csv.gz")
    csv = os.path.join(mimic_dir, "hosp", "admissions.csv")
    _print_header("hosp/admissions", gz)
    try:
        df = _load_csv(
            gz, csv,
            usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                     "hospital_expire_flag"],
        )
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    _print_snapshot(df)

    print(f"\nUnique patients:   {df['subject_id'].nunique():,}")
    print(f"Unique admissions: {df['hadm_id'].nunique():,}")
    adm_per_pt = df.groupby("subject_id")["hadm_id"].count()
    print(
        f"Admissions per patient — mean: {adm_per_pt.mean():.2f}, "
        f"max: {adm_per_pt.max()}"
    )
    print(f"\nIn-hospital mortality rate: {df['hospital_expire_flag'].mean() * 100:.2f}%")
    print(f"Deceased admissions: {df['hospital_expire_flag'].sum():,}")
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400
    print(f"\nLength of stay (days):\n{df['los_days'].describe().to_string()}")


def _inspect_patients(mimic_dir: str) -> None:
    gz = os.path.join(mimic_dir, "hosp", "patients.csv.gz")
    csv = os.path.join(mimic_dir, "hosp", "patients.csv")
    _print_header("hosp/patients", gz)
    try:
        df = _load_csv(
            gz, csv,
            usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
        )
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    _print_snapshot(df)

    print(f"\nUnique patients: {df['subject_id'].nunique():,}")
    _print_category_stats(df, "gender")
    print(f"\nAge distribution:\n{df['anchor_age'].describe().to_string()}")
    print(f"\nAge bins:")
    bins = [0, 18, 30, 45, 65, 75, 200]
    labels = ["<18", "18-29", "30-44", "45-64", "65-74", "75+"]
    print(
        pd.cut(df["anchor_age"], bins=bins, labels=labels, right=False)
        .value_counts()
        .sort_index()
        .to_string()
    )


def _inspect_omr(mimic_dir: str) -> None:
    gz = os.path.join(mimic_dir, "hosp", "omr.csv.gz")
    csv = os.path.join(mimic_dir, "hosp", "omr.csv")
    _print_header("hosp/omr", gz)
    try:
        df = _load_csv(
            gz, csv,
            usecols=["subject_id", "chartdate", "result_name", "result_value"],
        )
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    _print_snapshot(df)

    _print_value_counts(df, "result_name", top_n=30, label="result_name frequencies")
    for keyword in ["Height", "Weight", "BMI"]:
        subset = df[df["result_name"].str.contains(keyword, case=False, na=False)]
        print(f"\n{keyword} entries: {len(subset):,}")
        print(f"  Unique result_name values: {subset['result_name'].unique().tolist()}")
        vals = pd.to_numeric(subset["result_value"], errors="coerce").dropna()
        if len(vals):
            print(
                f"  Value range: {vals.min():.1f} – {vals.max():.1f},"
                f" mean: {vals.mean():.1f}"
            )
    before = len(df)
    after = len(df.drop_duplicates(subset=["subject_id", "chartdate", "result_name"]))
    print(
        f"\nDeduplication: {before:,} → {after:,} rows"
        f" (removed {before - after:,} seq_num duplicates)"
    )


def _inspect_diagnoses_icd(mimic_dir: str) -> None:
    gz = os.path.join(mimic_dir, "hosp", "diagnoses_icd.csv.gz")
    csv = os.path.join(mimic_dir, "hosp", "diagnoses_icd.csv")
    _print_header("hosp/diagnoses_icd", gz)
    try:
        df = _load_csv(
            gz, csv,
            usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        )
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    _print_snapshot(df)

    print(f"\nUnique patients:   {df['subject_id'].nunique():,}")
    print(f"Unique admissions: {df['hadm_id'].nunique():,}")
    _print_category_stats(df, "icd_version")
    print(
        f"\nDiagnoses per admission:\n"
        f"{df.groupby('hadm_id')['icd_code'].count().describe().to_string()}"
    )


def _inspect_d_icd_diagnoses(mimic_dir: str) -> None:
    gz = os.path.join(mimic_dir, "hosp", "d_icd_diagnoses.csv.gz")
    csv = os.path.join(mimic_dir, "hosp", "d_icd_diagnoses.csv")
    _print_header("hosp/d_icd_diagnoses", gz)
    try:
        df = _load_csv(
            gz, csv,
            usecols=["icd_code", "icd_version", "long_title"],
        )
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    _print_snapshot(df)

    _print_category_stats(df, "icd_version")
    print(
        f"\nSample long_titles:\n"
        f"{df['long_title'].dropna().head(10).to_string(index=False)}"
    )


def _inspect_d_labitems(mimic_dir: str) -> None:
    gz = os.path.join(mimic_dir, "hosp", "d_labitems.csv.gz")
    csv = os.path.join(mimic_dir, "hosp", "d_labitems.csv")
    _print_header("hosp/d_labitems", gz)
    try:
        df = _load_csv(
            gz, csv,
            usecols=["itemid", "label", "fluid", "category"],
        )
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    df["fluid"] = df["fluid"].str.strip()
    df["category"] = df["category"].str.strip()
    _print_snapshot(df)

    _print_category_stats(df, "fluid")
    _print_category_stats(df, "category")
    print(f"\nFluid × Category combinations:")
    print(
        df.groupby(["fluid", "category"])
        .size()
        .reset_index(name="count")
        .to_string(index=False)
    )
    print(
        f"\nArtifact rows (fluid in ['I','Q','fluid']): "
        f"{df[df['fluid'].isin(['I', 'Q', 'fluid'])].shape[0]}"
    )


def _inspect_labevents(mimic_dir: str) -> None:
    gz = os.path.join(mimic_dir, "hosp", "labevents.csv.gz")
    csv = os.path.join(mimic_dir, "hosp", "labevents.csv")
    _print_header("hosp/labevents (first 50,000 rows)", gz)
    try:
        df = _load_csv(
            gz, csv,
            usecols=[
                "subject_id", "hadm_id", "itemid", "charttime",
                "value", "valuenum", "valueuom",
                "ref_range_lower", "ref_range_upper", "flag", "priority",
            ],
            nrows=50_000,
        )
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    _print_snapshot(df)

    print(
        f"\nhadm_id — null: {df['hadm_id'].isna().sum():,}"
        f" ({100 * df['hadm_id'].isna().mean():.1f}% — outpatient rows)"
    )
    print(
        f"valuenum — null: {df['valuenum'].isna().sum():,}"
        f" ({100 * df['valuenum'].isna().mean():.1f}% — qualitative results)"
    )
    print(
        f"Both value and valuenum null:"
        f" {(df['value'].isna() & df['valuenum'].isna()).sum():,}"
    )
    _print_category_stats(df, "flag")
    _print_category_stats(df, "priority")
    print(f"\nvaluenum range (numeric results only):\n{df['valuenum'].describe().to_string()}")
    _print_value_counts(df, "valueuom", top_n=20, label="Unit of measure frequencies")
    linked = df.dropna(subset=["hadm_id"])
    print(f"\nAdmission-linked rows: {len(linked):,}")
    print(f"Unique itemids in sample: {df['itemid'].nunique():,}")


def _inspect_chartevents(mimic_dir: str) -> None:
    gz = os.path.join(mimic_dir, "icu", "chartevents.csv.gz")
    csv = os.path.join(mimic_dir, "icu", "chartevents.csv")
    _print_header("icu/chartevents (first 50,000 rows)", gz)
    try:
        df = _load_csv(
            gz, csv,
            usecols=["subject_id", "hadm_id", "itemid", "valuenum", "valueuom", "charttime"],
            nrows=50_000,
        )
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    _print_snapshot(df)

    VITAL_LABELS = {
        226707: "Height (Inch)",
        226730: "Height (cm)",
        226512: "Admission Weight (Kg)",
        224639: "Daily Weight (kg)",
        226531: "Admission Weight (lbs.)",
        226846: "Feeding Weight (kg)",
    }
    vitals = df[df["itemid"].isin(VITAL_LABELS.keys())].copy()
    vitals["label"] = vitals["itemid"].map(VITAL_LABELS)
    print(f"\nHeight/Weight rows in sample: {len(vitals):,}")
    if not vitals.empty:
        print(vitals[["itemid", "label", "valuenum", "valueuom"]].to_string(index=False))
        print(f"\nPer-itemid statistics:")
        print(
            vitals.groupby(["itemid", "label"])["valuenum"]
            .agg(["count", "mean", "min", "max"])
            .to_string()
        )
    _print_value_counts(df, "itemid", top_n=20, label="Most frequent itemids")


def _inspect_discharge(mimic_dir: str) -> None:
    gz_note = os.path.join(mimic_dir, "note", "discharge.csv.gz")
    csv_note = os.path.join(mimic_dir, "note", "discharge.csv")
    gz_hosp = os.path.join(mimic_dir, "hosp", "discharge.csv.gz")
    csv_hosp = os.path.join(mimic_dir, "hosp", "discharge.csv")
    # Try note/ first, then hosp/ as fallback
    path_gz = gz_note if os.path.exists(gz_note) else gz_hosp
    path_csv = csv_note if os.path.exists(csv_note) else csv_hosp
    _print_header("note/discharge (fallback hosp/discharge)", path_gz)
    try:
        df = _load_csv(
            path_gz, path_csv,
            usecols=["subject_id", "hadm_id", "charttime", "text"],
        )
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    _print_snapshot(df)

    print(f"\nUnique patients:   {df['subject_id'].nunique():,}")
    print(f"Unique admissions: {df['hadm_id'].nunique():,}")
    print(f"hadm_id null: {df['hadm_id'].isna().sum():,}")
    print(f"\nFirst 3 non-empty note previews (first 300 chars each):")
    for i, text in enumerate(df["text"].dropna().head(3)):
        print(f"\n  [{i + 1}] {str(text)[:300].strip()!r}")
    has_marker = df["text"].str.contains("Allergies:", na=False)
    print(
        f"\n'Allergies:' marker present: {has_marker.sum():,} / {len(df):,}"
        f" notes ({100 * has_marker.mean():.1f}%)"
    )


def _inspect_radiology(mimic_dir: str) -> None:
    gz_note = os.path.join(mimic_dir, "note", "radiology.csv.gz")
    csv_note = os.path.join(mimic_dir, "note", "radiology.csv")
    gz_hosp = os.path.join(mimic_dir, "hosp", "radiology.csv.gz")
    csv_hosp = os.path.join(mimic_dir, "hosp", "radiology.csv")
    path_gz = gz_note if os.path.exists(gz_note) else gz_hosp
    path_csv = csv_note if os.path.exists(csv_note) else csv_hosp
    _print_header("note/radiology (fallback hosp/radiology)", path_gz)
    try:
        df = _load_csv(
            path_gz, path_csv,
            usecols=["subject_id", "hadm_id", "charttime", "text"],
        )
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    _print_snapshot(df)

    print(f"\nUnique patients:   {df['subject_id'].nunique():,}")
    print(f"Unique admissions: {df['hadm_id'].nunique():,}")
    print(f"hadm_id null: {df['hadm_id'].isna().sum():,}")
    print(f"\nFirst 3 non-empty note previews (first 300 chars each):")
    for i, text in enumerate(df["text"].dropna().head(3)):
        print(f"\n  [{i + 1}] {str(text)[:300].strip()!r}")
    has_marker = df["text"].str.contains("EXAMINATION:", na=False)
    print(
        f"\n'EXAMINATION:' marker present: {has_marker.sum():,} / {len(df):,}"
        f" notes ({100 * has_marker.mean():.1f}%)"
    )


def _inspect_triage(mimic_dir: str) -> None:
    gz_ed = os.path.join(mimic_dir, "ed", "triage.csv.gz")
    csv_ed = os.path.join(mimic_dir, "ed", "triage.csv")
    gz_hosp = os.path.join(mimic_dir, "hosp", "triage.csv.gz")
    csv_hosp = os.path.join(mimic_dir, "hosp", "triage.csv")
    path_gz = gz_ed if os.path.exists(gz_ed) else gz_hosp
    path_csv = csv_ed if os.path.exists(csv_ed) else csv_hosp
    _print_header("ed/triage (fallback hosp/triage)", path_gz)
    try:
        df = _load_csv(path_gz, path_csv)
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    _print_snapshot(df)

    print(f"\nUnique patients:   {df['subject_id'].nunique():,}")
    if "hadm_id" in df.columns:
        print(
            f"hadm_id null: {df['hadm_id'].isna().sum():,}"
            f" ({100 * df['hadm_id'].isna().mean():.1f}%)"
        )
    if "chiefcomplaint" in df.columns:
        print(f"\nchiefcomplaint — null: {df['chiefcomplaint'].isna().sum():,}")
        _print_value_counts(df, "chiefcomplaint", top_n=20, label="Top chief complaints")
    for col in ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]:
        if col in df.columns:
            print(f"\n{col}:\n{df[col].describe().to_string()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect MIMIC-IV source files — diagnostic snapshots only, no output files.",
    )
    parser.add_argument(
        "--config",
        default=_CONFIG_PATH,
        help=f"Path to preprocessing.yaml (default: {_CONFIG_PATH})",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    mimic_dir = config["MIMIC_DATA_DIR"]

    print(f"MIMIC_DATA_DIR: {mimic_dir}")

    _inspect_admissions(mimic_dir)
    _inspect_patients(mimic_dir)
    _inspect_omr(mimic_dir)
    _inspect_diagnoses_icd(mimic_dir)
    _inspect_d_icd_diagnoses(mimic_dir)
    _inspect_d_labitems(mimic_dir)
    _inspect_labevents(mimic_dir)
    _inspect_chartevents(mimic_dir)
    _inspect_discharge(mimic_dir)
    _inspect_radiology(mimic_dir)
    _inspect_triage(mimic_dir)

    print("\n" + "=" * 70)
    print("Inspection complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
