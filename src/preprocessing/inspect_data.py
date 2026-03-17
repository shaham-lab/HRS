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
from typing import cast

import pandas as pd

from preprocessing_utils import _load_config, _load_csv

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.2f}".format)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "preprocessing.yaml")



# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_first_path(candidates: list[tuple[str, str]]) -> tuple[str, str]:
    """Return the first (gz, csv) pair whose gz or csv file exists.

    Falls back to ``candidates[0]`` when none of the candidates exist so that
    the subsequent ``_load_csv`` call raises a descriptive ``FileNotFoundError``.
    """
    if not candidates:
        raise ValueError("_find_first_path called with empty candidates list")
    for gz, csv in candidates:
        if os.path.exists(gz) or os.path.exists(csv):
            return gz, csv
    return candidates[0]


def _print_note_stats(df: pd.DataFrame, marker: str) -> None:
    """Print standard statistics for a notes DataFrame."""
    _print_snapshot(df)
    print(f"\nUnique patients:   {df['subject_id'].nunique():,}")
    print(f"Unique admissions: {df['hadm_id'].nunique():,}")
    print(f"hadm_id null: {df['hadm_id'].isna().sum():,}")
    print(f"\nFirst 3 non-empty note previews (first 300 chars each):")
    for i, text in enumerate(df["text"].dropna().head(3)):
        print(f"\n  [{i + 1}] {str(text)[:300].strip()!r}")
    avg_len = float(df["text"].dropna().str.len().mean())
    print(f"\nAverage note length: {avg_len:,.0f} characters (~{avg_len / 4:.0f} tokens estimated)")
    print("  Clinical_ModernBERT context window: 8,192 tokens. Notes exceeding this will be truncated.")
    has_marker = df["text"].str.contains(marker, na=False)
    print(
        f"\n{marker!r} marker present: {has_marker.sum():,} / {len(df):,}"
        f" notes ({100 * has_marker.mean():.1f}%)"
    )


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
        vals = cast(pd.Series, pd.to_numeric(subset["result_value"], errors="coerce")).dropna()
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

    null_hadm_count = df["hadm_id"].isna().sum()
    null_hadm_pct = 100 * df["hadm_id"].isna().mean()
    print(f"\nNull hadm_id: {null_hadm_count:,} ({null_hadm_pct:.1f}%) — these are outpatient/unlinked events")
    print("  → HADM_LINKAGE_STRATEGY in preprocessing.yaml controls how these are handled.")
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

    vital_labels = {
        226707: "Height (Inch)",
        226730: "Height (cm)",
        226512: "Admission Weight (Kg)",
        224639: "Daily Weight (kg)",
        226531: "Admission Weight (lbs.)",
        226846: "Feeding Weight (kg)",
    }
    vitals = df[df["itemid"].isin(vital_labels.keys())].copy()
    vitals["label"] = vitals["itemid"].map(vital_labels)
    print(f"\nHeight/Weight rows in sample: {len(vitals):,}")
    if not vitals.empty:
        print(vitals[["itemid", "label", "valuenum", "valueuom"]].to_string(index=False))
        print(f"\nPer-itemid statistics:")
        print(
            vitals.groupby(["itemid", "label"])["valuenum"]
            .agg(["count", "mean", "min", "max"])
            .to_string()
        )

    null_hadm_count = df["hadm_id"].isna().sum()
    null_hadm_pct = 100 * df["hadm_id"].isna().mean()
    print(f"\nNull hadm_id: {null_hadm_count:,} ({null_hadm_pct:.1f}%) — these are outpatient/unlinked events")
    print("  → HADM_LINKAGE_STRATEGY in preprocessing.yaml controls how these are handled.")

    _print_value_counts(df, "itemid", top_n=20, label="Most frequent itemids")


def _inspect_discharge(mimic_dir: str, note_dir: str) -> None:
    # Prefer MIMIC_NOTE_DIR/note/, then mimic_dir/note/, then mimic_dir/hosp/ as fallback
    candidates = [
        (os.path.join(note_dir, "note", "discharge.csv.gz"),
         os.path.join(note_dir, "note", "discharge.csv")),
        (os.path.join(mimic_dir, "note", "discharge.csv.gz"),
         os.path.join(mimic_dir, "note", "discharge.csv")),
        (os.path.join(mimic_dir, "hosp", "discharge.csv.gz"),
         os.path.join(mimic_dir, "hosp", "discharge.csv")),
    ]
    path_gz, path_csv = _find_first_path(candidates)
    _print_header("note/discharge (fallback hosp/discharge)", path_gz)
    try:
        df = _load_csv(
            path_gz, path_csv,
            usecols=["subject_id", "hadm_id", "charttime", "text"],
        )
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    _print_note_stats(df, "Allergies:")


def _inspect_radiology(mimic_dir: str, note_dir: str) -> None:
    # Prefer MIMIC_NOTE_DIR/note/, then mimic_dir/note/, then mimic_dir/hosp/ as fallback
    candidates = [
        (os.path.join(note_dir, "note", "radiology.csv.gz"),
         os.path.join(note_dir, "note", "radiology.csv")),
        (os.path.join(mimic_dir, "note", "radiology.csv.gz"),
         os.path.join(mimic_dir, "note", "radiology.csv")),
        (os.path.join(mimic_dir, "hosp", "radiology.csv.gz"),
         os.path.join(mimic_dir, "hosp", "radiology.csv")),
    ]
    path_gz, path_csv = _find_first_path(candidates)
    _print_header("note/radiology (fallback hosp/radiology)", path_gz)
    try:
        df = _load_csv(
            path_gz, path_csv,
            usecols=["subject_id", "hadm_id", "charttime", "text"],
        )
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    _print_note_stats(df, "EXAMINATION:")


def _inspect_triage(mimic_dir: str, ed_dir: str | None = None) -> None:
    # Prefer MIMIC_ED_DIR/ed/ → MIMIC_DATA_DIR/ed/ → MIMIC_DATA_DIR/hosp/
    candidates = []
    if ed_dir:
        candidates += [
            (os.path.join(ed_dir, "triage.csv.gz"),
             os.path.join(ed_dir, "triage.csv")),
        ]
    candidates += [
        (os.path.join(mimic_dir, "ed", "triage.csv.gz"),
         os.path.join(mimic_dir, "ed", "triage.csv")),
        (os.path.join(mimic_dir, "hosp", "triage.csv.gz"),
         os.path.join(mimic_dir, "hosp", "triage.csv")),
    ]
    path_gz, path_csv = _find_first_path(candidates)
    _print_header("ed/triage (mimic-iv-ed, fallback MIMIC_DATA_DIR/ed or hosp/triage)", path_gz)
    try:
        df = _load_csv(path_gz, path_csv)
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    _print_snapshot(df)

    print(f"\nUnique patients:   {df['subject_id'].nunique():,}")
    if "stay_id" in df.columns:
        print(f"Unique stay_ids:    {df['stay_id'].nunique():,}")
    if "hadm_id" in df.columns:
        print(
            f"hadm_id null: {df['hadm_id'].isna().sum():,}"
            f" ({100 * df['hadm_id'].isna().mean():.1f}%)"
        )
    else:
        print("\nhadm_id column absent — must be resolved via edstays.")
    if "chiefcomplaint" in df.columns:
        print(f"\nchiefcomplaint — null: {df['chiefcomplaint'].isna().sum():,}")
        _print_value_counts(df, "chiefcomplaint", top_n=20, label="Top chief complaints")
    for col in ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity"]:
        if col in df.columns:
            print(f"\n{col}:\n{df[col].describe().to_string()}")


def _inspect_edstays(mimic_dir: str, ed_dir: str | None = None) -> None:
    # Prefer MIMIC_ED_DIR/ed/ → MIMIC_DATA_DIR/ed/
    candidates = []
    if ed_dir:
        candidates += [
            (os.path.join(ed_dir, "edstays.csv.gz"),
             os.path.join(ed_dir, "edstays.csv")),
        ]
    candidates += [
        (os.path.join(mimic_dir, "ed", "edstays.csv.gz"),
         os.path.join(mimic_dir, "ed", "edstays.csv")),
    ]
    path_gz, path_csv = _find_first_path(candidates)
    _print_header("ed/edstays (mimic-iv-ed bridge table)", path_gz)
    try:
        df = _load_csv(
            path_gz, path_csv,
            usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "disposition"],
        )
    except FileNotFoundError:
        print("NOT FOUND — skipping.")
        return

    _print_snapshot(df)

    print(f"\nhadm_id null (non-admitted visits): {df['hadm_id'].isna().sum():,}"
          f" ({100 * df['hadm_id'].isna().mean():.1f}%)")
    _print_category_stats(df, "disposition")
    print(f"\nStay duration (hours):")
    df["intime"] = pd.to_datetime(df["intime"])
    df["outtime"] = pd.to_datetime(df["outtime"])
    df["los_hrs"] = (df["outtime"] - df["intime"]).dt.total_seconds() / 3600
    print(df["los_hrs"].describe().to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _inspect_lab_panel_config(classifications_dir: str) -> None:
    path = os.path.join(classifications_dir, "lab_panel_config.yaml")
    _print_header("lab_panel_config.yaml", path)
    if not os.path.exists(path):
        print("NOT FOUND — run build_lab_panel_config.py first.")
        return
    import yaml as _yaml
    with open(path, encoding="utf-8") as f:
        cfg = _yaml.safe_load(f)
    print(f"\nTotal lab groups: {len(cfg)}")
    for group_name, items in sorted(cfg.items()):
        print(f"  {group_name}: {len(items)} itemids")


def _inspect_hadm_linkage_stats(classifications_dir: str) -> None:
    path = os.path.join(classifications_dir, "hadm_linkage_stats.json")
    _print_header("hadm_linkage_stats.json", path)
    if not os.path.exists(path):
        print("NOT FOUND — will be created after pipeline runs.")
        return
    import json as _json
    with open(path, encoding="utf-8") as f:
        stats = _json.load(f)
    for module, tables in stats.items():
        print(f"\n  Module: {module}")
        for table, counts in tables.items():
            print(f"    {table}:")
            for k, v in counts.items():
                print(f"      {k}: {v:,}")


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
    mimic_dir: str = config["MIMIC_DATA_DIR"]
    note_dir: str = config.get("MIMIC_NOTE_DIR", mimic_dir)
    _ed_dir_raw = config.get("MIMIC_ED_DIR")
    ed_dir: str | None = os.path.join(str(_ed_dir_raw), "ed") if _ed_dir_raw else None

    print(f"MIMIC_DATA_DIR: {mimic_dir}")
    print(f"MIMIC_NOTE_DIR: {note_dir}")
    if ed_dir:
        print(f"MIMIC_ED_DIR/ed: {ed_dir}")

    _inspect_admissions(mimic_dir)
    _inspect_patients(mimic_dir)
    _inspect_omr(mimic_dir)
    _inspect_diagnoses_icd(mimic_dir)
    _inspect_d_icd_diagnoses(mimic_dir)
    _inspect_d_labitems(mimic_dir)
    _inspect_lab_panel_config(config.get("CLASSIFICATIONS_DIR", ""))
    _inspect_labevents(mimic_dir)
    _inspect_chartevents(mimic_dir)
    _inspect_discharge(mimic_dir, note_dir)
    _inspect_radiology(mimic_dir, note_dir)
    _inspect_triage(mimic_dir, ed_dir)
    _inspect_edstays(mimic_dir, ed_dir)
    _inspect_hadm_linkage_stats(config.get("CLASSIFICATIONS_DIR", ""))

    print("\n" + "=" * 70)
    print("Inspection complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
