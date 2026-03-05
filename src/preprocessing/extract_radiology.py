"""
extract_radiology.py – Most-recent radiology note per admission.

Text cleaning removes everything before the first "EXAMINATION:" marker.

Expected config keys:
    MIMIC_DATA_DIR  – root directory containing MIMIC-IV tables
    FEATURES_DIR    – output directory for feature parquets
    MIMIC_NOTE_DIR  – (optional) root of the mimic-iv-note module; radiology
                      notes are looked up under <MIMIC_NOTE_DIR>/note/ first
    HASH_REGISTRY_PATH – (optional) path to the source-file hash registry
"""

import logging
import os

import pandas as pd

from preprocessing_utils import (
    _gz_or_csv,
    _load_csv,
    _record_hashes,
    _sources_unchanged,
)

logger = logging.getLogger(__name__)


def _clean_note(text: str) -> str:
    """Remove all text before the first occurrence of 'EXAMINATION:'."""
    if not isinstance(text, str):
        return ""
    marker = "EXAMINATION:"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[idx:]


def _resolve_note_path(mimic_dir: str, note_dir: str, table: str) -> str:
    """Return the resolved path of a note CSV, checking note_dir/note/ first.

    Search order:
      1. <note_dir>/note/<table>.csv[.gz]
      2. <mimic_dir>/note/<table>.csv[.gz]
      3. <mimic_dir>/hosp/<table>.csv[.gz]  (last resort)
    """
    for base, subdir in [
        (note_dir, "note"),
        (mimic_dir, "note"),
        (mimic_dir, "hosp"),
    ]:
        gz = os.path.join(base, subdir, f"{table}.csv.gz")
        csv = os.path.join(base, subdir, f"{table}.csv")
        if os.path.exists(gz):
            return gz
        if os.path.exists(csv):
            return csv
    # Return default path for error messages
    return os.path.join(note_dir, "note", f"{table}.csv.gz")


def run(config: dict) -> None:
    """Extract the most recent radiology note for each admission."""
    required_keys = ["MIMIC_DATA_DIR", "FEATURES_DIR"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    mimic_dir = config["MIMIC_DATA_DIR"]
    features_dir = config["FEATURES_DIR"]
    hosp_dir = os.path.join(mimic_dir, "hosp")
    note_dir = config.get("MIMIC_NOTE_DIR", mimic_dir)
    registry_path = config.get("HASH_REGISTRY_PATH", "")

    # ------------------------------------------------------------------ #
    # Hash-based skip check
    # ------------------------------------------------------------------ #
    radiology_source = _resolve_note_path(mimic_dir, note_dir, "radiology")
    source_paths = [
        p for p in [
            radiology_source,
            _gz_or_csv(mimic_dir, "hosp", "admissions"),
        ] if os.path.exists(p)
    ]
    output_paths = [os.path.join(features_dir, "radiology_features.parquet")]

    if registry_path and not config.get("FORCE_RERUN", False):
        if _sources_unchanged("extract_radiology", source_paths,
                               output_paths, registry_path, logger):
            return

    # ------------------------------------------------------------------ #
    # Load radiology notes
    # ------------------------------------------------------------------ #
    note_path = radiology_source
    if not os.path.exists(note_path):
        raise FileNotFoundError(
            f"Radiology notes file not found. Searched under "
            f"{note_dir}/note/, {mimic_dir}/note/, and {mimic_dir}/hosp/. "
            "Expected radiology.csv[.gz]."
        )

    notes = pd.read_csv(
        note_path,
        usecols=["subject_id", "hadm_id", "charttime", "text"],
        parse_dates=["charttime"],
        dtype={"subject_id": int, "hadm_id": float},
    )

    notes["hadm_id"] = notes["hadm_id"].astype("Int64")
    notes = notes.dropna(subset=["hadm_id"])
    notes["hadm_id"] = notes["hadm_id"].astype(int)
    notes["text"] = notes["text"].apply(_clean_note)
    logger.info("Loaded %d radiology notes", len(notes))

    # ------------------------------------------------------------------ #
    # Load admissions for admission window filtering
    # ------------------------------------------------------------------ #
    admissions = _load_csv(
        os.path.join(hosp_dir, "admissions.csv.gz"),
        os.path.join(hosp_dir, "admissions.csv"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        parse_dates=["admittime", "dischtime"],
        dtype={"subject_id": int, "hadm_id": int},
    )

    # ------------------------------------------------------------------ #
    # Keep only notes that fall within each admission window
    # ------------------------------------------------------------------ #
    notes_merged = notes.merge(
        admissions[["subject_id", "hadm_id", "admittime", "dischtime"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )
    in_window = (
        notes_merged["charttime"].isna()
        | (
            (notes_merged["charttime"] >= notes_merged["admittime"])
            & (notes_merged["charttime"] <= notes_merged["dischtime"])
        )
    )
    notes_merged = notes_merged[in_window]

    # ------------------------------------------------------------------ #
    # Pick the most recent note per admission
    # ------------------------------------------------------------------ #
    most_recent = (
        notes_merged.sort_values("charttime")
        .groupby(["subject_id", "hadm_id"])["text"]
        .last()
        .reset_index()
        .rename(columns={"text": "radiology_text"})
    )

    out_df = admissions[["subject_id", "hadm_id"]].merge(
        most_recent, on=["subject_id", "hadm_id"], how="left"
    )
    out_df["radiology_text"] = out_df["radiology_text"].fillna("")

    # ------------------------------------------------------------------ #
    # Save output
    # ------------------------------------------------------------------ #
    os.makedirs(features_dir, exist_ok=True)
    output_path = os.path.join(features_dir, "radiology_features.parquet")
    out_df.to_parquet(output_path, index=False)
    logger.info("Saved radiology features to %s  (shape=%s)",
                output_path, out_df.shape)

    if registry_path:
        _record_hashes("extract_radiology", source_paths, registry_path)
