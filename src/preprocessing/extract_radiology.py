"""
extract_radiology.py – Most-recent radiology note per admission.

Text cleaning removes everything before the first "EXAMINATION:" marker.

Expected config keys:
    MIMIC_DATA_DIR  – root directory containing MIMIC-IV tables
    FEATURES_DIR    – output directory for feature parquets
"""

import logging
import os

from preprocessing_utils import _load_csv

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


def run(config: dict) -> None:
    """Extract the most recent radiology note for each admission."""
    required_keys = ["MIMIC_DATA_DIR", "FEATURES_DIR"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    mimic_dir = config["MIMIC_DATA_DIR"]
    features_dir = config["FEATURES_DIR"]

    hosp_dir = os.path.join(mimic_dir, "hosp")
    note_dir = os.path.join(mimic_dir, "note")

    # ------------------------------------------------------------------ #
    # Load radiology notes
    # ------------------------------------------------------------------ #
    for directory in [note_dir, hosp_dir]:
        gz = os.path.join(directory, "radiology.csv.gz")
        csv = os.path.join(directory, "radiology.csv")
        if os.path.exists(gz) or os.path.exists(csv):
            notes = _load_csv(
                gz, csv,
                usecols=["subject_id", "hadm_id", "charttime", "text"],
                parse_dates=["charttime"],
                dtype={"subject_id": int, "hadm_id": float},
            )
            break
    else:
        raise FileNotFoundError(
            f"Radiology notes file not found under {note_dir} or {hosp_dir}. "
            "Expected radiology.csv[.gz]."
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
