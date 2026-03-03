"""
extract_discharge_history.py – Prior-visit discharge summary text.

For each admission, retrieves discharge notes from all prior admissions of
the same patient (strictly before current admittime). Text cleaning removes
everything before the first "Allergies:" marker.

Expected config keys:
    MIMIC_DATA_DIR  – root directory containing MIMIC-IV tables
    FEATURES_DIR    – output directory for feature parquets
"""

import logging
import os

import pandas as pd

from utils import _load_csv

logger = logging.getLogger(__name__)


def _clean_note(text: str) -> str:
    """Remove all text before the first occurrence of 'Allergies:'."""
    if not isinstance(text, str):
        return ""
    marker = "Allergies:"
    idx = text.find(marker)
    if idx == -1:
        return text
    return text[idx:]


def run(config: dict) -> None:
    """Extract prior-visit discharge summary text for each admission."""
    required_keys = ["MIMIC_DATA_DIR", "FEATURES_DIR"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    mimic_dir = config["MIMIC_DATA_DIR"]
    features_dir = config["FEATURES_DIR"]
    hosp_dir = os.path.join(mimic_dir, "hosp")

    # ------------------------------------------------------------------ #
    # Load notes (discharge type)
    # ------------------------------------------------------------------ #
    note_path_gz = os.path.join(hosp_dir, "discharge.csv.gz")
    note_path_csv = os.path.join(hosp_dir, "discharge.csv")
    # MIMIC-IV note table may be stored under note/discharge.*
    note_dir = os.path.join(mimic_dir, "note")
    note_path_gz2 = os.path.join(note_dir, "discharge.csv.gz")
    note_path_csv2 = os.path.join(note_dir, "discharge.csv")

    for gz, csv in [
        (note_path_gz, note_path_csv),
        (note_path_gz2, note_path_csv2),
    ]:
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
            "Discharge notes file not found. Expected at "
            f"{note_path_gz} or {note_path_gz2}"
        )

    notes["hadm_id"] = notes["hadm_id"].astype("Int64")
    notes = notes.dropna(subset=["hadm_id"])
    notes["hadm_id"] = notes["hadm_id"].astype(int)
    notes["text"] = notes["text"].apply(_clean_note)

    logger.info("Loaded %d discharge notes", len(notes))

    # ------------------------------------------------------------------ #
    # Load admissions
    # ------------------------------------------------------------------ #
    admissions = _load_csv(
        os.path.join(hosp_dir, "admissions.csv.gz"),
        os.path.join(hosp_dir, "admissions.csv"),
        usecols=["subject_id", "hadm_id", "admittime"],
        parse_dates=["admittime"],
        dtype={"subject_id": int, "hadm_id": int},
    )

    # ------------------------------------------------------------------ #
    # Attach note admission time
    # ------------------------------------------------------------------ #
    notes_with_time = notes.merge(
        admissions[["subject_id", "hadm_id", "admittime"]].rename(
            columns={"hadm_id": "note_hadm_id", "admittime": "note_admittime"}
        ),
        left_on=["subject_id", "hadm_id"],
        right_on=["subject_id", "note_hadm_id"],
        how="left",
    )

    # ------------------------------------------------------------------ #
    # For each admission, concatenate notes from prior admissions only
    # ------------------------------------------------------------------ #
    logger.info(
        "Building prior-visit discharge text for %d admissions…", len(admissions)
    )
    merged = admissions.merge(
        notes_with_time[["subject_id", "note_hadm_id", "note_admittime", "text"]],
        on="subject_id",
        how="left",
    )
    prior_mask = merged["note_admittime"] < merged["admittime"]
    prior = merged[prior_mask]

    discharge_text = (
        prior.sort_values("note_admittime")
        .groupby(["subject_id", "hadm_id"])["text"]
        .apply(lambda note_texts: "\n\n---\n\n".join(n for n in note_texts if n))
        .reset_index()
        .rename(columns={"text": "discharge_history_text"})
    )

    out_df = admissions[["subject_id", "hadm_id"]].merge(
        discharge_text, on=["subject_id", "hadm_id"], how="left"
    )
    out_df["discharge_history_text"] = out_df["discharge_history_text"].fillna("")

    # ------------------------------------------------------------------ #
    # Save output
    # ------------------------------------------------------------------ #
    os.makedirs(features_dir, exist_ok=True)
    output_path = os.path.join(features_dir, "discharge_history_features.parquet")
    out_df.to_parquet(output_path, index=False)
    logger.info(
        "Saved discharge history features to %s  (shape=%s)",
        output_path, out_df.shape,
    )
