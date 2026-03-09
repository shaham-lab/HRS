"""
extract_discharge_history.py – Prior-visit discharge summary text.

For each admission, retrieves discharge notes from all prior admissions of
the same patient (strictly before current admittime). Text cleaning removes
everything before the first "Allergies:" marker. Notes are concatenated with
dated header lines.

Output format:
    Prior Discharge Summary (YYYY-MM-DD):
    Allergies: Penicillin
    [clinical note body...]

    Prior Discharge Summary (YYYY-MM-DD):
    Allergies: None known
    [clinical note body...]

Expected config keys:
    MIMIC_DATA_DIR  – root directory containing MIMIC-IV tables
    FEATURES_DIR    – output directory for feature parquets
    MIMIC_NOTE_DIR  – (optional) root of the mimic-iv-note module; discharge
                      notes are looked up under <MIMIC_NOTE_DIR>/note/ first
    HASH_REGISTRY_PATH – (optional) path to the source-file hash registry

Optional config keys:
    HADM_LINKAGE_STRATEGY – "drop" (default); null hadm_id records are always
                             dropped in this module (link strategy not applicable
                             as these tables lack charttime at the row level)
"""

import logging
import os

import pandas as pd
from tqdm import tqdm

from preprocessing_utils import (
    _gz_or_csv,
    _load_csv,
    _record_hashes,
    _sources_unchanged,
)

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
    """Extract prior-visit discharge summary text for each admission."""
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
    discharge_source = _resolve_note_path(mimic_dir, note_dir, "discharge")
    source_paths = [
        p for p in [
            discharge_source,
            _gz_or_csv(mimic_dir, "hosp", "admissions"),
        ] if os.path.exists(p)
    ]
    output_paths = [os.path.join(features_dir, "discharge_history_features.parquet")]

    if registry_path and not config.get("FORCE_RERUN", False):
        if _sources_unchanged("extract_discharge_history", source_paths,
                               output_paths, registry_path, logger):
            return

    # ------------------------------------------------------------------ #
    # Load notes (discharge type)
    # ------------------------------------------------------------------ #
    note_path = discharge_source
    if not os.path.exists(note_path):
        raise FileNotFoundError(
            "Discharge notes file not found. Searched under "
            f"{note_dir}/note/, {mimic_dir}/note/, and {mimic_dir}/hosp/"
        )

    notes = pd.read_csv(
        note_path,
        usecols=["subject_id", "hadm_id", "charttime", "text"],
        parse_dates=["charttime"],
        dtype={"subject_id": int, "hadm_id": float},
    )

    notes["hadm_id"] = notes["hadm_id"].astype("Int64")
    null_hadm_count = notes["hadm_id"].isna().sum()
    if null_hadm_count > 0:
        logger.info(
            "%s: %d rows (%.1f%%) have null hadm_id — dropping (strategy: %s)",
            "discharge notes", null_hadm_count,
            100 * null_hadm_count / len(notes),
            config.get("HADM_LINKAGE_STRATEGY", "drop"),
        )
    notes = notes.dropna(subset=["hadm_id"])
    notes["hadm_id"] = notes["hadm_id"].astype(int)
    tqdm.pandas(desc="Cleaning discharge notes")
    notes["text"] = notes["text"].progress_apply(_clean_note)

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
        .assign(
            _header=lambda df: df["note_admittime"].apply(
                lambda t: f"Prior Discharge Summary ({pd.to_datetime(t).strftime('%Y-%m-%d')}):"
            ),
            _entry=lambda df: df["_header"] + "\n" + df["text"],
        )
        .groupby(["subject_id", "hadm_id"])["_entry"]
        .apply(lambda entries: "\n\n".join(e for e in entries if e))
        .reset_index()
        .rename(columns={"_entry": "discharge_history_text"})
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

    if registry_path:
        _record_hashes("extract_discharge_history", source_paths, registry_path)
