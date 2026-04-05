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
    _setup_logging,
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


def _load_discharge_notes(note_path: str, config: dict) -> pd.DataFrame:
    """Load, validate, and clean discharge notes from *note_path*.

    Returns a DataFrame with subject_id, hadm_id, charttime, text.
    Rows with null hadm_id are dropped.
    """
    notes = pd.read_csv(
        note_path,
        usecols=["subject_id", "hadm_id", "charttime", "text"],
        parse_dates=["charttime"],
        dtype={"subject_id": int, "hadm_id": float},
    )
    notes["hadm_id"] = notes["hadm_id"].astype("Int64")
    n_null_hadm = int(notes["hadm_id"].isna().sum())
    if n_null_hadm > 0:
        logger.info(
            "%s: %d rows (%.1f%%) have null hadm_id — dropping (strategy: %s)",
            "discharge notes", n_null_hadm,
            100 * n_null_hadm / len(notes),
            config.get("HADM_LINKAGE_STRATEGY", "drop"),
        )
    notes = notes.dropna(subset=["hadm_id"])
    notes["hadm_id"] = notes["hadm_id"].astype(int)
    logger.info("  Loaded %d discharge notes for %d admissions",
                len(notes), notes["hadm_id"].nunique())
    if n_null_hadm:
        logger.info("  Dropped %d notes with null hadm_id (%.1f%%)",
                    n_null_hadm, 100 * n_null_hadm / (len(notes) + n_null_hadm))
    return notes


def _build_prior_discharge_text(
    notes: pd.DataFrame, admissions: pd.DataFrame
) -> pd.DataFrame:
    """Build prior-visit discharge text per admission.

    Returns a DataFrame with subject_id, hadm_id, discharge_history_text.
    """
    notes_with_time = notes.merge(
        admissions[["subject_id", "hadm_id", "admittime"]].rename(
            columns={"hadm_id": "note_hadm_id", "admittime": "note_admittime"}
        ),
        left_on=["subject_id", "hadm_id"],
        right_on=["subject_id", "note_hadm_id"],
        how="left",
    )

    logger.info(
        "Building prior-visit discharge text for %d admissions…", len(admissions)
    )
    merged = admissions.merge(
        notes_with_time[["subject_id", "note_hadm_id", "note_admittime", "text"]],
        on="subject_id",
        how="left",
    )
    prior = merged[merged["note_admittime"] < merged["admittime"]]

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
    logger.info("  Built discharge history for %d admissions (%d with prior notes)",
                len(out_df), int((out_df["discharge_history_text"] != "").sum()))
    return out_df


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

    # ------------------------------------------------------------------ #
    # Validate note file exists
    # ------------------------------------------------------------------ #
    note_path = discharge_source
    if not os.path.exists(note_path):
        raise FileNotFoundError(
            "Discharge notes file not found. Searched under "
            f"{note_dir}/note/, {mimic_dir}/note/, and {mimic_dir}/hosp/"
        )

    steps = [
        "Load discharge notes",
        "Clean note text",
        "Load admissions",
        "Build prior-visit text per admission",
        "Save discharge_history_features.parquet",
    ]
    with tqdm(total=len(steps), desc="extract_discharge_history", unit="step", dynamic_ncols=True) as pbar:
        # ------------------------------------------------------------------ #
        # Load notes
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_discharge_history — loading discharge notes")
        notes = _load_discharge_notes(note_path, config)
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Clean note text
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_discharge_history — cleaning note text")
        tqdm.pandas(desc="Cleaning discharge notes")
        notes["text"] = notes["text"].progress_apply(_clean_note)
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Load admissions
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_discharge_history — loading admissions")
        admissions = _load_csv(
            os.path.join(hosp_dir, "admissions.csv.gz"),
            os.path.join(hosp_dir, "admissions.csv"),
            usecols=["subject_id", "hadm_id", "admittime"],
            parse_dates=["admittime"],
            dtype={"subject_id": int, "hadm_id": int},
        )
        logger.info("  Loaded %d admissions", len(admissions))
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Build prior-visit discharge text per admission
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_discharge_history — building prior-visit text")
        out_df = _build_prior_discharge_text(notes, admissions)
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Save output
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_discharge_history — saving discharge_history_features.parquet")
        os.makedirs(features_dir, exist_ok=True)
        output_path = os.path.join(features_dir, "discharge_history_features.parquet")
        out_df.to_parquet(output_path, index=False)
        logger.info("  Saved %d rows to %s", len(out_df), output_path)
        pbar.update(1)

    if registry_path:
        _record_hashes("extract_discharge_history", source_paths, registry_path)


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
