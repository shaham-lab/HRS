"""
extract_diag_history.py – Prior-visit ICD diagnosis text.

For each admission, builds a structured text block of ICD diagnoses from all
prior admissions of the same patient (strictly before current admittime).
Formatted with dated section headers and one long_title per line per visit.
An empty string is produced if no prior admissions exist.

Output format:
    Past Diagnoses:

    Visit (YYYY-MM-DD):
    Chronic kidney disease, stage 3
    Hypertension

    Visit (YYYY-MM-DD):
    Acute kidney injury

Expected config keys:
    MIMIC_DATA_DIR  – root directory containing MIMIC-IV tables
    FEATURES_DIR    – output directory for feature parquets
"""

import logging
import os

import pandas as pd
from tqdm import tqdm

from preprocessing_utils import _gz_or_csv, _load_csv, _record_hashes, _sources_unchanged

logger = logging.getLogger(__name__)


def _format_diag_history(prior: pd.DataFrame) -> str:
    """Format prior diagnosis rows into a structured text block.

    Parameters
    ----------
    prior : pd.DataFrame
        Rows for a single (subject_id, hadm_id) pair, already filtered to
        strictly prior visits. Expected columns: diag_admittime, long_title.

    Returns
    -------
    str
        Structured text block; empty string if prior is empty.
    """
    if prior.empty:
        return ""

    lines = ["Past Diagnoses:"]
    for visit_date, visit_group in prior.sort_values("diag_admittime").groupby(
        "diag_admittime", sort=False
    ):
        date_str = pd.to_datetime(visit_date).strftime("%Y-%m-%d")
        lines.append("")
        lines.append(f"Visit ({date_str}):")
        for title in visit_group["long_title"]:
            if title:
                lines.append(title)
    return "\n".join(lines)


def run(config: dict) -> None:
    """Extract prior-visit diagnosis text for each admission."""
    required_keys = ["MIMIC_DATA_DIR", "FEATURES_DIR"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    mimic_dir = config["MIMIC_DATA_DIR"]
    features_dir = config["FEATURES_DIR"]
    hosp_dir = os.path.join(mimic_dir, "hosp")
    registry_path = config.get("HASH_REGISTRY_PATH", "")

    # ------------------------------------------------------------------ #
    # Hash-based skip check
    # ------------------------------------------------------------------ #
    source_paths = [p for p in [
        _gz_or_csv(mimic_dir, "hosp", "diagnoses_icd"),
        _gz_or_csv(mimic_dir, "hosp", "d_icd_diagnoses"),
        _gz_or_csv(mimic_dir, "hosp", "admissions"),
    ] if os.path.exists(p)]
    output_paths = [os.path.join(features_dir, "diag_history_features.parquet")]

    if registry_path and not config.get("FORCE_RERUN", False):
        if _sources_unchanged("extract_diag_history", source_paths,
                               output_paths, registry_path, logger):
            return

    # ------------------------------------------------------------------ #
    # Load source tables
    # ------------------------------------------------------------------ #
    logger.info("Loading diagnoses_icd…")
    diagnoses = _load_csv(
        os.path.join(hosp_dir, "diagnoses_icd.csv.gz"),
        os.path.join(hosp_dir, "diagnoses_icd.csv"),
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype={"subject_id": int, "hadm_id": int},
    )

    logger.info("Loading d_icd_diagnoses…")
    d_icd = _load_csv(
        os.path.join(hosp_dir, "d_icd_diagnoses.csv.gz"),
        os.path.join(hosp_dir, "d_icd_diagnoses.csv"),
        usecols=["icd_code", "icd_version", "long_title"],
    )

    logger.info("Loading admissions…")
    admissions = _load_csv(
        os.path.join(hosp_dir, "admissions.csv.gz"),
        os.path.join(hosp_dir, "admissions.csv"),
        usecols=["subject_id", "hadm_id", "admittime"],
        parse_dates=["admittime"],
        dtype={"subject_id": int, "hadm_id": int},
    )

    # ------------------------------------------------------------------ #
    # Attach long_title to each diagnosis
    # ------------------------------------------------------------------ #
    diagnoses = diagnoses.merge(d_icd, on=["icd_code", "icd_version"], how="left")
    diagnoses["long_title"] = diagnoses["long_title"].fillna("")

    # ------------------------------------------------------------------ #
    # For each admission, gather titles from strictly prior admissions
    # ------------------------------------------------------------------ #
    logger.info("Building prior-visit diagnosis text for %d admissions…",
                len(admissions))

    # Join diagnosis records to their admission time
    diag_with_time = diagnoses.merge(
        admissions[["subject_id", "hadm_id", "admittime"]].rename(
            columns={"hadm_id": "diag_hadm_id", "admittime": "diag_admittime"}
        ),
        left_on=["subject_id", "hadm_id"],
        right_on=["subject_id", "diag_hadm_id"],
        how="left",
    )

    # Cross-join via patient: merge admissions with diag_with_time on subject_id
    merged = admissions.merge(
        diag_with_time[["subject_id", "diag_hadm_id", "diag_admittime", "long_title"]],
        on="subject_id",
        how="left",
    )
    # Keep only strictly prior admissions
    prior_mask = merged["diag_admittime"] < merged["admittime"]
    prior = merged[prior_mask]

    # Build structured text block per admission
    tqdm.pandas(desc="Formatting diag history")
    diag_text = (
        prior.groupby(["subject_id", "hadm_id"])
        .progress_apply(lambda grp: _format_diag_history(grp))
        .reset_index()
        .rename(columns={0: "diag_history_text"})
    )

    out_df = admissions[["subject_id", "hadm_id"]].merge(
        diag_text, on=["subject_id", "hadm_id"], how="left"
    )
    out_df["diag_history_text"] = out_df["diag_history_text"].fillna("")

    # ------------------------------------------------------------------ #
    # Save output
    # ------------------------------------------------------------------ #
    os.makedirs(features_dir, exist_ok=True)
    output_path = os.path.join(features_dir, "diag_history_features.parquet")
    out_df.to_parquet(output_path, index=False)
    logger.info("Saved diagnosis history features to %s  (shape=%s)",
                output_path, out_df.shape)

    if registry_path:
        _record_hashes("extract_diag_history", source_paths, registry_path)
