"""
extract_y_data.py – In-hospital mortality (Y1) and 30-day readmission (Y2).

Y1: admissions.hospital_expire_flag
Y2: 1 if a subsequent admission starts within 30 days of dischtime.
    Patients who died (hospital_expire_flag == 1) receive NaN for Y2.

Expected config keys:
    MIMIC_DATA_DIR       – root directory containing MIMIC-IV tables
    CLASSIFICATIONS_DIR  – output directory for label parquets
"""

import logging
import os

import pandas as pd
from tqdm import tqdm

from preprocessing_utils import _check_required_keys, _gz_or_csv, _load_csv, _record_hashes, _sources_unchanged

logger = logging.getLogger(__name__)

_READMISSION_WINDOW_DAYS = 30


def run(config: dict) -> None:
    """Compute Y1 and Y2 labels and save to parquet."""
    _check_required_keys(config, ["MIMIC_DATA_DIR", "CLASSIFICATIONS_DIR"])

    mimic_dir = config["MIMIC_DATA_DIR"]
    classifications_dir = config["CLASSIFICATIONS_DIR"]
    hosp_dir = os.path.join(mimic_dir, "hosp")
    registry_path = config.get("HASH_REGISTRY_PATH", "")

    # ------------------------------------------------------------------ #
    # Hash-based skip check
    # ------------------------------------------------------------------ #
    source_paths = [p for p in [
        _gz_or_csv(mimic_dir, "hosp", "admissions"),
    ] if os.path.exists(p)]
    output_paths = [os.path.join(classifications_dir, "y_labels.parquet")]

    if registry_path and not config.get("FORCE_RERUN", False):
        if _sources_unchanged("extract_y_data", source_paths,
                               output_paths, registry_path, logger):
            return

    steps = ["Load admissions", "Compute Y1 (mortality)", "Compute Y2 (readmission)", "Save output"]
    with tqdm(total=len(steps), desc="extract_y_data", unit="step", dynamic_ncols=True) as pbar:
        # ------------------------------------------------------------------ #
        # Load admissions
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_y_data — loading admissions")
        logger.info("Loading admissions…")
        admissions = _load_csv(
            os.path.join(hosp_dir, "admissions.csv.gz"),
            os.path.join(hosp_dir, "admissions.csv"),
            usecols=["subject_id", "hadm_id", "admittime", "dischtime",
                     "hospital_expire_flag"],
            parse_dates=["admittime", "dischtime"],
            dtype={"subject_id": int, "hadm_id": int, "hospital_expire_flag": int},
        )
        logger.info("  Loaded %d admissions for %d patients",
                    len(admissions), admissions["subject_id"].nunique())
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Y1 – in-hospital mortality
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_y_data — computing Y1 (mortality)")
        labels = admissions[["subject_id", "hadm_id", "hospital_expire_flag"]].copy()
        labels = labels.rename(columns={"hospital_expire_flag": "y1_mortality"})
        logger.info("  Y1 positive rate: %.2f%%  (%d deceased admissions)",
                    100 * labels["y1_mortality"].mean(), labels["y1_mortality"].sum())
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Y2 – 30-day readmission
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_y_data — computing Y2 (30-day readmission)")
        logger.info("Computing 30-day readmission labels…")

        # Self-join on subject_id to find subsequent admissions
        adm_a = admissions[
            ["subject_id", "hadm_id", "dischtime", "hospital_expire_flag"]
        ].copy()
        adm_b = admissions[["subject_id", "hadm_id", "admittime"]].copy()
        adm_b = adm_b.rename(
            columns={"hadm_id": "next_hadm_id", "admittime": "next_admittime"}
        )

        cross = adm_a.merge(adm_b, on="subject_id", how="left")

        # Subsequent admission must come strictly after current discharge
        cross = cross[cross["next_admittime"] > cross["dischtime"]]

        # Within 30-day window
        window = pd.Timedelta(days=_READMISSION_WINDOW_DAYS)
        time_since_discharge: pd.Series = cross["next_admittime"] - cross["dischtime"]
        cross = cross[time_since_discharge <= window]

        readmitted_hadm_ids = cross["hadm_id"].unique()

        labels["y2_readmission"] = labels["hadm_id"].isin(readmitted_hadm_ids).astype(float)

        # Patients who died cannot be readmitted – set Y2 to NaN
        died_mask = labels["y1_mortality"] == 1
        labels.loc[died_mask, "y2_readmission"] = float("nan")

        logger.info("  Y2 positive rate (excl. deaths): %.2f%%  (%d readmitted)",
                    100 * labels.loc[~died_mask, "y2_readmission"].mean(),
                    int(labels.loc[~died_mask, "y2_readmission"].sum()))
        logger.info("  Y2 excluded (deceased): %d admissions", int(died_mask.sum()))
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Save output
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_y_data — saving y_labels.parquet")
        os.makedirs(classifications_dir, exist_ok=True)
        output_path = os.path.join(classifications_dir, "y_labels.parquet")
        labels.to_parquet(output_path, index=False)
        logger.info("  Saved %d label rows to %s", len(labels), output_path)
        pbar.update(1)

    if registry_path:
        _record_hashes("extract_y_data", source_paths, registry_path)
