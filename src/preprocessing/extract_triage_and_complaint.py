"""
extract_triage_and_complaint.py – Triage data and chief complaint.

Produces two separate output files:
  • triage_features.parquet        – natural-language triage template
  • chief_complaint_features.parquet – raw chief complaint text

Expected config keys:
    MIMIC_DATA_DIR  – root directory containing MIMIC-IV tables
    FEATURES_DIR    – output directory for feature parquets
"""

import logging
import os

import pandas as pd

from utils import _load_csv

logger = logging.getLogger(__name__)

# Template used to render triage structured fields as natural language
_TRIAGE_TEMPLATE = (
    "Triage assessment: temperature {temperature}°C, "
    "heart rate {heartrate} bpm, respiratory rate {resprate} breaths/min, "
    "O2 saturation {o2sat}%, blood pressure {sbp}/{dbp} mmHg, "
    "pain score {pain}/10, acuity level {acuity}."
)

# Chief complaint item ID in chartevents (MIMIC-IV ED)
_CHIEF_COMPLAINT_ITEMID = 223112


def _fmt(val) -> str:
    """Format a potentially missing value for the template."""
    if pd.isna(val):
        return "N/A"
    return str(val)


def run(config: dict) -> None:
    """Extract triage and chief complaint features."""
    required_keys = ["MIMIC_DATA_DIR", "FEATURES_DIR"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    mimic_dir = config["MIMIC_DATA_DIR"]
    features_dir = config["FEATURES_DIR"]

    # Possible locations for ED triage table
    ed_dir = os.path.join(mimic_dir, "ed")
    hosp_dir = os.path.join(mimic_dir, "hosp")

    # ------------------------------------------------------------------ #
    # Load triage table
    # ------------------------------------------------------------------ #
    for directory in [ed_dir, hosp_dir]:
        gz = os.path.join(directory, "triage.csv.gz")
        csv = os.path.join(directory, "triage.csv")
        if os.path.exists(gz) or os.path.exists(csv):
            triage = _load_csv(
                gz, csv,
                dtype={"subject_id": int, "hadm_id": float},
            )
            break
    else:
        raise FileNotFoundError(
            f"triage table not found under {ed_dir} or {hosp_dir}"
        )

    triage["hadm_id"] = triage["hadm_id"].astype("Int64")
    triage = triage.dropna(subset=["hadm_id"])
    triage["hadm_id"] = triage["hadm_id"].astype(int)
    logger.info("Loaded triage table with %d rows", len(triage))

    # ------------------------------------------------------------------ #
    # Build triage text via template
    # ------------------------------------------------------------------ #
    triage_cols = {
        "temperature": "temperature",
        "heartrate": "heartrate",
        "resprate": "resprate",
        "o2sat": "o2sat",
        "sbp": "sbp",
        "dbp": "dbp",
        "pain": "pain",
        "acuity": "acuity",
    }
    for col in triage_cols:
        if col not in triage.columns:
            triage[col] = float("nan")

    def _build_triage_text(row) -> str:
        return _TRIAGE_TEMPLATE.format(
            temperature=_fmt(row.get("temperature")),
            heartrate=_fmt(row.get("heartrate")),
            resprate=_fmt(row.get("resprate")),
            o2sat=_fmt(row.get("o2sat")),
            sbp=_fmt(row.get("sbp")),
            dbp=_fmt(row.get("dbp")),
            pain=_fmt(row.get("pain")),
            acuity=_fmt(row.get("acuity")),
        )

    triage["triage_text"] = triage.apply(_build_triage_text, axis=1)

    triage_out = (
        triage[["subject_id", "hadm_id", "triage_text"]]
        .drop_duplicates(subset=["subject_id", "hadm_id"])
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------ #
    # Chief complaint – try triage.chiefcomplaint first, fallback chartevents
    # ------------------------------------------------------------------ #
    if "chiefcomplaint" in triage.columns:
        logger.info("Extracting chief complaint from triage.chiefcomplaint…")
        complaint_out = (
            triage[["subject_id", "hadm_id", "chiefcomplaint"]]
            .drop_duplicates(subset=["subject_id", "hadm_id"])
            .rename(columns={"chiefcomplaint": "chief_complaint_text"})
            .reset_index(drop=True)
        )
        complaint_out["chief_complaint_text"] = (
            complaint_out["chief_complaint_text"].fillna("")
        )
    else:
        logger.info(
            "chiefcomplaint column not in triage – extracting from chartevents…"
        )
        # Attempt chartevents fallback
        icu_dir = os.path.join(mimic_dir, "icu")
        chart_gz = os.path.join(icu_dir, "chartevents.csv.gz")
        chart_csv = os.path.join(icu_dir, "chartevents.csv")

        if not (os.path.exists(chart_gz) or os.path.exists(chart_csv)):
            logger.warning(
                "chartevents not found – chief complaint will be empty strings"
            )
            complaint_out = triage[["subject_id", "hadm_id"]].copy()
            complaint_out["chief_complaint_text"] = ""
        else:
            chunks: list[pd.DataFrame] = []
            for chunk in pd.read_csv(
                chart_gz if os.path.exists(chart_gz) else chart_csv,
                usecols=["subject_id", "hadm_id", "itemid", "value"],
                dtype={"subject_id": int, "hadm_id": float, "itemid": int},
                chunksize=500_000,
            ):
                sub = chunk[chunk["itemid"] == _CHIEF_COMPLAINT_ITEMID]
                if not sub.empty:
                    chunks.append(sub)

            if chunks:
                cc = pd.concat(chunks, ignore_index=True)
                cc["hadm_id"] = cc["hadm_id"].astype("Int64")
                cc = cc.dropna(subset=["hadm_id"])
                cc["hadm_id"] = cc["hadm_id"].astype(int)
                cc = (
                    cc.groupby(["subject_id", "hadm_id"])["value"]
                    .first()
                    .reset_index()
                    .rename(columns={"value": "chief_complaint_text"})
                )
                complaint_out = triage[["subject_id", "hadm_id"]].merge(
                    cc, on=["subject_id", "hadm_id"], how="left"
                )
                complaint_out["chief_complaint_text"] = (
                    complaint_out["chief_complaint_text"].fillna("")
                )
            else:
                complaint_out = triage[["subject_id", "hadm_id"]].copy()
                complaint_out["chief_complaint_text"] = ""

        complaint_out = complaint_out.drop_duplicates(
            subset=["subject_id", "hadm_id"]
        ).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    os.makedirs(features_dir, exist_ok=True)

    triage_path = os.path.join(features_dir, "triage_features.parquet")
    triage_out.to_parquet(triage_path, index=False)
    logger.info("Saved triage features to %s  (shape=%s)", triage_path, triage_out.shape)

    complaint_path = os.path.join(features_dir, "chief_complaint_features.parquet")
    complaint_out.to_parquet(complaint_path, index=False)
    logger.info(
        "Saved chief complaint features to %s  (shape=%s)",
        complaint_path, complaint_out.shape,
    )
