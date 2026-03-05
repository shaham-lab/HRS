"""
extract_triage_and_complaint.py – Triage data and chief complaint.

Produces two separate output files:
  • triage_features.parquet        – natural-language triage template
  • chief_complaint_features.parquet – raw chief complaint text

The MIMIC-IV-ED triage table contains stay_id (ED stay identifier) but not
hadm_id. hadm_id is resolved in two steps:

  1. Primary linkage: join triage → edstays on stay_id. ED visits that
     resulted in a hospital admission will have hadm_id populated directly.

  2. Fallback linkage: for triage rows still missing hadm_id after the
     stay_id join, the closest hospital admission with admittime >= ED
     intime for the same subject_id is used as an approximate link.

  ED visits with no resolvable hadm_id (non-admitted visits) are excluded.

Expected config keys:
    MIMIC_DATA_DIR  – root directory containing MIMIC-IV tables
    MIMIC_ED_DIR    – (optional) root of the mimic-iv-ed module; triage and
                      edstays are expected at MIMIC_ED_DIR/ed/.
                      Falls back to MIMIC_DATA_DIR/ed/ then MIMIC_DATA_DIR/hosp/.
    FEATURES_DIR    – output directory for feature parquets
"""

import logging
import os

import pandas as pd

from preprocessing_utils import _gz_or_csv, _load_csv, _record_hashes, _sources_unchanged

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

    # Resolve triage/edstays search directories.
    # Priority: MIMIC_ED_DIR/ed/ → MIMIC_DATA_DIR/ed/ → MIMIC_DATA_DIR/hosp/
    ed_dirs = []
    if config.get("MIMIC_ED_DIR"):
        ed_dirs.append(os.path.join(config["MIMIC_ED_DIR"], "ed"))
    ed_dirs.append(os.path.join(mimic_dir, "ed"))
    hosp_dir = os.path.join(mimic_dir, "hosp")
    registry_path = config.get("HASH_REGISTRY_PATH", "")

    # ------------------------------------------------------------------ #
    # Hash-based skip check
    # ------------------------------------------------------------------ #
    # Resolve which triage / edstays file will actually be used
    def _resolve_ed_table(table_name: str) -> str | None:
        for directory in ed_dirs + [hosp_dir]:
            gz = os.path.join(directory, f"{table_name}.csv.gz")
            csv_p = os.path.join(directory, f"{table_name}.csv")
            if os.path.exists(gz):
                return gz
            if os.path.exists(csv_p):
                return csv_p
        return None

    triage_path_resolved = _resolve_ed_table("triage")
    edstays_path_resolved = _resolve_ed_table("edstays")

    source_paths = [p for p in [
        triage_path_resolved,
        edstays_path_resolved,
        _gz_or_csv(mimic_dir, "hosp", "admissions"),
    ] if p is not None and os.path.exists(p)]

    output_paths = [
        os.path.join(features_dir, "triage_features.parquet"),
        os.path.join(features_dir, "chief_complaint_features.parquet"),
    ]

    if registry_path and not config.get("FORCE_RERUN", False):
        if _sources_unchanged("extract_triage_and_complaint", source_paths,
                               output_paths, registry_path, logger):
            return

    # ------------------------------------------------------------------ #
    # Load triage table
    # ------------------------------------------------------------------ #
    for directory in ed_dirs + [hosp_dir]:
        gz = os.path.join(directory, "triage.csv.gz")
        csv = os.path.join(directory, "triage.csv")
        if os.path.exists(gz) or os.path.exists(csv):
            triage = _load_csv(
                gz, csv,
                dtype={"subject_id": int},
            )
            logger.info("Loaded triage from %s", gz if os.path.exists(gz) else csv)
            break
    else:
        raise FileNotFoundError(
            f"triage table not found under any of: {ed_dirs + [hosp_dir]}"
        )

    logger.info("Triage raw shape: %d rows", len(triage))

    # ------------------------------------------------------------------ #
    # hadm_id resolution via edstays bridge table
    # ------------------------------------------------------------------ #
    # The MIMIC-IV-ED triage table contains stay_id, not hadm_id.
    # We resolve hadm_id in two steps:
    #   1. Primary: join triage → edstays on stay_id
    #   2. Fallback: closest admission with admittime >= ED intime

    if "hadm_id" not in triage.columns:
        # Pure MIMIC-IV-ED format: only stay_id, no hadm_id column
        triage["hadm_id"] = float("nan")

    # --- Step 1: Primary linkage via edstays ---
    if "stay_id" in triage.columns:
        for directory in ed_dirs + [hosp_dir]:
            gz = os.path.join(directory, "edstays.csv.gz")
            csv = os.path.join(directory, "edstays.csv")
            if os.path.exists(gz) or os.path.exists(csv):
                edstays = _load_csv(
                    gz, csv,
                    usecols=["subject_id", "stay_id", "hadm_id", "intime"],
                    dtype={"subject_id": int, "stay_id": int},
                    parse_dates=["intime"],
                )
                edstays["hadm_id"] = pd.to_numeric(edstays["hadm_id"], errors="coerce")
                logger.info(
                    "Loaded edstays: %d rows, %d with hadm_id",
                    len(edstays), edstays["hadm_id"].notna().sum(),
                )
                break
        else:
            logger.warning(
                "edstays table not found — hadm_id resolution via stay_id will be skipped."
            )
            edstays = None

        if edstays is not None:
            triage = triage.merge(
                edstays[["stay_id", "hadm_id", "intime"]],
                on="stay_id",
                how="left",
                suffixes=("_triage", ""),
            )
            # Prefer the hadm_id from edstays; fall back to any existing value
            if "hadm_id_triage" in triage.columns:
                triage["hadm_id"] = triage["hadm_id"].combine_first(
                    triage["hadm_id_triage"]
                )
                triage = triage.drop(columns=["hadm_id_triage"])
            logger.info(
                "After stay_id join: %d / %d rows have hadm_id",
                triage["hadm_id"].notna().sum(), len(triage),
            )

    # --- Step 2: Fallback linkage via intime + subject_id ---
    if "intime" in triage.columns:
        null_mask = triage["hadm_id"].isna() & triage["intime"].notna()
    else:
        null_mask = pd.Series(False, index=triage.index)

    if null_mask.any():
        n_had_hadm_before = int(triage["hadm_id"].notna().sum())
        admissions = _load_csv(
            os.path.join(hosp_dir, "admissions.csv.gz"),
            os.path.join(hosp_dir, "admissions.csv"),
            usecols=["subject_id", "hadm_id", "admittime"],
            parse_dates=["admittime"],
            dtype={"subject_id": int, "hadm_id": int},
        )
        unlinked = triage[null_mask][["subject_id", "intime"]].copy()
        candidates = unlinked.merge(admissions, on="subject_id", how="left")
        candidates = candidates[candidates["admittime"] >= candidates["intime"]]
        closest = (
            candidates.sort_values("admittime")
            .groupby(["subject_id", "intime"])["hadm_id"]
            .first()
            .reset_index()
            .rename(columns={"hadm_id": "hadm_id_fallback"})
        )
        triage = triage.merge(closest, on=["subject_id", "intime"], how="left")
        triage["hadm_id"] = triage["hadm_id"].combine_first(triage["hadm_id_fallback"])
        triage = triage.drop(columns=["hadm_id_fallback"])
        n_resolved = int(triage["hadm_id"].notna().sum()) - n_had_hadm_before
        logger.info(
            "Fallback linkage resolved %d additional triage rows via intime+subject_id.",
            n_resolved,
        )

    # --- Drop rows with no resolvable hadm_id ---
    n_before = len(triage)
    triage = triage.dropna(subset=["hadm_id"])
    triage["hadm_id"] = triage["hadm_id"].astype(int)
    logger.info(
        "Triage: %d rows retained after hadm_id resolution (%d dropped — non-admitted ED visits).",
        len(triage), n_before - len(triage),
    )

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
            _CHART_CHUNK_SIZE = 1_000_000
            chart_path = chart_gz if os.path.exists(chart_gz) else chart_csv
            logger.info(
                "Streaming chartevents from %s in chunks of %d…",
                chart_path, _CHART_CHUNK_SIZE,
            )
            chunks: list[pd.DataFrame] = []
            for i, chunk in enumerate(pd.read_csv(
                chart_path,
                usecols=["subject_id", "hadm_id", "itemid", "value"],
                dtype={"subject_id": int, "hadm_id": float, "itemid": int},
                chunksize=_CHART_CHUNK_SIZE,
            )):
                sub = chunk[chunk["itemid"] == _CHIEF_COMPLAINT_ITEMID]
                if not sub.empty:
                    chunks.append(sub)
                if (i + 1) % 10 == 0:
                    logger.info("  Processed %d chunks…", i + 1)

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

    if registry_path:
        _record_hashes("extract_triage_and_complaint", source_paths, registry_path)
