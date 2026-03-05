"""
extract_labs.py – Lab events (current admission, long format).

Outputs one row per lab event per admission, formatted as a chronological
natural-language text line. No aggregation, no pivoting, no wide format.

Lab embedding is intentionally deferred to training/inference time. The MDP
agent selects a subset of itemids, their text lines are concatenated in
chronological order, and passed to the language model for encoding.

Expected config keys:
    MIMIC_DATA_DIR  – root directory containing MIMIC-IV tables
    FEATURES_DIR    – output directory for feature parquets
"""

import logging
import os

import pandas as pd

from preprocessing_utils import _gz_or_csv, _load_csv, _record_hashes, _sources_unchanged

logger = logging.getLogger(__name__)

# Rows to read per chunk from labevents
_CHUNK_SIZE = 1_000_000


def _build_lab_text_line(row) -> str:
    """Convert a single lab event row to a chronological text line."""
    # Value: prefer numeric formatted to 2dp, fall back to text value
    if pd.notna(row["valuenum"]):
        value_str = f"{row['valuenum']:.2f}"
    else:
        value_str = str(row["value"]).strip()

    # Unit
    uom = f" {row['valueuom']}" if pd.notna(row["valueuom"]) else ""

    # Reference range — only include when both bounds are present
    ref_str = ""
    if pd.notna(row["ref_range_lower"]) and pd.notna(row["ref_range_upper"]):
        ref_str = f" (ref: {row['ref_range_lower']}-{row['ref_range_upper']})"

    # Abnormal flag — only include when flagged
    flag_str = " [ABNORMAL]" if str(row["flag"]).strip().lower() == "abnormal" else ""

    # Priority — only include when STAT
    priority_str = " [STAT]" if str(row["priority"]).strip().upper() == "STAT" else ""

    time_str = pd.to_datetime(row["charttime"]).strftime("%H:%M")

    return (
        f"[{time_str}] {row['label']} ({row['fluid']}/{row['category']}): "
        f"{value_str}{uom}{ref_str}{flag_str}{priority_str}"
    )


def run(config: dict) -> None:
    """Extract lab events as long-format chronological text lines per admission."""
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
        _gz_or_csv(mimic_dir, "hosp", "labevents"),
        _gz_or_csv(mimic_dir, "hosp", "d_labitems"),
        _gz_or_csv(mimic_dir, "hosp", "admissions"),
    ] if os.path.exists(p)]
    output_paths = [os.path.join(features_dir, "labs_features.parquet")]

    if registry_path and not config.get("FORCE_RERUN", False):
        if _sources_unchanged("extract_labs", source_paths,
                               output_paths, registry_path, logger):
            return

    # ------------------------------------------------------------------ #
    # Load d_labitems for label, fluid, category mapping
    # ------------------------------------------------------------------ #
    logger.info("Loading d_labitems…")
    d_labitems = _load_csv(
        os.path.join(hosp_dir, "d_labitems.csv.gz"),
        os.path.join(hosp_dir, "d_labitems.csv"),
        usecols=["itemid", "label", "fluid", "category"],
    )
    # Clean d_labitems: strip whitespace, remove artifact rows
    d_labitems["fluid"]    = d_labitems["fluid"].str.strip()
    d_labitems["category"] = d_labitems["category"].str.strip()
    d_labitems = d_labitems[~d_labitems["fluid"].isin(["I", "Q", "fluid"])]

    d_labitems_indexed = d_labitems.set_index("itemid")
    item_to_label    = d_labitems_indexed["label"].to_dict()
    item_to_fluid    = d_labitems_indexed["fluid"].to_dict()
    item_to_category = d_labitems_indexed["category"].to_dict()

    # ------------------------------------------------------------------ #
    # Load admissions for window filtering
    # ------------------------------------------------------------------ #
    logger.info("Loading admissions…")
    admissions = _load_csv(
        os.path.join(hosp_dir, "admissions.csv.gz"),
        os.path.join(hosp_dir, "admissions.csv"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        parse_dates=["admittime", "dischtime"],
        dtype={"subject_id": int, "hadm_id": int},
    )

    # ------------------------------------------------------------------ #
    # Stream labevents in chunks, applying per-chunk filters
    # ------------------------------------------------------------------ #
    lab_gz = os.path.join(hosp_dir, "labevents.csv.gz")
    lab_csv = os.path.join(hosp_dir, "labevents.csv")
    if not (os.path.exists(lab_gz) or os.path.exists(lab_csv)):
        raise FileNotFoundError(
            f"labevents table not found under {hosp_dir}"
        )

    lab_path = lab_gz if os.path.exists(lab_gz) else lab_csv
    logger.info("Streaming labevents from %s in chunks of %d…",
                lab_path, _CHUNK_SIZE)

    all_chunks: list[pd.DataFrame] = []
    for i, chunk in enumerate(
        pd.read_csv(
            lab_path,
            usecols=[
                "subject_id", "hadm_id", "itemid", "charttime",
                "value", "valuenum", "valueuom",
                "ref_range_lower", "ref_range_upper", "flag", "priority",
            ],
            dtype={"subject_id": int, "hadm_id": float, "itemid": int},
            parse_dates=["charttime"],
            chunksize=_CHUNK_SIZE,
        )
    ):
        # 1. Filter out rows with no hadm_id (~70% of labevents are outpatient)
        chunk = chunk.dropna(subset=["hadm_id"]).copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)

        # 2. Filter out rows where both value and valuenum are null
        chunk = chunk[chunk["value"].notna() | chunk["valuenum"].notna()]

        # 3. Join label, fluid, category from d_labitems
        chunk["label"]    = chunk["itemid"].map(item_to_label)
        chunk["fluid"]    = chunk["itemid"].map(item_to_fluid)
        chunk["category"] = chunk["itemid"].map(item_to_category)

        # 4. Drop rows where itemid is not in d_labitems (unmapped items)
        chunk = chunk.dropna(subset=["label"])

        if not chunk.empty:
            all_chunks.append(chunk)

        if (i + 1) % 10 == 0:
            logger.info("  Processed %d chunks…", i + 1)

    if not all_chunks:
        logger.warning("No lab events found – saving empty labs feature file")
        empty = pd.DataFrame(columns=[
            "subject_id", "hadm_id", "charttime",
            "itemid", "label", "fluid", "category", "lab_text_line",
        ])
        os.makedirs(features_dir, exist_ok=True)
        empty.to_parquet(os.path.join(features_dir, "labs_features.parquet"),
                         index=False)
        return

    labs = pd.concat(all_chunks, ignore_index=True)

    # ------------------------------------------------------------------ #
    # Apply admission window filter after concatenating all chunks
    # ------------------------------------------------------------------ #
    labs = labs.merge(
        admissions[["subject_id", "hadm_id", "admittime", "dischtime"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    labs = labs[
        (labs["charttime"] >= labs["admittime"]) &
        (labs["charttime"] <= labs["dischtime"])
    ]

    # ------------------------------------------------------------------ #
    # Build chronological text line per event
    # ------------------------------------------------------------------ #
    logger.info("Building lab text lines…")
    labs["lab_text_line"] = labs.apply(_build_lab_text_line, axis=1)

    # ------------------------------------------------------------------ #
    # Sort chronologically within each admission
    # ------------------------------------------------------------------ #
    labs = labs.sort_values(["subject_id", "hadm_id", "charttime"])

    # ------------------------------------------------------------------ #
    # Output — long format, one row per lab event
    # ------------------------------------------------------------------ #
    out_df = labs[[
        "subject_id", "hadm_id", "charttime",
        "itemid", "label", "fluid", "category",
        "lab_text_line",
    ]].reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Save output
    # ------------------------------------------------------------------ #
    os.makedirs(features_dir, exist_ok=True)
    output_path = os.path.join(features_dir, "labs_features.parquet")
    out_df.to_parquet(output_path, index=False)
    n_admissions = out_df["hadm_id"].nunique()
    logger.info(
        "Saved labs features to %s  (%d rows, %d unique admissions)",
        output_path, len(out_df), n_admissions,
    )

    if registry_path:
        _record_hashes("extract_labs", source_paths, registry_path)
