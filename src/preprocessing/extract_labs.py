"""
extract_labs.py – Lab events for each admission.

Reads labevents in chunks for memory efficiency.
Aggregates lab results by clinical domain (d_labitems.category),
computing mean, min, max and last value per domain per admission.

Expected config keys:
    MIMIC_DATA_DIR  – root directory containing MIMIC-IV tables
    FEATURES_DIR    – output directory for feature parquets
"""

import logging
import os

import pandas as pd

from utils import _load_csv

logger = logging.getLogger(__name__)

# Rows to read per chunk from labevents
_CHUNK_SIZE = 500_000


def run(config: dict) -> None:
    """Extract and aggregate lab features per admission."""
    required_keys = ["MIMIC_DATA_DIR", "FEATURES_DIR"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    mimic_dir = config["MIMIC_DATA_DIR"]
    features_dir = config["FEATURES_DIR"]
    hosp_dir = os.path.join(mimic_dir, "hosp")

    # ------------------------------------------------------------------ #
    # Load d_labitems for domain mapping
    # ------------------------------------------------------------------ #
    logger.info("Loading d_labitems…")
    d_labitems = _load_csv(
        os.path.join(hosp_dir, "d_labitems.csv.gz"),
        os.path.join(hosp_dir, "d_labitems.csv"),
        usecols=["itemid", "category"],
    )
    item_to_category = d_labitems.set_index("itemid")["category"].to_dict()

    # ------------------------------------------------------------------ #
    # Load admissions to know valid admission windows
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
    # Stream labevents in chunks
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
            usecols=["subject_id", "hadm_id", "itemid", "valuenum", "charttime"],
            dtype={"subject_id": int, "hadm_id": float, "itemid": int},
            parse_dates=["charttime"],
            chunksize=_CHUNK_SIZE,
        )
    ):
        chunk = chunk.dropna(subset=["hadm_id", "valuenum"])
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk["category"] = chunk["itemid"].map(item_to_category).fillna("Other")
        all_chunks.append(chunk[["subject_id", "hadm_id", "category", "valuenum",
                                  "charttime"]])
        if (i + 1) % 10 == 0:
            logger.info("  Processed %d chunks…", i + 1)

    if not all_chunks:
        logger.warning("No lab events found – saving empty labs feature file")
        empty = admissions[["subject_id", "hadm_id"]].copy()
        os.makedirs(features_dir, exist_ok=True)
        empty.to_parquet(os.path.join(features_dir, "labs_features.parquet"),
                         index=False)
        return

    labs = pd.concat(all_chunks, ignore_index=True)
    logger.info("Total lab rows after filtering: %d", len(labs))

    # ------------------------------------------------------------------ #
    # Aggregate per (hadm_id, category)
    # ------------------------------------------------------------------ #
    logger.info("Aggregating lab results by admission and domain…")
    labs_sorted = labs.sort_values("charttime")

    agg = (
        labs_sorted.groupby(["subject_id", "hadm_id", "category"])["valuenum"]
        .agg(
            mean="mean",
            min="min",
            max="max",
            last="last",
        )
        .reset_index()
    )

    # Pivot to wide format: columns = category_mean, category_min, …
    agg_wide = agg.pivot_table(
        index=["subject_id", "hadm_id"],
        columns="category",
        values=["mean", "min", "max", "last"],
    )
    # Flatten multi-level column names
    agg_wide.columns = [
        f"{cat}_{stat}" for stat, cat in agg_wide.columns
    ]
    agg_wide = agg_wide.reset_index()

    # ------------------------------------------------------------------ #
    # Left-join onto admissions to preserve all admissions
    # ------------------------------------------------------------------ #
    out_df = admissions[["subject_id", "hadm_id"]].merge(
        agg_wide, on=["subject_id", "hadm_id"], how="left"
    )

    # ------------------------------------------------------------------ #
    # Save output
    # ------------------------------------------------------------------ #
    os.makedirs(features_dir, exist_ok=True)
    output_path = os.path.join(features_dir, "labs_features.parquet")
    out_df.to_parquet(output_path, index=False)
    logger.info("Saved labs features to %s  (shape=%s)", output_path, out_df.shape)
