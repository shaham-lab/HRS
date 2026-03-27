"""
extract_labs.py – Lab events (current admission, long format).

Outputs one row per lab event per admission, formatted as a chronological
natural-language text line. No aggregation, no pivoting, no wide format.

The admission window is controlled by LAB_ADMISSION_WINDOW:
  - Integer N: include events in [admittime, admittime + N hours]
  - "full": include all events in [admittime, dischtime]
  Default: 24 hours.

Expected config keys:
    MIMIC_DATA_DIR  – root directory containing MIMIC-IV tables
    FEATURES_DIR    – output directory for feature parquets

Optional config keys:
    LAB_ADMISSION_WINDOW – hours from admittime to include (default 24),
                           or "full" to include entire admission
    ED_LOOKBACK_HOURS            – hours before admittime to include, capturing
                                   pre-admission ED labs (default 24)
    HADM_LINKAGE_STRATEGY        – "drop" (default) or "link"; how to handle
                                   null hadm_id in labevents
    HADM_LINKAGE_TOLERANCE_HOURS – hours of tolerance for time-window linkage
                                   (default 2, only used when strategy is "link")
"""

import logging
import os

import pandas as pd
from tqdm import tqdm

from preprocessing_utils import _check_required_keys, _gz_or_csv, _load_csv, _load_d_labitems, _record_hashes, _sources_unchanged
from build_lab_text_lines import build_lab_text_line_row as _build_lab_text_line, build_lab_text_line_series

logger = logging.getLogger(__name__)

# Rows to read per chunk from labevents
_CHUNK_SIZE = 1_000_000


def _parse_lab_window(config: dict) -> int | None:
    """Parse LAB_ADMISSION_WINDOW from config.

    Returns an integer number of hours, or None for the "full" sentinel.
    Raises ValueError for invalid values.
    """
    raw_window = config.get("LAB_ADMISSION_WINDOW", 24)
    if str(raw_window).strip().lower() == "full":
        return None
    try:
        lab_window_hours = int(raw_window)
        if lab_window_hours <= 0:
            raise ValueError(f"LAB_ADMISSION_WINDOW must be positive, got {lab_window_hours}")
        return lab_window_hours
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"LAB_ADMISSION_WINDOW must be a positive integer or 'full', "
            f"got {raw_window!r}"
        ) from exc


def _stream_and_filter_labevents(
    lab_path: str,
    item_to_label: dict,
    item_to_fluid: dict,
    item_to_category: dict,
    admissions: pd.DataFrame,
    hadm_linkage_strategy: str,
    hadm_linkage_tolerance_hours: int,
) -> list[pd.DataFrame]:
    """Stream labevents CSV in chunks, applying hadm linkage and item filters.

    Returns a list of filtered DataFrames (one per non-empty chunk).
    """
    all_chunks: list[pd.DataFrame] = []
    i = 0
    for chunk in tqdm(
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
        ),
        desc="Streaming labevents",
        unit="chunk",
    ):
        chunk = chunk.copy()
        raw_chunk_len = len(chunk)
        null_hadm_mask = chunk["hadm_id"].isna()
        null_hadm_count = int(null_hadm_mask.sum())

        if hadm_linkage_strategy == "drop":
            chunk = chunk[~null_hadm_mask].copy()
        elif hadm_linkage_strategy == "link" and null_hadm_count > 0:
            null_rows = chunk[null_hadm_mask].sort_values("charttime")
            admissions_sorted = (
                admissions[["subject_id", "hadm_id", "admittime"]]
                .rename(columns={
                    "hadm_id": "hadm_id_matched",
                    "admittime": "admittime_matched",
                })
                .sort_values("admittime_matched")
            )
            tolerance = pd.Timedelta(hours=hadm_linkage_tolerance_hours)
            matched = pd.merge_asof(
                null_rows,
                admissions_sorted,
                left_on="charttime",
                right_on="admittime_matched",
                by="subject_id",
                direction="forward",
                tolerance=tolerance,
            )
            matched = matched[matched["hadm_id_matched"].notna()].copy()
            matched["hadm_id"] = matched["hadm_id_matched"].astype(int)
            matched = matched.drop(columns=["hadm_id_matched", "admittime_matched"])

            non_null = chunk[~null_hadm_mask].copy()
            if not matched.empty:
                chunk = pd.concat([non_null, matched], ignore_index=True)
            else:
                chunk = non_null

        chunk = chunk.dropna(subset=["hadm_id"]).copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)

        # Filter out rows where both value and valuenum are null
        chunk = chunk[chunk["value"].notna() | chunk["valuenum"].notna()]

        # Join label, fluid, category from d_labitems
        chunk["label"]    = chunk["itemid"].map(item_to_label)
        chunk["fluid"]    = chunk["itemid"].map(item_to_fluid)
        chunk["category"] = chunk["itemid"].map(item_to_category)

        # Drop rows where itemid is not in d_labitems
        chunk = chunk.dropna(subset=["label"])

        logger.info(
            "  Chunk %d: %d rows read  |  %d null hadm_id (%s)  |  %d retained after filters",
            i, raw_chunk_len, null_hadm_count,
            f"strategy: {hadm_linkage_strategy}", len(chunk),
        )
        i += 1

        if not chunk.empty:
            all_chunks.append(chunk)

    return all_chunks


def _apply_admission_window_filter(
    labs: pd.DataFrame,
    admissions: pd.DataFrame,
    lab_window_hours: int | None,
    ed_lookback_hours: int = 24,
) -> pd.DataFrame:
    """Filter lab events to the configured admission window.

    Parameters
    ----------
    labs : pd.DataFrame
        All lab events after chunk streaming (without admittime/dischtime).
    admissions : pd.DataFrame
        Admissions table with admittime and dischtime columns.
    lab_window_hours : int or None
        Hours from admittime to include, or None for full admission.
    ed_lookback_hours : int
        Hours before admittime to include, capturing pre-admission ED labs
        charted before formal admittime (e.g. sepsis lactates, troponins).
        Default: 24.

    Returns
    -------
    pd.DataFrame
        Lab events filtered to the admission window.
    """
    labs = labs.merge(
        admissions[["subject_id", "hadm_id", "admittime", "dischtime"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )

    lookback = pd.to_timedelta(ed_lookback_hours, unit="h")
    window_mask = labs["charttime"] >= (labs["admittime"] - lookback)
    if lab_window_hours is None:
        window_mask &= labs["charttime"] <= labs["dischtime"]
    else:
        cutoff = labs["admittime"] + pd.to_timedelta(lab_window_hours, unit="h")
        window_mask &= labs["charttime"] <= cutoff

    labs = labs[window_mask]
    logger.info("  After window filter (%s window, %dh ED lookback): %d rows for %d admissions",
                f"{lab_window_hours}h" if lab_window_hours else "full",
                ed_lookback_hours,
                len(labs), labs["hadm_id"].nunique())
    return labs


def run(config: dict) -> None:
    """Extract lab events as long-format chronological text lines per admission."""
    _check_required_keys(config, ["MIMIC_DATA_DIR", "FEATURES_DIR"])

    mimic_dir = config["MIMIC_DATA_DIR"]
    features_dir = config["FEATURES_DIR"]
    hosp_dir = os.path.join(mimic_dir, "hosp")
    registry_path = config.get("HASH_REGISTRY_PATH", "")

    lab_window_hours = _parse_lab_window(config)
    ed_lookback_hours: int = int(config.get("ED_LOOKBACK_HOURS", 24))

    hadm_linkage_strategy: str = str(config.get("HADM_LINKAGE_STRATEGY", "drop")).lower()
    hadm_linkage_tolerance_hours: int = int(config.get("HADM_LINKAGE_TOLERANCE_HOURS", 2))

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

    steps = [
        "Load d_labitems and admissions",
        "Stream and filter labevents",
        "Apply admission window filter",
        "Build lab text lines",
        "Sort and save labs_features.parquet",
    ]
    with tqdm(total=len(steps), desc="extract_labs", unit="step", dynamic_ncols=True) as pbar:
        # ------------------------------------------------------------------ #
        # Load d_labitems for label, fluid, category mapping
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_labs — loading d_labitems and admissions")
        logger.info("Loading d_labitems…")
        d_labitems = _load_d_labitems(hosp_dir)

        d_labitems_indexed = d_labitems.set_index("itemid")
        item_to_label    = d_labitems_indexed["label"].to_dict()
        item_to_fluid    = d_labitems_indexed["fluid"].to_dict()
        item_to_category = d_labitems_indexed["category"].to_dict()
        logger.info("  d_labitems: %d items across %d fluids and %d categories",
                    len(d_labitems),
                    d_labitems["fluid"].nunique(),
                    d_labitems["category"].nunique())

        logger.info("Loading admissions…")
        admissions = _load_csv(
            os.path.join(hosp_dir, "admissions.csv.gz"),
            os.path.join(hosp_dir, "admissions.csv"),
            usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
            parse_dates=["admittime", "dischtime"],
            dtype={"subject_id": int, "hadm_id": int},
        )
        logger.info("  Admissions: %d rows loaded for window filtering", len(admissions))

        if lab_window_hours is not None:
            logger.info(
                "Lab admission window: first %d hours from admittime", lab_window_hours
            )
        else:
            logger.info("Lab admission window: full admission (admittime → dischtime)")
        pbar.update(1)

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
        logger.info("Streaming labevents from %s…", lab_path)

        pbar.set_description("extract_labs — streaming labevents chunks")
        all_chunks = _stream_and_filter_labevents(
            lab_path, item_to_label, item_to_fluid, item_to_category,
            admissions, hadm_linkage_strategy, hadm_linkage_tolerance_hours,
        )

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

        total_rows_before_filter = sum(len(c) for c in all_chunks)
        logger.info("  labevents: %d rows retained after streaming and hadm_id handling",
                    total_rows_before_filter)
        labs = pd.concat(all_chunks, ignore_index=True)
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Apply admission window filter
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_labs — applying admission window filter")
        labs = _apply_admission_window_filter(labs, admissions, lab_window_hours, ed_lookback_hours)
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Build chronological text line per event
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_labs — building lab text lines")
        logger.info("Building lab text lines…")
        labs["lab_text_line"] = build_lab_text_line_series(labs)
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Sort chronologically and save
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_labs — saving labs_features.parquet")
        labs = labs.sort_values(["subject_id", "hadm_id", "charttime"])

        out_df = labs[[
            "subject_id", "hadm_id", "charttime",
            "itemid", "label", "fluid", "category",
            "lab_text_line",
        ]].reset_index(drop=True)

        os.makedirs(features_dir, exist_ok=True)
        output_path = os.path.join(features_dir, "labs_features.parquet")
        out_df.to_parquet(output_path, index=False)
        logger.info(
            "  Saved %d rows (%d unique admissions, %d unique itemids) to %s",
            len(out_df), out_df["hadm_id"].nunique(),
            out_df["itemid"].nunique(), output_path,
        )
        pbar.update(1)

    if registry_path:
        _record_hashes("extract_labs", source_paths, registry_path)
