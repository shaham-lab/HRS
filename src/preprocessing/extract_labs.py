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
    HADM_LINKAGE_STRATEGY        – "drop" (default) or "link"; how to handle
                                   null hadm_id in labevents
    HADM_LINKAGE_TOLERANCE_HOURS – hours of tolerance for time-window linkage
                                   (default 1, only used when strategy is "link")
"""

import logging
import os

import pandas as pd
from tqdm import tqdm

from preprocessing_utils import _check_required_keys, _gz_or_csv, _link_hadm_for_row, _load_csv, _record_hashes, _sources_unchanged
from build_lab_text_lines import _compute_row_abnormal_flag

logger = logging.getLogger(__name__)

# Rows to read per chunk from labevents
_CHUNK_SIZE = 1_000_000


def _build_lab_text_line(row) -> str:
    """Convert a single lab event row to a chronological text line.

    Format: [HH:MM] {label}: {value} {unit} (ref: lower-upper) [ABNORMAL]

    Where [HH:MM] is elapsed time since admittime (relative, not clock time).
    [ABNORMAL] is appended when flag == "abnormal" OR when valuenum falls
    outside [ref_range_lower, ref_range_upper].
    """
    # Elapsed time since admittime
    try:
        elapsed = pd.to_datetime(row["charttime"]) - pd.to_datetime(row["admittime"])
        total_minutes = int(elapsed.total_seconds() // 60)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        time_str = f"{hours:02d}:{minutes:02d}"
    except (TypeError, ValueError, OverflowError):
        time_str = "00:00"

    # Value: prefer numeric formatted to 2dp, fall back to text value
    if pd.notna(row["valuenum"]):
        value_str = f"{row['valuenum']:.2f}"
    else:
        value_str = str(row["value"]).strip()

    # Unit
    uom = f" {row['valueuom']}" if pd.notna(row["valueuom"]) else ""

    # Reference range — only include when both bounds are present
    ref_str = ""
    ref_lower = row.get("ref_range_lower", None)
    ref_upper = row.get("ref_range_upper", None)
    if pd.notna(ref_lower) and pd.notna(ref_upper):
        ref_str = f" (ref: {ref_lower}-{ref_upper})"

    # Abnormal flag: flagged as "abnormal" OR valuenum outside reference range
    flag_str = " [ABNORMAL]" if _compute_row_abnormal_flag(row) else ""

    return (
        f"[{time_str}] {row['label']}: "
        f"{value_str}{uom}{ref_str}{flag_str}"
    )


def run(config: dict) -> None:
    """Extract lab events as long-format chronological text lines per admission."""
    _check_required_keys(config, ["MIMIC_DATA_DIR", "FEATURES_DIR"])

    mimic_dir = config["MIMIC_DATA_DIR"]
    features_dir = config["FEATURES_DIR"]
    hosp_dir = os.path.join(mimic_dir, "hosp")
    registry_path = config.get("HASH_REGISTRY_PATH", "")

    # Parse LAB_ADMISSION_WINDOW: int (hours) or "full"
    raw_window = config.get("LAB_ADMISSION_WINDOW", 24)
    if str(raw_window).strip().lower() == "full":
        lab_window_hours: int | None = None  # sentinel: use dischtime
    else:
        try:
            lab_window_hours = int(raw_window)
            if lab_window_hours <= 0:
                raise ValueError(f"LAB_ADMISSION_WINDOW must be positive, got {lab_window_hours}")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"LAB_ADMISSION_WINDOW must be a positive integer or 'full', "
                f"got {raw_window!r}"
            ) from exc

    hadm_linkage_strategy: str = str(config.get("HADM_LINKAGE_STRATEGY", "drop")).lower()
    hadm_linkage_tolerance_hours: int = int(config.get("HADM_LINKAGE_TOLERANCE_HOURS", 1))

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
    steps = [
        "Load d_labitems and admissions",
        "Stream and filter labevents",
        "Apply admission window filter",
        "Build lab text lines",
        "Sort and save labs_features.parquet",
    ]
    with tqdm(total=len(steps), desc="extract_labs", unit="step", dynamic_ncols=True) as pbar:
        pbar.set_description("extract_labs — loading d_labitems and admissions")
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
        logger.info("  d_labitems: %d items across %d fluids and %d categories",
                    len(d_labitems),
                    d_labitems["fluid"].nunique(),
                    d_labitems["category"].nunique())

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
        logger.info("  Admissions: %d rows loaded for window filtering", len(admissions))

        if lab_window_hours is not None:
            logger.info(
                "Lab admission window: first %d hours from admittime", lab_window_hours
            )
        else:
            logger.info("Lab admission window: full admission (admittime → dischtime)")
        pbar.update(1)

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
        logger.info("Streaming labevents from %s…", lab_path)

        pbar.set_description("extract_labs — streaming labevents chunks")
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
            # 1. Handle rows with no hadm_id per strategy
            chunk = chunk.copy()
            raw_chunk_len = len(chunk)
            null_hadm_mask = chunk["hadm_id"].isna()
            null_hadm_count = int(null_hadm_mask.sum())

            if hadm_linkage_strategy == "drop":
                chunk = chunk[~null_hadm_mask].copy()
            elif hadm_linkage_strategy == "link" and null_hadm_count > 0:
                null_rows = chunk[null_hadm_mask].copy()
                resolved_rows = []
                tolerance = pd.Timedelta(hours=hadm_linkage_tolerance_hours)
                for _, row in null_rows.iterrows():
                    resolved_hadm = _link_hadm_for_row(row, admissions, tolerance)
                    if resolved_hadm is None:
                        continue
                    new_row = row.copy()
                    new_row["hadm_id"] = int(resolved_hadm)
                    resolved_rows.append(new_row)

                non_null = chunk[~null_hadm_mask].copy()
                if resolved_rows:
                    resolved_df = pd.DataFrame(resolved_rows)
                    chunk = pd.concat([non_null, resolved_df], ignore_index=True)
                else:
                    chunk = non_null

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

            logger.info(
                "  Chunk %d: %d rows read  |  %d null hadm_id (%s)  |  %d retained after filters",
                i,
                raw_chunk_len,
                null_hadm_count,
                f"strategy: {hadm_linkage_strategy}",
                len(chunk),
            )
            i += 1

            if not chunk.empty:
                all_chunks.append(chunk)

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
        # Apply admission window filter after concatenating all chunks
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_labs — applying admission window filter")
        labs = labs.merge(
            admissions[["subject_id", "hadm_id", "admittime", "dischtime"]],
            on=["subject_id", "hadm_id"],
            how="inner",
        )

        # Lower bound: charttime >= admittime (always)
        window_mask = labs["charttime"] >= labs["admittime"]

        # Upper bound: configurable
        if lab_window_hours is None:
            # "full" — use entire admission window
            window_mask &= labs["charttime"] <= labs["dischtime"]
        else:
            # Integer hours — cut off at admittime + window
            cutoff = labs["admittime"] + pd.to_timedelta(lab_window_hours, unit="h")
            window_mask &= labs["charttime"] <= cutoff

        labs = labs[window_mask]
        n_after_window = len(labs)
        logger.info("  After window filter (%s): %d rows for %d admissions",
                    f"{lab_window_hours}h" if lab_window_hours else "full",
                    n_after_window, labs["hadm_id"].nunique())
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Build chronological text line per event
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_labs — building lab text lines")
        logger.info("Building lab text lines…")
        tqdm.pandas(desc="Building lab text lines")
        labs["lab_text_line"] = labs.progress_apply(_build_lab_text_line, axis=1)
        pbar.update(1)

        # ------------------------------------------------------------------ #
        # Sort chronologically within each admission
        # ------------------------------------------------------------------ #
        pbar.set_description("extract_labs — saving labs_features.parquet")
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
        logger.info(
            "  Saved %d rows (%d unique admissions, %d unique itemids) to %s",
            len(out_df), out_df["hadm_id"].nunique(),
            out_df["itemid"].nunique(), output_path,
        )
        pbar.update(1)

    if registry_path:
        _record_hashes("extract_labs", source_paths, registry_path)
