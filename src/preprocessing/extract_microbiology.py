"""
extract_microbiology.py – Extract microbiology events as per-panel text parquets.

Reads microbiologyevents, applies panel assignment from micro_panel_config.yaml,
applies time window filtering and comment cleaning, then writes one parquet per
panel to FEATURES_DIR.

Output: FEATURES_DIR/micro_<panel_name>.parquet
  Columns: subject_id, hadm_id, text
  One row per admission with events in this panel within the time window.

Expected config keys:
    MIMIC_DATA_DIR          – root directory containing MIMIC-IV tables (hosp/)
    FEATURES_DIR            – output directory for feature parquets
    MICRO_PANEL_CONFIG_PATH – path to micro_panel_config.yaml (relative to repo root)

Optional config keys:
    MICRO_WINDOW_HOURS          – int hours from admittime, or "full_admission"
                                  Default: 72
    MICRO_NULL_HADM_STRATEGY    – "drop" (default) or "link"
    MICRO_LINK_TOLERANCE_HOURS  – hours tolerance for linkage (default: 2)
"""

import json
import logging
import os
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from preprocessing_utils import (
    _check_required_keys,
    _gz_or_csv,
    _link_hadm_for_row,
    _load_csv,
    _record_hashes,
    _sources_unchanged,
)
from build_micro_text import aggregate_panel_text

logger = logging.getLogger(__name__)


def _parse_micro_window(config: dict):
    """Parse MICRO_WINDOW_HOURS from config.

    Returns either an int (hours) or the string "full_admission".
    """
    raw = config.get("MICRO_WINDOW_HOURS", 72)
    if str(raw).strip().lower() == "full_admission":
        return "full_admission"
    try:
        hours = int(raw)
        if hours <= 0:
            raise ValueError(f"MICRO_WINDOW_HOURS must be > 0, got {hours}")
        return hours
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid MICRO_WINDOW_HOURS value: {raw!r}. "
            "Expected a positive integer or 'full_admission'."
        ) from exc


def run(config: dict) -> None:
    _check_required_keys(config, ["MIMIC_DATA_DIR", "FEATURES_DIR", "MICRO_PANEL_CONFIG_PATH"])

    mimic_dir = str(config["MIMIC_DATA_DIR"])
    features_dir = str(config["FEATURES_DIR"])
    registry_path = config.get("HASH_REGISTRY_PATH", "")

    micro_window = _parse_micro_window(config)
    ed_lookback_hours: int = int(config.get("ED_LOOKBACK_HOURS", 24))
    null_hadm_strategy = str(config.get("MICRO_NULL_HADM_STRATEGY", "drop")).lower()
    link_tolerance_hours = int(config.get("MICRO_LINK_TOLERANCE_HOURS", 2))

    hosp_dir = os.path.join(mimic_dir, "hosp")

    # Hash-based skip check
    source_paths = [
        _gz_or_csv(mimic_dir, "hosp", "microbiologyevents"),
        _gz_or_csv(mimic_dir, "hosp", "admissions"),
    ]
    output_paths = [os.path.join(features_dir, "micro_blood_culture_routine.parquet")]

    if registry_path and not config.get("FORCE_RERUN", False):
        if _sources_unchanged(
            "extract_microbiology", source_paths, output_paths, registry_path, logger
        ):
            return

    # Load micro panel config
    config_path = Path(config["MICRO_PANEL_CONFIG_PATH"]).resolve()
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"micro_panel_config.yaml not found at {config_path}. "
            "Ensure MICRO_PANEL_CONFIG_PATH in config/preprocessing.yaml points to a valid file."
        )
    with open(config_path, encoding="utf-8") as fh:
        micro_cfg = yaml.safe_load(fh)

    panels = micro_cfg.get("panels", {})
    excluded_tests = set(micro_cfg.get("excluded_tests", []))
    excluded_spec_types = set(micro_cfg.get("excluded_spec_types", []))
    cleaning_config = dict(micro_cfg.get("comment_cleaning", {}))

    # Allow config overrides for comment cleaning parameters
    if not config.get("MICRO_INCLUDE_COMMENTS", True):
        cleaning_config["max_chars"] = 0
    if "MICRO_COMMENT_MAX_SENTENCES" in config:
        cleaning_config["max_sentences"] = int(config["MICRO_COMMENT_MAX_SENTENCES"])
    if "MICRO_COMMENT_MAX_CHARS" in config:
        cleaning_config["max_chars"] = int(config["MICRO_COMMENT_MAX_CHARS"])

    # Build combo → panel lookup
    combo_to_panel = {
        (t, s): panel_name
        for panel_name, panel_data in panels.items()
        for t, s in panel_data["combos"]
    }

    # Load microbiologyevents
    micro = _load_csv(
        os.path.join(hosp_dir, "microbiologyevents.csv.gz"),
        os.path.join(hosp_dir, "microbiologyevents.csv"),
        usecols=["subject_id", "hadm_id", "charttime", "spec_type_desc", "test_name",
                 "org_name", "ab_name", "interpretation", "comments"],
        parse_dates=["charttime"],
        dtype={"subject_id": int, "hadm_id": float},
    )
    logger.info("Loaded %d microbiologyevents rows", len(micro))

    # Apply exclusions
    before_excl = len(micro)
    excl_mask = (
        micro["test_name"].isin(excluded_tests)
        | micro["spec_type_desc"].isin(excluded_spec_types)
    )
    micro = micro[~excl_mask].copy()
    logger.info(
        "After exclusions: dropped %d rows, %d remain",
        before_excl - len(micro), len(micro),
    )

    # Handle null hadm_id
    null_mask = micro["hadm_id"].isna()
    null_count = int(null_mask.sum())
    if null_hadm_strategy == "drop":
        micro = micro[~null_mask].copy()
        logger.info("  Dropped %d rows with null hadm_id", null_count)
    elif null_hadm_strategy == "link" and null_count > 0:
        admissions_for_link = _load_csv(
            os.path.join(hosp_dir, "admissions.csv.gz"),
            os.path.join(hosp_dir, "admissions.csv"),
            usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
            parse_dates=["admittime", "dischtime"],
            dtype={"subject_id": int, "hadm_id": int},
        )
        null_rows = micro[null_mask].copy()
        resolved = []
        tolerance = pd.Timedelta(hours=link_tolerance_hours)
        for _, row in null_rows.iterrows():
            resolved_hadm = _link_hadm_for_row(row, admissions_for_link, tolerance)
            if resolved_hadm is None:
                continue
            new_row = row.copy()
            new_row["hadm_id"] = int(resolved_hadm)
            resolved.append(new_row)
        non_null = micro[~null_mask].copy()
        if resolved:
            resolved_df = pd.DataFrame(resolved)
            micro = pd.concat([non_null, resolved_df], ignore_index=True)
            logger.info(
                "  Linked %d/%d null hadm_id rows", len(resolved), null_count
            )
        else:
            micro = non_null
            logger.info(
                "  No null hadm_id rows could be linked (%d dropped)", null_count
            )
        stats = {
            "null_count": null_count,
            "linked": len(resolved),
            "dropped": null_count - len(resolved),
        }
        stats_path = os.path.join(features_dir, "micro_linkage_stats.json")
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)

    micro = micro.dropna(subset=["hadm_id"]).copy()
    micro["hadm_id"] = micro["hadm_id"].astype(int)

    # Load admissions for time-window join
    admissions = _load_csv(
        os.path.join(hosp_dir, "admissions.csv.gz"),
        os.path.join(hosp_dir, "admissions.csv"),
        usecols=["subject_id", "hadm_id", "admittime", "dischtime"],
        parse_dates=["admittime", "dischtime"],
        dtype={"subject_id": int, "hadm_id": int},
    )

    # Join admissions
    micro = micro.merge(
        admissions[["subject_id", "hadm_id", "admittime", "dischtime"]],
        on=["subject_id", "hadm_id"],
        how="inner",
    )
    logger.info(
        "After joining admissions: %d rows, %d unique admissions",
        len(micro),
        micro["hadm_id"].nunique(),
    )

    # Apply time-window filter
    lookback = pd.to_timedelta(ed_lookback_hours, unit="h")
    window_mask = micro["charttime"] >= (micro["admittime"] - lookback)
    if isinstance(micro_window, int):
        cutoff = micro["admittime"] + pd.to_timedelta(micro_window, unit="h")
        window_mask &= micro["charttime"] <= cutoff
    else:  # "full_admission"
        window_mask &= micro["charttime"] <= micro["dischtime"]
    micro = micro[window_mask].copy()
    logger.info(
        "After window filter (%s window, %dh ED lookback): %d rows for %d admissions",
        micro_window,
        ed_lookback_hours,
        len(micro),
        micro["hadm_id"].nunique(),
    )

    # Assign panels via vectorized merge on stripped (test_name, spec_type_desc) pairs
    micro["_test_name_s"] = micro["test_name"].str.strip()
    micro["_spec_type_s"] = micro["spec_type_desc"].str.strip()
    combo_df = pd.DataFrame(
        [(t, s, p) for (t, s), p in combo_to_panel.items()],
        columns=["_test_name_s", "_spec_type_s", "panel"],
    )
    micro = micro.merge(combo_df, on=["_test_name_s", "_spec_type_s"], how="left")
    micro = micro.drop(columns=["_test_name_s", "_spec_type_s"])

    unassigned = micro[micro["panel"].isna()].copy()
    if not unassigned.empty:
        unique_combos = sorted(
            set(zip(
                unassigned["test_name"].astype(str),
                unassigned["spec_type_desc"].astype(str),
            ))
        )
        logger.info("  %d unassigned combos (not in any panel):", len(unique_combos))
        for t, s in unique_combos[:20]:
            logger.info("    (%r, %r)", t, s)
        if len(unique_combos) > 20:
            logger.info("    ... and %d more", len(unique_combos) - 20)
    micro = micro[micro["panel"].notna()].copy()

    # Process each panel
    os.makedirs(features_dir, exist_ok=True)
    panel_names = list(panels.keys())
    total_admissions = 0

    for panel_name in tqdm(panel_names, desc="extract_microbiology panels", unit="panel"):
        panel_df = micro[micro["panel"] == panel_name].copy()
        if panel_df.empty:
            result_df = pd.DataFrame(columns=["subject_id", "hadm_id", "text"])
        else:
            result_df = aggregate_panel_text(panel_df, cleaning_config)

        out_path = os.path.join(features_dir, f"micro_{panel_name}.parquet")
        result_df.to_parquet(out_path, index=False)

        n_admissions = len(result_df)
        total_admissions += n_admissions
        logger.info(
            "  micro_%s: %d admissions with events", panel_name, n_admissions
        )

    logger.info(
        "extract_microbiology complete: %d panels written, %d total admissions across panels",
        len(panel_names),
        total_admissions,
    )

    if registry_path:
        _record_hashes("extract_microbiology", source_paths, registry_path)
