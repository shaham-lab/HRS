"""
build_lab_panel_config.py – Build lab panel configuration from d_labitems.

Reads d_labitems, groups itemids by (fluid × category) combinations into 13
named lab groups, and writes the result to CLASSIFICATIONS_DIR/lab_panel_config.yaml.

Each group is a mapping from a snake_case group name (e.g. blood_chemistry) to a
list of integer itemids.

The 13 expected group names (derived from fluid × category combinations present
in MIMIC-IV d_labitems) are:
    blood_gas, blood_chemistry, blood_hematology,
    urine_chemistry, urine_hematology,
    other_body_fluid_chemistry, other_body_fluid_hematology,
    ascites, pleural, csf, bone_marrow, joint_fluid, stool

Some groups (e.g. ascites, pleural) span multiple categories (Chemistry +
Hematology) — the fluid name alone is used as the group key in those cases.

Must run before extract_labs.py.

Expected config keys:
    MIMIC_DATA_DIR      – root directory containing MIMIC-IV tables (hosp/)
    CLASSIFICATIONS_DIR – output directory for lab_panel_config.yaml
"""

import logging
import os

import pandas as pd
import yaml
from tqdm import tqdm

from preprocessing_utils import _gz_or_csv, _load_csv, _record_hashes, _sources_unchanged

logger = logging.getLogger(__name__)

# Fluids whose items are each grouped under a single group key regardless of
# their category (Chemistry, Hematology, etc.)
_SINGLE_GROUP_FLUIDS = frozenset({
    "Ascites", "Pleural", "CSF", "Bone Marrow", "Joint Fluid", "Stool",
})

# Map (fluid_lower, category_lower) → group name for multi-category fluids
_FLUID_CATEGORY_MAP: dict[tuple[str, str], str] = {
    ("blood", "blood gas"):     "blood_gas",
    ("blood", "chemistry"):     "blood_chemistry",
    ("blood", "hematology"):    "blood_hematology",
    ("urine", "chemistry"):     "urine_chemistry",
    ("urine", "hematology"):    "urine_hematology",
    ("other body fluid", "chemistry"):  "other_body_fluid_chemistry",
    ("other body fluid", "hematology"): "other_body_fluid_hematology",
}

# Artefact fluid values to exclude
_ARTIFACT_FLUIDS = frozenset({"I", "Q", "fluid"})


def _fluid_to_group_name(fluid: str) -> str:
    """Convert a single-group fluid name to snake_case group key."""
    return fluid.strip().lower().replace(" ", "_")


def run(config: dict) -> None:
    """Build lab panel config and save to CLASSIFICATIONS_DIR/lab_panel_config.yaml."""
    required_keys = ["MIMIC_DATA_DIR", "CLASSIFICATIONS_DIR"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    mimic_dir = config["MIMIC_DATA_DIR"]
    classifications_dir = config["CLASSIFICATIONS_DIR"]
    registry_path = config.get("HASH_REGISTRY_PATH", "")

    # ------------------------------------------------------------------ #
    # Hash-based skip check
    # ------------------------------------------------------------------ #
    source_paths = [p for p in [
        _gz_or_csv(mimic_dir, "hosp", "d_labitems"),
    ] if os.path.exists(p)]
    output_paths = [os.path.join(classifications_dir, "lab_panel_config.yaml")]

    if registry_path and not config.get("FORCE_RERUN", False):
        if _sources_unchanged("build_lab_panel_config", source_paths,
                               output_paths, registry_path, logger):
            return

    # ------------------------------------------------------------------ #
    # Load d_labitems
    # ------------------------------------------------------------------ #
    hosp_dir = os.path.join(mimic_dir, "hosp")
    logger.info("Loading d_labitems…")
    d_labitems = _load_csv(
        os.path.join(hosp_dir, "d_labitems.csv.gz"),
        os.path.join(hosp_dir, "d_labitems.csv"),
        usecols=["itemid", "label", "fluid", "category"],
    )

    # Strip whitespace and remove artefact rows
    d_labitems["fluid"] = d_labitems["fluid"].str.strip()
    d_labitems["category"] = d_labitems["category"].str.strip()
    d_labitems = d_labitems[~d_labitems["fluid"].isin(_ARTIFACT_FLUIDS)].copy()

    logger.info("d_labitems: %d rows after removing artefacts", len(d_labitems))

    # ------------------------------------------------------------------ #
    # Assign each itemid to a lab group
    # ------------------------------------------------------------------ #
    groups: dict[str, list[int]] = {}

    for _, row in tqdm(d_labitems.iterrows(), total=len(d_labitems),
                       desc="Assigning lab groups", unit="item"):
        fluid = str(row["fluid"]).strip()
        category = str(row["category"]).strip()
        itemid = int(row["itemid"])

        fluid_lower = fluid.lower()
        category_lower = category.lower()

        if fluid in _SINGLE_GROUP_FLUIDS:
            group_name = _fluid_to_group_name(fluid)
        else:
            group_name = _FLUID_CATEGORY_MAP.get(
                (fluid_lower, category_lower),
                # Fallback: combine fluid + category as snake_case
                f"{fluid_lower.replace(' ', '_')}_{category_lower.replace(' ', '_')}",
            )

        groups.setdefault(group_name, [])
        if itemid not in groups[group_name]:
            groups[group_name].append(itemid)

    # Sort itemids within each group for determinism
    for group_name in groups:
        groups[group_name].sort()

    logger.info("Lab panel config: %d groups, total %d itemids",
                len(groups), sum(len(v) for v in groups.values()))
    for group_name, items in sorted(groups.items()):
        logger.info("  %s: %d itemids", group_name, len(items))

    # ------------------------------------------------------------------ #
    # Write output
    # ------------------------------------------------------------------ #
    os.makedirs(classifications_dir, exist_ok=True)
    output_path = os.path.join(classifications_dir, "lab_panel_config.yaml")
    with open(output_path, "w", encoding="utf-8") as fh:
        yaml.dump(groups, fh, default_flow_style=False, sort_keys=True)
    logger.info("Saved lab panel config to %s", output_path)

    if registry_path:
        _record_hashes("build_lab_panel_config", source_paths, registry_path)
