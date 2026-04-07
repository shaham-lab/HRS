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

import yaml
from tqdm import tqdm

from preprocessing_utils import _gz_or_csv, _load_d_labitems, _record_hashes, _setup_logging

logger = logging.getLogger(__name__)

# Fluids whose items are each grouped under a single group key regardless of
# their category (Chemistry, Hematology, etc.)
_SINGLE_GROUP_FLUIDS = frozenset({
    "Ascites", "Pleural", "Cerebrospinal Fluid", "Bone Marrow", "Joint Fluid", "Stool",
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
    # Blood-gas panels for non-blood fluids: fold into the closest existing group
    # rather than creating spurious extra groups.
    ("urine", "blood gas"):             "urine_hematology",
    ("other body fluid", "blood gas"):  "other_body_fluid_hematology",
    ("fluid", "blood gas"):             "blood_gas",
}

# Explicit group-name overrides for single-group fluids whose snake_case
# conversion would differ from the canonical name in the spec.
_SINGLE_GROUP_NAME_OVERRIDES: dict[str, str] = {
    "cerebrospinal_fluid": "csf",
}


def _fluid_to_group_name(fluid: str) -> str:
    """Convert a single-group fluid name to snake_case group key."""
    raw = fluid.strip().lower().replace(" ", "_")
    return _SINGLE_GROUP_NAME_OVERRIDES.get(raw, raw)


def _assign_itemids_to_groups(d_labitems) -> dict[str, list[int]]:
    """Iterate over d_labitems and assign each itemid to its lab group.

    Returns a dict mapping group name → sorted list of integer itemids.
    """
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

    return groups


def run(config: dict) -> None:
    """Build lab panel config and save to CLASSIFICATIONS_DIR/lab_panel_config.yaml."""
    required_keys = ["MIMIC_DATA_DIR", "CLASSIFICATIONS_DIR"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    mimic_dir = config["MIMIC_DATA_DIR"]
    classifications_dir = config["CLASSIFICATIONS_DIR"]
    registry_path = config.get("HASH_REGISTRY_PATH", "")

    logger.info("Building lab panel config from d_labitems…")

    # ------------------------------------------------------------------ #
    # Hash-based skip check
    # ------------------------------------------------------------------ #
    source_paths = [p for p in [
        _gz_or_csv(mimic_dir, "hosp", "d_labitems"),
    ] if os.path.exists(p)]

    # ------------------------------------------------------------------ #
    # Load d_labitems
    # ------------------------------------------------------------------ #
    hosp_dir = os.path.join(mimic_dir, "hosp")
    logger.info("Loading d_labitems…")
    d_labitems = _load_d_labitems(hosp_dir)
    logger.info("d_labitems: %d rows after removing artefacts", len(d_labitems))

    # ------------------------------------------------------------------ #
    # Assign each itemid to a lab group
    # ------------------------------------------------------------------ #
    groups = _assign_itemids_to_groups(d_labitems)

    logger.info("Lab panel config: %d groups, %d total itemids",
                len(groups), sum(len(v) for v in groups.values()))
    logger.info("  %-45s  %s", "Group", "Item count")
    logger.info("  " + "-" * 55)
    for group_name in sorted(groups):
        logger.info("  %-45s  %d", group_name, len(groups[group_name]))

    # ------------------------------------------------------------------ #
    # Write output
    # ------------------------------------------------------------------ #
    os.makedirs(classifications_dir, exist_ok=True)
    output_path = os.path.join(classifications_dir, "lab_panel_config.yaml")
    with open(output_path, "w", encoding="utf-8") as fh:
        yaml.dump(groups, fh, default_flow_style=False, sort_keys=True)
    logger.info("Saved lab panel config → %s", output_path)

    if registry_path:
        _record_hashes("build_lab_panel_config", source_paths, registry_path)


if __name__ == "__main__":
    import argparse
    from preprocessing_utils import _load_config
    _setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/preprocessing.yaml")
    args = parser.parse_args()
    run(_load_config(args.config))

elif "snakemake" in dir():
    from preprocessing_utils import _normalize_config
    _setup_logging()
    run(_normalize_config(dict(snakemake.config)))
