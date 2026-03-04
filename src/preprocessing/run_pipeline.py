"""
run_pipeline.py – Orchestrator for the CDSS preprocessing pipeline.

Usage examples:
    # Run the full pipeline
    python preprocessing/run_pipeline.py --all

    # Run only specific steps
    python preprocessing/run_pipeline.py --create_splits
    python preprocessing/run_pipeline.py --extract_demographics --extract_labs
    python preprocessing/run_pipeline.py --embed_features
    python preprocessing/run_pipeline.py --combine_dataset

All configuration is loaded from preprocessing.yaml (located in the same
directory as this script). No module reads preprocessing.yaml directly.
"""

import argparse
import logging
import os
import sys

import yaml

logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "preprocessing.yaml")


_PATH_KEYS = {
    "MIMIC_DATA_DIR", "MIMIC_NOTE_DIR",
    "FEATURES_DIR", "EMBEDDINGS_DIR", "CLASSIFICATIONS_DIR",
    "HASH_REGISTRY_PATH",
}


def _load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Configuration file {config_path} must contain a YAML mapping."
        )
    for key in _PATH_KEYS:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = os.path.expanduser(cfg[key])
    return cfg


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _import_module(name: str):
    """Dynamically import a preprocessing sub-module by short name."""
    import importlib.util

    module_file = os.path.join(_SCRIPT_DIR, f"{name}.py")
    if not os.path.exists(module_file):
        raise FileNotFoundError(
            f"Module file not found: {module_file}"
        )
    spec = importlib.util.spec_from_file_location(name, module_file)
    if spec is None:
        raise ImportError(f"Cannot create module spec for: {module_file}")
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Module spec has no loader for: {module_file}")
    spec.loader.exec_module(module)
    return module


def _run_module(name: str, config: dict) -> None:
    logger.info("=" * 60)
    logger.info("Running module: %s", name)
    logger.info("=" * 60)
    module = _import_module(name)
    module.run(config)
    logger.info("Module '%s' completed successfully.", name)


def main() -> None:
    _setup_logging()

    # ------------------------------------------------------------------ #
    # Argument parsing
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(
        description="CDSS preprocessing pipeline orchestrator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the full pipeline in the correct order.",
    )
    parser.add_argument(
        "--create_splits",
        action="store_true",
        help="Run create_splits.py",
    )
    parser.add_argument(
        "--extract_demographics",
        action="store_true",
        help="Run extract_demographics.py",
    )
    parser.add_argument(
        "--extract_diag_history",
        action="store_true",
        help="Run extract_diag_history.py",
    )
    parser.add_argument(
        "--extract_discharge_history",
        action="store_true",
        help="Run extract_discharge_history.py",
    )
    parser.add_argument(
        "--extract_triage_and_complaint",
        action="store_true",
        help="Run extract_triage_and_complaint.py",
    )
    parser.add_argument(
        "--extract_labs",
        action="store_true",
        help="Run extract_labs.py",
    )
    parser.add_argument(
        "--extract_radiology",
        action="store_true",
        help="Run extract_radiology.py",
    )
    parser.add_argument(
        "--extract_y_data",
        action="store_true",
        help="Run extract_y_data.py",
    )
    parser.add_argument(
        "--embed_features",
        action="store_true",
        help="Run embed_features.py",
    )
    parser.add_argument(
        "--combine_dataset",
        action="store_true",
        help="Run combine_dataset.py",
    )
    parser.add_argument(
        "--config",
        default=_CONFIG_PATH,
        help=f"Path to preprocessing.yaml (default: {_CONFIG_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun of all selected modules even if sources are unchanged.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load configuration
    # ------------------------------------------------------------------ #
    config = _load_config(args.config)
    config["FORCE_RERUN"] = args.force
    logger.info("Loaded configuration from %s", args.config)

    # ------------------------------------------------------------------ #
    # Determine which modules to run
    # ------------------------------------------------------------------ #
    _EXTRACT_MODULES = [
        "extract_demographics",
        "extract_diag_history",
        "extract_discharge_history",
        "extract_triage_and_complaint",
        "extract_labs",
        "extract_radiology",
        "extract_y_data",
    ]

    # Full pipeline order
    _FULL_ORDER = (
        ["create_splits"]
        + _EXTRACT_MODULES
        + ["embed_features", "combine_dataset"]
    )

    if args.all:
        modules_to_run = _FULL_ORDER
    else:
        modules_to_run = [
            name for name in _FULL_ORDER
            if getattr(args, name, False)
        ]

    if not modules_to_run:
        parser.print_help()
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # Execute modules in order
    # ------------------------------------------------------------------ #
    for module_name in modules_to_run:
        _run_module(module_name, config)

    logger.info("Pipeline finished. Modules run: %s", modules_to_run)


if __name__ == "__main__":
    main()
