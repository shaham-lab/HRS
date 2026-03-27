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

All configuration is loaded from config/preprocessing.yaml (located in the
repository root). No module reads preprocessing.yaml directly.
"""

import argparse
import logging
import os
import sys
import time

from preprocessing_utils import _load_config, _PATH_KEYS  # noqa: F401  (re-exported for tests)

logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config", "preprocessing.yaml")


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


def _run_module(name: str, config: dict, idx: int, total: int) -> float:
    """Run a single module and return elapsed seconds."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  STEP %d/%d — %s", idx, total, name)
    logger.info("=" * 70)
    t0 = time.time()
    module = _import_module(name)
    module.run(config)
    elapsed = time.time() - t0
    logger.info("  STEP %d/%d — %s completed in %.1fs", idx, total, name, elapsed)
    return elapsed


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
        "--build_lab_panel_config",
        action="store_true",
        help="Run build_lab_panel_config.py",
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
        "--extract_microbiology",
        action="store_true",
        help="Run extract_microbiology.py",
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
        help=f"Path to config/preprocessing.yaml (default: {_CONFIG_PATH})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun of ALL selected modules even if sources are unchanged.",
    )
    parser.add_argument(
        "--force-module",
        dest="force_modules",
        nargs="+",
        metavar="MODULE",
        help=(
            "Force rerun of specific modules by name even if sources are unchanged. "
            "Example: --force-module extract_demographics extract_labs"
        ),
    )
    parser.add_argument(
        "--modules",
        dest="modules",
        nargs="+",
        metavar="MODULE",
        help=(
            "Run only the specified module(s) by name, in pipeline order. "
            "Example: --modules combine_dataset"
        ),
    )
    parser.add_argument(
        "--skip-modules",
        dest="skip_modules",
        nargs="+",
        default=[],
        metavar="MODULE",
        help=(
            "Skip specific modules even if they are in the run order. "
            "Example: --skip-modules embed_features"
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Load configuration
    # ------------------------------------------------------------------ #
    config = _load_config(args.config)
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
        "extract_microbiology",
        "extract_radiology",
        "extract_y_data",
    ]

    # Full pipeline order
    _FULL_ORDER = (
        ["create_splits", "build_lab_panel_config"]
        + _EXTRACT_MODULES
        + ["embed_features", "combine_dataset"]
    )

    if args.all:
        modules_to_run = _FULL_ORDER
    elif getattr(args, "modules", None):
        invalid = [m for m in args.modules if m not in _FULL_ORDER]
        if invalid:
            parser.error(
                f"Unknown module(s): {invalid}. Valid modules: {_FULL_ORDER}"
            )
        modules_to_run = [m for m in _FULL_ORDER if m in args.modules]
    else:
        modules_to_run = [
            name for name in _FULL_ORDER
            if getattr(args, name, False)
        ]

    if not modules_to_run:
        parser.print_help()
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # Print plan before execution
    # ------------------------------------------------------------------ #
    logger.info("")
    logger.info("Pipeline plan — %d module(s) to run:", len(modules_to_run))
    for idx, name in enumerate(modules_to_run, start=1):
        logger.info("  %d. %s", idx, name)
    logger.info("")

    # ------------------------------------------------------------------ #
    # Execute modules in order
    # ------------------------------------------------------------------ #
    t_pipeline_start = time.time()
    n = len(modules_to_run)
    skip_modules = args.skip_modules or []
    force_modules = args.force_modules or []

    for idx, module_name in enumerate(modules_to_run, start=1):
        if module_name in skip_modules:
            logger.info(
                "  STEP %d/%d — %s SKIPPED (--skip-modules)",
                idx, n, module_name,
            )
            continue
        config["FORCE_RERUN"] = args.force or (module_name in force_modules)
        _run_module(module_name, config, idx, n)

    total_elapsed = time.time() - t_pipeline_start
    logger.info("")
    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("  Modules run : %s", modules_to_run)
    logger.info("  Total time  : %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
