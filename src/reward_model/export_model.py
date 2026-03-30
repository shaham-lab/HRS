"""Export the frozen reward model to a self-contained artefact.

Produces a single ``.pt`` file that ``inference.py`` can load without access
to ``config/reward_model.yaml``.  The artefact embeds the model state dict,
feature index map snapshot, calibration temperatures, architecture config,
and input dimensionality.

Usage::

    python src/reward_model/export_model.py --config config/reward_model.yaml

See Detailed Design §5 (export_model.py) and Architecture §14 (Job Chain).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from reward_model_config import RewardModelConfig,load_and_validate_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Export dict assembly
# ---------------------------------------------------------------------------


def _build_export_dict(
    checkpoint_path: Path,
    calibration_params_path: Path,
) -> Dict[str, Any]:
    """Assemble a self-contained export dict from a checkpoint and calibration file.

    Loads ``best_model.pt`` (or any epoch checkpoint written by
    ``reward_model_manager.py``)
    and ``calibration_params.json`` and merges them into a single dict
    suitable for writing with ``torch.save``.  The result can be loaded by
    ``inference.py`` without access to ``config/reward_model.yaml``.

    Contents of the returned dict:

    - *model_state_dict* — Model weights, unwrapped from DDP by stripping
      the ``module.`` key prefix if present (applied directly to the state
      dict keys, since no live model object is available at export time).
    - *feature_index_map* — ``Dict[str, Tuple[int, int]]`` snapshot from the
      checkpoint; maps feature column name to ``(start, end)`` index range.
    - *T_0, T_1, …* — Scalar calibration temperature per target head.
    - *config_snapshot* — Architecture sub-dict containing only
      ``LAYER_WIDTHS``, ``DROPOUT_RATES``, ``ACTIVATION``, and
      ``NUM_TARGETS``.  Input dimensionality is stored separately as its own
      top-level key in the export dict.
    - *input_dim* — Input dimensionality derived from the feature index map
      (sum of all slot widths).

    Args:
        checkpoint_path: Absolute path to ``best_model.pt``.
        calibration_params_path: Absolute path to ``calibration_params.json``
            containing ``{'T_0': float, 'T_1': float, ...}``.

    Returns:
        Dict ready to be written with ``torch.save``.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    with open(calibration_params_path, "r") as f:
        calib = json.load(f)

    feature_index_map = ckpt["feature_index_map"]
    input_dim = sum(end - start for start, end in feature_index_map.values())

    raw_state_dict = ckpt["model_state_dict"]
    if any(key.startswith("module.") for key in raw_state_dict.keys()):
        model_state_dict = {k[len("module.") :]: v for k, v in raw_state_dict.items()}
    else:
        model_state_dict = raw_state_dict

    config_snapshot = ckpt["config"]
    num_targets = config_snapshot.get("NUM_TARGETS", 2)
    config_snapshot = {
        "LAYER_WIDTHS": config_snapshot["LAYER_WIDTHS"],
        "DROPOUT_RATES": config_snapshot["DROPOUT_RATES"],
        "ACTIVATION": config_snapshot["ACTIVATION"],
        "NUM_TARGETS": num_targets,
    }

    temperatures = {
        f"T_{i}": float(calib[f"T_{i}"])
        for i in range(num_targets)
    }

    return {
        "model_state_dict": model_state_dict,
        "feature_index_map": feature_index_map,
        **temperatures,
        "config_snapshot": config_snapshot,
        "input_dim": input_dim,
        "NUM_TARGETS": num_targets,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(config: RewardModelConfig) -> None:
    """Export the frozen model to a self-contained ``.pt`` artefact.

    Algorithm (Detailed Design §5, export_model.py):

    1. Load ``best_model.pt`` from ``CHECKPOINT_DIR``.
    2. Load ``calibration_params.json`` from ``CALIBRATION_PARAMS_PATH``.
    3. Assemble an export dict via ``_build_export_dict()``.
    3a. Creates the parent directory of ``EXPORT_PATH`` if it does not
        exist (``mkdir(parents=True, exist_ok=True)``).
    4. Write the export dict to ``EXPORT_PATH`` as a PyTorch ``.pt`` file.
    5. Log the export path and total model parameter count.

    Config keys used: ``CHECKPOINT_DIR``, ``CALIBRATION_PARAMS_PATH``,
    ``EXPORT_PATH``.

    Args:
        config: Validated ``RewardModelConfig`` instance.
    """
    checkpoint_path = Path(config.CHECKPOINT_DIR) / "best_model.pt"
    calibration_params_path = Path(config.CALIBRATION_PARAMS_PATH)
    export_path = Path(config.EXPORT_PATH)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    export_dict = _build_export_dict(checkpoint_path, calibration_params_path)
    torch.save(export_dict, export_path)

    param_count = sum(v.numel() for v in export_dict["model_state_dict"].values())
    logger.info("Exported frozen model to %s (params=%d)", export_path, param_count)


def main() -> int:
    """CLI entry point for ``export_model.py``."""
    parser = argparse.ArgumentParser(
        description="Export the frozen CDSS-ML reward model for RL agent consumption."
    )
    parser.add_argument("--config", required=True, help="Path to reward_model.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = load_and_validate_config(args.config)
    run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
