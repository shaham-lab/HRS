"""Temperature-scaling calibration for the CDSS-ML reward model.

Applies temperature scaling to the best trained model on the dev split.
Single GPU — no DDP.  Writes calibration parameters to
``CALIBRATION_PARAMS_PATH`` as JSON.

Usage::

    python src/reward_model/calibrate.py --config config/reward_model.yaml

See Detailed Design §5 (calibrate.py) and Architecture §8.7
(Post-Training Calibration).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.reward_model import load_dataset
from src.reward_model.model import RewardModel
from src.reward_model.reward_model_utils import (
    ParquetDataset,
    RewardModelConfig,
    get_device,
    load_and_validate_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple[RewardModel, Dict[str, Tuple[int, int]], Dict[str, Any]]:
    """Load a frozen ``RewardModel`` from a checkpoint file.

    Instantiates ``RewardModel`` from the architecture config embedded in the
    checkpoint (not the current ``config/reward_model.yaml`` — the checkpoint
    config is authoritative for architecture).  Loads the state dict, moves
    the model to ``device``, and calls ``model.eval()``.

    Args:
        checkpoint_path: Absolute path to ``best_model.pt`` (or any epoch
            checkpoint written by ``train.py``).
        device: Target device for the loaded model.

    Returns:
        Three-tuple of:

        - *model* — Loaded ``RewardModel`` in eval mode on ``device``.
        - *feature_index_map* — ``Dict[str, Tuple[int, int]]`` snapshot saved
          inside the checkpoint (maps feature column name to ``(start, end)``
          byte-range in the flat input tensor).
        - *config_snapshot* — Raw ``Dict`` from ``config.model_dump()`` as
           saved by ``train.py``; authoritative for architecture keys
           ``LAYER_WIDTHS``, ``DROPOUT_RATE``, and ``ACTIVATION``.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    config_snapshot: Dict = ckpt["config"]
    feature_index_map: Dict[str, Tuple[int, int]] = ckpt["feature_index_map"]
    input_dim = max(end for _, end in feature_index_map.values())

    model = RewardModel(
        input_dim=input_dim,
        layer_widths=config_snapshot["LAYER_WIDTHS"],
        dropout_rate=config_snapshot["DROPOUT_RATE"],
        activation=config_snapshot["ACTIVATION"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model, feature_index_map, config_snapshot


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def _run_forward_pass(
    model: RewardModel,
    dataset: ParquetDataset,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a full forward pass over *dataset* with ``torch.no_grad()``.

    Iterates the dataset in a ``DataLoader`` with ``num_workers=0`` (no worker
    processes) — calibration is single-pass on a single GPU and spawning
    workers adds overhead without benefit.  Accumulates raw (un-calibrated)
    logits for both heads together with ground-truth labels.  No masking is
    applied — the full feature vector is passed to the model.

    Args:
        model: Frozen ``RewardModel`` in eval mode on ``device``.
        dataset: ``ParquetDataset`` instance for the target split (typically
            the dev split).
        device: Device on which batch tensors are placed before the forward
            pass.
        batch_size: Number of samples per DataLoader batch.  Pass
            ``config.BATCH_SIZE_PER_GPU`` from ``run()``.

    Returns:
        Four-tuple of flat NumPy arrays, each of shape ``(N,)`` where *N* is
        the total number of rows in *dataset*:

        - *logits_y1* — Raw logits for the mortality head (Y1), float32.
        - *logits_y2* — Raw logits for the readmission head (Y2), float32.
        - *y1* — Ground-truth mortality labels (int8, values 0 or 1).
        - *y2* — Ground-truth readmission labels (float32, ``NaN`` for
           deceased patients).
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    all_logits_y1 = []
    all_logits_y2 = []
    all_y1 = []
    all_y2 = []

    with torch.no_grad():
        for X, y1, y2 in dataloader:
            X = X.to(device)
            y1_t = y1.to(device)
            y2_t = y2.to(device)

            logits_y1, logits_y2 = model(X)

            all_logits_y1.append(logits_y1.detach().cpu().numpy().reshape(-1))
            all_logits_y2.append(logits_y2.detach().cpu().numpy().reshape(-1))
            all_y1.append(y1_t.detach().cpu().numpy().reshape(-1))
            all_y2.append(y2_t.detach().cpu().numpy().reshape(-1))

    logits_y1_np = np.concatenate(all_logits_y1, axis=0)
    logits_y2_np = np.concatenate(all_logits_y2, axis=0)
    y1_np = np.concatenate(all_y1, axis=0)
    y2_np = np.concatenate(all_y2, axis=0)

    return logits_y1_np, logits_y2_np, y1_np, y2_np


# ---------------------------------------------------------------------------
# Temperature fitting
# ---------------------------------------------------------------------------


def _fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Fit a scalar temperature *T* via L-BFGS minimisation of NLL.

    Temperature scaling: the calibrated probability is
    ``sigmoid(logit / T)``, equivalent to ``sigmoid(logit * (1/T))``.
    Optimisation is performed only on the subset of samples selected by
    *mask*.

    Args:
        logits: Raw model logits of shape ``(N,)``, float32.
        labels: Binary ground-truth labels of shape ``(N,)`` with values in
            ``{0, 1}``.
        mask: Boolean array of shape ``(N,)`` selecting which samples to
            include.  For Y1 this is all-True (full dev split); for Y2 this
            is the survivor mask ``~np.isnan(y2)``.

    Returns:
        Fitted scalar temperature ``T``, clamped to a minimum of
        ``1e-8`` before being returned to prevent downstream division
        by zero in ``sigmoid(logit / T)``.
    """
    masked_logits = logits[mask]
    masked_labels = labels[mask]

    logits_t = torch.tensor(masked_logits, dtype=torch.float32)
    labels_t = torch.tensor(masked_labels, dtype=torch.float32)

    T = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled_logits = logits_t / T.clamp(min=1e-8)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled_logits, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(max(T.item(), 1e-8))


# ---------------------------------------------------------------------------
# ECE helper
# ---------------------------------------------------------------------------


def _compute_ece_from_logits(
    logits: np.ndarray,
    labels: np.ndarray,
    T: float,
    mask: np.ndarray,
) -> float:
    """Compute Expected Calibration Error (ECE) after applying temperature *T*.

    Uses equal-width probability binning (15 bins) on the masked subset of
    samples.  Probabilities are clipped to ``[0, 1]`` before binning and the
    rightmost bin is closed on both sides so that samples with probability
    exactly ``1.0`` are not silently excluded.  Intended for logging pre- and
    post-calibration ECE for audit; not used in the optimisation itself.

    Args:
        logits: Raw model logits of shape ``(N,)``, float32.
        labels: Binary ground-truth labels of shape ``(N,)`` with values in
            ``{0, 1}``.
        T: Scalar temperature applied before sigmoid.  Pass ``T = 1.0`` to
            compute the pre-calibration ECE.
        mask: Boolean array of shape ``(N,)`` selecting which samples to
            include in the ECE computation.

    Returns:
        ECE as a float in ``[0, 1]``.
    """
    masked_logits = logits[mask]
    masked_labels = labels[mask]

    probs = 1.0 / (1.0 + np.exp(-masked_logits / T))

    bin_edges = np.linspace(0.0, 1.0, num=16)
    ece = 0.0
    for i in range(len(bin_edges) - 1):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs >= lower) & (probs < upper) if i < len(bin_edges) - 2 else (probs >= lower) & (probs <= upper)
        if not np.any(in_bin):
            continue
        conf = probs[in_bin].mean()
        acc = masked_labels[in_bin].mean()
        ece += np.abs(conf - acc) * (np.sum(in_bin) / len(probs))
    return float(ece)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(config: RewardModelConfig) -> None:
    """Run temperature scaling calibration on the dev split.

    Algorithm (Detailed Design §5, calibrate.py):

    1. Load ``best_model.pt`` from ``CHECKPOINT_DIR``.  Extract model state
       dict and config snapshot; instantiate ``RewardModel`` from the
       checkpoint config snapshot (not the current YAML).
    2. Load the dev split from ``DATASET_PATH`` via ``load_dataset.run()``.
    3. Run a full forward pass with ``torch.no_grad()`` to collect raw logits
       for Y1 and Y2.
    4. Fit ``T_y1`` on the full dev split via L-BFGS on NLL.
    5. Fit ``T_y2`` on the survivor subset (``~isnan(y2)`` rows) via L-BFGS
       on NLL.
    6. Log pre- and post-calibration ECE for both heads for audit.
    7. Write ``{'T_y1': float(T_y1), 'T_y2': float(T_y2)}`` to
       ``CALIBRATION_PARAMS_PATH`` as JSON.

    Config keys used: ``CHECKPOINT_DIR``, ``DATASET_PATH``,
    ``CALIBRATION_PARAMS_PATH``.

    Args:
        config: Validated ``RewardModelConfig`` instance.
    """
    device = get_device(local_rank=0)
    checkpoint_path = Path(config.CHECKPOINT_DIR) / "best_model.pt"
    model, feature_index_map, config_snapshot = _load_model_from_checkpoint(checkpoint_path, device)

    bundle = load_dataset.run(config)
    logits_y1, logits_y2, y1, y2 = _run_forward_pass(
        model,
        bundle.dev_dataset,
        device,
        config.BATCH_SIZE_PER_GPU,
    )

    survivor_mask = ~np.isnan(y2)
    full_mask = np.ones(len(y1), dtype=bool)

    pre_ece_y1 = _compute_ece_from_logits(logits_y1, y1, 1.0, full_mask)
    pre_ece_y2 = _compute_ece_from_logits(logits_y2, y2, 1.0, survivor_mask)
    logger.info("Pre-calibration ECE — Y1: %.6f, Y2: %.6f", pre_ece_y1, pre_ece_y2)

    T_y1 = _fit_temperature(logits_y1, y1, full_mask)
    T_y2 = _fit_temperature(logits_y2, y2, survivor_mask)

    post_ece_y1 = _compute_ece_from_logits(logits_y1, y1, T_y1, full_mask)
    post_ece_y2 = _compute_ece_from_logits(logits_y2, y2, T_y2, survivor_mask)
    logger.info("Post-calibration ECE — Y1: %.6f, Y2: %.6f", post_ece_y1, post_ece_y2)

    params = {"T_y1": T_y1, "T_y2": T_y2}
    calibration_path = Path(config.CALIBRATION_PARAMS_PATH)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(params))
    logger.info(
        "Wrote calibration parameters to %s (T_y1=%.6f, T_y2=%.6f)",
        calibration_path,
        T_y1,
        T_y2,
    )


def main() -> int:
    """CLI entry point for ``calibrate.py``."""
    parser = argparse.ArgumentParser(
        description="Temperature-scaling calibration for the CDSS-ML reward model."
    )
    parser.add_argument("--config", required=True, help="Path to reward_model.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    config = load_and_validate_config(args.config)
    run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
