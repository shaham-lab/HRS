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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import RewardModel
from parquet_dataset import ParquetDataset
from reward_model_config import RewardModelConfig,load_and_validate_config
from reward_model_utils import get_device
from mimic4_data_loader import Mimic4DataLoader

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
           ``LAYER_WIDTHS``, ``DROPOUT_RATES``, ``ACTIVATION``, and
           ``NUM_TARGETS``.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config_snapshot: Dict[str, Any] = ckpt["config"]
    feature_index_map: Dict[str, Tuple[int, int]] = ckpt["feature_index_map"]
    input_dim = sum(end - start for start, end in feature_index_map.values())

    model = RewardModel(
        input_dim=input_dim,
        layer_widths=config_snapshot["LAYER_WIDTHS"],
        dropout_rates=config_snapshot["DROPOUT_RATES"],
        activation=config_snapshot.get("ACTIVATION", "relu"),
        num_targets=config_snapshot.get("NUM_TARGETS", 2),
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
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Run a full forward pass over *dataset* with ``torch.no_grad()``.

    Iterates the dataset in a ``DataLoader`` with ``num_workers=0`` (no worker
    processes) — calibration is single-pass on a single GPU and spawning
    workers adds overhead without benefit.  Accumulates raw (un-calibrated)
    logits for all T heads together with ground-truth labels.  No masking is
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
        Two-tuple of lists, each of length T (number of model heads):

        - *logits_list* — Raw logits per target, each a flat NumPy array of
          shape ``(N,)``, float32.
        - *labels_list* — Ground-truth labels per target, each a flat NumPy
          array of shape ``(N,)``, float32, with NaN where a target is not
          applicable for a sample.
    """
    num_targets = len(model.heads)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    all_logits: List[List[np.ndarray]] = [[] for _ in range(num_targets)]
    all_labels: List[List[np.ndarray]] = [[] for _ in range(num_targets)]

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            model_outputs = model(X)
            for i in range(num_targets):
                all_logits[i].append(model_outputs[i].detach().cpu().numpy().reshape(-1))
                all_labels[i].append(batch[i + 1].cpu().numpy().reshape(-1))

    logits_list = [np.concatenate(all_logits[i], axis=0) for i in range(num_targets)]
    labels_list = [np.concatenate(all_labels[i], axis=0) for i in range(num_targets)]
    return logits_list, labels_list


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
    Optimisation is performed in log-space (optimising log_T, recovering
    ``T = exp(log_T)``) to guarantee ``T > 0`` throughout, avoiding erratic
    gradients at the clamp boundary that occur with direct optimisation.  Only
    the subset of samples selected by *mask* is used.

    Args:
        logits: Raw model logits of shape ``(N,)``, float32.
        labels: Binary ground-truth labels of shape ``(N,)`` with values in
            ``{0, 1}``.
        mask: Boolean array of shape ``(N,)`` selecting which samples to
            include.  For targets without NaN labels this is all-True (full
            dev split); for targets with NaN labels this is
            ``~np.isnan(labels)``.

    Returns:
        Fitted scalar temperature ``T``, clamped to a minimum of
        ``1e-8`` before being returned to prevent downstream division
        by zero in ``sigmoid(logit / T)``.
    """
    logits_t = torch.tensor(logits[mask], dtype=torch.float32)
    labels_t = torch.tensor(labels[mask], dtype=torch.float32)

    log_T = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_T], lr=0.01, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        T_pos = torch.exp(log_T)
        scaled_logits = logits_t / T_pos
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled_logits, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(max(torch.exp(log_T).item(), 1e-8))


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

    Uses equal-mass (adaptive) probability binning via ``np.percentile`` on
    the masked subset of samples.  Bin edges are derived from the empirical
    distribution of predicted probabilities, so each bin contains
    approximately the same number of samples.  Intended for logging pre- and
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
        ECE as a float in ``[0, 1]``, or ``nan`` if ``mask`` selects no samples.
    """
    if not np.any(mask):
        return float("nan")
    masked_logits = logits[mask]
    masked_labels = labels[mask]
    probs = 1.0 / (1.0 + np.exp(-masked_logits / T))

    n_bins = 10
    bin_edges = np.percentile(probs, range(0, 101, 100 // n_bins))
    bin_edges[0] = 0.0
    bin_edges[-1] = 1.0
    bin_edges = np.unique(bin_edges)
    inds = np.searchsorted(bin_edges[:-1], probs, side="right") - 1
    inds = np.clip(inds, 0, len(bin_edges) - 2)

    ece = 0.0
    total = len(probs)
    for i in range(len(bin_edges) - 1):
        in_bin = inds == i
        if not np.any(in_bin):
            continue
        avg_conf = probs[in_bin].mean()
        avg_acc = masked_labels[in_bin].mean()
        ece += (np.sum(in_bin) / total) * abs(avg_conf - avg_acc)
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
    2. Load the dev split from ``DATASET_PATH`` via ``Mimic4DataLoader(config).load()``.
    3. Run a full forward pass with ``torch.no_grad()`` to collect raw logits
       for all T heads.
    4. For each target i in range(T): construct the valid-sample mask
       (``~np.isnan(labels_list[i])``), fit ``T_i`` via L-BFGS on NLL.
    5. Log pre- and post-calibration ECE for all heads for audit.
    6. Write ``{f"T_{i}": float(T_i) for i in range(T)}`` to
       ``CALIBRATION_PARAMS_PATH`` as JSON.

    Config keys used: ``CHECKPOINT_DIR``, ``DATASET_PATH``,
    ``CALIBRATION_PARAMS_PATH``.

    Args:
        config: Validated ``RewardModelConfig`` instance.
    """
    device = get_device(local_rank=0)
    checkpoint_path = Path(config.CHECKPOINT_DIR) / "best_model.pt"
    model, feature_index_map, config_snapshot = _load_model_from_checkpoint(checkpoint_path, device)
    num_targets = config_snapshot.get("NUM_TARGETS", 2)

    bundle = Mimic4DataLoader(config).load()
    logits_list, labels_list = _run_forward_pass(
        model,
        bundle.dev_dataset,
        device,
        config.BATCH_SIZE_PER_GPU,
    )

    temperatures: List[float] = []
    for i in range(num_targets):
        mask_i = ~np.isnan(labels_list[i])
        pre_ece = _compute_ece_from_logits(logits_list[i], labels_list[i], 1.0, mask_i)
        logger.info("Pre-calibration ECE — target %d: %.6f", i, pre_ece)
        T_i = _fit_temperature(logits_list[i], labels_list[i], mask_i)
        temperatures.append(T_i)
        post_ece = _compute_ece_from_logits(logits_list[i], labels_list[i], T_i, mask_i)
        logger.info("Post-calibration ECE — target %d: %.6f (T=%.6f)", i, post_ece, T_i)

    params = {f"T_{i}": float(T_i) for i, T_i in enumerate(temperatures)}
    calibration_path = Path(config.CALIBRATION_PARAMS_PATH)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(params))
    logger.info("Wrote calibration parameters to %s: %s", calibration_path, params)


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
