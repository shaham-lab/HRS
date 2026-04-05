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
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from reward_model import RewardModel
from reward_model_config import RewardModelConfig, load_and_validate_config
from mimic4_data_loader import Mimic4DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_model_from_checkpoint(
    config: RewardModelConfig,
    checkpoint_path: Path,
    device: torch.device,
) -> RewardModel:
    """Load a frozen ``RewardModel`` from a checkpoint file.

    Instantiates ``RewardModel`` from ``config``, loads the state dict, moves
    the model to ``device``, sets eval mode, and freezes all parameters.

    Args:
        config: Validated ``RewardModelConfig`` instance; authoritative for
            architecture and ``INPUT_DIM``.
        checkpoint_path: Absolute path to ``best_model.pt``.
        device: Target device for the loaded model.

    Returns:
        Loaded ``RewardModel`` in eval mode on ``device`` with frozen gradients.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = RewardModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


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


class TemperatureCalibrator:
    """Manages dev-split forward pass and per-head temperature scaling."""

    def __init__(self, config: RewardModelConfig, device: torch.device):
        self.config = config
        self.device = device

        checkpoint_path = Path(self.config.CHECKPOINT_DIR) / "best_model.pt"
        self.model = _load_model_from_checkpoint(self.config, checkpoint_path, self.device)
        self.num_targets = self.config.NUM_TARGETS

        bundle = Mimic4DataLoader(self.config).load()
        self.feature_index_map = bundle.feature_index_map
        self.dev_dataset = bundle.dev_dataset

    def _collect_logits(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run dev forward pass to collect logits and labels per head."""
        dataloader = DataLoader(
            self.dev_dataset, batch_size=self.config.BATCH_SIZE_PER_GPU, num_workers=0, shuffle=False
        )
        all_logits: List[List[torch.Tensor]] = [[] for _ in range(self.num_targets)]
        all_labels: List[List[torch.Tensor]] = [[] for _ in range(self.num_targets)]

        with torch.no_grad():
            for batch in dataloader:
                X = batch[0].to(self.device)
                model_outputs = self.model(X)
                for i in range(self.num_targets):
                    all_logits[i].append(model_outputs[i].detach().cpu().reshape(-1))
                    all_labels[i].append(batch[i + 1].detach().cpu().reshape(-1))

        logits_list = [torch.cat(all_logits[i], dim=0) for i in range(self.num_targets)]
        labels_list = [torch.cat(all_labels[i], dim=0) for i in range(self.num_targets)]
        return logits_list, labels_list

    def calibrate_and_save(self) -> None:
        """Fit per-head temperatures, log ECE, and write calibration JSON."""
        logits_list, labels_list = self._collect_logits()

        temperatures: List[float] = []
        for i in range(self.num_targets):
            logits_np = logits_list[i].cpu().numpy()
            labels_np = labels_list[i].cpu().numpy()
            mask_i = ~np.isnan(labels_np)

            pre_ece = _compute_ece_from_logits(logits_np, labels_np, 1.0, mask_i)
            logger.info("Pre-calibration ECE — target %d: %.6f", i, pre_ece)

            T_i = _fit_temperature(logits_np, labels_np, mask_i)
            temperatures.append(T_i)

            post_ece = _compute_ece_from_logits(logits_np, labels_np, T_i, mask_i)
            logger.info("Post-calibration ECE — target %d: %.6f (T=%.6f)", i, post_ece, T_i)

        params = {f"T_{i}": float(T_i) for i, T_i in enumerate(temperatures)}
        calibration_path = Path(self.config.CALIBRATION_PARAMS_PATH)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calibrator = TemperatureCalibrator(config, device)
    calibrator.calibrate_and_save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
