"""Frozen inference for the CDSS-ML reward model.

Provides ``RewardModelInference``, which is loaded once per RL session and
called once per episode step.  Applies pre-fitted temperature-scaling
parameters to return calibrated probability estimates.

Usage::

    from src.reward_model.inference import RewardModelInference

    inf = RewardModelInference(
        checkpoint_path="data/reward_model/frozen_model.pt",
        calibration_params_path="data/reward_model/calibration_params.json",
    )
    probs = inf.predict(X)

See Detailed Design §5 (inference.py) and Architecture §8.7
(Post-Training Calibration).
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

import torch

from reward_model import RewardModel
from reward_model_config import RewardModelConfig
from reward_model_utils import get_device

logger = logging.getLogger(__name__)


class RewardModelInference:
    """Frozen reward model for RL agent consumption.

    Loaded once per RL session.  ``predict()`` is called once per episode
    step.  No gradient computation ever occurs in this class.

    The constructor loads the frozen model weights, feature index map,
    and calibration parameters ``T_0 … T_{T-1}`` from disk.
    It moves the model to the target device, calls ``model.eval()``, and
    calls ``requires_grad_(False)`` on all model parameters in
    ``__init__``, and uses ``torch.no_grad()`` in ``predict()`` to
    prevent gradient tracking during forward passes.
    """

    def __init__(
        self,
        config: RewardModelConfig,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
    ) -> None:
        """Load frozen model and calibration parameters.

        Loads the exported artefact at ``checkpoint_path`` (written by
        ``export_model.py``) which embeds calibration temperatures, the
        feature index map, and input dimensionality.

        Args:
            config: Validated ``RewardModelConfig`` instance.
            checkpoint_path: Path to the self-contained ``frozen_model.pt``
                artefact written by ``export_model.py``.
            device: Target device for the loaded model.  Defaults to
                ``cuda:0`` if a GPU is available, otherwise ``cpu``.
        """
        if device is None:
            device = get_device(local_rank=0)

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self._feature_index_map: Dict[str, Tuple[int, int]] = ckpt["feature_index_map"]
        num_targets = config.NUM_TARGETS
        self._temperatures: List[float] = [
            max(float(ckpt[f"T_{i}"]), 1e-8) for i in range(num_targets)
        ]

        model = RewardModel(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        self._model = model
        self._device = device

        logger.info(
            "Loaded calibration temperatures: %s",
            {f"T_{i}": t for i, t in enumerate(self._temperatures)},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Return calibrated probabilities for all T targets.

        Runs a forward pass under ``torch.no_grad()``.  Temperature scaling
        is applied to each head: ``p = sigmoid(logit / T)``.
        ``T`` values are clamped to a minimum of ``1e-8`` before division
        to prevent numerical overflow.

        Args:
            X: Input feature tensor of shape ``(N, input_dim)``, dtype
               float32, on the same device as the model.  Masking (zeroing
               of unavailable feature slots) is the caller's responsibility.

        Returns:
            Tuple of T tensors, each of shape ``(N, 1)``, one per target head.
            ``p[i] = sigmoid(logit_i / T_i)``.
        """
        with torch.no_grad():
            model_output = self._model(X)
            return tuple(
                torch.sigmoid(logits / T)
                for logits, T in zip(model_output, self._temperatures)
            )

    def get_feature_index_map(self) -> Dict[str, Tuple[int, int]]:
        """Return the feature index map snapshot loaded from the checkpoint.

        The RL agent uses this map to construct correctly masked input tensors
        for each episode step without needing access to
        ``full_cdss_dataset.parquet``.

        Returns:
            Dict mapping feature column name to ``(start, end)`` index range
            within the flat input vector of dimensionality ``input_dim``.
            Matches the canonical column order defined in
            ``PREPROCESSING_DATA_MODEL.md`` Section 3.12.
        """
        return self._feature_index_map
