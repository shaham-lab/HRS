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
    p_mortality, p_readmission = inf.predict(X)

See Detailed Design §5 (inference.py) and Architecture §8.7
(Post-Training Calibration).
"""

import json
import logging
from typing import Dict, Optional, Tuple

import torch

from src.reward_model.model import RewardModel

logger = logging.getLogger(__name__)


class RewardModelInference:
    """Frozen reward model for RL agent consumption.

    Loaded once per RL session.  ``predict()`` is called once per episode
    step.  No gradient computation ever occurs in this class.

    The constructor loads the frozen model weights, feature index map
    snapshot, and calibration parameters ``T_y1`` and ``T_y2`` from disk.
    It moves the model to the target device, calls ``model.eval()``, and
    freezes all parameters via ``torch.no_grad()`` at inference time.

    Config keys used: None — all parameters come from the checkpoint and
    calibration files passed to the constructor.
    """

    def __init__(
        self,
        checkpoint_path: str,
        calibration_params_path: str,
        device: Optional[torch.device] = None,
    ) -> None:
        """Load frozen model and calibration parameters.

        Supports both the epoch/best checkpoint format written by ``train.py``
        and the self-contained export format written by ``export_model.py``.
        All parameters are loaded from the provided files — no access to
        ``config/reward_model.yaml`` is required.

        If the exported artefact at ``checkpoint_path`` already embeds
        calibration parameters, ``calibration_params_path`` is ignored and
        the embedded values are used instead.

        Args:
            checkpoint_path: Path to ``best_model.pt`` written by
                ``train.py``, or to the self-contained ``frozen_model.pt``
                artefact written by ``export_model.py``.
            calibration_params_path: Path to ``calibration_params.json``
                containing ``{'T_y1': float, 'T_y2': float}``.  Used only
                when ``checkpoint_path`` does not embed calibration
                parameters.
            device: Target device for the loaded model.  Defaults to
                ``cuda:0`` if a GPU is available, otherwise ``cpu``.
        """
        ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return calibrated mortality and readmission probabilities.

        Runs a forward pass under ``torch.no_grad()``.  Temperature scaling
        is applied to both heads: ``p = sigmoid(logit / T)``.

        Args:
            X: Input feature tensor of shape ``(N, input_dim)``, dtype
               float32, on the same device as the model.  Masking (zeroing
               of unavailable feature slots) is the caller's responsibility.

        Returns:
            Two-tuple of tensors, each of shape ``(N, 1)``:

            - *p_mortality* — Calibrated ``P(in-hospital mortality)`` from
              ``sigmoid(logit_y1 / T_y1)``.
            - *p_readmission* — Calibrated
              ``P(30-day readmission | survived)`` from
              ``sigmoid(logit_y2 / T_y2)``.
        """
        ...

    def get_feature_index_map(self) -> Dict[str, Tuple[int, int]]:
        """Return the feature index map snapshot loaded from the checkpoint.

        The RL agent uses this map to construct correctly masked input tensors
        for each episode step without needing access to
        ``final_cdss_dataset.parquet``.

        Returns:
            Dict mapping feature column name to ``(start, end)`` index range
            within the flat input vector of dimensionality ``input_dim``.
            Matches the canonical column order defined in
            ``PREPROCESSING_DATA_MODEL.md`` Section 3.12.
        """
        ...
