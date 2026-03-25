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
from src.reward_model.reward_model_utils import get_device, unwrap_ddp

logger = logging.getLogger(__name__)


class RewardModelInference:
    """Frozen reward model for RL agent consumption.

    Loaded once per RL session.  ``predict()`` is called once per episode
    step.  No gradient computation ever occurs in this class.

    The constructor loads the frozen model weights, feature index map
    snapshot, and calibration parameters ``T_y1`` and ``T_y2`` from disk.
    It moves the model to the target device, calls ``model.eval()``, and
    calls ``requires_grad_(False)`` on all model parameters in
    ``__init__``, and uses ``torch.no_grad()`` in ``predict()`` to
    prevent gradient tracking during forward passes.

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
        if device is None:
            device = get_device(local_rank=0)

        ckpt = torch.load(checkpoint_path, map_location=device)
        self._feature_index_map: Dict[str, Tuple[int, int]] = ckpt["feature_index_map"]
        input_dim = ckpt.get("input_dim") or sum(end - start for start, end in self._feature_index_map.values())

        if "T_y1" in ckpt and "T_y2" in ckpt:
            self._T_y1 = max(float(ckpt["T_y1"]), 1e-8)
            self._T_y2 = max(float(ckpt["T_y2"]), 1e-8)
            config_snapshot = ckpt["config_snapshot"]
        else:
            with open(calibration_params_path, "r") as f:
                calib = json.load(f)
            self._T_y1 = max(float(calib["T_y1"]), 1e-8)
            self._T_y2 = max(float(calib["T_y2"]), 1e-8)
            config_snapshot = ckpt["config"]

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

        self._model = model
        self._device = device

        logger.info("Loaded calibration temperatures: T_y1=%.6f, T_y2=%.6f", self._T_y1, self._T_y2)

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
        ``T`` values are clamped to a minimum of ``1e-8`` before division
        to prevent numerical overflow.

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
        with torch.no_grad():
            logits_y1, logits_y2 = self._model(X)
            p_mortality = torch.sigmoid(logits_y1 / self._T_y1)
            p_readmission = torch.sigmoid(logits_y2 / self._T_y2)
        return p_mortality, p_readmission

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
        return self._feature_index_map
