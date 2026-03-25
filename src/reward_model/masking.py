"""Curriculum masking schedule for the CDSS-ML reward model training loop.

Implements MaskingSchedule, which maintains the masking curriculum state and
applies the correct masking mode to each mini-batch.  Three modes:

  random      — zero k feature slots selected uniformly at random per sample
  adversarial — zero the highest-L2-norm gradient slot per sample
                (Shaham et al. 2016, adapted for discrete feature-slot masking)
  none        — return X unchanged

The probability of each mode evolves via a sigmoid crossover schedule driven by
sigmoid_crossover() in reward_model_utils.py.

See Detailed Design §5 (masking.py) and §6.3 (adversarial masking under DDP).
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.reward_model.reward_model_utils import sigmoid_crossover

logger = logging.getLogger(__name__)


class MaskingSchedule:
    """Curriculum masking schedule: random, adversarial, and no-mask modes.

    Maintains state across the training loop.  Epoch advancement is a stateful
    operation — the same schedule instance is used for the entire run and is
    serialised into every checkpoint.

    Config keys consumed (via constructor args, not direct config access):
        MASKING_START_RATIOS, MASKING_END_RATIOS, MASKING_TRANSITION_SHAPE,
        MASKING_TRANSITION_MIDPOINT_EPOCH, MASKING_K.
    """

    def __init__(
        self,
        feature_index_map: Dict[str, Tuple[int, int]],
        start_ratios: Dict[str, float],
        end_ratios: Dict[str, float],
        transition_shape: str,
        transition_midpoint_epoch: int,
        total_epochs: int,
        k: int = 1,
    ) -> None:
        """Initialise the masking schedule.

        Args:
            feature_index_map: Mapping of feature column name to ``(start, end)``
                byte-range within the flat input tensor, as derived by
                ``load_dataset.run()`` from the canonical Parquet column order.
                Defines both the slot count and the index boundaries used for
                zeroing.
            start_ratios: Dict with keys ``'random'``, ``'adversarial'``,
                ``'none'`` giving the masking mode probabilities at epoch 0.
                Values must sum to 1.0.
            end_ratios: Dict with keys ``'random'``, ``'adversarial'``,
                ``'none'`` giving the masking mode probabilities at the final
                epoch.  Values must sum to 1.0.
            transition_shape: Crossover curve shape.  Only ``'sigmoid'`` is
                supported; validated by RewardModelConfig at startup.
            transition_midpoint_epoch: Epoch at the sigmoid crossover inflection
                point (``MASKING_TRANSITION_MIDPOINT_EPOCH`` in config).
            total_epochs: Total training epochs (``MAX_EPOCHS`` in config).
            k: Number of feature slots zeroed per sample in random mode.
                Default 1 (``MASKING_K`` in config).
        """
        ...

    # ------------------------------------------------------------------
    # Curriculum schedule
    # ------------------------------------------------------------------

    def get_mode_probabilities(self, epoch: int) -> Tuple[float, float, float]:
        """Return ``(p_random, p_adversarial, p_none)`` for the given epoch.

        Delegates to ``sigmoid_crossover()`` in ``reward_model_utils.py``.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            Three-tuple of floats summing to 1.0.
        """
        ...

    def sample_mode(self, epoch: int) -> str:
        """Draw a masking mode string for this mini-batch.

        Samples from ``['random', 'adversarial', 'none']`` according to the
        probabilities returned by ``get_mode_probabilities(epoch)``.

        Args:
            epoch: Current training epoch (0-indexed).

        Returns:
            One of ``'random'``, ``'adversarial'``, or ``'none'``.
        """
        ...

    # ------------------------------------------------------------------
    # Masking operators
    # ------------------------------------------------------------------

    def apply_random_mask(self, X: torch.Tensor) -> torch.Tensor:
        """Zero *k* randomly-selected feature slots per sample without replacement.

        Slot indices are drawn uniformly at random from all entries in
        ``feature_index_map``.  The original tensor is not modified in place —
        a clone is returned.

        Args:
            X: Input batch tensor of shape ``(batch_size, input_dim)``,
               dtype float32.

        Returns:
            Cloned tensor of same shape with *k* slots zeroed per sample.
        """
        ...

    def apply_adversarial_mask(
        self, X: torch.Tensor, grad_X: torch.Tensor
    ) -> torch.Tensor:
        """Zero the highest-L2-norm gradient slot per sample.

        Importance score per slot: L2 norm of ``grad_X`` over the slot's index
        range.  Using the norm (rather than the raw per-dimension maximum) makes
        slots of different sizes comparable — ``demographic_vec`` spans 8
        dimensions while all embedding slots span 768.

        The gradient ``∂L/∂X`` is computed by ``train.py`` via a first
        forward/backward pass inside ``model.no_sync()`` before this method is
        called (Detailed Design §7).  The original tensor is not modified in
        place — a clone is returned.

        Args:
            X: Input batch tensor of shape ``(batch_size, input_dim)``,
               dtype float32.
            grad_X: Gradient ``∂L/∂X`` of shape ``(batch_size, input_dim)``,
                dtype float32, accumulated from the first forward/backward pass
                inside ``model.no_sync()``.

        Returns:
            Cloned tensor of same shape with the highest-norm slot zeroed per
            sample.
        """
        ...

    def apply_no_mask(self, X: torch.Tensor) -> torch.Tensor:
        """Return *X* unchanged (no masking applied).

        Args:
            X: Input batch tensor of shape ``(batch_size, input_dim)``,
               dtype float32.

        Returns:
            *X* unchanged (not cloned).
        """
        ...
