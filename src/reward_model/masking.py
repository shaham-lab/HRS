"""Curriculum masking schedule for the CDSS-ML reward model training loop.

Implements MaskingSchedule, which maintains the masking curriculum state and
applies the correct masking mode to each mini-batch.  Three modes:

  random      — zero k feature slots selected uniformly at random per sample,
                where k is drawn per sample from a configured fraction range
  adversarial — zero the top-k highest-RMS-gradient slots per sample
                (Shaham et al. 2016, adapted for discrete feature-slot masking)
  none        — return X unchanged

The probability of each mode evolves via a sigmoid crossover schedule computed
in ``MaskingSchedule.get_mode_probabilities``.

See Detailed Design §5 (masking.py) and §6.3 (adversarial masking).
"""

import logging
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch

from reward_model_config import RewardModelConfig


logger = logging.getLogger(__name__)


class MaskingSchedule:
    """Curriculum masking schedule: random, adversarial, and no-mask modes.

    Maintains state across the training loop. Three modes:

      random      — zero k feature slots selected uniformly at random per sample
      adversarial — zero the top-k highest-gradient slots per sample
      none        — return X unchanged

    The probability of each mode evolves via a sigmoid crossover schedule
    computed in ``get_mode_probabilities``. The first
    ``NUM_ALWAYS_VISIBLE_FEATURES`` slots (in insertion order) are never
    masked; all remaining slots are maskable.
    """

    def __init__(
        self,
        config: RewardModelConfig,
        feature_index_map: Dict[str, Tuple[int, int]],
    ) -> None:
        self._mode_rng = np.random.default_rng(42)
        self._start_ratios = config.MASKING_START_RATIOS
        self._end_ratios = config.MASKING_END_RATIOS
        self._transition_midpoint_epoch = config.MASKING_TRANSITION_MIDPOINT_EPOCH
        self._total_epochs = config.MAX_EPOCHS
        self._random_k_min_fraction = config.MASKING_RANDOM_K_MIN_FRACTION
        self._random_k_max_fraction = config.MASKING_RANDOM_K_MAX_FRACTION
        self._adversarial_k_min_fraction = config.MASKING_ADVERSARIAL_K_MIN_FRACTION
        self._adversarial_k_max_fraction = config.MASKING_ADVERSARIAL_K_MAX_FRACTION

        all_slots: List[str] = list(feature_index_map.keys())
        self._always_visible_slots: List[str] = all_slots[:config.NUM_ALWAYS_VISIBLE_FEATURES]
        self._maskable_slots: List[str] = all_slots[config.NUM_ALWAYS_VISIBLE_FEATURES:]
        self._M: int = len(self._maskable_slots)
        self._feature_index_map: Dict[str, Tuple[int, int]] = feature_index_map

    # ------------------------------------------------------------------
    # Curriculum schedule
    # ------------------------------------------------------------------

    def get_mode_probabilities(self, epoch: int) -> Tuple[float, float, float]:
        """Return ``(p_random, p_adversarial, p_none)`` for the given epoch."""
        clamped = max(0, min(epoch, self._total_epochs))
        scale = max(self._total_epochs * 0.1, 1.0)
        progress = 1.0 / (1.0 + math.exp(-(clamped - self._transition_midpoint_epoch) / scale))
        probs = [
            s + (e - s) * progress
            for s, e in (
                (self._start_ratios[k], self._end_ratios[k])
                for k in ("random", "adversarial", "none")
            )
        ]
        return probs[0], probs[1], probs[2]

    def sample_mode(self, epoch: int) -> str:
        """Draw a masking mode string for this mini-batch."""
        probs = np.array(self.get_mode_probabilities(epoch), dtype=np.float64)
        probs /= probs.sum()
        return str(self._mode_rng.choice(["random", "adversarial", "none"], p=probs))
        #return str(np.random.choice(["random", "adversarial", "none"], p=probs))

    # ------------------------------------------------------------------
    # k sampling
    # ------------------------------------------------------------------

    def _sample_k(self, min_fraction: float, max_fraction: float) -> int:
        """Draw an integer k from [lower, upper] for the configured fraction range.

        Algorithm (DD §5 masking.py):
            lower = floor(min_fraction * M)
            upper = ceil(max_fraction * M)
            lower = max(1, lower)
            upper = min(M - 1, upper)
            if lower > upper: lower = upper
            return randint(lower, upper) inclusive
        """
        lower = math.floor(min_fraction * self._M)
        upper = math.ceil(max_fraction * self._M)
        lower = max(1, lower)
        upper = min(self._M - 1, upper)
        if lower > upper:
            lower = upper
        return random.randint(lower, upper)

    # ------------------------------------------------------------------
    # Masking operators
    # ------------------------------------------------------------------

    def apply_random_mask(self, X: torch.Tensor) -> torch.Tensor:
        """Zero k randomly-selected maskable slots per sample without replacement.

        k is drawn independently per sample from the configured random fraction
        range.  The original tensor is not modified — a clone is returned.
        """
        if self._M == 0:
            return X.clone()
        masked = X.clone()
        for i in range(X.shape[0]):
            k = self._sample_k(self._random_k_min_fraction, self._random_k_max_fraction)
            selected = np.random.choice(len(self._maskable_slots), size=k, replace=False)
            for slot_idx in selected:
                slot = self._maskable_slots[slot_idx]
                start, end = self._feature_index_map[slot]
                masked[i, start:end] = 0.0
        return masked

    def apply_adversarial_mask(
        self, X: torch.Tensor, grad_X: torch.Tensor
    ) -> torch.Tensor:
        """Zero the top-k highest-RMS-gradient maskable slots per sample.

        k is drawn independently per sample from the configured adversarial
        fraction range.  Importance score per slot: RMS gradient magnitude,
        computed as L2 norm divided by sqrt(slot_dim), to normalise across
        slots of different widths.  The original tensor is not modified — a
        clone is returned.
        """
        if self._M == 0:
            return X.clone()
        masked = X.clone()
        #loop per sample
        for i in range(X.shape[0]):
            #per sample decide how many features to mask
            k = self._sample_k(

                self._adversarial_k_min_fraction, self._adversarial_k_max_fraction
            )
            # Compute RMS importance for every maskable slot.
            #list to hold sorted gradients by importance
            importances: List[Tuple[float, str]] = []
            #loop per maskable feature
            for slot in self._maskable_slots:
                start, end = self._feature_index_map[slot]
                slot_dim = end - start
                #calculate L2 norm of gradient vector / sqrt(feature dimension)
                norm = float(torch.linalg.norm(grad_X[i, start:end])) / math.sqrt(slot_dim)
                importances.append((norm, slot))
            # Sort descending by importance; zero the top k.
            importances.sort(key=lambda pair: pair[0], reverse=True)
            for _, slot in importances[:k]:
                #find feature start and end indexes in data
                start, end = self._feature_index_map[slot]
                #mask all feature related indexes
                masked[i, start:end] = 0.0
        return masked

