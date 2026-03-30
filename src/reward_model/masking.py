"""Curriculum masking schedule for the CDSS-ML reward model training loop.

Implements MaskingSchedule, which maintains the masking curriculum state and
applies the correct masking mode to each mini-batch.  Three modes:

  random      — zero k feature slots selected uniformly at random per sample,
                where k is drawn per sample from a configured fraction range
  adversarial — zero the top-k highest-RMS-gradient slots per sample
                (Shaham et al. 2016, adapted for discrete feature-slot masking)
  none        — return X unchanged

The probability of each mode evolves via a sigmoid crossover schedule driven by
``sigmoid_crossover()`` defined in this module.

See Detailed Design §5 (masking.py) and §6.3 (adversarial masking under DDP).
"""

import logging
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch


def sigmoid_crossover(
    epoch: int,
    total_epochs: int,
    start_ratios: Dict[str, float],
    end_ratios: Dict[str, float],
    midpoint: float,
) -> Tuple[float, float, float]:
    """Compute masking probabilities for the given epoch using a sigmoid crossover."""
    clamped_epoch = max(0, min(epoch, total_epochs))
    scale = max(total_epochs * 0.1, 1.0)
    progress = 1.0 / (1.0 + math.exp(-(clamped_epoch - midpoint) / scale))
    probs = []
    for key in ("random", "adversarial", "none"):
        start = start_ratios[key]
        end = end_ratios[key]
        probs.append(start + (end - start) * progress)
    return (probs[0], probs[1], probs[2])


logger = logging.getLogger(__name__)


class MaskingSchedule:
    """Curriculum masking schedule: random, adversarial, and no-mask modes.

    Maintains state across the training loop.  Epoch advancement is a stateful
    operation — the same schedule instance is used for the entire run and is
    serialised into every checkpoint.

    Config keys consumed (via constructor args, not direct config access):
        MASKING_START_RATIOS, MASKING_END_RATIOS, MASKING_TRANSITION_SHAPE,
        MASKING_TRANSITION_MIDPOINT_EPOCH, NUM_ALWAYS_VISIBLE_FEATURES,
        MASKING_RANDOM_K_MIN_FRACTION, MASKING_RANDOM_K_MAX_FRACTION,
        MASKING_ADVERSARIAL_K_MIN_FRACTION, MASKING_ADVERSARIAL_K_MAX_FRACTION.
    """

    def __init__(
        self,
        feature_index_map: Dict[str, Tuple[int, int]],
        start_ratios: Dict[str, float],
        end_ratios: Dict[str, float],
        transition_shape: str,
        transition_midpoint_epoch: int,
        total_epochs: int,
        num_always_visible: int = 5,
        random_k_min_fraction: float = 0.5,
        random_k_max_fraction: float = 1.0,
        adversarial_k_min_fraction: float = 0.3,
        adversarial_k_max_fraction: float = 0.7,
    ) -> None:
        """Initialise the masking schedule.

        Args:
            feature_index_map: Mapping of feature column name to ``(start, end)``
                index range within the flat input tensor.  The first
                ``num_always_visible`` entries (in insertion order) are treated
                as always-visible; all remaining entries are maskable.
            start_ratios: Mode probabilities at epoch 0.
            end_ratios: Mode probabilities at the final epoch.
            transition_shape: Crossover curve shape (only ``'sigmoid'`` supported).
            transition_midpoint_epoch: Epoch at sigmoid crossover inflection point.
            total_epochs: Total training epochs.
            num_always_visible: Number of leading slots in ``feature_index_map``
                that are never masked.  Positional — must match the leading
                always-visible slots enforced by the upstream preprocessing
                pipeline column order.
            random_k_min_fraction: Minimum fraction of maskable slots zeroed per
                sample in random mode.
            random_k_max_fraction: Maximum fraction of maskable slots zeroed per
                sample in random mode (effective upper bound is M−1).
            adversarial_k_min_fraction: Minimum fraction of maskable slots zeroed
                per sample in adversarial mode.
            adversarial_k_max_fraction: Maximum fraction of maskable slots zeroed
                per sample in adversarial mode.
        """
        self._feature_index_map = feature_index_map
        self._start_ratios = start_ratios
        self._end_ratios = end_ratios
        self._transition_shape = transition_shape
        self._transition_midpoint_epoch = transition_midpoint_epoch
        self._total_epochs = total_epochs
        self._random_k_min_fraction = random_k_min_fraction
        self._random_k_max_fraction = random_k_max_fraction
        self._adversarial_k_min_fraction = adversarial_k_min_fraction
        self._adversarial_k_max_fraction = adversarial_k_max_fraction

        # Derive always-visible and maskable slots positionally from the map.
        all_slots: List[str] = list(feature_index_map.keys())
        self._always_visible_slots: List[str] = all_slots[:num_always_visible]
        self._maskable_slots: List[str] = all_slots[num_always_visible:]
        self._M: int = len(self._maskable_slots)

    # ------------------------------------------------------------------
    # Curriculum schedule
    # ------------------------------------------------------------------

    def get_mode_probabilities(self, epoch: int) -> Tuple[float, float, float]:
        """Return ``(p_random, p_adversarial, p_none)`` for the given epoch."""
        return sigmoid_crossover(
            epoch=epoch,
            total_epochs=self._total_epochs,
            start_ratios=self._start_ratios,
            end_ratios=self._end_ratios,
            midpoint=self._transition_midpoint_epoch,
        )

    def sample_mode(self, epoch: int) -> str:
        """Draw a masking mode string for this mini-batch."""
        probs = np.array(self.get_mode_probabilities(epoch), dtype=np.float64)
        probs /= probs.sum()
        return str(np.random.choice(["random", "adversarial", "none"], p=probs))

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

    def apply_no_mask(self, X: torch.Tensor) -> torch.Tensor:
        """Return a clone of X with no masking applied."""
        return X.clone()
