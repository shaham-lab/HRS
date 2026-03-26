import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import torch

# Re-exports for backward compatibility
from src.reward_model.dataset_bundle import DatasetBundle  # noqa: F401
from src.reward_model.parquet_dataset import ParquetDataset  # noqa: F401
from src.reward_model.reward_model_config import RewardModelConfig, load_and_validate_config  # noqa: F401
from src.reward_model.row_group_block_sampler import RowGroupBlockSampler  # noqa: F401

logger = logging.getLogger(__name__)


ALWAYS_VISIBLE_SLOTS: frozenset = frozenset(
    {
        "demographic_vec",
        "diag_history_embedding",
        "discharge_history_embedding",
        "triage_embedding",
        "chief_complaint_embedding",
    }
)


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


def get_device(local_rank: int) -> torch.device:
    """Return CUDA device at local_rank if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap a DDP-wrapped model to its underlying module."""
    return model.module if hasattr(model, "module") else model


def broadcast_tensor(tensor: torch.Tensor, src_rank: int) -> torch.Tensor:
    """Broadcast a tensor from src_rank to all ranks if distributed is initialised."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(tensor, src=src_rank)
    return tensor


def getenv_int(name: str, default: int) -> int:
    """Return env var as int with default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

