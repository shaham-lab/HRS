import logging
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


def get_device(local_rank: int) -> torch.device:
    """Return CUDA device at local_rank if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")
