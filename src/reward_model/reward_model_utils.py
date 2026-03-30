"""Shared helpers and backward-compatibility re-exports for the reward model.
"""

import torch


def get_device(local_rank: int) -> torch.device:
    """Return CUDA device at local_rank if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Return underlying module if wrapped in DistributedDataParallel."""
    return model.module if hasattr(model, "module") else model
