"""Shared helpers and backward-compatibility re-exports for the reward model.
"""

import torch


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")

