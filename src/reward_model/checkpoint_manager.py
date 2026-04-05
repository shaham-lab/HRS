import os
from pathlib import Path
from typing import Optional

import torch

from reward_model_utils import unwrap_ddp


class CheckpointManager:
    """Checkpoint lifecycle manager for reward model training.

    Maintains two files:
    - best_model.pt       : model weights only, for inference.
    - best_model_train.pt : model + optimizer + epoch + best_dev_loss, for resuming.
    Both are overwritten in place whenever dev loss improves.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def find_latest(self) -> Optional[Path]:
        path = self.checkpoint_dir / "best_model_train.pt"
        return path if path.exists() else None

    def load(self, path: Path) -> dict:
        return torch.load(path, map_location="cpu")

    def save_train_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_dev_loss: float,
    ) -> None:
        state = {
            "model_state_dict": unwrap_ddp(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_dev_loss": best_dev_loss,
        }
        target_path = self.checkpoint_dir / "best_model_train.pt"
        tmp_path = target_path.with_suffix(".pt.tmp")
        torch.save(state, tmp_path)
        os.replace(tmp_path, target_path)

    def save_best_model(
        self,
        model: torch.nn.Module,
        epoch: int,
        best_dev_loss: float,
    ) -> None:
        state = {
            "model_state_dict": unwrap_ddp(model).state_dict(),
            "epoch": epoch,
            "best_dev_loss": best_dev_loss,
        }
        target_path = self.checkpoint_dir / "best_model.pt"
        tmp_path = target_path.with_suffix(".pt.tmp")
        torch.save(state, tmp_path)
        os.replace(tmp_path, target_path)
