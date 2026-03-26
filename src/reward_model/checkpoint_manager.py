import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

from src.reward_model.schema_error import SchemaError


class CheckpointManager:
    """Checkpoint lifecycle manager for reward model training.

    Responsibilities:
    - Discover the latest ``epoch_<N>.pt`` checkpoint.
    - Save rolling epoch checkpoints and ``best_model.pt`` atomically.
    - Prune old epoch checkpoints beyond ``keep_n``.
    - Validate feature-index map consistency when resuming.
    """

    def __init__(self, checkpoint_dir: Path, keep_n: int) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.keep_n = keep_n
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def validate_feature_index_map(
        checkpoint_map: Dict[str, Tuple[int, int]], current_map: Dict[str, Tuple[int, int]]
    ) -> None:
        if checkpoint_map != current_map:
            raise SchemaError("Feature index map mismatch between checkpoint and current dataset")

    def find_latest(self) -> Optional[Path]:
        checkpoints = []
        for path in self.checkpoint_dir.glob("epoch_*.pt"):
            try:
                epoch_str = path.stem.split("_", maxsplit=1)[1]
                epoch_num = int(epoch_str)
                checkpoints.append((epoch_num, path))
            except (IndexError, ValueError):
                continue
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda pair: pair[0], reverse=True)
        return checkpoints[0][1]

    def load(self, path: Path, current_feature_index_map: Dict[str, Tuple[int, int]]) -> dict:
        state = torch.load(path, map_location="cpu")
        ckpt_feature_map = state["feature_index_map"]
        self.validate_feature_index_map(ckpt_feature_map, current_feature_index_map)
        return state

    def save_epoch_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int,
        best_dev_loss: float,
        feature_index_map: Dict[str, Tuple[int, int]],
        config,
    ) -> Path:
        state = {
            "model_state_dict": self._unwrap_ddp(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "masking_schedule_state": {
                "start_ratios": config.MASKING_START_RATIOS,
                "end_ratios": config.MASKING_END_RATIOS,
                "transition_shape": config.MASKING_TRANSITION_SHAPE,
                "transition_midpoint_epoch": config.MASKING_TRANSITION_MIDPOINT_EPOCH,
                "total_epochs": config.MAX_EPOCHS,
                "random_k_min_fraction": config.MASKING_RANDOM_K_MIN_FRACTION,
                "random_k_max_fraction": config.MASKING_RANDOM_K_MAX_FRACTION,
                "adversarial_k_min_fraction": config.MASKING_ADVERSARIAL_K_MIN_FRACTION,
                "adversarial_k_max_fraction": config.MASKING_ADVERSARIAL_K_MAX_FRACTION,
                "num_always_visible": config.NUM_ALWAYS_VISIBLE_FEATURES,
            },
            "best_dev_loss": best_dev_loss,
            "feature_index_map": feature_index_map,
            "config": config.model_dump(),
        }

        target_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
        tmp_path = target_path.with_suffix(".pt.tmp")
        torch.save(state, tmp_path)
        os.replace(tmp_path, target_path)
        return target_path

    def save_best_model(
        self,
        model: torch.nn.Module,
        epoch: int,
        best_dev_loss: float,
        feature_index_map: Dict[str, Tuple[int, int]],
        config,
    ) -> None:
        state = {
            "model_state_dict": self._unwrap_ddp(model).state_dict(),
            "epoch": epoch,
            "best_dev_loss": best_dev_loss,
            "feature_index_map": feature_index_map,
            "config": config.model_dump(),
        }
        target_path = self.checkpoint_dir / "best_model.pt"
        tmp_path = target_path.with_suffix(".pt.tmp")
        torch.save(state, tmp_path)
        os.replace(tmp_path, target_path)

    def prune_old_checkpoints(self) -> None:
        checkpoints = []
        for path in self.checkpoint_dir.glob("epoch_*.pt"):
            try:
                epoch_num = int(path.stem.split("_", maxsplit=1)[1])
                checkpoints.append((epoch_num, path))
            except (IndexError, ValueError):
                continue

        checkpoints.sort(key=lambda pair: pair[0], reverse=True)
        for _, path in checkpoints[self.keep_n :]:
            try:
                path.unlink()
            except FileNotFoundError:
                continue
