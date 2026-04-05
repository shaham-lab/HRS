"""torchrun entry point for training the CDSS-ML reward model."""

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist

from reward_model_config import load_and_validate_config
from reward_model_utils import get_device
from reward_model_manager import RewardModelManager

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CDSS-ML reward model")
    parser.add_argument("--config", required=True, help="Path to reward_model.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return parser.parse_args()


def _setup_logging(initial_rank: int) -> None:
    if initial_rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            stream=sys.stdout,
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def _init_ddp() -> tuple[int, int, int, bool]:
    """Initialise the DDP process group from torchrun environment variables."""
    if torch.cuda.device_count() < 2:
        logger.warning("Insufficient CUDA devices for DDP — running in single-process mode")
        return 0, 0, 1, False

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if world_size <= 1:
        logger.warning("WORLD_SIZE <= 1 — running in single-process mode")
        return 0, 0, 1, False

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return rank, local_rank, world_size, True


def _init_runtime(config) -> tuple[int, int, int, bool, torch.device]:
    rank, local_rank, world_size, is_ddp = _init_ddp()
    device = get_device(local_rank)
    return rank, local_rank, world_size, is_ddp, device


def main() -> int:
    args = _parse_args()

    initial_rank = int(os.environ.get("RANK", "0"))
    _setup_logging(initial_rank)

    config = load_and_validate_config(args.config)
    rank, local_rank, world_size, is_ddp, device = _init_runtime(config)

    manager = RewardModelManager(config, rank, local_rank, world_size, is_ddp, device)
    ckpt_state = None
    start_epoch = 0
    best_dev_loss = float("inf")

    if args.resume:
        ckpt_state, start_epoch, best_dev_loss = manager.resume_from_checkpoint()
    manager.setup_training_state(ckpt_state, start_epoch)
    manager.train_epochs(start_epoch, best_dev_loss)

    if is_ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
