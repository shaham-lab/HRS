"""torchrun entry point for training the CDSS-ML reward model."""

import os
import sys

import torch.distributed as dist

from reward_model_config import load_and_validate_config
from train import (
    RewardModelManager,
    _init_runtime,
    _parse_args,
    _resume_from_checkpoint,
    _setup_logging,
)


def main() -> int:
    args = _parse_args()

    initial_rank = int(os.environ.get("RANK", "0"))
    _setup_logging(initial_rank)

    config = load_and_validate_config(args.config)
    rank, local_rank, world_size, is_ddp, device = _init_runtime(config)

    manager = RewardModelManager(config, rank, local_rank, world_size, is_ddp, device)
    ckpt_state, start_epoch, best_dev_loss = _resume_from_checkpoint(
        args, manager.checkpoint_manager, manager.feature_index_map, config, rank, is_ddp
    )
    manager.setup_training_state(ckpt_state, start_epoch)
    manager.train_epochs(start_epoch, best_dev_loss)

    if is_ddp:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
