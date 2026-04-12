"""Entry point for training the CDSS-ML reward model."""

import argparse
import logging
import sys

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

from reward_model_config import load_and_validate_config
from reward_model_manager import RewardModelManager

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CDSS-ML reward model")
    parser.add_argument("--config", required=True, help="Path to reward_model.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return parser.parse_args()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        stream=sys.stdout,
    )


def main() -> int:
    args = _parse_args()
    process_group_kwargs = InitProcessGroupKwargs(backend="gloo")
    accelerator = Accelerator(
        kwargs_handlers=[process_group_kwargs]
    )
    if accelerator.is_local_main_process:
        _setup_logging()

    config = load_and_validate_config(args.config)

    manager = RewardModelManager(config, accelerator)
    ckpt_state = None
    start_epoch = 0
    best_dev_loss = float("inf")

    if args.resume:
        ckpt_state, start_epoch, best_dev_loss = manager.resume_from_checkpoint()
    manager.setup_training_state(ckpt_state, start_epoch)
    manager.train_epochs(start_epoch, best_dev_loss)
    return 0


if __name__ == "__main__":
    sys.exit(main())
