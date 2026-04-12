import logging
from typing import Tuple

import torch
from torch import nn

from reward_model_config import RewardModelConfig

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    def __init__(self, config: RewardModelConfig) -> None:
        super().__init__()
        activations = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}

        layers = []
        in_dim = config.INPUT_DIM
        self.pad_size = (16 - in_dim % 16) % 16
        in_dim += self.pad_size
        for width, rate in zip(config.LAYER_WIDTHS, config.DROPOUT_RATES):
            layers += [nn.Linear(in_dim, width), nn.BatchNorm1d(width), activations[config.ACTIVATION](), nn.Dropout(rate)]
            in_dim = width

        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(config.NUM_TARGETS)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute logits for each target head given input features."""
        features = self.backbone(x)
        return tuple(head(features) for head in self.heads)



