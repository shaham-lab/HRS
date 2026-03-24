import logging
from typing import Iterable

import torch
from torch import nn

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layer_widths: Iterable[int],
        dropout_rate: float,
        activation: str = "relu",
    ) -> None:
        """Initialize the reward model MLP with configurable widths and activation."""
        super().__init__()
        widths = list(layer_widths)
        if not widths:
            raise ValueError("layer_widths must contain at least one layer width")

        layers = []
        in_dim = input_dim
        for width in widths:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.BatchNorm1d(width))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError("activation must be 'relu' or 'leaky_relu'")
            layers.append(nn.Dropout(dropout_rate))
            in_dim = width

        self.backbone = nn.Sequential(*layers)
        self.head_y1 = nn.Linear(in_dim, 1)
        self.head_y2 = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor):
        """Compute logits for Y1 and Y2 given input features."""
        features = self.backbone(x)
        return self.head_y1(features), self.head_y2(features)
