import logging
from typing import Iterable, List, Tuple, Union

import torch
from torch import nn

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        layer_widths: Iterable[int],
        dropout_rates: Union[float, List[float]],
        activation: str = "relu",
        num_targets: int = 2,
    ) -> None:
        """Initialize the reward model MLP with configurable widths and activation."""
        super().__init__()
        widths = list(layer_widths)
        if not widths:
            raise ValueError("layer_widths must contain at least one layer width")

        # Normalise dropout_rates to a per-layer list.
        if isinstance(dropout_rates, float):
            rates = [dropout_rates] * len(widths)
        else:
            rates = list(dropout_rates)
        if len(rates) != len(widths):
            raise ValueError(
                f"dropout_rates length ({len(rates)}) must equal "
                f"layer_widths length ({len(widths)})"
            )

        layers = []
        in_dim = input_dim
        for i, width in enumerate(widths):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.BatchNorm1d(width))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError("activation must be 'relu' or 'leaky_relu'")
            layers.append(nn.Dropout(rates[i]))
            in_dim = width

        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(in_dim, 1) for _ in range(num_targets)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute logits for each target head given input features."""
        features = self.backbone(x)
        return tuple(head(features) for head in self.heads)
