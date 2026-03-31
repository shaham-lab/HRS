import os
import sys
from typing import Dict, List

import pytest
import torch
from torch import nn

# Add reward_model module path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "reward_model"))
from reward_model import RewardModel  # noqa: E402


@pytest.fixture
def synthetic_data() -> Dict[str, object]:
    batch_size = 4
    input_dim = 50
    num_targets = 2

    # Wider value range to make raw logits clearly differ from probabilities
    X = torch.randn(batch_size, input_dim, dtype=torch.float32) * 3.0
    labels: List[torch.Tensor] = [
        torch.randint(0, 2, (batch_size, 1), dtype=torch.float32) for _ in range(num_targets)
    ]

    return {
        "batch_size": batch_size,
        "input_dim": input_dim,
        "num_targets": num_targets,
        "X": X,
        "labels": labels,
    }


def test_reward_model_initialization():
    model = RewardModel(input_dim=50, layer_widths=[32, 16], dropout_rates=[0.2, 0.2], num_targets=2)
    assert len(model.heads) == 2

    # Single dropout value should broadcast without error
    model = RewardModel(input_dim=50, layer_widths=[32, 16], dropout_rates=0.3, num_targets=2)
    assert isinstance(model.backbone[3], nn.Dropout)

    # Mismatched dropout list length should raise
    with pytest.raises(ValueError):
        RewardModel(input_dim=50, layer_widths=[32, 16], dropout_rates=[0.2], num_targets=2)


def test_reward_model_forward_pass(synthetic_data):
    model = RewardModel(
        input_dim=synthetic_data["input_dim"],
        layer_widths=[32, 16],
        dropout_rates=[0.2, 0.2],
        num_targets=synthetic_data["num_targets"],
    )

    outputs = model(synthetic_data["X"])

    assert isinstance(outputs, tuple)
    assert len(outputs) == synthetic_data["num_targets"]
    for out in outputs:
        assert out.shape == (synthetic_data["batch_size"], 1)
        # Raw logits: ensure not all values are constrained to [0, 1]
        assert ((out < 0) | (out > 1)).any()


def test_reward_model_gradient_flow(synthetic_data):
    model = RewardModel(
        input_dim=synthetic_data["input_dim"],
        layer_widths=[32, 16],
        dropout_rates=[0.2, 0.2],
        num_targets=synthetic_data["num_targets"],
    )

    logits_list = model(synthetic_data["X"])
    criterion = nn.BCEWithLogitsLoss()
    loss = sum(criterion(logit, label) for logit, label in zip(logits_list, synthetic_data["labels"]))
    loss.backward()

    assert model.backbone[0].weight.grad is not None
    assert model.heads[0].weight.grad is not None
    assert not torch.isnan(model.backbone[0].weight.grad).any()
    assert not torch.isnan(model.heads[0].weight.grad).any()
