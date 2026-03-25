"""Reward model configuration schema and loader."""

import math
import os
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from src.reward_model.schema_error import SchemaError


class RewardModelConfig(BaseModel):
    LAYER_WIDTHS: List[int]
    DROPOUT_RATE: float
    ACTIVATION: str

    MAX_EPOCHS: int
    BATCH_SIZE_PER_GPU: int
    NUM_GPUS: int = 2
    LEARNING_RATE: float
    WEIGHT_DECAY: float
    ADAM_BETA1: float
    ADAM_BETA2: float
    LR_WARMUP_EPOCHS: int
    LR_MIN: float
    EARLY_STOPPING_PATIENCE: int
    CHECKPOINT_KEEP_N: int

    LOSS_WEIGHT_Y1: float
    LOSS_WEIGHT_Y2: float
    POS_WEIGHT_Y1: Optional[float] = Field(default=None)
    POS_WEIGHT_Y2: Optional[float] = Field(default=None)

    MASKING_START_RATIOS: Dict[str, float]
    MASKING_END_RATIOS: Dict[str, float]
    MASKING_TRANSITION_MIDPOINT_EPOCH: int
    MASKING_TRANSITION_SHAPE: str
    MASKING_K: int

    DATASET_PATH: str
    DATASET_ROW_GROUP_CACHE_SIZE: int = 2
    DATALOADER_NUM_WORKERS: int = 4

    CHECKPOINT_DIR: str
    METRICS_PATH: str
    CALIBRATION_PARAMS_PATH: str
    EXPORT_PATH: str

    @field_validator("ACTIVATION")
    @classmethod
    def _validate_activation(cls, value: str) -> str:
        if value not in {"relu", "leaky_relu"}:
            raise ValueError("ACTIVATION must be one of {'relu', 'leaky_relu'}")
        return value

    @field_validator("MASKING_START_RATIOS", "MASKING_END_RATIOS")
    @classmethod
    def _validate_ratios(cls, value: Dict[str, float]) -> Dict[str, float]:
        expected_keys = {"random", "adversarial", "none"}
        if set(value.keys()) != expected_keys:
            raise ValueError(
                f"Masking ratios must define exactly {expected_keys}, received {set(value.keys())}"
            )
        return value

    @model_validator(mode="after")
    def _expand_paths(self) -> "RewardModelConfig":
        for field_name, field_value in self.__dict__.items():
            if field_value is None:
                continue
            if field_name.endswith(("_PATH", "_DIR", "_FILE")):
                expanded = os.path.abspath(os.path.expanduser(str(field_value)))
                object.__setattr__(self, field_name, expanded)
        start_sum = sum(self.MASKING_START_RATIOS.values())
        end_sum = sum(self.MASKING_END_RATIOS.values())
        if not math.isclose(start_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"MASKING_START_RATIOS must sum to 1.0, found {start_sum}")
        if not math.isclose(end_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"MASKING_END_RATIOS must sum to 1.0, found {end_sum}")
        return self

    class Config:
        extra = "forbid"


def load_and_validate_config(path: str) -> RewardModelConfig:
    """Load YAML config and validate it against the RewardModelConfig schema."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    try:
        return RewardModelConfig(**data)
    except ValidationError as exc:
        raise SchemaError(f"Invalid reward model configuration: {exc}") from exc


