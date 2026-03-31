import os
import sys
from typing import Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

# Add reward_model module path for direct import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "reward_model"))
from mimic4_data_loader import Mimic4DataLoader  # noqa: E402
from parquet_dataset import ParquetDataset  # noqa: E402
from reward_model_config import RewardModelConfig  # noqa: E402
from schema_error import SchemaError  # noqa: E402


@pytest.fixture
def valid_synthetic_parquet(tmp_path) -> str:
    expected_cols = Mimic4DataLoader._EXPECTED_COLUMNS
    num_rows = 10

    splits = ["train", "train", "dev", "test", "train", "dev", "test", "train", "dev", "train"]
    y1_values = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.int64)
    y2_values = np.array([0, np.nan, 1, 0, np.nan, 1, 0, 1, np.nan, 0], dtype=np.float32)

    data: Dict[str, List] = {}
    for col in expected_cols:
        if col == "subject_id":
            data[col] = np.arange(1000, 1000 + num_rows, dtype=np.int64)
        elif col == "hadm_id":
            data[col] = np.arange(2000, 2000 + num_rows, dtype=np.int64)
        elif col == "split":
            data[col] = splits
        elif col == "y1_mortality":
            data[col] = y1_values
        elif col == "y2_readmission":
            data[col] = y2_values
        elif col == "demographic_vec":
            data[col] = [np.linspace(0, 1, 8, dtype=np.float64).tolist() for _ in range(num_rows)]
        elif col.endswith("_embedding"):
            data[col] = [np.linspace(0, 0.3, 4, dtype=np.float32).tolist() for _ in range(num_rows)]
        else:
            # Should not hit due to exhaustive expected columns
            data[col] = [0 for _ in range(num_rows)]

    arrays = {}
    for col, values in data.items():
        if col in {"subject_id", "hadm_id"}:
            arrays[col] = pa.array(values, type=pa.int64())
        elif col == "split":
            arrays[col] = pa.array(values, type=pa.string())
        elif col == "y1_mortality":
            arrays[col] = pa.array(values, type=pa.int64())
        elif col == "y2_readmission":
            arrays[col] = pa.array(values, type=pa.float32())
        elif col == "demographic_vec":
            arrays[col] = pa.array(values, type=pa.list_(pa.float64()))
        elif col.endswith("_embedding"):
            arrays[col] = pa.array(values, type=pa.list_(pa.float32()))
        else:
            arrays[col] = pa.array(values)

    table = pa.Table.from_arrays([arrays[col] for col in expected_cols], names=expected_cols)
    path = tmp_path / "synthetic_cdss.parquet"
    pq.write_table(table, path)
    return str(path)


def _build_mock_config(dataset_path: str) -> RewardModelConfig:
    return RewardModelConfig(
        LAYER_WIDTHS=[32, 16],
        DROPOUT_RATES=0.1,
        ACTIVATION="relu",
        NUM_TARGETS=2,
        MAX_EPOCHS=1,
        BATCH_SIZE_PER_GPU=1,
        NUM_GPUS=1,
        LEARNING_RATE=1e-3,
        WEIGHT_DECAY=0.0,
        ADAM_BETA1=0.9,
        ADAM_BETA2=0.999,
        LR_WARMUP_EPOCHS=0,
        LR_MIN=1e-5,
        EARLY_STOPPING_PATIENCE=1,
        CHECKPOINT_KEEP_N=1,
        LOSS_WEIGHTS=[0.5, 0.5],
        POS_WEIGHTS=None,
        MASKING_START_RATIOS={"random": 1.0, "adversarial": 0.0, "none": 0.0},
        MASKING_END_RATIOS={"random": 1.0, "adversarial": 0.0, "none": 0.0},
        MASKING_TRANSITION_MIDPOINT_EPOCH=1,
        MASKING_TRANSITION_SHAPE="linear",
        MASKING_RANDOM_K_MIN_FRACTION=0.0,
        MASKING_RANDOM_K_MAX_FRACTION=0.0,
        MASKING_ADVERSARIAL_K_MIN_FRACTION=0.0,
        MASKING_ADVERSARIAL_K_MAX_FRACTION=0.0,
        NUM_ALWAYS_VISIBLE_FEATURES=5,
        INPUT_DIM=228,
        DATASET_PATH=dataset_path,
        DATASET_ROW_GROUP_CACHE_SIZE=1,
        DATALOADER_NUM_WORKERS=0,
        CHECKPOINT_DIR="/tmp/ckpt",
        METRICS_PATH="/tmp/metrics",
        CALIBRATION_PARAMS_PATH="/tmp/calibration",
        EXPORT_PATH="/tmp/export",
    )


def test_mimic4_data_loader_valid_schema(valid_synthetic_parquet):
    config = _build_mock_config(valid_synthetic_parquet)
    loader = Mimic4DataLoader(config)

    try:
        bundle = loader.load()
    except SchemaError as exc:
        pytest.fail(f"Schema validation unexpectedly failed: {exc}")

    assert isinstance(bundle.train_dataset, ParquetDataset)
    assert isinstance(bundle.dev_dataset, ParquetDataset)
    assert isinstance(bundle.test_dataset, ParquetDataset)


def test_parquet_dataset_lazy_load(valid_synthetic_parquet):
    config = _build_mock_config(valid_synthetic_parquet)
    loader = Mimic4DataLoader(config)
    bundle = loader.load()

    train_dataset = bundle.train_dataset
    assert len(train_dataset) > 0

    X, y1, y2 = train_dataset[0]
    assert isinstance(X, torch.Tensor)
    assert X.dtype == torch.float32
    assert X.shape[0] == 228

    assert isinstance(y1, torch.Tensor)
    assert isinstance(y2, torch.Tensor)
