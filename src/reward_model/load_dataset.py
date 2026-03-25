import logging
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

from src.reward_model.reward_model_utils import (
    DatasetBundle,
    ParquetDataset,
    RewardModelConfig,
    SchemaError,
    build_feature_index_map,
    compute_pos_weights,
    validate_schema,
)

logger = logging.getLogger(__name__)


def _validate_y2_alignment(y_table: pa.Table) -> None:
    df = y_table.to_pandas()
    mortality = df["y1_mortality"].astype(float)
    readmit = df["y2_readmission"].astype(float)
    deceased_mask = mortality == 1.0
    survivor_mask = mortality == 0.0

    if readmit[deceased_mask].notna().any():
        raise SchemaError(
            "y2_readmission must be NaN for deceased rows; produced by extract_y_data.py"
        )
    if survivor_mask.any() and readmit[survivor_mask].isna().any():
        raise SchemaError(
            "y2_readmission must be non-null for survivors; produced by extract_y_data.py"
        )


def _build_split_indices(split_table: pa.Table) -> Dict[str, List[int]]:
    splits: Dict[str, List[int]] = {"train": [], "dev": [], "test": []}
    split_values = split_table.column("split").to_pylist()
    for idx, value in enumerate(split_values):
        if value not in splits:
            raise SchemaError(f"Unexpected split label '{value}' in split column")
        splits[value].append(idx)
    return splits


def run(config: RewardModelConfig) -> DatasetBundle:
    """Load dataset splits lazily, returning DatasetBundle with metadata."""
    parquet_file = pq.ParquetFile(config.DATASET_PATH)

    validate_schema(parquet_file)

    y_table = parquet_file.read(columns=["y1_mortality", "y2_readmission"])
    _validate_y2_alignment(y_table)

    feature_index_map = build_feature_index_map(parquet_file.schema_arrow.names)

    split_table = parquet_file.read(columns=["split"])
    split_indices = _build_split_indices(split_table)

    train_rows = split_indices["train"]
    if config.POS_WEIGHT_Y1 is not None and config.POS_WEIGHT_Y2 is not None:
        pos_weight_y1 = float(config.POS_WEIGHT_Y1)
        pos_weight_y2 = float(config.POS_WEIGHT_Y2)
    else:
        labels_df = y_table.to_pandas()
        train_y_df = labels_df.iloc[train_rows].reset_index(drop=True)
        pos_weight_y1, pos_weight_y2 = compute_pos_weights(train_y_df)

    derived_dim = max(end for _, end in feature_index_map.values())

    cache_size = config.DATASET_ROW_GROUP_CACHE_SIZE
    train_dataset = ParquetDataset(parquet_file, config.DATASET_PATH, train_rows, feature_index_map, cache_size)
    dev_dataset = ParquetDataset(parquet_file, config.DATASET_PATH, split_indices["dev"], feature_index_map, cache_size)
    test_dataset = ParquetDataset(parquet_file, config.DATASET_PATH, split_indices["test"], feature_index_map, cache_size)

    return DatasetBundle(
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        feature_index_map=feature_index_map,
        pos_weight_y1=pos_weight_y1,
        pos_weight_y2=pos_weight_y2,
        input_dim=derived_dim,
    )
