import logging
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.reward_model.reward_model_utils import (
    DatasetBundle,
    ParquetDataset,
    RewardModelConfig,
    SchemaError,
    build_feature_index_map,
    compute_pos_weights,
    get_expected_columns,
)

logger = logging.getLogger(__name__)


def _validate_column_order(schema: pa.Schema, expected: List[str]) -> None:
    if list(schema.names) != expected:
        raise SchemaError(
            "Column order or presence mismatch; expected columns per PREPROCESSING_DATA_MODEL.md Section 3.12"
        )


def _assert_dtype_matches(field: pa.Field, allowed_types: Tuple[pa.DataType, ...], producer: str) -> None:
    if not any(field.type == allowed for allowed in allowed_types):
        allowed_str = ", ".join(str(t) for t in allowed_types)
        raise SchemaError(f"{field.name} dtype mismatch (expected one of {allowed_str}) produced by {producer}")


def _validate_label_columns(schema: pa.Schema) -> None:
    y1_field = schema.field("y1_mortality")
    y2_field = schema.field("y2_readmission")
    _assert_dtype_matches(
        y1_field, (pa.int8(), pa.float32()), "extract_y_data.py (y1_mortality)"
    )
    _assert_dtype_matches(y2_field, (pa.float32(),), "extract_y_data.py (y2_readmission)")


def _validate_embedding_columns(schema: pa.Schema) -> None:
    for name in schema.names:
        if not name.endswith("_embedding"):
            continue
        field = schema.field(name)
        if not (pa.types.is_fixed_size_list(field.type) and pa.types.is_float32(field.type.value_type)):
            raise SchemaError(
                f"{name} type mismatch; expected float32[768] produced by combine_dataset.py"
            )
        if field.type.list_size != 768:
            raise SchemaError(
                f"{name} length mismatch; expected fixed_size_list[768] produced by combine_dataset.py"
            )


def _validate_null_counts(parquet_file: pq.ParquetFile, columns: List[str]) -> None:
    for col in columns:
        nulls = 0
        for rg in range(parquet_file.metadata.num_row_groups):
            column_index = parquet_file.schema_arrow.get_field_index(col)
            stats = parquet_file.metadata.row_group(rg).column(column_index).statistics
            if stats is None:
                raise SchemaError(f"Missing statistics for column {col}; cannot validate null counts")
            nulls += stats.null_count
        if nulls != 0:
            producer = "combine_dataset.py" if col.endswith("_embedding") else "extract_y_data.py"
            raise SchemaError(f"Null values found in {col} produced by {producer}")


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
    parquet_file = pq.ParquetFile(config.DATASET_PATH)
    schema = parquet_file.schema_arrow

    expected_columns = get_expected_columns()
    _validate_column_order(schema, expected_columns)
    _validate_label_columns(schema)
    _validate_embedding_columns(schema)

    _validate_null_counts(parquet_file, ["y1_mortality"])
    embedding_columns = [name for name in expected_columns if name.endswith("_embedding")]
    _validate_null_counts(parquet_file, embedding_columns)

    y_table = parquet_file.read(columns=["y1_mortality", "y2_readmission"])
    _validate_y2_alignment(y_table)

    feature_index_map = build_feature_index_map(expected_columns)

    split_table = parquet_file.read(columns=["split"])
    split_indices = _build_split_indices(split_table)

    train_rows = split_indices["train"]
    if config.POS_WEIGHT_Y1 is not None and config.POS_WEIGHT_Y2 is not None:
        pos_weight_y1 = float(config.POS_WEIGHT_Y1)
        pos_weight_y2 = float(config.POS_WEIGHT_Y2)
    else:
        pos_weight_y1, pos_weight_y2 = compute_pos_weights(y_table.to_pandas().iloc[train_rows])

    derived_dim = max(end for _, end in feature_index_map.values())
    if config.INPUT_DIM != derived_dim:
        raise SchemaError(f"INPUT_DIM {config.INPUT_DIM} does not match derived dimension from feature index map")

    cache_size = config.DATASET_ROW_GROUP_CACHE_SIZE
    train_dataset = ParquetDataset(parquet_file, train_rows, feature_index_map, cache_size)
    dev_dataset = ParquetDataset(parquet_file, split_indices["dev"], feature_index_map, cache_size)
    test_dataset = ParquetDataset(parquet_file, split_indices["test"], feature_index_map, cache_size)

    return DatasetBundle(
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        feature_index_map=feature_index_map,
        pos_weight_y1=pos_weight_y1,
        pos_weight_y2=pos_weight_y2,
    )
