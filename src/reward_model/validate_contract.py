import argparse
import logging
import sys
from typing import List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from src.reward_model.reward_model_utils import (
    SchemaError,
    get_expected_columns,
    load_and_validate_config,
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
    _assert_dtype_matches(y1_field, (pa.int8(), pa.float32()), "extract_y_data.py (y1_mortality)")
    _assert_dtype_matches(y2_field, (pa.float32(),), "extract_y_data.py (y2_readmission)")


def _validate_embedding_columns(schema: pa.Schema) -> None:
    for name in schema.names:
        if not name.endswith("_embedding"):
            continue
        field = schema.field(name)
        if not (pa.types.is_fixed_size_list(field.type) and pa.types.is_float32(field.type.value_type)):
            raise SchemaError(f"{name} type mismatch; expected float32[768] produced by combine_dataset.py")
        if field.type.list_size != 768:
            raise SchemaError(f"{name} length mismatch; expected fixed_size_list[768] produced by combine_dataset.py")


def _validate_null_counts(parquet_file: pq.ParquetFile, columns: List[str]) -> None:
    for col in columns:
        nulls = 0
        for rg in range(parquet_file.metadata.num_row_groups):
            idx = parquet_file.schema_arrow.get_field_index(col)
            stats = parquet_file.metadata.row_group(rg).column(idx).statistics
            if stats is None:
                raise SchemaError(f"Missing statistics for column {col}; cannot validate null counts")
            nulls += stats.null_count
        if nulls != 0:
            producer = "combine_dataset.py" if col.endswith("_embedding") else "extract_y_data.py"
            raise SchemaError(f"Null values found in {col} produced by {producer}")


def _validate_y2_alignment_metadata(parquet_file: pq.ParquetFile) -> None:
    y1_index = parquet_file.schema_arrow.get_field_index("y1_mortality")
    y2_index = parquet_file.schema_arrow.get_field_index("y2_readmission")

    for rg in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(rg)
        num_rows = row_group.num_rows
        y1_stats = row_group.column(y1_index).statistics
        y2_stats = row_group.column(y2_index).statistics

        if y1_stats is None or y2_stats is None:
            raise SchemaError("Missing statistics for y1_mortality or y2_readmission; cannot validate alignment")

        if y1_stats.null_count != 0:
            raise SchemaError("y1_mortality contains nulls; produced by extract_y_data.py")

        if y1_stats.min == y1_stats.max == 1:
            if y2_stats.null_count != num_rows:
                raise SchemaError(
                    "y2_readmission expected to be NaN for deceased rows; produced by extract_y_data.py"
                )
        elif y1_stats.min == y1_stats.max == 0:
            if y2_stats.null_count != 0:
                raise SchemaError(
                    "y2_readmission expected to be non-null for survivor rows; produced by extract_y_data.py"
                )


def _run_assertions(parquet_file: pq.ParquetFile) -> List[Tuple[str, bool, str]]:
    results: List[Tuple[str, bool, str]] = []
    schema = parquet_file.schema_arrow
    expected_columns = get_expected_columns()

    checks = [
        ("Column order", lambda: _validate_column_order(schema, expected_columns)),
        ("Label dtypes", lambda: _validate_label_columns(schema)),
        ("Embedding dtypes", lambda: _validate_embedding_columns(schema)),
        ("y1_mortality nulls", lambda: _validate_null_counts(parquet_file, ["y1_mortality"])),
        (
            "Embedding nulls",
            lambda: _validate_null_counts(
                parquet_file, [name for name in expected_columns if name.endswith("_embedding")]
            ),
        ),
        ("y2_readmission alignment", lambda: _validate_y2_alignment_metadata(parquet_file)),
    ]

    for name, fn in checks:
        try:
            fn()
            results.append((name, True, "PASS"))
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, str(exc)))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate reward model dataset contract.")
    parser.add_argument("--config", required=True, help="Path to reward_model.yaml")
    args = parser.parse_args()

    config = load_and_validate_config(args.config)
    parquet_file = pq.ParquetFile(config.DATASET_PATH)

    results = _run_assertions(parquet_file)
    failed = False
    for name, ok, message in results:
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {name} - {message}")
        if not ok:
            failed = True

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
