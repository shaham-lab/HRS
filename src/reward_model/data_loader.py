"""Dataset loader infrastructure for the reward model."""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from src.reward_model.dataset_bundle import DatasetBundle
from src.reward_model.parquet_dataset import ParquetDataset
from src.reward_model.reward_model_config import RewardModelConfig
from src.reward_model.reward_model_utils import build_feature_index_map, compute_pos_weights, validate_schema
from src.reward_model.schema_error import SchemaError

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Generic dataset loader template.

    Subclasses provide dataset-specific schema validation, label handling, and
    feature index map construction.
    """

    def __init__(self, config: RewardModelConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------ #
    # Template hooks — override in subclasses for dataset-specific logic #
    # ------------------------------------------------------------------ #
    def _open_parquet(self) -> pq.ParquetFile:
        return pq.ParquetFile(self._config.DATASET_PATH)

    @abstractmethod
    def _validate_schema(self, parquet_file: pq.ParquetFile) -> None:  # pragma: no cover - abstract hook
        ...

    @abstractmethod
    def _read_label_table(self, parquet_file: pq.ParquetFile) -> pa.Table:  # pragma: no cover - abstract hook
        ...

    @abstractmethod
    def _validate_labels(self, label_table: pa.Table) -> None:  # pragma: no cover - abstract hook
        ...

    @abstractmethod
    def _build_feature_index_map(self, parquet_file: pq.ParquetFile) -> Dict[str, Tuple[int, int]]:  # pragma: no cover - abstract hook
        ...

    @abstractmethod
    def _compute_pos_weights(self, label_table, train_rows: List[int]) -> Tuple[float, float]:  # pragma: no cover - abstract hook
        ...

    # ---------------------------- #
    # Shared helpers / defaults    #
    # ---------------------------- #
    @staticmethod
    def _build_split_indices(split_table: pa.Table) -> Dict[str, List[int]]:
        splits: Dict[str, List[int]] = {"train": [], "dev": [], "test": []}
        split_values = split_table.column("split").to_pylist()
        for idx, value in enumerate(split_values):
            if value not in splits:
                raise SchemaError(f"Unexpected split label '{value}' in split column")
            splits[value].append(idx)
        return splits

    def load(self) -> DatasetBundle:
        """Load dataset splits lazily, returning DatasetBundle with metadata."""
        parquet_file = self._open_parquet()

        self._validate_schema(parquet_file)

        label_table = self._read_label_table(parquet_file)
        self._validate_labels(label_table)

        feature_index_map = self._build_feature_index_map(parquet_file)

        split_table = parquet_file.read(columns=["split"])
        split_indices = self._build_split_indices(split_table)

        train_rows = split_indices["train"]
        pos_weight_y1, pos_weight_y2 = self._compute_pos_weights(label_table, train_rows)

        derived_dim = max(end for _, end in feature_index_map.values())

        cache_size = self._config.DATASET_ROW_GROUP_CACHE_SIZE
        train_dataset = ParquetDataset(
            parquet_file, self._config.DATASET_PATH, train_rows, feature_index_map, cache_size
        )
        dev_dataset = ParquetDataset(
            parquet_file, self._config.DATASET_PATH, split_indices["dev"], feature_index_map, cache_size
        )
        test_dataset = ParquetDataset(
            parquet_file, self._config.DATASET_PATH, split_indices["test"], feature_index_map, cache_size
        )

        return DatasetBundle(
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            feature_index_map=feature_index_map,
            pos_weight_y1=pos_weight_y1,
            pos_weight_y2=pos_weight_y2,
            input_dim=derived_dim,
        )


class Mimic4DataLoader(DataLoader):
    """MIMIC-IV–specific dataset loader."""

    @staticmethod
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

    def _validate_schema(self, parquet_file: pq.ParquetFile) -> None:
        validate_schema(parquet_file)

    def _read_label_table(self, parquet_file: pq.ParquetFile) -> pa.Table:
        return parquet_file.read(columns=["y1_mortality", "y2_readmission"])

    def _validate_labels(self, label_table: pa.Table) -> None:
        self._validate_y2_alignment(label_table)

    def _build_feature_index_map(self, parquet_file: pq.ParquetFile) -> Dict[str, Tuple[int, int]]:
        return build_feature_index_map(parquet_file.schema_arrow.names)

    def _compute_pos_weights(self, label_table: pa.Table, train_rows: List[int]) -> Tuple[float, float]:
        if self._config.POS_WEIGHT_Y1 is not None and self._config.POS_WEIGHT_Y2 is not None:
            return float(self._config.POS_WEIGHT_Y1), float(self._config.POS_WEIGHT_Y2)

        labels_df = label_table.to_pandas()
        train_y_df = labels_df.iloc[train_rows].reset_index(drop=True)
        return compute_pos_weights(train_y_df)
