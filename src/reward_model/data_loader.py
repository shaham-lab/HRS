"""Dataset loader infrastructure for the reward model."""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from dataset_bundle import DatasetBundle
from .parquet_dataset import ParquetDataset
from reward_model_config import RewardModelConfig
from schema_error import SchemaError

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

    # pragma: no cover  # abstract hook
    @abstractmethod
    def _validate_schema(self, parquet_file: pq.ParquetFile) -> None:
        """Validate dataset-level schema/columns; raise ``SchemaError`` on violation."""

    # pragma: no cover  # abstract hook
    @abstractmethod
    def _read_label_table(self, parquet_file: pq.ParquetFile) -> pa.Table:
        """Read the label columns into a ``pyarrow.Table`` for downstream validation/weights."""

    # pragma: no cover  # abstract hook
    @abstractmethod
    def _validate_labels(self, label_table: pa.Table) -> None:
        """Validate label consistency (dtypes, NaN rules) and raise ``SchemaError`` on failure."""

    # pragma: no cover  # abstract hook
    @abstractmethod
    def _build_feature_index_map(self, parquet_file: pq.ParquetFile) -> Dict[str, Tuple[int, int]]:
        """Construct the feature index map defining (start, end) for each feature column."""

    # pragma: no cover  # abstract hook
    @abstractmethod
    def _compute_pos_weights(self, label_table: pa.Table, train_rows: List[int]) -> List[float]:
        """Compute positive class weights from the training split (or use config overrides)."""

    # pragma: no cover  # abstract hook
    @abstractmethod
    def _get_label_columns(self) -> List[str]:
        """Return the ordered list of label column names for this dataset."""

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
        label_columns = self._get_label_columns()

        split_table = parquet_file.read(columns=["split"])
        split_indices = self._build_split_indices(split_table)

        train_rows = split_indices["train"]
        pos_weights = self._compute_pos_weights(label_table, train_rows)

        derived_dim = max(end for _, end in feature_index_map.values())

        cache_size = self._config.DATASET_ROW_GROUP_CACHE_SIZE
        train_dataset = ParquetDataset(
            parquet_file, self._config.DATASET_PATH, train_rows, feature_index_map,
            cache_size, label_columns,
        )
        dev_dataset = ParquetDataset(
            parquet_file, self._config.DATASET_PATH, split_indices["dev"], feature_index_map,
            cache_size, label_columns,
        )
        test_dataset = ParquetDataset(
            parquet_file, self._config.DATASET_PATH, split_indices["test"], feature_index_map,
            cache_size, label_columns,
        )

        n_train = len(train_rows)
        n_dev = len(split_indices["dev"])
        n_test = len(split_indices["test"])
        logger.info("Dataset loaded: %d train, %d dev, %d test admissions",
                    n_train, n_dev, n_test)
        n_slots = len(feature_index_map)
        logger.info("Input dim: %d  (%d feature slots: 1 structured + 55 embeddings)",
                    derived_dim, n_slots)

        return DatasetBundle(
            train_dataset=train_dataset,
            dev_dataset=dev_dataset,
            test_dataset=test_dataset,
            feature_index_map=feature_index_map,
            pos_weights=pos_weights,
            input_dim=derived_dim,
        )
