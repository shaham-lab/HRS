"""Dataset bundle definition for reward model datasets."""

from typing import Dict, NamedTuple, Tuple

from src.reward_model.parquet_dataset import ParquetDataset


class DatasetBundle(NamedTuple):
    train_dataset: ParquetDataset
    dev_dataset: ParquetDataset
    test_dataset: ParquetDataset
    feature_index_map: Dict[str, Tuple[int, int]]
    pos_weight_y1: float
    pos_weight_y2: float
    input_dim: int


