"""Dataset bundle definition for reward model datasets."""

from typing import Dict, List, NamedTuple, Tuple

from parquet_dataset import ParquetDataset


class DatasetBundle(NamedTuple):
    train_dataset: ParquetDataset
    dev_dataset: ParquetDataset
    test_dataset: ParquetDataset
    feature_index_map: Dict[str, Tuple[int, int]]
    pos_weights: List[float]
    input_dim: int
