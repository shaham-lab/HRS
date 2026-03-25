"""Lazy Parquet-backed dataset with LRU row-group cache."""

from bisect import bisect_right
from collections import OrderedDict
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch


class ParquetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        parquet_file: pq.ParquetFile,
        dataset_path: str,
        row_indices: List[int],
        feature_index_map: Dict[str, Tuple[int, int]],
        cache_size: int,
    ) -> None:
        """Create a lazy Parquet-backed dataset with LRU row-group cache."""
        self._parquet_file = parquet_file
        self._dataset_path = dataset_path
        self._row_indices = list(row_indices)
        self._feature_index_map = feature_index_map
        self._cache_size = max(1, cache_size)
        self._cache: OrderedDict[int, pa.Table] = OrderedDict()
        self._columns_needed = list(feature_index_map.keys()) + ["y1_mortality", "y2_readmission"]

        metadata = parquet_file.metadata
        self._row_group_boundaries: List[Tuple[int, int]] = []
        self._rg_starts: List[int] = []
        start = 0
        for i in range(metadata.num_row_groups):
            num_rows = metadata.row_group(i).num_rows
            self._row_group_boundaries.append((start, start + num_rows))
            self._rg_starts.append(start)
            start += num_rows

    def __getstate__(self) -> dict:
        """Return picklable state — exclude the non-picklable file handle."""
        state = self.__dict__.copy()
        state["_parquet_file"] = None
        state["_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state in worker process — reopen the Parquet file."""
        self.__dict__.update(state)
        self._parquet_file = pq.ParquetFile(self._dataset_path)
        self._cache = OrderedDict()

    def __len__(self) -> int:
        return len(self._row_indices)

    def __getitem__(self, idx: int):
        """Return feature tensor and labels for the idx-th split row."""
        row_idx = self._row_indices[idx]
        rg_index, rg_start = self._locate_row_group(row_idx)
        table = self._get_row_group(rg_index)
        offset = row_idx - rg_start
        row = table.slice(offset, 1)

        features = []
        for col in self._feature_index_map.keys():
            value = row[col].to_pylist()[0]
            features.append(torch.tensor(value, dtype=torch.float32))
        X = torch.cat(features, dim=0)

        y1_value = row["y1_mortality"].to_pylist()[0]
        y1_tensor = torch.tensor(y1_value, dtype=torch.int8)

        y2_value = row["y2_readmission"].to_pylist()[0]
        y2_value = float("nan") if y2_value is None else y2_value
        y2_tensor = torch.tensor(y2_value, dtype=torch.float32)

        return X, y1_tensor, y2_tensor

    def _locate_row_group(self, row_idx: int) -> Tuple[int, int]:
        i = bisect_right(self._rg_starts, row_idx) - 1
        if i < 0 or i >= len(self._row_group_boundaries):
            raise IndexError(f"Row index {row_idx} out of bounds for dataset")
        start, end = self._row_group_boundaries[i]
        if not (start <= row_idx < end):
            raise IndexError(f"Row index {row_idx} out of bounds for dataset")
        return i, start

    def _get_row_group(self, rg_index: int) -> pa.Table:
        if rg_index in self._cache:
            table = self._cache.pop(rg_index)
            self._cache[rg_index] = table
            return table

        table = self._parquet_file.read_row_group(rg_index, columns=self._columns_needed)
        self._cache[rg_index] = table
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return table


