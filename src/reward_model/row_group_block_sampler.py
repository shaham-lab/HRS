"""Row-group-aware sampler for ParquetDataset."""

import random
from typing import Dict, List

from torch.utils.data import Sampler

from parquet_dataset import ParquetDataset


class RowGroupBlockSampler(Sampler[int]):
    """Sampler that yields indices grouped by row group for cache locality.

    Indices are emitted in row-group order to maximise the ParquetDataset LRU
    cache hit rate.  Within each row group the order is shuffled per epoch.
    """

    def __init__(
        self,
        dataset: ParquetDataset,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

        self._rg_to_local_indices: Dict[int, List[int]] = {}
        for local_idx, global_row_idx in enumerate(dataset._row_indices):
            rg_index, _ = dataset._locate_row_group(global_row_idx)
            self._rg_to_local_indices.setdefault(rg_index, []).append(local_idx)

    def __iter__(self):
        rng = random.Random(self._seed + self._epoch)
        row_groups = sorted(self._rg_to_local_indices.keys())
        if self._shuffle:
            rng.shuffle(row_groups)
        for rg in row_groups:
            local = list(self._rg_to_local_indices[rg])
            if self._shuffle:
                rng.shuffle(local)
            yield from local

    def __len__(self) -> int:
        return sum(len(v) for v in self._rg_to_local_indices.values())

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
