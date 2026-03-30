"""Row-group-aware sampler for ParquetDataset."""

import random
from typing import Dict, List

from torch.utils.data import Sampler

from parquet_dataset import ParquetDataset


class RowGroupBlockSampler(Sampler[int]):
    """Row-group-aware sampler that shuffles at the row group level to prevent I/O thrashing on the ParquetDataset LRU cache.

    Consecutive indices yielded by this sampler come from the same or adjacent
    row groups, maximising cache hit rate.  Partitions row groups across DDP
    ranks in round-robin order.
    """

    def __init__(
        self,
        dataset: ParquetDataset,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self._dataset = dataset
        self._rank = rank
        self._world_size = world_size
        self._shuffle = shuffle
        self._seed = seed
        self._epoch = 0

        self._rg_to_local_indices: Dict[int, List[int]] = {}
        for local_idx, global_row_idx in enumerate(self._dataset._row_indices):
            rg_index, _ = self._dataset._locate_row_group(global_row_idx)
            self._rg_to_local_indices.setdefault(rg_index, []).append(local_idx)

        self._num_samples = sum(len(indices) for indices in self._rg_to_local_indices.values())

    def __iter__(self):
        rng = random.Random(self._seed + self._epoch)

        row_groups = [rg for rg, indices in self._rg_to_local_indices.items() if indices]
        if self._shuffle:
            rng.shuffle(row_groups)

        row_groups_for_rank = [rg for idx, rg in enumerate(row_groups) if idx % self._world_size == self._rank]

        for rg in row_groups_for_rank:
            local_indices = list(self._rg_to_local_indices[rg])
            if self._shuffle:
                rng.shuffle(local_indices)
            for idx in local_indices:
                yield idx

    def __len__(self) -> int:
        rng = random.Random(self._seed + self._epoch)
        row_groups = [rg for rg, indices in self._rg_to_local_indices.items() if indices]
        if self._shuffle:
            rng.shuffle(row_groups)
        row_groups_for_rank = [rg for idx, rg in enumerate(row_groups) if idx % self._world_size == self._rank]
        return sum(len(self._rg_to_local_indices[rg]) for rg in row_groups_for_rank)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch


