# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from contextlib import closing
from dataclasses import replace
from functools import partial
from typing import cast

import grain.python as grain
from torchtitan.components.data.dataset import GrainDataset, SingleDatasetConfig
from torchtitan.components.data.types import DatasetBuildContext, DatasetIterationPolicy


class _CheckedMapShardIterator(grain.DatasetIterator):
    def __init__(self, parent: grain.DatasetIterator, check_after: int, error: str, probe: Callable[[], bool] | None):
        super().__init__(parent)
        self._source = parent
        self._check_after = check_after
        self._error = error
        self._probe = probe

    def __next__(self):
        for rows_read, sample in enumerate(self._source, start=1):
            if sample is not None:
                return sample
            if rows_read == self._check_after:
                if self._probe is None or not self._probe():
                    raise ValueError(self._error)
                # Only the finite probe is disposable. Keep the shuffled cursor.
                self._check_after = 0
                self._probe = None
        raise StopIteration

    def get_state(self):
        return self._source.get_state()

    def set_state(self, state):
        self._source.set_state(state)


class _CheckedMapShard(grain.IterDataset):
    def __init__(self, parent: grain.IterDataset, check_after: int, error: str, probe: Callable[[], bool] | None):
        super().__init__(parent)
        self._source = parent
        self._check_after = check_after
        self._error = error
        self._probe = probe

    def __iter__(self) -> grain.DatasetIterator:
        return _CheckedMapShardIterator(iter(self._source), self._check_after, self._error, self._probe)


class SFTMapDatasetConfig(SingleDatasetConfig):
    """Repeat a map shard while detecting rows entirely removed by SFT filters."""

    def build(self, *, context: DatasetBuildContext, dataset_iteration_policy: DatasetIterationPolicy) -> GrainDataset:
        policy = dataset_iteration_policy
        shard = super().build(context=context, dataset_iteration_policy=replace(policy, repeat=False))
        if not policy.repeat:
            return shard

        shard = cast("grain.MapDataset", shard)
        size = len(shard)
        check_after = 2 * size - 1 if policy.shuffle else size
        probe = None
        if policy.shuffle and policy.dp_world_size > 1:
            # Each epoch reshuffles before sharding, so consecutive misses do
            # not imply that the next epoch will leave this rank empty.
            probe = partial(self._has_valid_sample, context=context, policy=policy)
        repeated = shard.repeat().to_iter_dataset(read_options=context.read_options, allow_nones=True)
        error = f"SFT data shard {policy.dp_rank}/{policy.dp_world_size} has no valid samples"
        return _CheckedMapShard(repeated, check_after, error, probe)

    def _has_valid_sample(self, *, context: DatasetBuildContext, policy: DatasetIterationPolicy) -> bool:
        finite = replace(policy, shuffle=False, repeat=False, dp_rank=0, dp_world_size=1)
        dataset = cast("grain.MapDataset", super().build(context=context, dataset_iteration_policy=finite))
        rows = dataset.to_iter_dataset(read_options=context.read_options, allow_nones=True)
        with closing(iter(rows)) as iterator:
            return any(sample is not None for sample in iterator)
