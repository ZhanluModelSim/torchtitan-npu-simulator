# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Own a resumable training stream and isolate forward-only side effects."""

import copy
import time
from dataclasses import dataclass

from torchtitan.components.dataloader import DataloaderExhaustedError


def copy_batch(batch, *, device):
    """Make an independent working copy for forward-only preprocessing."""
    import torch
    from torch.utils._pytree import tree_map

    def copy_leaf(value):
        if not isinstance(value, torch.Tensor):
            return copy.deepcopy(value)
        return value.detach().to(device=device, copy=True)

    return tree_map(copy_leaf, batch)


@dataclass
class PrefetchedStep:
    batches: list
    loading_times: list[float]


class TrainingDataIterator:
    """Keep one outer stream while recreating the loader iterator after rollback."""

    def __init__(self, loader):
        self.loader = loader
        self.iterator = None
        self.exhausted = False
        self.loading_times = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.exhausted:
            raise DataloaderExhaustedError()
        if self.iterator is None:
            self.iterator = iter(self.loader)
        start = time.perf_counter()
        try:
            batch = next(self.iterator)
        except StopIteration as error:
            self.exhausted = True
            raise DataloaderExhaustedError() from error
        self.loading_times.append(time.perf_counter() - start)
        return batch

    def reset(self):
        # Never close the loader itself: some loaders own a persistent iterator.
        self.iterator = None
        self.exhausted = False
        self.loading_times.clear()

    def fetch_step(self, count, device):
        """Retain loader batches as-is; device is used for the availability collective."""
        import torch
        import torch.distributed as dist

        state = copy.deepcopy(self.loader.state_dict())
        self.loading_times.clear()
        batches = []
        available = True
        try:
            for _ in range(count):
                batches.append(next(self))
        except DataloaderExhaustedError:
            available = False
        # All ranks either record/train a complete step or drain their queues.
        flag = torch.tensor(int(available), dtype=torch.int32, device=device)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if not bool(flag.item()):
            self.loader.load_state_dict(state)
            self.reset()
            self.exhausted = True
            raise DataloaderExhaustedError()
        return PrefetchedStep(batches, list(self.loading_times))
