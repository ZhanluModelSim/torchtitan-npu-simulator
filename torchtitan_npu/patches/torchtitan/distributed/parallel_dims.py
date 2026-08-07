# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3864

import functools
import logging

from torchtitan.distributed.parallel_dims import ParallelDims

logger = logging.getLogger(__name__)

_original_post_init = ParallelDims.__post_init__


@functools.wraps(_original_post_init)
def _patched_post_init(self):
    _original_post_init(self)
    ParallelDims._global_instance = self


def _patched_get(cls) -> ParallelDims:
    assert cls._global_instance is not None, "ParallelDims has not been initialized."
    return cls._global_instance


def apply() -> None:
    ParallelDims._global_instance = None
    ParallelDims.__post_init__ = _patched_post_init
    ParallelDims.get = classmethod(_patched_get)

    logger.info("[PATCH] ParallelDims._global_instance + get()")


apply()
