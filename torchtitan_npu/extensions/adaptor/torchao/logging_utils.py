# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped logging helpers for TorchAO-NPU config converters."""

from __future__ import annotations

__all__ = ["GroupedConverterLogger"]

import logging
from collections import defaultdict

# TODO: Switch to torchtitan.config.format once upstream pytorch/torchtitan#6315 lands.
from torchtitan_npu.patches.torchtitan.config.format import fold_indexed_paths


class GroupedConverterLogger:
    """Aggregate per-node converter logs at INFO while preserving DEBUG details."""

    def __init__(self, converter: object) -> None:
        cls = converter if isinstance(converter, type) else type(converter)
        self._logger = logging.getLogger(cls.__module__)
        self._name = f"{cls.__module__}.{cls.__qualname__}"
        self._groups: dict[str, list[str]] = defaultdict(list)

    def record(self, detail: str, fqn: str, *, debug_detail: str | None = None) -> None:
        self._logger.debug("[Converter] %s: %s %s", self._name, fqn, debug_detail or detail)
        self._groups[detail].append(fqn)

    def flush(self) -> None:
        for detail, fqns in self._groups.items():
            self._logger.info(
                "[Converter] %s: %s matched=%d fqns=%s",
                self._name,
                detail,
                len(fqns),
                fold_indexed_paths(fqns),
            )
        self._groups.clear()
