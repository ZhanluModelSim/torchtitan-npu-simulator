# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pattern replacements applied before AOTAutograd."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch._inductor.config as inductor_config
from torch._inductor.custom_graph_pass import (
    CustomGraphPass,
    get_custom_graph_passes,
    get_hash_for_files,
)
from torch.fx.subgraph_rewriter import replace_pattern_with_filters

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PatternReplacement:
    """One forward pattern and its replacement."""

    search_fn: Callable[..., Any]
    replacement_fn: Callable[..., Any]
    ignore_literals: bool = False


class _PreAOTPatternPass(CustomGraphPass):
    """Replace forward fragments before AOTAutograd traces backward."""

    def __init__(self) -> None:
        self._patterns: dict[str, PatternReplacement] = {}

    def __call__(self, graph: torch.fx.Graph) -> None:
        graph_module = graph.owning_module
        assert graph_module is not None

        for name, pattern in self._patterns.items():
            matches = replace_pattern_with_filters(
                graph_module,
                pattern.search_fn,
                pattern.replacement_fn,
                ignore_literals=pattern.ignore_literals,
            )
            if matches:
                logger.info(
                    "Pre-AOT pattern %s replaced %d subgraph(s)",
                    name,
                    len(matches),
                )

    def uuid(self) -> bytes | None:
        pattern_files = {__file__}
        pattern_key: list[str] = []
        for name, pattern in self._patterns.items():
            pattern_key.extend((name, str(pattern.ignore_literals)))
            for pattern_fn in (pattern.search_fn, pattern.replacement_fn):
                if source_file := inspect.getsourcefile(pattern_fn):
                    pattern_files.add(source_file)
                pattern_key.extend(
                    f"{key}={value!r}" for key, value in sorted(inspect.getclosurevars(pattern_fn).nonlocals.items())
                )
        return get_hash_for_files(
            tuple(sorted(pattern_files)),
            extra="\n".join(pattern_key),
        )


_PRE_AOT_PATTERN_PASS = _PreAOTPatternPass()


def register_pre_aot_patterns(
    patterns: Mapping[str, PatternReplacement],
) -> None:
    """Register patterns and install the shared pass for Inductor backends."""

    if not patterns:
        return
    _PRE_AOT_PATTERN_PASS._patterns.update(patterns)
    installed = get_custom_graph_passes(inductor_config.pre_grad_custom_pass)
    if _PRE_AOT_PATTERN_PASS not in installed:
        inductor_config.pre_grad_custom_pass = (
            *installed,
            _PRE_AOT_PATTERN_PASS,
        )
