# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Resolve model-owned parameter dtype-preservation patterns before FSDP."""

import logging
import re

from torch import nn

FSDP_PARAMETER_PRESERVE_DTYPE_ATTR = "_torchtitan_npu_fsdp_preserve_parameter_dtype"

logger = logging.getLogger(__name__)


def _has_fsdp_extension(param: nn.Parameter) -> bool:
    return hasattr(param, "fsdp_pre_all_gather") or hasattr(
        param,
        "fsdp_post_all_gather",
    )


def _compile_parameter_pattern(pattern: str) -> re.Pattern[str]:
    if not pattern:
        raise ValueError("FSDP parameter precision patterns must not be empty")

    segments = pattern.split(".")
    if any(not segment for segment in segments):
        raise ValueError(f"invalid FSDP parameter precision pattern {pattern!r}: FQN segments must not be empty")

    # Match ".segment" units so ** can consume zero or more complete FQN segments.
    regex_parts: list[str] = []
    for segment in segments:
        if segment == "**":
            regex_parts.append(r"(?:\.[^.]+)*")
            continue
        if "**" in segment:
            raise ValueError(
                f"invalid FSDP parameter precision pattern {pattern!r}: '**' must occupy a complete FQN segment"
            )

        segment_regex = re.escape(segment).replace(r"\*", "[^.]*")
        regex_parts.append(rf"\.{segment_regex}")
    return re.compile("".join(regex_parts))


def apply_fsdp_parameter_precision(
    module: nn.Module,
    patterns: list[str],
) -> tuple[str, ...]:
    """Mark final ordinary Parameters that must retain their original dtype.

    This must run after model converters and TP/EP transforms, but before the
    first ``fully_shard()`` call. TorchAO wrapper tensors own their FSDP
    extension protocol and reject this ordinary-parameter marker.

    ``*`` matches within one FQN segment and ``**`` matches zero or more
    complete FQN segments. Matching always covers the entire parameter FQN.
    """

    if not patterns:
        return ()

    compiled_patterns = [_compile_parameter_pattern(pattern) for pattern in patterns]

    matched_pattern_counts = [0] * len(compiled_patterns)
    preserved_by_param: dict[int, tuple[nn.Parameter, list[str]]] = {}
    for name, param in module.named_parameters(remove_duplicate=False):
        matched_pattern_indices = [
            index
            for index, compiled_pattern in enumerate(compiled_patterns)
            if compiled_pattern.fullmatch(f".{name}") is not None
        ]
        if not matched_pattern_indices:
            continue

        for index in matched_pattern_indices:
            matched_pattern_counts[index] += 1
        if _has_fsdp_extension(param):
            raise ValueError(f"FSDP precision pattern matched a TorchAO wrapper parameter: {name}")

        entry = preserved_by_param.get(id(param))
        if entry is None:
            preserved_by_param[id(param)] = (param, [name])
        else:
            entry[1].append(name)

    if not preserved_by_param:
        raise ValueError(
            "FSDP parameter precision patterns matched no parameters: "
            + ", ".join(repr(pattern) for pattern in patterns)
        )

    for pattern, count in zip(patterns, matched_pattern_counts, strict=True):
        if count == 0:
            logger.warning("FSDP parameter precision pattern %r matched 0 parameters", pattern)

    marked_names: list[str] = []
    for param, names in preserved_by_param.values():
        existing = getattr(param, FSDP_PARAMETER_PRESERVE_DTYPE_ATTR, False)
        if not isinstance(existing, bool):
            raise ValueError(f"parameter {names[0]} has an invalid FSDP dtype-preservation marker")
        setattr(param, FSDP_PARAMETER_PRESERVE_DTYPE_ATTR, True)
        marked_names.extend(names)
    return tuple(marked_names)
