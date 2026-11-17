# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility boundary for the pinned TorchTitan/PyTorch LoRA APIs."""

from torch.compiler import _is_non_strict_tracing
from torchtitan.components.lora import _lora_adapter_sharding
from torchtitan.protocols.sharding import ShardingConfig


def is_non_strict_tracing() -> bool:
    return _is_non_strict_tracing()


def upstream_adapter_sharding(
    base_sharding: ShardingConfig | None,
) -> tuple[ShardingConfig | None, ShardingConfig | None]:
    return _lora_adapter_sharding(base_sharding)
