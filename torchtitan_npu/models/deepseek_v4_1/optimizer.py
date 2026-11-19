# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass

from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import SparseAdam


@dataclass(kw_only=True)
class EngramTableOptimizerConfig(SparseAdam.Config):
    """The recipe-owned table group, distinct from user-defined groups."""
