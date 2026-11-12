# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""extensions for torchtitan-npu."""

# The import order is the extension activation order. Do not sort this block.
# ruff: noqa: I001,F401
from .torchtitan import distributed, ep_process_group
from .torchtitan.models import common
from .torchtitan import trainer
from .torchtitan.components import metrics
from .torchtitan.tools import utils
from .torchtitan import eager_flex
from .compile import dynamo_config
