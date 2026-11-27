# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-factory swap contracts of the common MoE patch.

``torchtitan_npu.patches.torchtitan.models.common.moe.apply()`` swaps the
router/MoE classes on the patched modules and rebinds ``config_utils``'s
module-level names. A ``config_utils`` imported earlier in the process keeps
its own ``from ... import`` bindings frozen on the originals, so the rebind is
what keeps ``make_router_config`` building the ``HashRouter.Config`` that the
DeepSeek-V4.1 factories extend with ``vision_enabled`` / ``sorted_topk``.
"""

import dataclasses

from torchtitan.models.common import config_utils
from torchtitan.models.common.config_utils import make_router_config

from torchtitan_npu.patches.torchtitan.models.common.moe import HashMoE, HashRouter


def test_config_utils_names_track_the_patched_classes() -> None:
    assert config_utils.TokenChoiceTopKRouter is HashRouter
    assert config_utils.MoE is HashMoE


def test_router_config_survives_the_multimodal_extension() -> None:
    router = make_router_config(dim=8, num_experts=4, gate_param_init={}, top_k=1)
    assert type(router) is HashRouter.Config
    extended = dataclasses.replace(router, vision_enabled=True, sorted_topk=True)
    assert extended.vision_enabled and extended.sorted_topk
