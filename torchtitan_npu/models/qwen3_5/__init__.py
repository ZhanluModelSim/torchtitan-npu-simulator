# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from torchtitan_npu.models.qwen3_5._fla_compat import ensure_qwen3_5_importable

ensure_qwen3_5_importable()

from torchtitan_npu.patches.torchtitan.hf_datasets.multimodal import (
    mm_collator as _mm_collator_patch,
)
from torchtitan_npu.patches.torchtitan.models.qwen3_5 import (
    vision_encoder as _vision_patch,
)

_mm_collator_patch.apply()
_vision_patch.apply()

import torchtitan.models.qwen3_5.config_registry as _upstream_config_registry
from torchtitan.models.qwen3_5 import model_registry as _upstream_model_registry

from torchtitan_npu.override.qwen3_5.parallelize import parallelize_qwen3_5_npu


def model_registry(*args, **kwargs):
    """Return upstream Qwen3.5 specs with the NPU parallelizer."""
    spec = _upstream_model_registry(*args, **kwargs)
    spec.parallelize_fn = parallelize_qwen3_5_npu
    return spec


_upstream_config_registry.model_registry = model_registry
