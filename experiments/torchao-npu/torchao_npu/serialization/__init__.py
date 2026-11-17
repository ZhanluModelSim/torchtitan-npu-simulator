# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Register ``torchao_npu`` quantized tensors and configs with torchao's safetensors allowlists.

torchao/transformers gate safetensors support purely by class name: an unknown
name in ``ALLOWED_TENSORS_SUBCLASSES`` makes ``is_metadata_torchao`` reject the
whole header, and ``ALLOWED_CLASSES`` maps names to the classes used to rebuild
each entry.
"""

__all__ = ["register_torchao_safetensors"]

from torchao.prototype.safetensors import safetensors_utils

from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
from torchao_npu.quantized_tensors import MXTensor


def register_torchao_safetensors() -> None:
    """Add the NPU MX tensor and quant configs to torchao's safetensors allowlists."""
    safetensors_utils.ALLOWED_CLASSES["MXTensor"] = MXTensor

    # Quant configs: stored as MXTensor attributes, looked up by name on load.
    safetensors_utils.ALLOWED_CLASSES["MXQuantizeConfig"] = MXQuantizeConfig
    safetensors_utils.ALLOWED_CLASSES["BlockMXQuantizeConfig"] = BlockMXQuantizeConfig


# Register on import so ``import torchao_npu.serialization`` is sufficient.
register_torchao_safetensors()
