# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Patch for torchtitan.components.quantization.mx.has_cuda_capability

Target:
- torchtitan.components.quantization.mx.has_cuda_capability

Reason:
- MXFP8Converter.__init__ calls has_cuda_capability(10, 0) to guard
  MXFP8 behind Blackwell+ (SM100). On Ascend NPU there is no CUDA device, so the
  check always fails even though torchao + torchtitan_npu provide NPU-native
  MXFP8 kernels.

- This patch replaces has_cuda_capability in the mx module only,
  avoiding side effects on other callers (attention.py, float8.py, etc.).
"""

from torchtitan.tools.logging import logger

try:
    from torchtitan.components.quantization import mx as mx_module

    from torchtitan_npu.tools.device import get_npu_device_type

    def has_mx_capability(major: int, minor: int) -> bool:
        """Check NPU capability equivalent to CUDA SM capability for MXFP8.

        Ascend950 (A5) provides MXFP8 support equivalent to NVIDIA SM100 (Blackwell).
        Returns True on A5 regardless of the requested SM version, so that upstream
        MXFP8 checks pass even if torchtitan changes the (major, minor) parameters.
        """
        device_type = get_npu_device_type()
        if device_type == "A5":
            return True
        logger.warning(
            f"Patched cuda capability check does not acknowledge cuda {major},{minor} "
            f"abilities on current NPU platform '{device_type}'."
        )
        return False

    mx_module.has_cuda_capability = has_mx_capability
except (ImportError, AttributeError):
    logger.warning("Failed to patch torchtitan.components.quantization.mx.has_cuda_capability.")
