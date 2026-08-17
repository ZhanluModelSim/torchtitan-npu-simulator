# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Patch:
#   torch.use_deterministic_algorithms — wrap it so that enabling determinism
#   also sets the NPU env vars (HCCL_DETERMINISTIC, CLOSE_MATMUL_K_SHIFT) that
#   the HCCL / op layer reads but torch_npu's Python layer does not set.
#
# Why:
#   Upstream set_determinism() enables determinism via use_deterministic_algorithms
#   and sets CUDA's deterministic env (CUBLAS_WORKSPACE_CONFIG) at the same spot.
#   Hooking the same call adds the NPU equivalents, so determinism stays driven
#   purely by --debug.deterministic with no separate injection point.

import os
from collections.abc import Callable

import torch
from torchtitan.tools.logging import logger

# NPU deterministic env vars and the values we set when determinism is enabled.
# Values follow the torch_npu documented convention ("true" / "1").
_NPU_DETERMINISTIC_ENV: dict[str, str] = {
    "HCCL_DETERMINISTIC": "true",
    "CLOSE_MATMUL_K_SHIFT": "1",
}

_original_use_deterministic_algorithms: Callable | None = None


def _set_env_with_warning(name: str, value: str) -> None:
    """Set ``os.environ[name] = value``, warning on a genuine conflict."""
    existing = os.environ.get(name)
    if name == "HCCL_DETERMINISTIC" and existing is not None and existing.lower() == "strict":
        logger.warning(
            "Both deterministic and batch-invariant modes are enabled; only batch-invariant mode takes effect."
        )
        return
    if existing is not None and existing.lower() != value.lower():
        logger.warning(
            "Overriding existing env %s=%r with %r for deterministic training.",
            name,
            existing,
            value,
        )
    os.environ[name] = value


def _use_deterministic_algorithms(mode, *args, **kwargs):
    if mode:
        for name, value in _NPU_DETERMINISTIC_ENV.items():
            _set_env_with_warning(name, value)
        logger.info(
            "NPU deterministic env enabled (%s).",
            ", ".join(_NPU_DETERMINISTIC_ENV),
        )
    if _original_use_deterministic_algorithms is None:
        raise RuntimeError("determinism patch wrapper called before apply().")
    return _original_use_deterministic_algorithms(mode, *args, **kwargs)


def apply() -> None:
    """Wrap ``torch.use_deterministic_algorithms`` to also set NPU env."""
    global _original_use_deterministic_algorithms
    if _original_use_deterministic_algorithms is not None:
        return
    _original_use_deterministic_algorithms = torch.use_deterministic_algorithms
    torch.use_deterministic_algorithms = _use_deterministic_algorithms


apply()
