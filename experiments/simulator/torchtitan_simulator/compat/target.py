# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Select override geometry from the requested logical device, not the host."""

import sys
from importlib import import_module


def install_target_selection(scope, target):
    if target != "non_a5":
        raise ValueError("A5 backend acceptance is pending; this runtime currently verifies non_a5")
    owner = import_module("torchtitan_npu.override")
    if not hasattr(owner, "_IS_A5"):
        raise RuntimeError("Native override device selection contract requires review")
    count = 0
    for name, module in list(sys.modules.items()):
        if (name == "torchtitan_npu.override" or name.startswith("torchtitan_npu.override.")) and (
            module is not None and "_IS_A5" in vars(module)
        ):
            scope.replace(module, "_IS_A5", False)
            count += 1
    return {"target_npu_device_type": target, "device_selection_bindings": count}
