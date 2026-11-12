# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the device-aware runtime estimation patch."""

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

_PATCH_PATH = (
    Path(__file__).resolve().parents[4] / "torchtitan_npu" / "patches" / "torch_npu" / "inductor_runtime_estimation.py"
)


def test_apply_routes_runtime_estimation_by_device():
    # PrivateUse1 backend renaming is process-global and cannot be undone.
    subprocess.run([sys.executable, __file__], check=True)


def _check_in_subprocess():
    import torch
    import torch._inductor
    import torch._inductor.utils
    import torch.utils._runtime_estimation

    torch.utils.rename_privateuse1_backend("npu")
    torch.npu = types.SimpleNamespace(is_available=lambda: False)
    current_device = [torch.device("npu:0")]
    torch.accelerator.current_accelerator = lambda **kwargs: current_device[0]
    device_calls = []
    gpu_calls = []
    compile_calls = []

    def original_device(device=None):
        device_calls.append(device)
        return 42.0

    def original_gpu():
        gpu_calls.append(True)
        return 84.0

    def original_compile(*args, **kwargs):
        compile_calls.append((args, kwargs))
        return "compiled"

    torch._inductor.utils.get_device_dram_gbps = original_device
    torch._inductor.utils.get_gpu_dram_gbps = original_gpu
    torch._inductor.standalone_compile = original_compile
    # Reproduce an already captured binding before applying the patch.
    torch.utils._runtime_estimation.get_device_dram_gbps = original_device
    spec = importlib.util.spec_from_file_location("runtime_estimation_patch", _PATCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert torch._inductor.utils.get_device_dram_gbps is original_device

    torch.npu.is_available = lambda: True
    module.apply()
    patched_device = torch._inductor.utils.get_device_dram_gbps
    patched_gpu = torch._inductor.utils.get_gpu_dram_gbps
    patched_compile = torch._inductor.standalone_compile
    module.apply()

    assert torch._inductor.utils.get_device_dram_gbps is patched_device
    assert torch.utils._runtime_estimation.get_device_dram_gbps is patched_device
    assert torch._inductor.utils.get_gpu_dram_gbps is patched_gpu
    assert torch._inductor.standalone_compile is patched_compile
    # The real consumer computes bytes divided by the fallback bandwidth.
    tensor = torch.ones(4, dtype=torch.float32)
    assert torch.utils._runtime_estimation.get_transfer_time([tensor], [], device="npu:0") == 16 / 1200
    assert patched_device(torch.device("npu:0")) == 1200
    assert patched_device() == 1200
    assert patched_gpu() == 1200
    assert device_calls == [] and gpu_calls == []

    assert patched_device("cpu") == 42.0
    assert patched_device(torch.device("cuda:0")) == 42.0
    current_device[0] = torch.device("cuda:0")
    assert patched_device() == 42.0
    assert patched_gpu() == 84.0
    assert device_calls == ["cpu", torch.device("cuda:0"), None]
    assert gpu_calls == [True]

    assert patched_compile("graph") == "compiled"
    assert patched_compile("graph", donate_graph_module=False) == "compiled"
    assert compile_calls == [
        (("graph",), {"donate_graph_module": True}),
        (("graph",), {"donate_graph_module": False}),
    ]


if __name__ == "__main__":
    _check_in_subprocess()
