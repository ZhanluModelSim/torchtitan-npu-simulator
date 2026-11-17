# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run with pytest on an Ascend 950 with the repository runtime installed."""

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file


@pytest.mark.smoke
@pytest.mark.parametrize("fp4", [False, True], ids=["mxfp8", "mxfp4"])
def test_mx_hf_loading_on_npu(tmp_path, fp4):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("NPU unavailable")
    device_name = torch_npu.npu.get_device_name()
    if not device_name.startswith("Ascend950"):
        pytest.skip(f"AntiMxQuant requires Ascend 950; found {device_name}")
    import torchtitan_npu  # noqa: F401
    from torchtitan_npu.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter

    from torchtitan_npu.extensions.mx_storage_reader.hf_storage import MXHuggingFaceStorageReader

    qdata = torch.full((2, 48), 0xE2, dtype=torch.uint8) if fp4 else torch.ones((2, 96)).to(torch.float8_e4m3fn)
    scale = torch.tensor([[126, 127, 128], [129, 128, 127]], dtype=torch.uint8)
    save_file({"linear.weight": qdata, "linear.scale": scale}, tmp_path / "model.safetensors")
    adapter = object.__new__(DeepSeekV4StateDictAdapter)
    reader = adapter.get_hf_storage_reader(str(tmp_path), from_quantized=True)
    assert isinstance(reader, MXHuggingFaceStorageReader)
    target = {"linear.weight": torch.empty((2, 96), device="npu", dtype=torch.bfloat16)}

    dcp.load(target, storage_reader=reader)

    values = torch.tensor([1., -4.]).repeat(2, 48) if fp4 else torch.ones(2, 96)
    expected = values * torch.tensor([[.5, 1., 2.], [4., 2., 1.]]).repeat_interleave(32, -1)
    torch.testing.assert_close(target["linear.weight"].cpu(), expected.bfloat16(), rtol=0, atol=0)
