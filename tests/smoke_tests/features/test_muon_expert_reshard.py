# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.multiprocessing as mp

from tests.testing.muon_reshard import run_npu_expert_matrix_worker, run_npu_expert_strided_worker

pytestmark = pytest.mark.smoke


def test_expert_matrix_reshard_on_two_npus(tmp_path):
    if not hasattr(torch, "npu") or torch.npu.device_count() < 2:
        pytest.skip("requires two real NPU devices")
    mp.spawn(run_npu_expert_matrix_worker, args=(f"file://{tmp_path / 'hccl_store'}",), nprocs=2, join=True)


def test_expert_strided_dim0_on_four_npus(tmp_path):
    if not hasattr(torch, "npu") or torch.npu.device_count() < 4:
        pytest.skip("requires four real NPU devices")
    mp.spawn(run_npu_expert_strided_worker, args=(f"file://{tmp_path / 'hccl_strided_store'}",), nprocs=4, join=True)
