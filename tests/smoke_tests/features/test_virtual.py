# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from types import SimpleNamespace

import pytest
import torch.nn as nn
import torchtitan.components.optimizer as tt_optimizer

from torchtitan_npu.patches.optimizer.virtual_optimizer import virtual_optimizer_step

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def _reset_virtual_optimizer_global_state():
    yield


def _virtual_optimizer_config():
    return SimpleNamespace(
        virtual_optimizer=True,
        virtual_optimizer_size=10.0,
        swap_optimizer=False,
        name="AdamW",
        lr=1e-3,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.01,
        implementation="fused",
        early_step_in_backward=False,
        loss_scale=None,
        dtype=None,
        gradient_clipping=None,
    )


def test_virtual_optimizer_builds_optimizer_correctly(npu_device):
    model = nn.Linear(32, 32).to(npu_device)

    container = tt_optimizer.build_optimizers(
        model_parts=[model],
        optimizer_config=_virtual_optimizer_config(),
        parallel_dims=object(),
        ft_manager=None,
    )
    optimizer = container.optimizers[0]

    assert len(container.optimizers) == 1
    assert optimizer.step.__name__ == virtual_optimizer_step.__name__
    assert hasattr(optimizer, "_allocator_config")


def test_virtual_optimizer_unsupported_swap_optimizer(npu_device):
    model = nn.Linear(8, 8).to(npu_device)
    cfg = _virtual_optimizer_config()
    cfg.swap_optimizer = True

    with pytest.raises(
        ValueError, match="Virtual optimizer does not support swap_optimizer"
    ):
        tt_optimizer.build_optimizers(
            model_parts=[model],
            optimizer_config=cfg,
            parallel_dims=object(),
            ft_manager=None,
        )
