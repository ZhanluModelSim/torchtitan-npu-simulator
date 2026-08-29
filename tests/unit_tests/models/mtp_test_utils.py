# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from unittest.mock import Mock, patch

import torch

BATCH_SIZE = 1
SEQ_LEN = 128
VOCAB_SIZE = 64
_DOCUMENT_LEN = SEQ_LEN // 2
PACKED_POSITIONS = torch.cat([torch.arange(_DOCUMENT_LEN), torch.arange(_DOCUMENT_LEN)]).unsqueeze(0)


@contextmanager
def _single_process_parallel_dims():
    mesh = Mock()
    mesh.size.return_value = 1
    parallel_dims = Mock()
    parallel_dims.get_optional_mesh.return_value = mesh
    with patch(
        "torchtitan_npu.patches.torchtitan.models.common.aux_loss.ParallelDims.get",
        return_value=parallel_dims,
    ):
        yield


def build_cpu_model(config):
    torch.manual_seed(42)
    with _single_process_parallel_dims():
        model = config.build()
        model.init_states(buffer_device=torch.device("cpu"))
    return model.train()


def assert_packed_mtp_training(model) -> None:
    from torchtitan.models.deepseek_v3.mtp import MTPLoss

    tokens = (torch.arange(SEQ_LEN) % VOCAB_SIZE).view(BATCH_SIZE, -1)
    positions = PACKED_POSITIONS.clone()
    labels = (tokens + 1) % VOCAB_SIZE
    tokens, labels, extra_kwargs = model.build_attention_masks(tokens, labels, {"positions": positions})
    output = model(tokens, positions, extra_kwargs["attention_masks"])

    if not isinstance(output, list):
        raise AssertionError(f"Expected MTP outputs as a list, got {type(output).__name__}.")
    if len(output) != 2:
        raise AssertionError(f"Expected two MTP outputs, got {len(output)}.")
    expected_shape = (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    if output[0].shape != expected_shape:
        raise AssertionError(f"Expected main output shape {expected_shape}, got {tuple(output[0].shape)}.")
    if output[1].shape != output[0].shape:
        raise AssertionError(
            f"Expected matching main and MTP output shapes, got {tuple(output[0].shape)} and {tuple(output[1].shape)}."
        )

    loss, _ = MTPLoss(MTPLoss.Config(mtp_scale=0.3, global_vocab_size=VOCAB_SIZE))(output, labels, positions=positions)
    if not torch.isfinite(loss).item():
        raise AssertionError(f"Expected a finite MTP loss, got {loss.item()}.")
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.mtp_layers.parameters()):
        raise AssertionError("Expected at least one MTP parameter to receive a gradient.")
