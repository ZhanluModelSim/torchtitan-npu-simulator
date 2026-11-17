# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch

BATCH_SIZE = 1
SEQ_LEN = 128
VOCAB_SIZE = 64
_DOCUMENT_LEN = SEQ_LEN // 2
PACKED_POSITIONS = torch.cat([torch.arange(_DOCUMENT_LEN), torch.arange(_DOCUMENT_LEN)]).unsqueeze(0)

# ``preprocess_inputs`` (the seam under test) takes the parallelism context;
# these models run a single process with no CP and the default SPMD backend.
_NO_CP_DIMS = SimpleNamespace(cp_enabled=False, get_mesh=lambda name: None)
_NO_SPMD = SimpleNamespace(
    spmd_backend="partial_dtensor",
    context_parallel_load_balancer="headtail",
    context_parallel_ptrr_mask_key=None,
)


def build_cpu_model(config):
    torch.manual_seed(42)
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    return model.train()


def preprocess(model, tokens, labels, positions):
    """Run the model's own input preprocessing, whatever seam it uses.

    Migrated stacks hand back an ``attention_metadata`` map, the others the
    pinned single ``attention_masks``; the forward takes either.
    """
    inputs, labels, extra_kwargs = model.preprocess_inputs(
        {"input": tokens, "positions": positions},
        labels,
        parallel_dims=_NO_CP_DIMS,
        device=tokens.device,
        parallelism=_NO_SPMD,
    )
    assert "attention_masks" in extra_kwargs or "attention_metadata" in extra_kwargs, (
        "expected the model to build attention metadata"
    )
    return inputs, labels, extra_kwargs


def assert_packed_mtp_training(model) -> None:
    from torchtitan.models.deepseek_v3.mtp import MTPLoss

    tokens = (torch.arange(SEQ_LEN) % VOCAB_SIZE).view(BATCH_SIZE, -1)
    positions = PACKED_POSITIONS.clone()
    labels = (tokens + 1) % VOCAB_SIZE
    tokens, labels, extra_kwargs = preprocess(model, tokens, labels, positions)
    output = model(tokens, **extra_kwargs)

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
