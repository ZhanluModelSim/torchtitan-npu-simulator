# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

from torchtitan.config.parallelism import ParallelismConfig

import torch
from torchtitan.distributed.spmd_types import spmd_dense_mesh

BATCH_SIZE = 1
# The DSV4 stack aligns every packed document to lcm(compress_ratios) -- 128 for
# the debug stack's (1, 4, 128) -- so the two packed documents are whole 128-token
# multiples (they used to be two 64-token documents, which the aligned metadata
# contract refuses).
SEQ_LEN = 256
VOCAB_SIZE = 64
_DOCUMENT_LEN = SEQ_LEN // 2
# The folded-token form: one flat packed row, two documents, positions resetting
# at the document boundary.
PACKED_POSITIONS = torch.cat([torch.arange(_DOCUMENT_LEN), torch.arange(_DOCUMENT_LEN)])

# ``preprocess_inputs`` (the seam under test) takes the parallelism context;
# these models run a single process with no CP and the default SPMD backend.
_NO_CP_CONTEXT = SimpleNamespace(cp_enabled=False, get_mesh=lambda name: None, spmd_dense_mesh=spmd_dense_mesh)
_PARALLELISM = ParallelismConfig()


def build_cpu_model(config):
    torch.manual_seed(42)
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    return model.train()


def preprocess(model, tokens, labels, positions):
    """Run the model's own input preprocessing, whatever seam it uses.

    Migrated stacks hand back an ``attention_metadata`` map, the others the
    pinned single ``attention_masks``; the forward takes either.  The labels
    ride inside the batch dict, as the 0.4.0 trainer hands it over.
    """
    inputs, labels, extra_kwargs = model.preprocess_inputs(
        {"input": tokens, "labels": labels, "positions": positions},
        parallelism_context=_NO_CP_CONTEXT,
        parallelism=_PARALLELISM,
    )
    assert "attention_masks" in extra_kwargs or "attention_metadata" in extra_kwargs, (
        "expected the model to build attention metadata"
    )
    return inputs, labels, extra_kwargs


def assert_packed_mtp_training(model) -> None:
    from torchtitan.models.deepseek_v3.mtp import MTPLoss

    tokens = torch.arange(SEQ_LEN) % VOCAB_SIZE
    positions = PACKED_POSITIONS.clone()
    labels = (tokens + 1) % VOCAB_SIZE
    tokens, labels, extra_kwargs = preprocess(model, tokens, labels, positions)
    output = model(tokens, **extra_kwargs)

    if not isinstance(output, list):
        raise AssertionError(f"Expected MTP outputs as a list, got {type(output).__name__}.")
    if len(output) != 2:
        raise AssertionError(f"Expected two MTP outputs, got {len(output)}.")
    expected_shape = (SEQ_LEN, VOCAB_SIZE)
    if output[0].shape != expected_shape:
        raise AssertionError(f"Expected main output shape {expected_shape}, got {tuple(output[0].shape)}.")
    if output[1].shape != output[0].shape:
        raise AssertionError(
            f"Expected matching main and MTP output shapes, got {tuple(output[0].shape)} and {tuple(output[1].shape)}."
        )

    prepared = extra_kwargs["mtp_batch"].labels
    aligned_labels = (labels, *prepared.unbind(dim=-1))
    loss, _ = MTPLoss(MTPLoss.Config(mtp_scale=0.3, global_vocab_size=VOCAB_SIZE))(tuple(output), aligned_labels)
    if not torch.isfinite(loss).item():
        raise AssertionError(f"Expected a finite MTP loss, got {loss.item()}.")
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.mtp_layers.parameters()):
        raise AssertionError("Expected at least one MTP parameter to receive a gradient.")


def build_v4_model_config(
    *,
    compress_ratios=(1, 1, 1),
    rope_head_dim=4,
    original_seq_len=SEQ_LEN,
    num_mtp_layers=1,
    n_hash_layers=0,
):
    from torchtitan_npu.models.deepseek_v4 import _make_v4_config

    return _make_v4_config(
        dim=32,
        n_layers=3,
        vocab_size=VOCAB_SIZE,
        n_heads=4,
        head_dim=8,
        rope_head_dim=rope_head_dim,
        q_lora_rank=16,
        o_lora_rank=8,
        n_groups=2,
        compress_ratios=compress_ratios,
        window_size=SEQ_LEN,
        norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        moe_inter_dim=16,
        num_experts=2,
        num_shared_experts=1,
        top_k=1,
        n_hash_layers=n_hash_layers,
        route_norm=False,
        route_scale=1.0,
        load_balance_coeff=1e-3,
        hc_mult=2,
        sinkhorn_iters=2,
        hc_eps=1e-6,
        max_seq_len=SEQ_LEN,
        compress_rope_theta=10000.0,
        original_seq_len=original_seq_len,
        num_mtp_layers=num_mtp_layers,
    )
