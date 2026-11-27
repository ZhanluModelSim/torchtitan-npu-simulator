# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention.cp_attention import CPInnerAttention

from torchtitan_npu.models.common.metadata import KernelFrame
from torchtitan_npu.override.deepseek_v4.cp_metadata import DocumentCPLoadBalancer, DocumentCPPlan


@pytest.mark.parametrize(
    "head_tail, owned, windows, reaches",
    [
        (False, [0, 1, 2, 3, 8, 9, 10, 11], [[0, 4, 8]], [[2, 2]]),
        (True, [0, 1, 8, 9, 6, 7, 14, 15], [[0, 2, 4], [0, 4, 8]], [[1, 1], [4, 4]]),
    ],
    ids=["chunk", "head-tail"],
)
def test_document_permutation_builds_the_frames_read_by_sparse_kernels(head_tail, owned, windows, reaches):
    offsets = torch.tensor([0, 8, 16], dtype=torch.int32)
    row = SimpleNamespace(
        ratio=2,
        q=KernelFrame(cu_seqlens=offsets, seqused=offsets.diff()),
        cmp_k=KernelFrame(cu_seqlens=offsets // 2, seqused=offsets.diff() // 2),
        position_k=torch.arange(8).repeat(2),
    )
    layout = DocumentCPLoadBalancer.Config(head_tail=head_tail).build(
        seq_len=16, cp_size=2, attention_metadata=SimpleNamespace(row=row)
    )
    permutation = layout.generate_permutation()
    assert permutation[0, :8].tolist() == owned
    plan = DocumentCPPlan.build(permutation, offsets, cp_size=2, cp_rank=0, head_tail=head_tail)
    metadata = plan.prepare(row, window_size=3, overlap=False, build_entry=SimpleNamespace)
    assert [entry.swa_k.cu_seqlens.tolist() for entry in metadata.entries] == windows
    assert [entry.cmp_k.seqused.tolist() for entry in metadata.entries] == reaches


def test_global_head_tail_balancer_is_rejected():
    from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer

    from torchtitan_npu.override.deepseek_v4.context_parallel import configure_context_parallel

    parallelism = ParallelismConfig(context_parallel_degree=2)
    parallelism.context_parallel_load_balancer = HeadTailCPLoadBalancer.Config()
    with pytest.raises(ValueError, match="DocumentCPLoadBalancer"):
        configure_context_parallel(None, parallelism)


@pytest.mark.parametrize("family", ["v4", "v41"])
@pytest.mark.parametrize("cp_degree", [1, 2])
def test_recipe_selects_cp_inner_attention(family, cp_degree):
    from torchtitan_npu_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel
    from torchtitan_npu_recipes.tests.models.deepseek_v4_1 import _make_trainer_config

    parallelism = ParallelismConfig(context_parallel_degree=cp_degree, data_parallel_shard_degree=1)
    if family == "v4":
        config = deepseek_v4_debugmodel(parallelism=parallelism)
    else:
        config = _make_trainer_config(
            "deepseek_v4_1_debugmodel_text",
            parallelism=parallelism,
            engram_enabled=False,
        )
    assert all(
        isinstance(layer.attention.inner_attention, CPInnerAttention.Config) == (cp_degree > 1)
        for layer in config.model.layers
    )
    if family == "v41" and cp_degree > 1:
        with torch.device("meta"):
            model = config.model.build()
        for layer in model.layers.values():
            inner_attention = layer.attention.inner_attention
            assert inner_attention.aux_loss is inner_attention.inner.aux_loss
