# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch._dynamo.config
import torch._inductor.config
from torch.distributed.tensor import DTensor

from torchtitan_npu.models.deepseek_v4.parallelize import _chunk_dsv4_moe_forward
from torchtitan_npu_recipes.graph_trainer.deepseek_v4 import graph_trainer_deepseek_v4_debugmodel


def test_graph_trainer_parallelize_initializes_singleton_model_parallel_axes(tmp_path, monkeypatch, single_rank_spmd):
    # EP annotations wrap shared dispatcher methods for the lifetime of the process.
    monkeypatch.setattr("torchtitan_npu.models.deepseek_v4.parallelize.annotate_moe_ep_regions", lambda: None)
    # The production compiler sets these flags; restore their caller-owned values after the test.
    monkeypatch.setattr(torch._dynamo.config, "capture_scalar_outputs", torch._dynamo.config.capture_scalar_outputs)
    monkeypatch.setattr(
        torch._inductor.config, "reorder_for_peak_memory", torch._inductor.config.reorder_for_peak_memory
    )
    config = graph_trainer_deepseek_v4_debugmodel()
    config.training.mixed_precision_param = "float32"
    config.model.layers = config.model.layers[:1]
    config.model.n_layers = 1
    config.model.compress_ratios = (1,)
    config.model.vocab_size = 128
    config.model.tok_embeddings.num_embeddings = 128
    config.model.lm_head.out_features = 128
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = config.model.build()
    model = model.parallelize(
        parallelism_context=single_rank_spmd,
        training=config.training,
        parallelism=config.parallelism,
        compile_config=config.compile,
        activation_remat=config.activation_remat,
        dump_folder=str(tmp_path),
    )
    projection = model.layers["0"].attention.wq_a
    assert projection._parallelized
    assert isinstance(projection._parameters["weight"], DTensor)
    assert not isinstance(projection.weight, DTensor)
    projection.weight.square().sum().backward()
    assert isinstance(projection._parameters["weight"].grad, DTensor)


@pytest.mark.parametrize("hash_routing", [True, False], ids=["hash-routing", "learned-routing"])
def test_graph_trainer_moe_chunks_share_the_aux_loss_denominator(hash_routing):
    seen = []
    denominator = torch.tensor(16)
    tokens = torch.arange(4)
    mask = torch.tensor([False, False, False, True])

    def forward(x, *, padding_mask_T, aux_loss_denominator, **router_kwargs):
        seen.append(aux_loss_denominator)
        assert ("input_ids_T" in router_kwargs) == hash_routing
        ids = router_kwargs.get("input_ids_T", torch.zeros_like(padding_mask_T))
        return x + ids[:, None] - padding_mask_T[:, None].int()

    chunked = _chunk_dsv4_moe_forward(forward, "layers.0.feed_forward")
    x = torch.zeros(4, 1)
    router_kwargs = {"input_ids_T": tokens} if hash_routing else {}
    result = chunked(x, padding_mask_T=mask, aux_loss_denominator=denominator, **router_kwargs)

    assert len(seen) == 2
    assert all(value is denominator for value in seen)
    expected = [[0.0], [1.0], [2.0], [2.0]] if hash_routing else [[0.0], [0.0], [0.0], [-1.0]]
    torch.testing.assert_close(result, torch.tensor(expected))
