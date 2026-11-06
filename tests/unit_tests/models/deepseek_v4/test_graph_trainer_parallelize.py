# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch._dynamo.config
import torch._inductor.config
from torch.distributed.tensor import DTensor

from torchtitan_npu.models.deepseek_v4.config_registry import graph_trainer_deepseek_v4_debugmodel


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
    config.model.update_from_config(config=config)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        model = config.model.build()
    model = model.parallelize(
        parallelism_context=single_rank_spmd,
        training=config.training,
        parallelism=config.parallelism,
        compile_config=config.compile,
        ac_config=config.activation_checkpoint,
        dump_folder=str(tmp_path),
    )
    projection = model.layers["0"].attention.wq_a
    assert projection._parallelized
    assert isinstance(projection._parameters["weight"], DTensor)
    assert not isinstance(projection.weight, DTensor)
    projection.weight.square().sum().backward()
    assert isinstance(projection._parameters["weight"].grad, DTensor)
