# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import get_spmd_backend, get_spmd_context, set_spmd_backend

from torchtitan_npu.models.deepseek_v4.config_registry import graph_trainer_deepseek_v4_debugmodel


def test_graph_trainer_spmd_types_initializes_singleton_model_parallel_axes(tmp_path, monkeypatch):
    monkeypatch.setattr("torchtitan.distributed.parallel_dims.device_type", "cpu")
    # EP annotations wrap shared dispatcher methods for the lifetime of the process.
    monkeypatch.setattr("torchtitan_npu.models.deepseek_v4.parallelize.annotate_moe_ep_regions", lambda: None)
    previous_backend = get_spmd_backend()
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", world_size=1, rank=0)
    try:
        set_spmd_backend("spmd_types")
        config = graph_trainer_deepseek_v4_debugmodel()
        config.parallelism.spmd_backend = "spmd_types"
        config.training.mixed_precision_param = "float32"
        config.model_spec.model.layers = config.model_spec.model.layers[:1]
        config.model_spec.model.n_layers = 1
        config.model_spec.model.compress_ratios = (1,)
        config.model_spec.model.vocab_size = 128
        config.model_spec.model.tok_embeddings.num_embeddings = 128
        config.model_spec.model.lm_head.out_features = 128
        config.model_spec.model.update_from_config(config=config)
        dims = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1)
        with get_spmd_context(parallel_dims=dims)():
            model = config.model_spec.model.build()
            model = config.model_spec.parallelize_fn(
                model,
                parallel_dims=dims,
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
    finally:
        set_spmd_backend(previous_backend)
        dist.destroy_process_group()
