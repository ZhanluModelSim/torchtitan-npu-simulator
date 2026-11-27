# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

from tests.integration_tests import OverrideDefinitions


def build_deepseek_cp_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            test_name=f"{model}_cp2_{layout}",
            test_descr=f"Fused {model} CP2 with {layout} document layout",
            ngpu=2,
            python_config=True,
            override_args=[()],
            env_vars={"MODULE": __name__, "CONFIG": f"{model}_cp2_{layout}"},
            expected_steps=((1, 2),),
            use_golden=False,
            check_loss=False,
        )
        for model in ("dsv4", "dsv41")
        for layout in ("chunk",)
    ]


def _config(family):
    from torchtitan.components.optim import AdamW
    from torchtitan.config.parallelism import ParallelismConfig

    from torchtitan_npu.config import OptimizerConfig
    from torchtitan_npu_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel
    from torchtitan_npu_recipes.tests.models.deepseek_v4_1 import _make_trainer_config

    parallelism = ParallelismConfig(
        context_parallel_degree=2,
        context_parallel_load_balancer=None,
        data_parallel_shard_degree=1,
        expert_parallel_degree=1,
    )
    if family == "deepseek_v4":
        config = deepseek_v4_debugmodel(parallelism=parallelism, seq_len=2048)
        geometry = {"num_heads": 16, "index_num_heads": 8}
    else:
        config = _make_trainer_config(
            "deepseek_v4_1_flash_40layers_16experts_text",
            parallelism=parallelism,
            seq_len=2048,
            hf_assets_path="tests/assets/deepseek_v3",
            engram_enabled=False,
        )
        # Keep the released fused-kernel widths and include both source and reuse layers.
        model = config.model
        model.layers = model.layers[:4]
        model.n_layers = len(model.layers)
        model.compress_ratios = model.compress_ratios[: model.n_layers]
        model.kv_source_layers = (2,)
        model.index_source_layers = (2,)
        geometry = {}
    config.override.imports += [
        (f"torchtitan_npu.override.{family}.sparse_attn.asc", geometry),
        f"torchtitan_npu.override.{family}.lightning_indexer.asc",
    ]
    config.optim.optimizer = OptimizerConfig(optimizers=[AdamW.Config(pattern=".*", lr=1e-4, fused=True)])
    config.training.steps = 2
    config.training.num_tokens_per_train_step = config.training.max_context_length
    config.metrics.log_freq = 1
    config.metrics.enable_tensorboard = True
    config.metrics.save_tb_folder = "tb_phase_0"
    return config


dsv4_cp2_chunk = partial(_config, "deepseek_v4")
dsv41_cp2_chunk = partial(_config, "deepseek_v4_1")
