# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan.models.deepseek_v3.parallelize as titan_deepseekv3_parallelize
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)

from torchtitan.distributed import ParallelDims
from torchtitan.models.deepseek_v3 import DeepSeekV3Model
from torchtitan.protocols import ModelConvertersContainer


def _is_ulysses_enabled(parallelism: ParallelismConfig) -> bool:
    return (
        getattr(parallelism, "enable_custom_context_parallel", False)
        and parallelism.context_parallel_degree > 1
    )


def parallelize_deepseekv3(
    model: DeepSeekV3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    def _call_upstream_parallelize():
        return titan_deepseekv3_parallelize.parallelize_deepseekv3(
            model,
            parallel_dims=parallel_dims,
            training=training,
            model_converters=model_converters,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )

    if parallel_dims.cp_enabled and _is_ulysses_enabled(parallelism):
        cp_mesh = parallel_dims.get_mesh("cp")
        model_config = model.model_config if hasattr(model, "model_config") else None

        n_heads = model_config.layers[0].attention.n_heads if model_config else 128
        cp_degree = cp_mesh.size()

        if n_heads % cp_degree != 0:
            raise ValueError(
                f"[Ulysses CP] n_heads={n_heads} must be divisible by "
                f"context_parallel_degree={cp_degree}."
            )

        from torchtitan.distributed.context_parallel import apply_cp_to_attention_module

        orig_apply_cp = titan_deepseekv3_parallelize.apply_cp_to_attention_module

        def _force_ulysses_cp(attention_modules, cp_mesh):
            apply_cp_to_attention_module(
                attention_modules,
                cp_mesh,
                attention_type="ulysses",  # pyrefly: ignore [unexpected-keyword]
                model_args=getattr(
                    model, "model_args", None
                ),  # pyrefly: ignore [unexpected-keyword]
                tp_mesh=parallel_dims.get_mesh(
                    "tp"
                )  # pyrefly: ignore [unexpected-keyword]
                if parallel_dims.tp_enabled
                else None,
            )

        titan_deepseekv3_parallelize.apply_cp_to_attention_module = _force_ulysses_cp
        try:
            return _call_upstream_parallelize()
        finally:
            titan_deepseekv3_parallelize.apply_cp_to_attention_module = orig_apply_cp

    return _call_upstream_parallelize()
