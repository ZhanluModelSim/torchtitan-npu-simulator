# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    annotate_moe_ep_regions,
    apply_simple_fsdp,
)
from torchtitan.experiments.graph_trainer.compile import (
    apply_compile as apply_graph_trainer_compile,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3

from torchtitan_npu.models.deepseek_v4.model import GraphTrainerDeepSeekV4Model

# The standard (eager) path reuses the DeepSeek V3 parallelization, which the
# sparse-attention sharding is built on. The GraphTrainer path below is DSV4
# specific because it must not reorder sparse-attention compute under FSDP.
parallelize_deepseek_v4 = parallelize_deepseekv3


def annotate_deepseek_v4(model: GraphTrainerDeepSeekV4Model) -> None:
    """Attach annotations to FX graph nodes for DeepSeek V4.

    - Expert Parallel (EP) annotations: Tags "dispatch", "combine", and "compute"
      regions in MoE for debugging purposes.
    - Module FQN annotation: Tags each submodule's forward with its
      fully-qualified name for downstream passes (bucketing, SAC region
      boundaries, etc.).
    """
    annotate_moe_ep_regions()
    annotate_module_fqns(model)


def parallelize_graph_trainer_deepseek_v4(
    model: GraphTrainerDeepSeekV4Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    # The graph_trainer simple_fsdp wrapper is built on raw DTensor ops
    # (``distribute_tensor`` / ``tensor._spec``) and errors under the
    # spmd_types backend; only the eager path supports that backend for now.
    assert get_spmd_backend() != "spmd_types", "The GraphTrainer path does not yet support the spmd_types backend."

    # TP currently cannot handle uneven seq_len because we set
    # ``use_local_output=True`` to use plain Tensors for legacy reasons.
    assert training.seq_len % parallel_dims.seq_len_divisor == 0, f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}), i.e. {parallel_dims.seq_len_divisor}.
        """

    # DeepSeek V4 sparse attention does not yet support context parallelism.
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Context Parallel is not yet supported for DeepSeek V4 sparse attention in the GraphTrainer path."
        )

    annotate_deepseek_v4(model)

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

    # Apply simple_fsdp unconditionally. The ``fsdp`` mesh always exists with a
    # real backend (see ParallelDims._mesh_exist), even at degree 1, so that
    # MixedPrecisionPolicy's param_dtype cast still applies in single-GPU runs.
    # pyrefly: ignore [bad-assignment]
    model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)

    # Apply compilation based on mode
    # pyrefly: ignore [bad-assignment]
    model = apply_graph_trainer_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )

    return model
