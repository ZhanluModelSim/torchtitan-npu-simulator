# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Whether a fused DeepSeek stack can be compiled at this degree, and what to say when it cannot.

arch22's fused attention and selector size their kernel options by a **device** row count -- the key
compaction is a ``nonzero`` -- so a compiled region cannot guard on it
(``Could not guard on data-dependent expression``, raised from inside the vendor op's meta
function).  There is nowhere to break the graph either: torchtitan compiles every TransformerBlock
with ``fullgraph=True``, and under full activation checkpointing the block is traced inside the
``checkpoint`` higher-order operator, which admits no break of its own.

arch35 reads ``seqused``, its metadata carries no device row count, and it compiles as usual; a
single degree moves nothing, so ``CP=1`` compiles on every architecture.

So the stack states the fact where its degree and its arch are both known -- in the model's
``parallelize`` -- and the flag stays accepted, which is what lets one recipe serve both
architectures: on A5 it compiles, on arch22 it warns once and runs eagerly.
"""

import logging
from dataclasses import replace

import torch._dynamo.config

from torchtitan_npu.patches.torch_npu.device import is_a5

logger = logging.getLogger(__name__)

__all__ = ["compiles_under_cp", "skip_compile_under_cp"]


def compiles_under_cp(parallelism_context, *, a5: bool | None = None) -> bool:
    """Whether this run's blocks can each be one graph (see the module docstring).

    ``a5`` states the architecture; ``None`` asks the device.  The parameter exists so the
    decision can be pinned in a test without an NPU.
    """
    return parallelism_context.cp <= 1 or (is_a5() if a5 is None else a5)


def skip_compile_under_cp(model, compile_config, parallelism_context, *, a5: bool | None = None):
    """Drop only requested model compilation when CP metadata cannot be traced."""
    if (
        compile_config is None
        or getattr(compile_config, "model", None) is None
        or compiles_under_cp(parallelism_context, a5=a5)
    ):
        return compile_config
    logger.warning(
        "Skipping torch.compile for %s: arch22 fused kernels read device row counts "
        "under context parallelism. Running the model eagerly.",
        type(model).__name__,
    )
    return replace(compile_config, model=None)


def configure_model_compile() -> None:
    """Retain the upstream compiler settings for token-choice MoE and AC."""
    torch._dynamo.config.capture_scalar_outputs = True
    # As in the pre-migration apply_compile, balanced Python mesh scopes and
    # cache updates are traced once rather than replayed by compiled AC.
    setattr(torch._dynamo.config, "skip_fwd_side_effects_in_bwd_under_checkpoint", True)  # noqa: B010


def compile_transformer_blocks(model, model_compile) -> None:
    """Compile decoder blocks through the NPU-owned model compile policy."""
    configure_model_compile()
    for block in model.layers.values():
        block.compile(backend=model_compile.backend, fullgraph=model_compile.fullgraph)


def parallelize_with_model_compile(
    model,
    *,
    parallelism_context,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
    skip_dp=False,
):
    """Run the upstream model lifecycle with the explicit NPU compile component."""
    with parallelism_context.activate_spmd():
        model._parallelize(parallelism_context)
        if ac_config is not None:
            policy = ac_config.build(dump_folder=dump_folder)
            policy.apply(model)
            apply_extensions = getattr(model, "apply_activation_checkpointing_extensions", None)
            if apply_extensions is not None:
                apply_extensions(policy)
            for fqn in getattr(model, "multimodal_encoder_fqns", ()):
                encoder = getattr(model, fqn)
                if encoder is not None:
                    policy.apply(encoder)
        model_compile = getattr(compile_config, "model", None)
        if model_compile is not None:
            compiler = getattr(model, "apply_model_compile", None)
            if compiler is None:
                compile_transformer_blocks(model, model_compile)
            else:
                compiler(model_compile)
        if not skip_dp:
            model._apply_fsdp(parallelism_context=parallelism_context, training=training, parallelism=parallelism)
    return model
