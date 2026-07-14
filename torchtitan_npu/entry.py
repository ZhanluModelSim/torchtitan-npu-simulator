# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import TYPE_CHECKING, cast

import torch
import torch_npu
from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger, logger

import torchtitan_npu  # noqa: F401
from torchtitan_npu.train import (
    _patch_for_garbage_collection_run,
    _patch_for_parallel_dims_build_mesh,
)

if TYPE_CHECKING:
    from torchtitan.trainer import Trainer

_SKIP_FLEX_TO_SDPA_REWRITE_MODELS = {"vlm"}


def main() -> None:
    """Main entry point for NPU training with new config system."""
    init_logger()

    config_manager = ConfigManager()
    config = cast("Trainer.Config", config_manager.parse_args())

    # Set NPU HF32 backend globals before trainer construction and operator execution.
    allow_hf32 = getattr(config.training, "allow_hf32", True)
    torch_npu.npu.matmul.allow_hf32 = allow_hf32
    torch_npu.npu.conv.allow_hf32 = allow_hf32
    torch_npu.npu.aclnn.allow_hf32 = allow_hf32

    trainer = None

    comm_mode = os.getenv("COMM_MODE", None)
    if comm_mode not in ("fake_backend", "local_tensor"):
        # only when COMM_MODE is not set, patch is apply
        _patch_for_garbage_collection_run()
        _patch_for_parallel_dims_build_mesh()

    model_name = config.model_spec.name if config.model_spec else "unknown"

    from torchtitan.models.common import FlexAttention, ScaledDotProductAttention
    from torchtitan.models.common.decoder import Decoder

    if (
        config.model_spec
        # Some models replace FlexAttention in their own parallelize path because
        # they require model-specific dense masks instead of the default causal mask.
        and model_name not in _SKIP_FLEX_TO_SDPA_REWRITE_MODELS
        and isinstance(
            config.model_spec.model,
            Decoder.Config,
        )
    ):
        for layer_cfg in config.model_spec.model.layers:
            if isinstance(layer_cfg.attention.inner_attention, FlexAttention.Config):
                layer_cfg.attention.inner_attention = ScaledDotProductAttention.Config()
                layer_cfg.attention.mask_type = "causal"
                logger.info("Replaced FlexAttention with ScaledDotProductAttention for NPU compatibility")

    if config.compile.enable and config.activation_checkpoint.mode != "none":
        logger.warning("There might be performance issues with activation checkpointing and torch.compile enabled!")

    if config.compile.enable:
        if model_name == "deepseek_v3":
            # MLA performs shape inference according to the value tensor;
            # patch the meta registration so dynamo traces the right shapes.
            try:
                # pyrefly: ignore [missing-import]
                from torch_npu.op_plugin.meta._meta_registrations import (
                    npu_fusion_attention_forward as original_meta_func,
                )
            except ImportError:
                logger.info("torch_npu meta registrations not available, skipping compile patch")
            else:
                from torchtitan_npu.patches.torch_npu._meta_registrations import (
                    npu_fusion_attention_forward,
                )

                original_meta_func.__code__ = npu_fusion_attention_forward.__code__

        try:
            # pyrefly: ignore [missing-import]
            import inductor_npu_ext  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                f"compile.enable is True for {model_name} model but inductor_npu_ext is not available. "
                "Please install inductor_npu_ext before enabling compile. "
                "See docs/torch_compile.md for installation instructions."
            ) from e

    if model_name in ("deepseek_v32", "deepseek_v4"):
        from torchtitan_npu.train import (
            _patch_init_for_dsa_set_loss_scale,
            _patch_train_step_for_dsa_indexer_loss,
        )

        _patch_train_step_for_dsa_indexer_loss()
        _patch_init_for_dsa_set_loss_scale()

    if model_name == "deepseek_v3":
        logger.warning("deepseek_v3 checkpoint patch is temporarily disabled due to config system migration.")

    try:
        trainer = config.build()

        if config.checkpoint.create_seed_checkpoint:
            assert int(os.environ["WORLD_SIZE"]) == 1, (
                "Must create seed checkpoint using a single device, to disable sharding."
            )
            assert config.checkpoint.enable, "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main()
