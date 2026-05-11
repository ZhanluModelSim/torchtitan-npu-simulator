# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

import torchtitan.distributed.activation_checkpoint as activation_checkpoint_module

from torchtitan.config import ConfigManager
from torchtitan.tools.logging import init_logger, logger

from torchtitan_npu.patches.torchtitan.activation_checkpoint import (
    _patched_apply_full_ac,
)


def main() -> None:
    """Main entry point for NPU training with new config system."""
    init_logger()

    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer = None

    model_name = config.model_spec.name if config.model_spec else "unknown"

    from torchtitan.models.common import FlexAttention, ScaledDotProductAttention
    from torchtitan.models.common.decoder import Decoder

    if config.model_spec and isinstance(config.model_spec.model, Decoder.Config):
        for layer_cfg in config.model_spec.model.layers:
            if isinstance(layer_cfg.attention.inner_attention, FlexAttention.Config):
                layer_cfg.attention.inner_attention = ScaledDotProductAttention.Config()
                layer_cfg.attention.mask_type = "causal"
                logger.info(
                    "Replaced FlexAttention with ScaledDotProductAttention for NPU compatibility"
                )

    if config.compile.enable and config.activation_checkpoint.mode != "none":
        logger.warning(
            "There might be performance issues with activation checkpointing and torch.compile enabled!"
        )
    else:
        apply_full_ac_attr = "_apply_full_ac"
        setattr(
            activation_checkpoint_module, apply_full_ac_attr, _patched_apply_full_ac
        )

    if config.compile.enable:
        if model_name == "deepseek_v3":
            try:
                # pyrefly: ignore [missing-import]
                from torch_npu.op_plugin.meta._meta_registrations import (
                    npu_fusion_attention_forward as original_meta_func,
                )
            except ImportError:
                logger.info(
                    "torch_npu meta registrations not available, skipping compile patch"
                )
            else:
                from torchtitan_npu.patches.torch_npu._meta_registrations import (
                    npu_fusion_attention_forward,
                )

                original_meta_func.__code__ = npu_fusion_attention_forward.__code__

            try:
                # pyrefly: ignore [missing-import]
                import inductor_npu_ext  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    "compile.enable is True for deepseek_v3 model but inductor_npu_ext is not available. "
                    "Please install inductor_npu_ext before enabling compile. "
                    "See docs/torch_compile.md for installation instructions."
                ) from e

            converter_names = []
            if config.model_converters and hasattr(
                config.model_converters, "converters"
            ):
                converter_names = [
                    c if isinstance(c, str) else type(c).__name__
                    for c in config.model_converters.converters
                ]
            if "npu_bypass_triton_codegen" in converter_names:
                raise RuntimeError(
                    "deepseek_v3 model with compile.enable=True should not use npu_bypass_triton_codegen. "
                    "Please remove 'npu_bypass_triton_codegen' from model.converters in your config."
                )

    if model_name == "deepseek_v32":
        logger.warning(
            "deepseek_v32 patches are temporarily disabled due to config system migration. "
            "Some features (indexer loss, NPU memory config) may not work correctly."
        )

    if model_name == "llama4":
        from torchtitan_npu.tools.checkpoint_patch import (
            patch_llama4_checkpoint_support,
        )

        patch_llama4_checkpoint_support()

    if model_name == "deepseek_v3":
        logger.warning(
            "deepseek_v3 checkpoint patch is temporarily disabled due to config system migration."
        )

    try:
        trainer = config.build()

        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
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
