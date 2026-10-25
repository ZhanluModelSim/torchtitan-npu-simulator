# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native training configuration factories for the Kimi K3 text decoder."""

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import default_adamw, register_moe_load_balancing_hook
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from . import kimi_k3_configs
from .parallelize import parallelize_kimi_k3
from .state_dict_adapter import KimiK3StateDictAdapter


def _trainer_config(flavor: str, *, num_mtp_layers: int = 0) -> Trainer.Config:
    if num_mtp_layers:
        raise ValueError("The migrated Kimi K3 text model has no MTP modules")
    model = kimi_k3_configs[flavor]()
    return Trainer.Config(
        model_spec=ModelSpec(
            name="kimi_k3",
            flavor=flavor,
            model=model,
            parallelize_fn=parallelize_kimi_k3,
            pipelining_fn=None,
            post_optimizer_build_fn=register_moe_load_balancing_hook,
            state_dict_adapter=KimiK3StateDictAdapter,
        ),
        loss=ChunkedLossWrapper.Config(loss_fn=CrossEntropyLoss.Config(global_vocab_size=model.vocab_size)),
        optimizer=default_adamw(lr=8e-4),
        parallelism=ParallelismConfig(context_parallel_load_balancer=None),
        training=TrainingConfig(local_batch_size=1, seq_len=128, steps=1, dtype="float32"),
    )


def kimi_k3_debugmodel(*, num_mtp_layers: int = 0) -> Trainer.Config:
    return _trainer_config("debug", num_mtp_layers=num_mtp_layers)


def kimi_k3_full(*, num_mtp_layers: int = 0) -> Trainer.Config:
    return _trainer_config("full", num_mtp_layers=num_mtp_layers)


def kimi_k3_16layer_reduced(*, num_mtp_layers: int = 0) -> Trainer.Config:
    return _trainer_config("16layer_reduced", num_mtp_layers=num_mtp_layers)
