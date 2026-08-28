# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    to_graph_trainer_config,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.profiler import Profiler
from torchtitan.trainer import Trainer

from . import (
    memory_policy,  # noqa: F401
    model_registry,
)
from .model import GraphTrainerDeepSeekV4Model
from .parallelize import parallelize_graph_trainer_deepseek_v4


def _make_trainer_config(
    flavor: str,
    *,
    local_batch_size: int,
    seq_len: int,
) -> Trainer.Config:
    model_spec = model_registry(flavor)
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        profiler=Profiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_spec,
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=default_adamw(
            lr=1e-5,
            eps=1e-6,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=20,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.01,
        ),
        training=TrainingConfig(
            local_batch_size=local_batch_size,
            seq_len=seq_len,
            steps=100,
        ),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
            fsdp_reshard_after_forward="always",
        ),
        activation_checkpoint=FullAC.Config(),
        compile=CompileConfig(enable=False),
        checkpoint=CheckpointManager.Config(
            enable=False,
            interval=100,
        ),
    )


def deepseek_v4_debugmodel() -> Trainer.Config:
    return _make_trainer_config("debugmodel", local_batch_size=1, seq_len=2048)


def deepseek_v4_flash() -> Trainer.Config:
    return _make_trainer_config("deepseek_v4_flash", local_batch_size=1, seq_len=4096)


def deepseek_v4_flash_43layers_16experts() -> Trainer.Config:
    return _make_trainer_config(
        "deepseek_v4_flash_43layers_16experts",
        local_batch_size=1,
        seq_len=4096,
    )


def deepseek_v4_pro() -> Trainer.Config:
    return _make_trainer_config("deepseek_v4_pro", local_batch_size=1, seq_len=4096)


# --- GraphTrainer config factories ---


def _graph_trainer_model_registry(flavor: str) -> ModelSpec:
    """Build a ModelSpec for the graph_trainer DeepSeek V4 path.

    Wraps the base model config in ``GraphTrainerDeepSeekV4Model.Config`` and
    points to ``parallelize_graph_trainer_deepseek_v4``. ``to_graph_trainer_config``
    re-wraps the base config's model fields into this Config class, so the
    values copied here only carry the class identity and parallelize_fn.
    """
    spec = model_registry(flavor)
    graph_model = GraphTrainerDeepSeekV4Model.Config(
        **{f.name: getattr(spec.model, f.name) for f in fields(spec.model)}
    )
    return ModelSpec(
        name=spec.name,
        flavor=flavor,
        model=graph_model,
        parallelize_fn=parallelize_graph_trainer_deepseek_v4,
        pipelining_fn=spec.pipelining_fn,
        post_optimizer_build_fn=spec.post_optimizer_build_fn,
        state_dict_adapter=spec.state_dict_adapter,
    )


def _graph_trainer_compile_config() -> GraphTrainerCompileConfig:
    """Compile settings shared by the GraphTrainer flash/pro config factories."""
    return GraphTrainerCompileConfig(
        enable=True,
        mode="aot_fx_trace",
        memory_policy="full",
        disable_passes=[
            "cudagraph_pass",
        ],
    )


def graph_trainer_deepseek_v4_debugmodel() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 debug model"""
    config = to_graph_trainer_config(
        deepseek_v4_debugmodel(),
        _graph_trainer_model_registry,
    )
    config.compile = GraphTrainerCompileConfig(
        enable=True,
        mode="aot_fx_trace",
        memory_policy="full",
        enable_passes=True,
        disable_passes=[
            "cudagraph_pass",
        ],
    )
    return config


def graph_trainer_deepseek_v4_flash() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 Flash model"""
    config = to_graph_trainer_config(
        deepseek_v4_flash(),
        _graph_trainer_model_registry,
    )
    config.compile = _graph_trainer_compile_config()
    return config


def graph_trainer_deepseek_v4_flash_43layers_16experts() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 Flash 43 Layers 16 experts model"""
    config = to_graph_trainer_config(
        deepseek_v4_flash_43layers_16experts(),
        _graph_trainer_model_registry,
    )
    config.compile = _graph_trainer_compile_config()
    return config


def graph_trainer_deepseek_v4_pro() -> GraphTrainer.Config:
    """GraphTrainer config for the DeepSeek V4 Pro model"""
    config = to_graph_trainer_config(
        deepseek_v4_pro(),
        _graph_trainer_model_registry,
    )
    config.compile = _graph_trainer_compile_config()
    return config
