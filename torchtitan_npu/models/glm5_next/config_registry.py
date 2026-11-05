# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Training config registry for GLM-5.3-Flash (torchtitan 0.3.0).

Capture flavors use modest sequence lengths that satisfy the uniform MoE
routing contract: ``seq_len * top_k`` must be divisible by the expert count
(288 routed experts at top-8 in the production flavors, so seq_len must be a
multiple of 36; 216 = 36 * 6 works for every flavor).
"""

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import derive
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan_npu.config import OptimizerConfig, TrainingConfig
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx

from . import model_registry
from .parallelize import parallelize_glm5_next
from .state_dict_adapter import Glm5NextStateDictAdapter

_CAPTURE_SEQ_LEN = 216  # 36 * 6: uniform routing holds for top_k=2 and 8


def model_spec_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="glm5_next",
        flavor=flavor,
        model=model_registry(flavor),
        parallelize_fn=parallelize_glm5_next,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=Glm5NextStateDictAdapter,
    )


def _make_trainer_config(flavor: str, *, seq_len: int = _CAPTURE_SEQ_LEN) -> TrainerEx.Config:
    model_spec = model_spec_registry(flavor)
    return TrainerEx.Config(
        model_spec=model_spec,
        training=TrainingConfig(local_batch_size=1, seq_len=seq_len, steps=1),
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=model_spec.model.vocab_size),
        ),
        tokenizer=HuggingFaceTokenizer.Config(),
        dataloader=_text_dataloader_config(),
        optimizer=derive(default_adamw(lr=8e-4, eps=1e-8), OptimizerConfig, name="AdamW"),
        activation_checkpoint=FullAC.Config(),
    )


def _text_dataloader_config():
    """The packed text loader (per-document pooling alignment 1: GLM has no
    compression ratios to align to)."""
    from torchtitan_npu.patches.torchtitan.hf_datasets.text_datasets import (
        AlignedHuggingfaceDataloader,
    )

    return AlignedHuggingfaceDataloader.Config()


def glm5_next_debugmodel_text(*, num_mtp_layers: int = 0) -> TrainerEx.Config:
    if num_mtp_layers:
        raise ValueError("GLM-5.3-Flash has no MTP modules; num_mtp_layers must be 0")
    return _make_trainer_config("debugmodel_text")


def glm5_next_debugmodel_multimodal(*, num_mtp_layers: int = 0) -> TrainerEx.Config:
    if num_mtp_layers:
        raise ValueError("GLM-5.3-Flash has no MTP modules; num_mtp_layers must be 0")
    return _make_trainer_config("debugmodel")


def glm5_next_reduced_2l(*, num_mtp_layers: int = 0) -> TrainerEx.Config:
    if num_mtp_layers:
        raise ValueError("GLM-5.3-Flash has no MTP modules; num_mtp_layers must be 0")
    return _make_trainer_config("reduced-2L")


def glm5_next(*, num_mtp_layers: int = 0) -> TrainerEx.Config:
    if num_mtp_layers:
        raise ValueError("GLM-5.3-Flash has no MTP modules; num_mtp_layers must be 0")
    return _make_trainer_config("GLM-5.3-Flash")


def glm5_next_910b_test(*, num_mtp_layers: int = 0) -> TrainerEx.Config:
    if num_mtp_layers:
        raise ValueError("GLM-5.3-Flash has no MTP modules; num_mtp_layers must be 0")
    return _make_trainer_config("910b-test")
