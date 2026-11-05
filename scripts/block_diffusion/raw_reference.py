# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Load the canonical raw reference without importing the NPU plugin package.

This loader is for standalone tooling and comparative tests. Native model code
must implement the raw contract independently and must not import this loader.
The shared cache key preserves dataclass/class identity across multiple tools.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_REFERENCE_MODULE_NAME = "_torchtitan_block_diffusion_raw_reference"
_REFERENCE_PATH = (
    Path(__file__).resolve().parents[2] / "torchtitan_npu" / "simulator" / "raw_model" / "block_diff" / "model.py"
)


def load_raw_reference() -> ModuleType:
    """Return the one canonical reference module for this checkout."""
    if _REFERENCE_MODULE_NAME in sys.modules:
        loaded = sys.modules[_REFERENCE_MODULE_NAME]
        loaded_path = getattr(loaded, "__file__", None)
        if loaded_path is None or Path(loaded_path).resolve() != _REFERENCE_PATH:
            raise RuntimeError("a different checkout already occupies the raw reference module cache")
        return loaded
    spec = importlib.util.spec_from_file_location(_REFERENCE_MODULE_NAME, _REFERENCE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load canonical raw model: {_REFERENCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_REFERENCE_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[_REFERENCE_MODULE_NAME]
        raise
    return module


if TYPE_CHECKING:
    from torchtitan_npu.simulator.raw_model.block_diff.model import (
        ALODLM_CONTRACT_VERSION,
        MODEL_CONTRACT_VERSION,
        STANDARD_BLOCK_DIFFUSION_CONTRACT_VERSION,
        ALoDLMBatch,
        ALoDLMDecision,
        ALoDLMModel,
        ALoDLMObjective,
        ALoDLMOuterOutput,
        ALoDLMOutput,
        ALoDLMSpec,
        Attention,
        BlockDiffusionModel,
        KVCache,
        LayerKV,
        ModelConfig,
        MoE,
        PhaseOutput,
        RMSNorm,
        RoutingPlan,
        SwiGLU,
        TrainingOutput,
        TrainingSpec,
        TransformerBlock,
        alodlm_attention_mask,
        alodlm_decision,
        alodlm_training_objective,
        build_alodlm_batch,
        training_attention_mask,
    )
else:
    _reference = load_raw_reference()
    MODEL_CONTRACT_VERSION = _reference.MODEL_CONTRACT_VERSION
    STANDARD_BLOCK_DIFFUSION_CONTRACT_VERSION = _reference.STANDARD_BLOCK_DIFFUSION_CONTRACT_VERSION
    ALODLM_CONTRACT_VERSION = _reference.ALODLM_CONTRACT_VERSION
    ALoDLMSpec = _reference.ALoDLMSpec
    ALoDLMBatch = _reference.ALoDLMBatch
    build_alodlm_batch = _reference.build_alodlm_batch
    alodlm_attention_mask = _reference.alodlm_attention_mask
    ALoDLMOutput = _reference.ALoDLMOutput
    ALoDLMOuterOutput = _reference.ALoDLMOuterOutput
    ALoDLMModel = _reference.ALoDLMModel
    ALoDLMObjective = _reference.ALoDLMObjective
    alodlm_training_objective = _reference.alodlm_training_objective
    ALoDLMDecision = _reference.ALoDLMDecision
    alodlm_decision = _reference.alodlm_decision
    Attention = _reference.Attention
    BlockDiffusionModel = _reference.BlockDiffusionModel
    KVCache = _reference.KVCache
    LayerKV = _reference.LayerKV
    ModelConfig = _reference.ModelConfig
    MoE = _reference.MoE
    PhaseOutput = _reference.PhaseOutput
    RMSNorm = _reference.RMSNorm
    RoutingPlan = _reference.RoutingPlan
    SwiGLU = _reference.SwiGLU
    TrainingOutput = _reference.TrainingOutput
    TrainingSpec = _reference.TrainingSpec
    TransformerBlock = _reference.TransformerBlock
    training_attention_mask = _reference.training_attention_mask

__all__ = [
    "ALODLM_CONTRACT_VERSION",
    "MODEL_CONTRACT_VERSION",
    "STANDARD_BLOCK_DIFFUSION_CONTRACT_VERSION",
    "ALoDLMBatch",
    "ALoDLMDecision",
    "ALoDLMModel",
    "ALoDLMObjective",
    "ALoDLMOuterOutput",
    "ALoDLMOutput",
    "ALoDLMSpec",
    "Attention",
    "BlockDiffusionModel",
    "KVCache",
    "LayerKV",
    "MoE",
    "ModelConfig",
    "PhaseOutput",
    "RMSNorm",
    "RoutingPlan",
    "SwiGLU",
    "TrainingOutput",
    "TrainingSpec",
    "TransformerBlock",
    "alodlm_attention_mask",
    "alodlm_decision",
    "alodlm_training_objective",
    "build_alodlm_batch",
    "load_raw_reference",
    "training_attention_mask",
]
