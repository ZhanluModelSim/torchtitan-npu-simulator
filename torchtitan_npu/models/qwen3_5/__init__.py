# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torchtitan.config import derive
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import ModelConfigConverter
from torchtitan.models.common.decoder_sharding import colwise_sharding_config, residual_activation_placement
from torchtitan.models.common.feed_forward import DenseFeedForward, FeedForward
from torchtitan.models.common.linear import ColumnParallelLinear, Linear
from torchtitan.models.qwen3_5 import build_model_config as _upstream_build_model_config
from torchtitan.models.qwen3_5.model import Qwen35Model
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

from torchtitan_npu.patches.torchtitan.models.qwen3_5 import (
    vision_encoder as _vision_patch,
)

_vision_patch.apply()


class Qwen35TextStateDictAdapter(Qwen35StateDictAdapter):
    """Map Qwen text weights using the actual HF initialization checkpoint."""

    def __init__(self, model_config: Qwen35Model.Config, hf_assets_path: str | None):
        if model_config.vision_encoder is not None:
            raise ValueError("The text checkpoint adapter requires vision_encoder=None.")
        # The tokenizer assets can live outside the HF weight directory.
        super().__init__(model_config, None)
        self._source: dict[str, str] = {"path": "", "layout": ""}
        self._export_auxiliary = False
        self._auxiliary: dict[str, str] = {}

    def prepare_hf_load(self, path: str) -> None:
        root = Path(path)
        index = root / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            if not isinstance(weight_map, dict):
                raise ValueError(f"Invalid Qwen HF weight index: {index}")
        else:
            single = root / "model.safetensors"
            with safe_open(single, framework="pt", device="cpu") as shard:
                weight_map = dict.fromkeys(shard.keys(), single.name)

        has_text = "model.embed_tokens.weight" in weight_map
        has_conditional = "model.language_model.embed_tokens.weight" in weight_map
        if has_text == has_conditional:
            raise ValueError(f"Expected exactly one Qwen language-model prefix in {root}")
        layout = hashlib.sha256(json.dumps(weight_map, sort_keys=True).encode()).hexdigest()
        if self._source["path"] == str(root) and self._source["layout"] != layout:
            raise ValueError(f"Qwen HF weight layout changed at {root}")

        # Reuse upstream's index and native-to-HF key mapping for this source.
        model_config = cast("Qwen35Model.Config", self.model_config)
        super().__init__(model_config, str(root) if index.is_file() else None)
        if has_conditional:
            self.hf_language_model_prefix = "model.language_model"
            self.from_hf_map = {
                key.replace("model.", "model.language_model.", 1) if key.startswith("model.") else key: value
                for key, value in self.from_hf_map.items()
            }
        self._source = {"path": str(root), "layout": layout}
        auxiliary_prefixes = ("model.visual.", "mtp.")
        self._auxiliary = {key: filename for key, filename in weight_map.items() if key.startswith(auxiliary_prefixes)}

    def state_dict(self) -> dict[str, str]:
        return self._source.copy()

    def load_state_dict(self, state_dict: dict[str, str]) -> None:
        self._source = state_dict.copy()

    @contextmanager
    def export_hf(self, *, fallback_path: str | None = None):
        path = self._source["path"] or fallback_path
        if not path:
            raise ValueError("Qwen HF export requires the original HF weight source in the checkpoint.")
        self.prepare_hf_load(path)
        self._export_auxiliary = True
        try:
            yield
        finally:
            self._export_auxiliary = False

    def to_hf(self, state_dict: dict[str, Any], *, to_quantized_layout: bool = False) -> dict[str, Any]:
        converted = super().to_hf(state_dict, to_quantized_layout=to_quantized_layout)
        if self._export_auxiliary:
            root = Path(self._source["path"])
            for filename in sorted(set(self._auxiliary.values())):
                with safe_open(root / filename, framework="pt", device="cpu") as shard:
                    for key, source in self._auxiliary.items():
                        if source == filename and key not in converted:
                            converted[key] = shard.get_tensor(key)
        return converted


class NPUQwen35Model(Qwen35Model):
    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Model.Config):
        text_only: bool = False

    def __init__(self, config):
        super().__init__(config)
        self.text_only = config.text_only

    def forward(self, *args, **kwargs):
        if self.text_only:
            kwargs.setdefault("special_tokens", {"image_id": -1, "video_id": -1})
        return super().forward(*args, **kwargs)

    def _apply_fsdp(self, **kwargs):
        from torchtitan_npu.override.qwen3_5.parallelize import apply_qwen3_5_npu_fsdp

        apply_qwen3_5_npu_fsdp(self, **kwargs)


def model_registry(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> NPUQwen35Model.Config:
    """Build the NPU Qwen3.5 config with the upstream flavor parameters."""
    config = _upstream_build_model_config(
        flavor,
        parallelism=parallelism,
        seq_len=seq_len,
        attn_backend=attn_backend,
        converters=converters,
    )
    return derive(config, NPUQwen35Model.Config)


def checkpoint_model_config(flavor: str, *, has_vision: bool, separate_gate_up: bool) -> NPUQwen35Model.Config:
    """Build the weight layout present in a Qwen DCP for offline export."""
    from torchtitan.models.common.attention.gdn import GDN, InnerGDN

    from torchtitan_npu.override.qwen3_5.gated_delta import npu

    model = model_registry(flavor, parallelism=ParallelismConfig())
    if not has_vision:
        model = derive(model, Qwen35TextModel.Config, text_only=True, vision_encoder=None)
    if separate_gate_up:
        split_gate_up(model, enable_sequence_parallel=True)
    for _, attention, _, _ in model.traverse(GDN.Config):
        if isinstance(attention.inner_attention, InnerGDN.Config):
            attention.inner_attention = npu(attention.inner_attention)
    return model


class _SeparateGateUp(FeedForward):
    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        w1: Linear.Config
        w2: Linear.Config
        w3: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.w1 = config.w1.build()
        self.w2 = config.w2.build()
        self.w3 = config.w3.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def split_gate_up(model: NPUQwen35Model.Config, *, enable_sequence_parallel: bool) -> None:
    """Use separate gate and up parameters with the master FFN layout."""
    for layer in model.layers:
        ffn = layer.feed_forward
        if not isinstance(ffn, DenseFeedForward.Config) or ffn.w13.num_linears != 2:
            raise ValueError("The Qwen master FFN layout requires two dense gate/up projections.")
        projection = ColumnParallelLinear.Config(
            in_features=ffn.w13.in_features,
            out_features=ffn.w13.out_features,
            param_init=ffn.w2.param_init,
            sharding_config=colwise_sharding_config(
                input_layout=residual_activation_placement(enable_sp=enable_sequence_parallel)
            ),
            inplace_wgrad_accum=False,
        )
        layer.feed_forward = _SeparateGateUp.Config(
            w1=projection, w2=ffn.w2, w3=replace(projection), sharding_config=ffn.sharding_config
        )


class Qwen35TextModel(NPUQwen35Model):
    """Keep the native model and select its text-checkpoint adapter explicitly."""

    state_dict_adapter_cls = Qwen35TextStateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(NPUQwen35Model.Config):
        pass

    def __init__(self, config: Config):
        if config.vision_encoder is not None:
            raise ValueError("The text model requires vision_encoder=None.")
        super().__init__(config)

    def parallelize(self, **kwargs):
        context, parallelism = kwargs["parallelism_context"], kwargs["parallelism"]
        if context.cp_enabled:
            if context.tp_enabled or context.ep_enabled:
                raise NotImplementedError("Qwen NPU head exchange requires TP1 and EP1.")
            if parallelism.context_parallel_load_balancer is not None:
                raise ValueError("Qwen NPU head exchange requires contiguous CP sharding without a load balancer.")
        return super().parallelize(**kwargs)
