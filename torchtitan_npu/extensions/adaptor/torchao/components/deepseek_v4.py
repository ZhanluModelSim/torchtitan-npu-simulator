# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4 special-mechanism components + executors.

Hosts the V4 ``lightning_indexer`` and the V4-only ``kv_norm`` components, their
hardware executors, and the V4 lightning-indexer adapter machinery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Protocol, cast

import torch
import tyro
from torchao_npu.configs import (
    DeepSeekV4QuantLightningIndexerConfig,
    quant_mode_for_config,
)
from torchao_npu.quantization.quant_configs import (
    FP8QuantizeConfig,
    HiF8QuantizeConfig,
    MXQuantizeConfig,
)
from torchao_npu.quantization.quant_primitives.mx import mx_fake_quantize
from torchtitan.config import derive
from torchtitan.config.transform.quantization import QuantizationConverter
from torchtitan.protocols.module import Module

from ..configs import ModelType
from ..configs.deepseek_v4 import V4FilterKeys
from ..filters import _replace_config, match_config_fqn_suffix
from . import QuantComponent, component

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torchtitan_npu.config.configs import LIQuantization

    from ..recipes import RecipeContext

_V4_LI_LAYOUT = "TND"
_V4_LI_MASK_MODE = 3
_V4_LI_CMP_RATIO = 4

# The V4 inner attention, at both places the stack carries it: the decoder builds the metadata
# map from `attention.inner_attention`, while `Attention` reads its key and builds the module from
# `attention.inner_attention`.  The quantized backend replaces the
# whole node, so both copies have to become its config -- otherwise the map is keyed by the
# quantized class while every layer asks for the fused one and reads no metadata at all.
_v4_inner_attention_filter = match_config_fqn_suffix(
    ".attention.inner_attention",
    ".attention.inner_attention",
)

_DYNAMIC_NORM_CLASS_CACHE: dict[type[Module], type[Module]] = {}


def _prepare_quant_lightning_indexer_inputs(
    module,
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    idx_w: torch.Tensor,
    attention_metadata,
):
    from torchao_npu.quantized_modules.deepseek_v4.attention_metadata import QuantV4Metadata
    from torchao_npu.quantized_modules.deepseek_v4.lightning_indexer import (
        QuantizedLightningIndexerInputs as DeepSeekV4QuantizedLightningIndexerInputs,
    )

    if type(attention_metadata) is not QuantV4Metadata:
        raise TypeError(
            "the quantized selector consumes its own backend's entry, got "
            f"{type(attention_metadata).__name__}; build the stack with the quantized "
            "sparse-attention selector (quantized_attention) instead of the BF16 record."
        )
    asc = attention_metadata.aicpu_metadata
    li_metadata = None if asc is None else asc.lightning_indexer.fwd
    if li_metadata is None:
        raise RuntimeError("Quantized LI requires the quantized selector record provider")
    cmp_k = attention_metadata.cmp_k
    if cmp_k is None or cmp_k.residual is None:
        raise ValueError("Quantized LI requires the geometry's compressed frame.")
    # The quantized selector is the unsharded backend (its CP hook refuses a re-cut), and the
    # compressor emits exactly the row's compressed blocks -- there is no container to strip.
    return DeepSeekV4QuantizedLightningIndexerInputs(
        query=idx_q,
        key=idx_k.unsqueeze(1).contiguous(),
        weights=idx_w.float(),
        topk=module.index_topk,
        metadata=li_metadata,
        cu_seqlens_q=attention_metadata.q.cu_seqlens,
        cu_seqlens_k=cmp_k.cu_seqlens,
        cmp_residual_k=cmp_k.residual,
        backward_metadata=asc.lightning_indexer.bwd,
        seqused_q=attention_metadata.q.seqused,
        seqused_k=cmp_k.seqused,
    )


def _quant_deepseek_v4_lightning_indexer(
    *,
    li_quantization: LIQuantization,
    dst_type_max: float = 0.0,
) -> DeepSeekV4QuantLightningIndexerConfig:
    qk_config: FP8QuantizeConfig | HiF8QuantizeConfig | MXQuantizeConfig
    if li_quantization == "fp8":
        qk_config = FP8QuantizeConfig()
    elif li_quantization == "mxfp4":
        qk_config = MXQuantizeConfig(elem_dtype=torch.float4_e2m1fn_x2)
    elif li_quantization == "mxfp8":
        qk_config = MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn)
    elif li_quantization == "hif8":
        qk_config = HiF8QuantizeConfig(dst_type_max=dst_type_max)
    else:
        raise ValueError("Quantized LI requires mxfp4, mxfp8, fp8, or hif8")
    quant_mode = quant_mode_for_config(qk_config)
    return DeepSeekV4QuantLightningIndexerConfig(
        layout_q=_V4_LI_LAYOUT,
        layout_k=_V4_LI_LAYOUT,
        mask_mode=_V4_LI_MASK_MODE,
        cmp_ratio=_V4_LI_CMP_RATIO,
        query_config=qk_config,
        key_config=qk_config,
        quant_mode=quant_mode,
        input_adapter=_prepare_quant_lightning_indexer_inputs,
    )


def _v4_li_quant_mode(li_quantization: LIQuantization, dst_type_max: float) -> str:
    """The quant-mode the quantized V4 selector and its sparse-attention backend share.

    Kept as one pure derivation so the ``lightning_indexer`` and
    ``sparse_attention`` components can never disagree on the backend's format.
    """
    return _quant_deepseek_v4_lightning_indexer(
        li_quantization=li_quantization,
        dst_type_max=dst_type_max,
    ).quant_mode


def _override_kwargs(override_config, suffix: str) -> dict[str, object] | None:
    """The kwargs a recipe states for the override entry ending in ``suffix``.

    Quantization runs before the override pass, so a config-class replacement has to state the
    arguments the port's own factory would have received (the port entry is listed, but its
    factory never sees the node).  ``None`` means the recipe did not select that entry.
    """
    if override_config is None:
        return None
    suffix_parts = suffix.split(".")
    found: dict[str, object] | None = None
    for entry in override_config.imports:
        target, kwargs = entry if isinstance(entry, tuple) else (entry, {})
        target_parts = str(target).split(".")
        if len(target_parts) < len(suffix_parts) or target_parts[-len(suffix_parts) :] != suffix_parts:
            continue
        if found is not None:
            raise ValueError(f"multiple override entries end with {suffix!r}: the recipe must select it once.")
        found = dict(kwargs)
    return found


def _v4_attention_numbers(model_config) -> dict[str, int]:
    """The widths the quantized V4 selector's entry is sized with, off the model config.

    One config describes one geometry, so they are read once here instead of being restated by
    every layer; a stack that mixes geometries is refused, since one quantized backend cannot
    describe two.  The selector's record operator and the fused builder both read these, and the
    kernels are where a wrong statement surfaces.
    """
    layers = [layer for layer in model_config.layers if layer.attention.indexer is not None]
    if not layers:
        raise ValueError("the quantized selector needs a layer that runs one (compress_ratio 4)")
    attention = layers[0].attention
    indexer = attention.indexer
    first = (attention.n_heads, attention.head_dim, indexer.num_index_heads, indexer.index_head_dim)
    for layer in layers[1:]:
        other = (
            layer.attention.n_heads,
            layer.attention.head_dim,
            layer.attention.indexer.num_index_heads,
            layer.attention.indexer.index_head_dim,
        )
        if other != first:
            raise ValueError(
                f"the stack mixes attention geometries ({first} vs {other}), which one quantized "
                "backend cannot describe"
            )
    return {
        "num_heads": first[0],
        "head_dim": first[1],
        "index_num_heads": first[2],
        "index_head_dim": first[3],
    }


@component("lightning_indexer", models=(ModelType.V4,))
class DeepSeekV4QuantLightningIndexerComponent(QuantComponent):
    """DeepSeek-V4 Lightning-Indexer Q/K quantization."""

    def should_trigger(self, filters: dict, ctx: RecipeContext) -> bool:
        return ctx.li_quantization is not None

    def build_converters(self, filters: dict, ctx: RecipeContext):
        from ..recipes import _quantization_converter

        assert ctx.li_quantization is not None
        logger.info(
            "[Component] DeepSeekV4QuantLightningIndexer: building LI converter (li_quantization=%s)",
            ctx.li_quantization,
        )
        return [
            _quantization_converter(
                _quant_deepseek_v4_lightning_indexer(
                    li_quantization=ctx.li_quantization,
                    dst_type_max=ctx.dst_type_max,
                ),
                filters[V4FilterKeys.LIGHTNING_INDEXER],
                model_compile_enabled=ctx.model_compile_enabled,
                unified_npu_quant_modules=ctx.unified_npu_quant_modules,
                specific_npu_quant_modules=ctx.specific_npu_quant_modules,
            )
        ]


@component("v4_sparse_attention", models=(ModelType.V4,))
class DeepSeekV4QuantSparseAttentionComponent(QuantComponent):
    """Install the quantized V4 sparse-attention backend that carries the LI records.

    Paired with the ``lightning_indexer`` component: it fires on exactly the same
    trigger (``ctx.li_quantization is not None``) so the selector's records and
    this backend always agree on one ``quant_mode`` (shared via ``_v4_li_quant_mode``).
    """

    def should_trigger(self, filters: dict, ctx: RecipeContext) -> bool:
        return ctx.li_quantization is not None

    def build_converters(self, filters: dict, ctx: RecipeContext):
        from torchao_npu.quantized_modules.deepseek_v4.sparse_attention import (
            QuantizedV4SparseAttention,
        )

        from ..recipes import _quantization_converter

        assert ctx.li_quantization is not None
        logger.info("[Component] DeepSeekV4QuantSparseAttention: installing quantized V4 backend")

        # The quantized selector is a backend of its own: its config owns the entry's record
        # operator and replaces the fused inner attention, so nothing is injected into the
        # fused config and the fused backend never learns that a quantized selector exists.
        #
        # The recipe lists the port entry; quantization runs before the override pass, so
        # this replacement *is* the port (it subclasses the fused config) and the entry's
        # arguments are read from the recipe rather than from the node.
        asc_kwargs = _override_kwargs(ctx.override_config, "sparse_attn.asc")
        if asc_kwargs is None:
            raise ValueError(
                "V4 LightningIndexer quantization requires the fused stack: add "
                "torchtitan_npu.override.deepseek_v4.sparse_attn.asc (and .lightning_indexer.asc) to "
                "--override.imports."
            )
        geometry = _v4_attention_numbers(ctx.model_config)
        for name, value in geometry.items():
            if name in asc_kwargs and asc_kwargs[name] != value:
                raise ValueError(
                    f"the fused attention entry states {name}={asc_kwargs[name]!r} but the model "
                    f"config runs {value}: the quantized selector's records are sized by the "
                    "model's own geometry"
                )
        return [
            _quantization_converter(
                None,
                _v4_inner_attention_filter,
                model_compile_enabled=ctx.model_compile_enabled,
                replacement_config_type=QuantizedV4SparseAttention.Config,
                replacement_kwargs={
                    **geometry,
                    "quant_mode": _v4_li_quant_mode(ctx.li_quantization, ctx.dst_type_max),
                },
                unified_npu_quant_modules=(Module.Config,),
                specific_npu_quant_modules=ctx.specific_npu_quant_modules,
                recurse=True,
            )
        ]


@component("kv_norm", models=(ModelType.V4,))
class KVNormComponent(QuantComponent):
    """Fake-quantizes a model's KV norms (declared via target FQNs). Opt-in per model."""

    def should_trigger(self, filters: dict, ctx: RecipeContext) -> bool:
        k = ctx.kv_norm_quantization
        return k is not None and getattr(k, "format", None) is not None

    def build_converters(self, filters: dict, ctx: RecipeContext):
        k = ctx.kv_norm_quantization
        block_size = getattr(k, "block_size", None)
        fqns = tuple(getattr(k, "fqns", ()) or ())
        logger.info("[Component] KVNorm: fake-quantizing %d norm(s), block_size=%s", len(fqns), block_size)
        return [
            KVNormFakeQuantConverter.Config(
                base_config=MXQuantizeConfig(elem_dtype=torch.float8_e4m3fn, block_size=block_size),
                target_fqns=fqns,
                model_compile_enabled=ctx.model_compile_enabled,
            )
        ]


class KVNormFakeQuantConverter(QuantizationConverter):
    """Install :class:`FakeQuantizedNormFactory` in front of the selected norms."""

    class _KVRopeGeometry(Protocol):
        head_dim: int
        rope_head_dim: int
        rope: Module.Config

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        base_config: Annotated[MXQuantizeConfig | None, tyro.conf.Suppress] = None
        target_fqns: tuple[str, ...] = ()
        require_match: bool = True

        def __post_init__(self) -> None:
            if self.base_config is None:
                raise ValueError("KVNormFakeQuantConverter.Config requires base_config")
            if len(self.target_fqns) == 0:
                raise ValueError("Must specify fqns of target Modules")

    def __init__(self, config: Config) -> None:
        self.config = config
        self.filter = match_config_fqn_suffix(*config.target_fqns)

    def convert(self, model_config):
        matches = [
            (fqn, config, parent, attr)
            for fqn, config, parent, attr in model_config.traverse(Module.Config, recurse=True)
            if self.filter(cast("Module.Config", config), fqn)
        ]

        converted = 0
        for fqn, config, parent, attr in matches:
            if parent is None:
                raise ValueError(f"the norm fake quantization does not support the root config ({fqn!r})")
            parent_cfg = cast("KVNormFakeQuantConverter._KVRopeGeometry", parent)
            head_dim = getattr(parent_cfg, "head_dim", None)
            rope_head_dim = getattr(parent_cfg, "rope_head_dim", None)
            if head_dim is None or rope_head_dim is None:
                raise ValueError(
                    f"target norm at {fqn!r} has no rope geometry on its parent; the nope span cannot be derived"
                )
            nope_head_dim = head_dim - rope_head_dim
            split = getattr(getattr(parent_cfg, "rope", None), "split", nope_head_dim)
            if not 0 < nope_head_dim <= head_dim:
                raise ValueError(
                    f"KV norm at {fqn!r} has an invalid nope span: head_dim={head_dim}, nope_head_dim={nope_head_dim}"
                )
            if split != nope_head_dim:
                raise ValueError(
                    f"KV norm at {fqn!r} disagrees with its rope split: "
                    f"nope_head_dim={nope_head_dim}, rope split={split}"
                )
            replacement = derive(
                config,
                FakeQuantizedNormFactory.Config,
                norm=config,
                _fake_quant_config=self.config.base_config,
                head_dim=head_dim,
                nope_head_dim=nope_head_dim,
            )
            model_config = _replace_config(model_config, parent, attr, replacement)
            converted += 1
            logger.info(
                "[Converter] %s.%s: model.%s %s -> %s",
                type(self).__module__,
                type(self).__qualname__,
                fqn,
                type(config).__qualname__,
                type(replacement).__qualname__,
            )

        if converted == 0 and self.config.require_match:
            raise ValueError("KVNormFakeQuantConverter did not match any config nodes")
        logger.info("Converted %d config node(s) for KV norm fake quantization", converted)
        return model_config


def _dynamic_norm_class(owner: type[Module]) -> type[Module]:
    """Generates a FakeQuantizedNorm via subclassing ``owner`` (cached)."""

    if owner in _DYNAMIC_NORM_CLASS_CACHE:
        return _DYNAMIC_NORM_CLASS_CACHE[owner]

    class FakeQuantizedNorm(owner):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(owner.Config):  # type: ignore[misc, valid-type]
            fake_quant_config: MXQuantizeConfig
            head_dim: int
            nope_head_dim: int

        def __init__(self, config: Config) -> None:
            super().__init__(config)
            self.fake_quant_config = config.fake_quant_config
            self.head_dim = config.head_dim
            self.nope_head_dim = config.nope_head_dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.shape[-1] != self.head_dim:
                raise ValueError(f"{type(self).__name__} expects a width-{self.head_dim} input, got {tuple(x.shape)}")
            y = super().forward(x)
            return torch.cat(
                [
                    mx_fake_quantize(y[..., : self.nope_head_dim], -1, self.fake_quant_config),
                    y[..., self.nope_head_dim :],
                ],
                dim=-1,
            )

    FakeQuantizedNorm.__name__ = FakeQuantizedNorm.__qualname__ = f"FakeQuantized{owner.__name__}"
    _DYNAMIC_NORM_CLASS_CACHE[owner] = FakeQuantizedNorm
    return FakeQuantizedNorm


class FakeQuantizedNormFactory(Module):
    """Config node that builds a fake-quantizing norm in place of a norm config."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm: Module.Config
        _fake_quant_config: Annotated[MXQuantizeConfig | None, tyro.conf.Suppress] = None
        head_dim: int
        nope_head_dim: int

        def build(self, **kwargs):
            owner = type(self.norm)._owner
            if owner is None:
                raise TypeError(
                    f"{type(self.norm).__qualname__} has no owning module class; "
                    "FakeQuantizedNormFactory cannot build it"
                )
            if self._fake_quant_config is None:
                raise ValueError("FakeQuantizedNormFactory.Config requires _fake_quant_config")
            return derive(
                self.norm,
                _dynamic_norm_class(owner).Config,
                fake_quant_config=self._fake_quant_config,
                head_dim=self.head_dim,
                nope_head_dim=self.nope_head_dim,
                param_init=self.param_init,
                sharding_config=self.sharding_config,
            ).build(**kwargs)
