# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchAO-NPU converter integration hosted by ``torchtitan-npu``.

TorchTitan builds models from a tree of ``Module.Config`` objects.  Quantization
must therefore be expressed in that tree before the model is built: a converter
replaces selected configs with configs whose modules install torchao-npu
parameter wrappers or module-level QAT behavior in ``__init__``.  This keeps
the transforms in place before
TP/EP/FSDP and optimizer construction without modifying TorchTitan's trainer.

Only this adapter lives in ``torchtitan_npu``.  Quantization configs, parameter
wrappers and NPU kernels continue to come from the separately installed
``torchao_npu`` package.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol, cast

import torch
import tyro
from torchao.core.config import AOBaseConfig  # noqa: TC002
from torchao.quantization.quant_api import quantize_
from torchao_npu.configs import (
    DeepSeekV4QuantLightningIndexerConfig,
    DeepSeekV41QuantCompressorConfig,
    DeepSeekV41QuantLightningIndexerConfig,
    DeepSeekV41QuantSparseAttentionConfig,
    ParamSwapConfig,
    quant_mode_for_config,
)
from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    FP8QuantizeConfig,
    HiF8QuantizeConfig,
    MXQuantizeConfig,
)
from torchao_npu.quantization.quant_primitives.mx import mx_fake_quantize
from torchtitan.components.quantization import QuantizationConverter
from torchtitan.config import derive
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger

from torchtitan_npu.models.common.metadata_extension import (
    LightningIndexerKernelConfig as DeepSeekV4LightningIndexerKernelConfig,
)
from torchtitan_npu.models.common.metadata_extension import (
    LightningIndexerMetadata as DeepSeekV4LightningIndexerMetadata,
)
from torchtitan_npu.models.common.metadata_extension import register_pytree_node_for_dataclass
from torchtitan_npu.models.deepseek_v4.compressor import LightningIndexer as DeepSeekV4LightningIndexer
from torchtitan_npu.models.deepseek_v4_1.attention import (
    CompressedSparseInnerAttention2 as DeepSeekV41CompressedSparseInnerAttention2,
)
from torchtitan_npu.models.deepseek_v4_1.attention import (
    DeepSeekV41Metadata as _DeepSeekV41Metadata,
)
from torchtitan_npu.models.deepseek_v4_1.compressor import Compressor as DeepSeekV41Compressor
from torchtitan_npu.models.deepseek_v4_1.indexer import Selector as DeepSeekV41Selector
from torchtitan_npu.override.common.swiglu_group.ascendc import AscGroupedExperts, _ensure_cann_ops_loaded
from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

if TYPE_CHECKING:
    from torchtitan.protocols.model_spec import ModelSpec

    from torchtitan_npu.config.configs import (
        KVNormQuantizationConfig,
        LIQuantization,
        QuantizationExtensionConfig,
    )


class ConfigFilterFn(Protocol):
    """Predicate over a TorchTitan model-config node and its config-tree FQN."""

    def __call__(self, config: Module.Config, fqn: str) -> bool: ...


def any_config_filter(*filters: ConfigFilterFn) -> ConfigFilterFn:
    """Combine config-tree filters with logical OR."""

    def _filter(config: Module.Config, fqn: str) -> bool:
        return any(filter_fn(config, fqn) for filter_fn in filters)

    return _filter


def match_config_fqn_suffix(*suffixes: str) -> ConfigFilterFn:
    """Match config-tree FQNs by a dotted suffix or an exact root name.

    ``Config.traverse`` reports nested fields as dotted FQNs, but a field on
    the model config itself has no leading component (for example,
    ``lightning_indexer_metadata``).  Accepting both forms keeps filters
    useful for root-owned extension configs as well as nested modules.
    """

    normalized = tuple(suffix.lstrip(".") for suffix in suffixes)

    def _filter(config: Module.Config, fqn: str) -> bool:
        return any(fqn == suffix or fqn.endswith(f".{suffix}") for suffix in normalized)

    return _filter


def _replace_config(model_config, parent: object | None, attr: str | int | None, replacement):
    """Replace one config-tree node, including the uncommon root-node case."""

    if parent is None:
        return replacement
    if isinstance(parent, list):
        assert isinstance(attr, int)
        parent[attr] = replacement
    else:
        assert isinstance(attr, str)
        setattr(parent, attr, replacement)
    return model_config


_npu_quantized_module_cache: dict[type[Module], type[Module]] = {}
_DEFAULT_TARGET_CONFIG_TYPES = (
    Linear.Config,
    BatchedLinear.Config,
    GroupedExperts.Config,
    DeepSeekV4LightningIndexer.Config,
    DeepSeekV4LightningIndexerMetadata.Config,
    DeepSeekV41CompressedSparseInnerAttention2.Config,
    DeepSeekV41Compressor.Config,
    DeepSeekV41Selector.Config,
)


def _get_npu_quantized_module_cls(parent_cls: type[Module]) -> type[Module]:
    """Create a quantized subclass while preserving host behavior.

    The generated class inherits the concrete config owner so custom forward
    methods and config fields remain intact. Standard GroupedExperts and
    AscGroupedExperts use the dedicated subclass selected by NpuQuantizeConverter.
    """

    if parent_cls in _npu_quantized_module_cache:
        return _npu_quantized_module_cache[parent_cls]

    parent_config_cls = parent_cls.Config

    class NpuQuantizedModule(parent_cls):  # type: ignore[valid-type, misc]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc, valid-type]
            # TorchAO configs contain Python objects and are supplied by the
            # recipe, not by Tyro CLI parsing or config serialization.
            _torchao_npu_config: Annotated[AOBaseConfig | None, tyro.conf.Suppress] = None

        def __init__(self, config: Config) -> None:
            super().__init__(config)
            if config._torchao_npu_config is None:
                raise ValueError(f"{type(self).__name__}.Config requires _torchao_npu_config")
            quantize_(
                self,
                config._torchao_npu_config,
                filter_fn=lambda candidate, _fqn: candidate is self,
            )

    NpuQuantizedModule.__name__ = f"NpuQuantized{parent_cls.__name__}"
    NpuQuantizedModule.__qualname__ = f"NpuQuantized{parent_cls.__name__}"
    _npu_quantized_module_cache[parent_cls] = NpuQuantizedModule
    return NpuQuantizedModule


class NpuQuantizedAscGroupedExpertsModule(AscGroupedExperts):
    """Parameter-quantized routed experts with the AscGroupedExperts forward."""

    @dataclass(kw_only=True, slots=True)
    class Config(AscGroupedExperts.Config):
        _torchao_npu_config: Annotated[AOBaseConfig | None, tyro.conf.Suppress] = None

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if config._torchao_npu_config is None:
            raise ValueError(f"{type(self).__name__}.Config requires _torchao_npu_config")
        _ensure_cann_ops_loaded()
        quantize_(
            self,
            config._torchao_npu_config,
            filter_fn=lambda candidate, _fqn: candidate is self,
        )


class NpuQuantizeConverter(QuantizationConverter):
    """Replace selected quantizable module or metadata config nodes.

    A recipe may instantiate this converter more than once with different
    parameter-swap policies.  For example, one instance can select attention
    and shared-expert Linear nodes for MX training while another selects routed
    GroupedExperts nodes for Block FP8 training.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(QuantizationConverter.Config):
        base_config: Annotated[AOBaseConfig | None, tyro.conf.Suppress] = None
        filter_fn: Annotated[ConfigFilterFn | None, tyro.conf.Suppress] = None
        replacement_config_type: Annotated[type | None, tyro.conf.Suppress] = None
        replacement_kwargs: Annotated[dict[str, object], tyro.conf.Suppress] = field(default_factory=dict)
        require_match: bool = True
        prequantize_fqns: Annotated[tuple[str, ...] | None, tyro.conf.Suppress] = None
        """Whitelist of FQN suffix patterns that keep ``fsdp_prequantize=True``.
        Matched nodes keep the flag, unmatched nodes get an explicit per-node
        ``fsdp_prequantize=False``. ``None`` disables the policy (the base
        config's flag applies to every node). Shard-alignment violations are
        caught at runtime by the fail-fast guard in
        :class:`BlockMXTrainingWeightWrapperTensor`.
        """

        def __post_init__(self) -> None:
            if self.base_config is None and self.replacement_config_type is None:
                raise ValueError("NpuQuantizeConverter.Config requires base_config or replacement_config_type")

    def __init__(self, config: Config) -> None:
        self.config = config
        # Filled by ``convert``: whitelist patterns that matched at least one
        # pre-quantizing node here. ``apply_quantization_converter`` aggregates
        # them across converters: a pattern matching nothing anywhere is a
        # silent no-op (typo / stale name) and must fail fast.
        self.matched_prequantize_fqns: set[str] = set()

    def _apply_prequantize_policy(self, base_config: AOBaseConfig, fqn: str) -> AOBaseConfig:
        """Resolve the per-node ``fsdp_prequantize`` flag for one matched node.

        Only nodes whose weight config is a pre-quantizing
        :class:`BlockMXQuantizeConfig` participate. Whitelisted nodes keep the
        flag and record their patterns in ``matched_prequantize_fqns``;
        unmatched nodes get an explicit ``fsdp_prequantize=False`` copy.
        Returns the (possibly replaced) config for this node.
        """
        if self.config.prequantize_fqns is None:
            return base_config
        weight_config = getattr(base_config, "weight_config", None)
        if not _is_prequantizing_block_mx(weight_config):
            return base_config

        patterns = self.config.prequantize_fqns
        matched = [pattern for pattern in patterns if _match_fqn_suffix(fqn, pattern)]
        if not matched:
            logger.info("[Converter] %s: fsdp_prequantize=disabled (not in whitelist)", fqn)
            return _param_swap_without_prequantize(base_config)

        logger.info("[Converter] %s: fsdp_prequantize=enabled (whitelist %s)", fqn, matched[0])
        self.matched_prequantize_fqns.update(matched)
        return base_config

    def convert(self, model_config):
        converted = 0
        matches = []
        seen_fqns = set()
        for config_type in _DEFAULT_TARGET_CONFIG_TYPES:
            for match in model_config.traverse(config_type):
                if match[0] not in seen_fqns:
                    matches.append(match)
                    seen_fqns.add(match[0])
        for fqn, config, parent, attr in matches:
            if self.config.filter_fn is not None and not self.config.filter_fn(config, fqn):
                continue
            if self.config.replacement_config_type is not None:
                replacement = derive(config, self.config.replacement_config_type, **self.config.replacement_kwargs)
                replacement_name = self.config.replacement_config_type.__qualname__
            else:
                # Metadata is a Configurable provider, not an nn.Module.  It
                # shares the target set only so the replacement branch can
                # handle it; parameter quantization must skip it.
                if isinstance(config, DeepSeekV4LightningIndexerMetadata.Config):
                    continue
                assert self.config.base_config is not None
                parent_cls = cast("type[Module] | None", type(config)._owner)
                if parent_cls is None:
                    raise TypeError(f"Config at {fqn!r} has no owning module class")
                if (
                    isinstance(config, NpuQuantizedAscGroupedExpertsModule.Config)
                    or parent_cls in _npu_quantized_module_cache.values()
                ):
                    continue

                # Preserve model-specific expert subclasses and their config fields.
                if type(config) in (GroupedExperts.Config, AscGroupedExperts.Config):
                    quantized_cls = NpuQuantizedAscGroupedExpertsModule
                else:
                    quantized_cls = _get_npu_quantized_module_cls(parent_cls)
                node_base_config = self._apply_prequantize_policy(self.config.base_config, fqn)
                replacement = derive(
                    config,
                    quantized_cls.Config,
                    _torchao_npu_config=node_base_config,
                )
                replacement_name = f"{quantized_cls.__qualname__}.Config"
            model_config = _replace_config(model_config, parent, attr, replacement)
            converted += 1
            logger.info(
                "[Converter] %s.%s: model_spec.model.%s %s -> %s",
                type(self).__module__,
                type(self).__qualname__,
                fqn,
                type(config).__qualname__,
                replacement_name,
            )

        if converted == 0 and self.config.require_match:
            raise ValueError("NpuQuantizeConverter did not match any config nodes")
        logger.info("Converted %d config node(s) for torchao-npu", converted)
        return model_config


class FakeQuantizedNormFactory(Module):
    """Config node that builds a fake-quantizing norm in place of a norm config.

    The converter installs this node where a target config used to be; the node
    keeps that config in ``norm`` so the override pass (e.g. ``rms_norm.asc``)
    still finds and rewrites it.  ``build()`` then resolves the module class from
    the wrapped config's owner -- the fused replacement when such an override ran,
    the original module class otherwise -- and builds a generated subclass of it
    that fake-quantizes the nope prefix of its output.

    Consequences worth knowing:

    - the built module keeps the norm's parameter names (``...kv_norm.weight``),
      so checkpoints and state-dict adapters are unaffected;
    - the built module is *not* ``type(config)._owner``;
    - ``FakeQuantizedNormFactory`` itself is never instantiated.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm: Module.Config
        """The config this node replaced (selected by the converter's targets)."""
        _fake_quant_config: Annotated[MXQuantizeConfig | None, tyro.conf.Suppress] = None
        """How the nope prefix is fake-quantized."""
        head_dim: int
        """Width of the norm's input and output; validated at run time."""
        nope_head_dim: int
        """Width of the fake-quantized prefix (the channels RoPE does not rotate)."""

        def build(self, **kwargs):
            owner = type(self.norm)._owner
            if owner is None:
                raise TypeError(
                    f"{type(self.norm).__qualname__} has no owning module class; "
                    "FakeQuantizedNormFactory cannot build it"
                )
            if self._fake_quant_config is None:
                raise ValueError("FakeQuantizedNormFactory.Config requires _fake_quant_config")
            # The generated Config's owner is the generated module class, so the
            # framework's Module.Config.build does the construction and attaches
            # ``_param_init``/``_sharding_config`` (the model writes the sharding
            # declaration at this node, not on the wrapped config).
            return derive(
                self.norm,
                _dynamic_norm_class(owner).Config,
                fake_quant_config=self._fake_quant_config,
                head_dim=self.head_dim,
                nope_head_dim=self.nope_head_dim,
                param_init=self.param_init,
                sharding_config=self.sharding_config,
            ).build(**kwargs)


_DYNAMIC_NORM_CLASS_CACHE: dict[type[Module], type[Module]] = {}


def _dynamic_norm_class(owner: type[Module]) -> type[Module]:
    """
    Generates a FakeQuantizedNorm via subclassing ``owner``.

    The generated class is cached using owner as the key. So repeated
    builds neither re-create classes nor re-trigger ``torch.compile``.

    A FakeQuantizedNorm fake-quantizes the nope prefix of the output of
    the Norm class it subclasses. Each FakeQuantizedNorm carries its own
    widths and quantization config on the config,
    """

    if owner in _DYNAMIC_NORM_CLASS_CACHE:
        return _DYNAMIC_NORM_CLASS_CACHE[owner]

    class FakeQuantizedNorm(owner):  # type: ignore[valid-type, misc]
        """``owner`` followed by a fake-quantization of the nope prefix."""

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
            # Compute the norm first, then fake-quantize the nope prefix only: the
            # trailing channels must reach RoPE untouched. The fake quantization
            # carries straight-through gradients and ``cat`` keeps the whole
            # path free of in-place writes.
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


class KVNormFakeQuantConverter(QuantizationConverter):
    """Install :class:`FakeQuantizedNormFactory` in front of the selected norms.

    The nodes are selected by config-tree FQN suffix (``target_fqns``), so any
    norm node can be targeted.
    """

    class _KVRopeGeometry(Protocol):
        """The parent-config surface the converter reads."""

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
        # Collect before mutating: traversal never re-visits a replaced node, and
        # the factory keeps the wrapped config one level down.  There is no common
        # norm config base class, so the walk is generic and the selection is
        # entirely FQN-driven (``recurse=True`` walks the whole tree).
        matches = [
            (fqn, config, parent, attr)
            for fqn, config, parent, attr in model_config.traverse(Module.Config, recurse=True)
            if self.filter(cast("Module.Config", config), fqn)
        ]

        converted = 0
        for fqn, config, parent, attr in matches:
            if parent is None:
                raise ValueError(f"the norm fake quantization does not support the root config ({fqn!r})")
            # The parent carries the rope geometry: the nope span is exactly the
            # prefix RoPE leaves untouched, so it must agree with the rope split.
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
                "[Converter] %s.%s: model_spec.model.%s %s -> %s",
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


_DENSE_SCOPE_SUFFIXES = (
    ".attention.wq_a",
    ".attention.wq_b",
    ".attention.wkv",
    ".attention.wo_a",
    ".attention.wo_b",
    ".attention.indexer.wq_b",
    ".moe.shared_experts.w1",
    ".moe.shared_experts.w2",
    ".moe.shared_experts.w3",
)
_ROUTED_EXPERT_SCOPE_SUFFIXES = (".moe.routed_experts.inner_experts",)

# DeepSeek-V4 config-tree filters for the current model hierarchy.
_DSV4_CONFIG_FILTERS = {
    "dense": match_config_fqn_suffix(*_DENSE_SCOPE_SUFFIXES),
    "routed_expert": match_config_fqn_suffix(*_ROUTED_EXPERT_SCOPE_SUFFIXES),
    "lightning_indexer": match_config_fqn_suffix(".attention.compressed_sparse_attention.lightning_indexer"),
    "lightning_indexer_metadata": match_config_fqn_suffix(".lightning_indexer_metadata"),
}

_v41_compressor_fqn_filter = match_config_fqn_suffix(".attention.compressor")


def _is_v41_kv_source(config: Module.Config, fqn: str) -> bool:
    return (
        _v41_compressor_fqn_filter(config, fqn)
        and isinstance(config, DeepSeekV41Compressor.Config)
        and config.is_source
    )


_DSV41_CONFIG_FILTERS = {
    # V4.1 keeps the V4 attention and MoE config-tree suffixes for the dense
    # parameter quantizers and adds Compressor/sparse-attention module swaps.
    "dense": _DSV4_CONFIG_FILTERS["dense"],
    "routed_expert": _DSV4_CONFIG_FILTERS["routed_expert"],
    "compressor": _is_v41_kv_source,
    "lightning_indexer": match_config_fqn_suffix(".attention.indexer.selector"),
    "sparse_attention": match_config_fqn_suffix(".attention.inner_attention"),
}

_KIMI_CONFIG_FILTERS = {
    # Retain the legacy Kimi default: quantize routed/shared experts only.
    "dense": match_config_fqn_suffix(".moe.shared_experts.w1", ".moe.shared_experts.w2", ".moe.shared_experts.w3"),
    "routed_expert": match_config_fqn_suffix(*_ROUTED_EXPERT_SCOPE_SUFFIXES),
}

_ModelType = Literal["v4", "v41", "kimi_k3"]
_CONFIG_FILTERS_BY_MODEL_TYPE: dict[_ModelType, dict[str, ConfigFilterFn]] = {
    "v4": _DSV4_CONFIG_FILTERS,
    "v41": _DSV41_CONFIG_FILTERS,
    "kimi_k3": _KIMI_CONFIG_FILTERS,
}
_MODEL_TYPE_BY_MODEL_NAME: dict[str, _ModelType] = {
    "deepseek_v4": "v4",
    "deepseek_v4_1": "v41",
    "kimi_k3": "kimi_k3",
}


_SUPPORTED_RECIPES = ("all_mxfp8", "all_hif8", "mix", "all_block_fp8")

_PREQUANTIZE_DEFAULT_FQNS: dict[str, tuple[str, ...]] = {
    "dense": _DENSE_SCOPE_SUFFIXES,
    "routed_expert": _ROUTED_EXPERT_SCOPE_SUFFIXES,
}


def _match_fqn_suffix(fqn: str, pattern: str) -> bool:
    """Match a config-tree FQN by a dotted suffix pattern (``.attention.wkv``)."""
    normalized = pattern.lstrip(".")
    return fqn == normalized or fqn.endswith(f".{normalized}")


def _is_prequantizing_block_mx(weight_config: object) -> bool:
    """Whether a weight config is a Block MX config with pre-quantize enabled."""
    return isinstance(weight_config, BlockMXQuantizeConfig) and weight_config.fsdp_prequantize


def _param_swap_without_prequantize(param_swap: ParamSwapConfig) -> ParamSwapConfig:
    """Copy a ``ParamSwapConfig`` with the node-local ``fsdp_prequantize`` off.

    Used for weights outside the whitelist: the BF16 communication path is
    then an explicit configuration decision, not an implicit runtime fallback.
    Every constructor field is carried over so the copy cannot silently drop
    state if ``ParamSwapConfig`` grows new parameters.
    """
    weight_config = param_swap.weight_config
    assert isinstance(weight_config, BlockMXQuantizeConfig)
    return ParamSwapConfig(
        base_config=param_swap.base_config,
        activation_config=param_swap.activation_config,
        weight_config=replace(weight_config, fsdp_prequantize=False),
        step=param_swap.step,
        params_filter_fn=param_swap.params_filter_fn,
    )


def _check_prequantize_policy(
    converters: list[object],
    prequantize_fqns: tuple[str, ...],
    *,
    strict: bool,
) -> None:
    """Validate whitelist coverage after conversion.

    ``strict`` distinguishes the two whitelist sources. A *user-provided*
    whitelist is a promise that every pattern exists, so a pattern matching no
    pre-quantizing node is a typo or stale name and must fail fast. The
    *recipe default* whitelist enumerates the full projection set including
    optional modules (e.g. factories with ``shared_experts=None``), so a
    zero-match default entry is a legitimate absent module and only logs.
    """
    if not prequantize_fqns:
        return
    matched: set[str] = set()
    for converter in converters:
        matched |= getattr(converter, "matched_prequantize_fqns", set())
    unmatched = [pattern for pattern in prequantize_fqns if pattern not in matched]
    if not unmatched:
        return
    if not strict:
        logger.info(
            "fsdp_prequantize default whitelist entries not present in this model, skipped: %s",
            unmatched,
        )
        return
    raise ValueError(
        f"fsdp_prequantize_fqns patterns matched no quantized weight: {unmatched}. "
        f"Check the spelling against the model's quantized projections "
        f"(all patterns: {list(prequantize_fqns)})."
    )


def _model_type_for_spec(model_spec: ModelSpec) -> _ModelType:
    """Return the quantization family recorded by the model registry."""
    try:
        return _MODEL_TYPE_BY_MODEL_NAME[model_spec.name]
    except KeyError as exc:
        raise ValueError(
            "TorchAO-NPU quantization supports DeepSeek V4, V4.1 and Kimi K3 model specs, "
            f"got {getattr(model_spec, 'name', type(model_spec).__qualname__)!r}"
        ) from exc


def _prepare_quant_lightning_indexer_inputs(
    module,
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    idx_w: torch.Tensor,
    attention_masks,
):
    from torchao_npu.quantized_modules.deepseek_v4.lightning_indexer import (
        QuantizedLightningIndexerInputs as DeepSeekV4QuantizedLightningIndexerInputs,
    )

    ratio = module._torchao_npu_module_swap_config.cmp_ratio
    plan = attention_masks.plans.get(ratio)
    if plan is None:
        raise ValueError(f"LI compression ratio {ratio} is not present in attention metadata")
    if plan.li_metadata is None:
        raise RuntimeError("Quantized LI requires the quantized LI metadata provider")
    key = idx_k.flatten(0, 1)
    if plan.cmp_k_global_gather_indices is not None:
        key = key[plan.cmp_k_global_gather_indices]
    else:
        key = key[: plan.n_cmp_blocks_host]
    return DeepSeekV4QuantizedLightningIndexerInputs(
        query=idx_q.flatten(0, 1),
        key=key.unsqueeze(1).contiguous(),
        weights=idx_w.flatten(0, 1).float(),
        topk=module.index_topk,
        metadata=plan.li_metadata,
        cu_seqlens_q=attention_masks.varlen.cu_seq_q,
        cu_seqlens_k=plan.cu_seqlens_cmp_k,
        cmp_residual_k=plan.block_remainder,
    )


@dataclass(frozen=True, kw_only=True, slots=True)
class _V41KernelFrame:
    """One tensor's coordinate frame, as the AI_CPU metadata operators read it.

    The same three fields as the unquantized entry's ``KernelFrame``, restated rather than imported:
    that class lives in ``torchtitan_npu.override``, which this package may not depend on (see
    ``docs/architecture.md``), and the quantized entry is this package's own.
    """

    cu_seqlens: torch.Tensor
    seqused: torch.Tensor
    residual: torch.Tensor | None = None


@dataclass(frozen=True, kw_only=True, slots=True)
class _V41QuantAttentionMetadata(_DeepSeekV41Metadata):
    """The row, this geometry's frames, and the quantized kernels' metadata.

    Same interface as the unquantized stack's entry (``aicpu_metadata.sparse_attn`` /
    ``.lightning_indexer``), built by this package because the quantized attention is its own
    backend: it is the fused attention's copy with the sliding-window KV fake-quantized.
    """

    ratio: int
    q: _V41KernelFrame
    swa_k: _V41KernelFrame
    cmp_k: _V41KernelFrame | None
    aicpu_metadata: Any

    @staticmethod
    def prepare_cp_metadata(  # pyrefly: ignore [bad-override]
        attention_metadata: _V41QuantAttentionMetadata,
        *,
        permutation,
    ):
        """The quantized stack does not rebuild its metadata per rank yet (context parallelism)."""
        del attention_metadata, permutation
        raise ValueError(
            "the quantized sparse attention does not rebuild its metadata per rank yet: run it "
            "with --parallelism.context-parallel-degree 1."
        )


def _v41_frames(
    row: _DeepSeekV41Metadata, ratio: int
) -> tuple[_V41KernelFrame, _V41KernelFrame, _V41KernelFrame | None]:
    """This geometry's three frames, read off the row the reference builder produced.

    ``cu_seq_k`` is already folded at this ratio by the backend that built the row, so the
    compressed frame needs nothing but the per-document lengths for ``seqused``/``residual``.
    """
    cu_seqlens = row.cu_seq_q
    expected = cu_seqlens if ratio <= 1 else (cu_seqlens // ratio).to(torch.int32)
    if not torch.equal(row.cu_seq_k, expected):
        raise ValueError(
            f"the row's cu_seq_k was not folded at compress_ratio {ratio}: build the metadata "
            "through the attention backend that owns this layer's ratio."
        )
    seqused = torch.diff(cu_seqlens).to(torch.int32)
    q = _V41KernelFrame(cu_seqlens=cu_seqlens, seqused=seqused)
    swa_k = _V41KernelFrame(cu_seqlens=cu_seqlens, seqused=seqused)
    cmp_k = None
    if ratio > 0:
        cmp_k = _V41KernelFrame(
            cu_seqlens=row.cu_seq_k,
            seqused=(seqused // ratio).to(torch.int32),
            # Ratio one addresses the row itself: the axis is not folded, so the kernels reject a
            # trailing-group value for it.
            residual=None if ratio == 1 else (seqused % ratio).to(torch.int32),
        )
        if cmp_k.residual is not None and bool(cmp_k.residual.any()):
            raise ValueError(
                f"compress_ratio {ratio} has a partial trailing group: a document length is not a "
                f"multiple of the pooling ratio ({cmp_k.residual.tolist()}). The dataloader's "
                "document alignment must be a multiple of every ratio the stack pools at."
            )
    return q, swa_k, cmp_k


class QuantV41SparseAttention(DeepSeekV41CompressedSparseInnerAttention2):
    """The quantized sparse attention: the reference attention's node, its forward swapped.

    This is the *node* a recipe's config ends up with; the forward it runs lives in
    ``torchao_npu.quantized_modules.deepseek_v4_1.sparse_attention``, which happens to give its
    class the same name and is swapped in by ``quantize_`` below.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV41CompressedSparseInnerAttention2.Config):
        """The reference attention plus the numbers this backend's metadata is sized with."""

        num_heads: int
        head_dim: int
        index_topk: int
        index_num_heads: int
        index_head_dim: int
        candidate_topk_blocks: int
        candidate_block_size: int
        _torchao_npu_config: Annotated[Any, tyro.conf.Suppress] = None

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> _V41QuantAttentionMetadata:
            """The row this backend already builds, plus its own kernels' metadata."""
            from torchao_npu.quantized_modules.deepseek_v4_1.attention_metadata import (
                build_attention_metadata as build_quant_metadata,
            )

            # ``slots=True`` recreates the dataclass, so the parent is named explicitly rather
            # than reached through a zero-argument ``super()``.
            row = DeepSeekV41CompressedSparseInnerAttention2.Config.build_attention_metadata(
                self,
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            q, swa_k, cmp_k = _v41_frames(row, self.compress_ratio)
            return _V41QuantAttentionMetadata(
                cu_seq_q=row.cu_seq_q,
                cu_seq_k=row.cu_seq_k,
                positions_q=row.positions_q,
                exchange=row.exchange,
                ratio=self.compress_ratio,
                q=q,
                swa_k=swa_k,
                cmp_k=cmp_k,
                aicpu_metadata=build_quant_metadata(
                    ratio=self.compress_ratio,
                    q=q,
                    swa_k=swa_k,
                    cmp_k=cmp_k,
                    window_size=self.window_size,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    index_topk=self.index_topk,
                    index_num_heads=self.index_num_heads,
                    index_head_dim=self.index_head_dim,
                    topk_blocks=self.candidate_topk_blocks,
                    block_size=self.candidate_block_size,
                    num_rows=row.positions_q.numel(),
                ),
            )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if config._torchao_npu_config is None:
            raise ValueError(f"{type(self).__name__}.Config requires _torchao_npu_config")
        from torchao.quantization import quantize_

        quantize_(
            self,
            config._torchao_npu_config,
            filter_fn=lambda candidate, _fqn: candidate is self,
        )


register_pytree_node_for_dataclass(_V41QuantAttentionMetadata)


def _v41_attention_numbers(model_config) -> dict[str, int]:
    """The numbers the quantized attention's metadata is sized with, off the model config.

    One config describes one geometry: ``test_fused_geometry`` pins the same five widths (plus the
    candidate pool) on every layer, so they are read once here rather than restated on the
    replacement node by each layer.  Mixed stacks and disagreeing pools are refused instead.
    """
    layers = list(model_config.layers)
    attention = layers[0].attention
    indexer = attention.indexer
    first = (
        attention.n_heads,
        attention.head_dim,
        indexer.index_topk,
        indexer.num_index_heads,
        indexer.index_head_dim,
    )
    for layer in layers[1:]:
        other = (
            layer.attention.n_heads,
            layer.attention.head_dim,
            layer.attention.indexer.index_topk,
            layer.attention.indexer.num_index_heads,
            layer.attention.indexer.index_head_dim,
        )
        if other != first:
            raise ValueError(
                f"the stack mixes geometries ({first} vs {other}), which one quantized attention config cannot describe"
            )
    # The pool is declared per layer, on the selector that builds or searches it, so it is read
    # there; the model-level pair is the same statement and is checked against it rather than
    # silently preferred.  A missing field is an error, not a default: ``-1`` here would size the
    # metadata for no pool while the launches still pass one.
    pools = {
        (layer.attention.indexer.selector.candidate_topk_blocks, layer.attention.indexer.selector.candidate_block_size)
        for layer in layers
    }
    positive = {(blocks, size) for blocks, size in pools if blocks > 0}
    if len(positive) > 1:
        raise ValueError(f"the stack declares more than one candidate pool: {sorted(positive)}")
    pool = positive.pop() if positive else (-1, -1)
    declared = (int(model_config.candidate_topk_blocks), int(model_config.candidate_block_size))
    if pool[0] > 0 and declared != pool:
        raise ValueError(
            f"the candidate pool the layers declare ({pool}) disagrees with the model config's "
            f"({declared}): state one value."
        )
    return {
        "num_heads": first[0],
        "head_dim": first[1],
        "index_topk": first[2],
        "index_num_heads": first[3],
        "index_head_dim": first[4],
        "candidate_topk_blocks": pool[0],
        "candidate_block_size": pool[1],
    }


class _DeepSeekV4QuantizedLightningIndexerMetadataAdapter(DeepSeekV4LightningIndexerMetadata):
    """Bridge TorchTitan metadata to the TorchAO-NPU DeepSeek-V4 LI provider."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4LightningIndexerMetadata.Config):
        index_n_heads: int  # pyrefly: ignore [bad-override]
        index_head_dim: int  # pyrefly: ignore [bad-override]
        index_topk: int  # pyrefly: ignore [bad-override]
        quant_mode: int

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        from torchao_npu.quantized_modules.deepseek_v4.lightning_indexer_metadata import (
            QuantizedLightningIndexerMetadata as DeepSeekV4QuantizedLightningIndexerMetadata,
        )
        from torchao_npu.quantized_modules.deepseek_v4.lightning_indexer_metadata import (
            QuantizedLightningIndexerMetadataConfig as DeepSeekV4QuantizedLightningIndexerMetadataConfig,
        )

        cfg = cast("_DeepSeekV4QuantizedLightningIndexerMetadataAdapter.Config", config)
        kernel = cfg.li_kernel_config
        self._provider = DeepSeekV4QuantizedLightningIndexerMetadata(
            DeepSeekV4QuantizedLightningIndexerMetadataConfig(
                index_n_heads=cfg.index_n_heads,
                index_head_dim=cfg.index_head_dim,
                index_topk=cfg.index_topk,
                layout_q=kernel.layout_q,
                layout_k=kernel.layout_k,
                mask_mode=kernel.mask_mode,
                cmp_ratio=kernel.cmp_ratio,
                quant_mode=cfg.quant_mode,
            )
        )

    def __call__(self, metadata):
        ratio = self.config.li_kernel_config.cmp_ratio
        plan = metadata.plans.get(ratio)
        if plan is None:
            raise ValueError(f"LI compression ratio {ratio} is not present in attention metadata")
        if plan.gather_indices.numel() == 0:
            raise ValueError("batch has no complete compression block for the configured LI ratio")
        plan.li_metadata = self._provider(
            cu_seqlens_q=metadata.varlen.cu_seq_q,
            cu_seqlens_k=plan.cu_seqlens_cmp_k,
            cmp_residual_k=plan.block_remainder,
        )
        return metadata


def _quant_deepseek_v4_lightning_indexer(
    kernel_config: DeepSeekV4LightningIndexerKernelConfig,
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
        layout_q=kernel_config.layout_q,
        layout_k=kernel_config.layout_k,
        mask_mode=kernel_config.mask_mode,
        cmp_ratio=kernel_config.cmp_ratio,
        query_config=qk_config,
        key_config=qk_config,
        quant_mode=quant_mode,
        input_adapter=_prepare_quant_lightning_indexer_inputs,
    )


def _mxfp8_param_swap() -> ParamSwapConfig:
    mx_config = MXQuantizeConfig()
    return ParamSwapConfig(
        weight_config=mx_config,
        activation_config=mx_config,
    )


def _block_fp8_param_swap(
    *,
    enable_mxfp4_qat: bool = False,
    dst_type_max: float = 0.0,
    enable_fsdp_prequantize: bool = False,
    fsdp_prequantize_dtype: Literal["fp8", "mxfp4"] = "fp8",
) -> ParamSwapConfig:
    mxfp4_config = None
    if enable_mxfp4_qat:
        mxfp4_config = MXQuantizeConfig(
            elem_dtype=torch.float4_e2m1fn_x2,
            dst_type_max=dst_type_max,
        )
    return ParamSwapConfig(
        weight_config=BlockMXQuantizeConfig(
            mxfp4_fake_quantize_config=mxfp4_config,
            fsdp_prequantize=enable_fsdp_prequantize,
            fsdp_prequantize_dtype=fsdp_prequantize_dtype,
        ),
        activation_config=MXQuantizeConfig(),
    )


# extension.quantization.enable-hif8-save-quant-codes ops added to the
# selective-AC save list.
_QUANT_CODE_SAVE_OPS = (
    "npu.npu_quantize.default",
    "npu.npu_dynamic_quant.default",
    "npu.npu_grouped_matmul.default",
    "npu.npu_quant_matmul.default",
)

_AC_QUANT_CODES_PATCHED = [False]


def _save_quant_codes_under_selective_ac(enable: bool) -> None:
    """extension.quantization.enable-hif8-save-quant-codes: keep the quantized
    matmul's operands and result across the selective-AC boundary instead of
    re-deriving them in the backward.

    Must be called at model-setup time, before parallelization wraps the
    blocks: torchtitan reads ``_get_default_save_ops`` when it installs
    activation checkpointing.
    """
    if not enable or _AC_QUANT_CODES_PATCHED[0]:
        return
    try:
        from torchtitan.distributed import activation_checkpoint as ac
    except ImportError:
        return
    _AC_QUANT_CODES_PATCHED[0] = True
    orig = ac._get_default_save_ops

    def _get_default_save_ops():
        ops = orig()
        added, missing = [], []
        for path in _QUANT_CODE_SAVE_OPS:
            obj = torch.ops
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
            except (AttributeError, RuntimeError):
                missing.append(path)
                continue
            ops.add(obj)
            added.append(path)
        logger.info(
            "enable_hif8_save_quant_codes: added %s to the selective-AC save list%s",
            ", ".join(added) or "nothing",
            f" (not registered: {', '.join(missing)})" if missing else "",
        )
        return ops

    ac._get_default_save_ops = _get_default_save_ops


_AC_BLOCK_OPS_PATCHED = [False]

# extension.quantization.save-block-ops-level levels -> the transformer-block ops
# added to the selective-AC save list, grouped by the module they belong to
# and ordered by critical-path ms per saved GiB (best first).
_BLOCK_SAVE_OPS = (
    # 1: attention
    (1, "cann_ops_transformer.sparse_flash_mla.default"),
    (1, "cann_ops_transformer.lightning_indexer.default"),
    # 2: mHC
    (2, "npu.npu_mhc_pre.default"),
    (2, "cann_ops_transformer.mhc_post.default"),
    (2, "npu.npu_mhc_sinkhorn.default"),
    # 3: MoE routing
    (3, "npu.npu_moe_token_permute.default"),
    (3, "torchtitan_npu.npu_moe_re_routing.default"),
    (3, "torchtitan_npu.npu_moe_token_unpermute.default"),
)


def _resolve_torch_op(path: str):
    """``torch.ops`` lookup by dotted path, or None if not registered yet."""
    obj = torch.ops
    try:
        for part in path.split("."):
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


def _save_block_ops_under_selective_ac(level: int) -> None:
    """extension.quantization.save-block-ops-level: keep the attention / mHC /
    MoE-routing results across the selective-AC boundary instead of
    re-deriving them in the backward.

    torchtitan's ``_get_default_save_ops()`` doesn't know about custom NPU
    ops, so the min-cut partitioner always recomputes them. ``_BLOCK_SAVE_OPS``
    orders them best-first by profiled win, an upper bound rather than a
    guarantee -- a recomputed kernel off the critical path may already be
    overlapped.

    NOT bit-exact (forward and recompute don't compile to the same fused
    kernel), so this needs a loss A/B before being trusted.

    Resolution is deferred to call time (inside the returned closure, not
    at patch time) because some ops register only once model modules have
    imported ``cann_ops_transformer``.
    """
    if level <= 0 or _AC_BLOCK_OPS_PATCHED[0]:
        return
    try:
        from torchtitan.distributed import activation_checkpoint as ac
    except ImportError:
        return
    wanted = tuple(path for lvl, path in _BLOCK_SAVE_OPS if lvl <= level)
    if not wanted:
        return
    _AC_BLOCK_OPS_PATCHED[0] = True
    orig = ac._get_default_save_ops

    def _get_default_save_ops():
        ops = orig()
        added, missing = [], []
        for path in wanted:
            op = _resolve_torch_op(path)
            if op is None:
                # Not registered in this build/config (e.g. the lightning
                # indexer is off for this model). A miss is a lost
                # optimization, never a correctness problem.
                missing.append(path)
            else:
                ops.add(op)
                added.append(path)
        logger.info(
            "save_block_ops_level=%d: added %s to the selective-AC save list%s",
            level,
            ", ".join(added) or "nothing",
            f" (not registered: {', '.join(missing)})" if missing else "",
        )
        return ops

    ac._get_default_save_ops = _get_default_save_ops


def _hif8_param_swap(save_quant_codes: bool) -> ParamSwapConfig:
    # Current/constant tensor scaling only: every quantize call recomputes
    # its scale via npu_dynamic_quant, no persisted per-parameter state.
    hif8_config = HiF8QuantizeConfig()
    _save_quant_codes_under_selective_ac(save_quant_codes)
    return ParamSwapConfig(
        weight_config=hif8_config,
        activation_config=hif8_config,
    )


def _quantization_converter(
    base_config: AOBaseConfig | None,
    filter_fn: ConfigFilterFn | None = None,
    *,
    model_compile_enabled: bool,
    replacement_config_type: type | None = None,
    replacement_kwargs: dict[str, object] | None = None,
    prequantize_fqns: tuple[str, ...] | None = None,
) -> NpuQuantizeConverter.Config:
    return NpuQuantizeConverter.Config(
        base_config=base_config,
        filter_fn=filter_fn,
        model_compile_enabled=model_compile_enabled,
        replacement_config_type=replacement_config_type,
        replacement_kwargs=replacement_kwargs or {},
        prequantize_fqns=prequantize_fqns,
    )


def _resolve_prequantize_fqns(
    user_fqns: list[str] | None,
    *,
    enabled: bool,
) -> dict[str, tuple[str, ...]] | None:
    """Resolve the per-scope pre-quantize whitelists for the recipe converters.

    Returns ``None`` when pre-quantize is off (no policy is applied). With the
    master switch on, an unset user list falls back to the recipe default
    whitelist per scope (logged), while an explicit list applies to every
    scope. An explicitly empty list is a contradictory configuration and is
    rejected by ``QuantizationExtensionConfig.validate`` upstream. The
    default/user distinction also drives the zero-match strictness in
    :func:`_check_prequantize_policy`.
    """
    if not enabled:
        return None
    if user_fqns is None:
        logger.info(
            "fsdp_prequantize whitelist = recipe default (%s)",
            {scope: list(fqns) for scope, fqns in _PREQUANTIZE_DEFAULT_FQNS.items()},
        )
        return dict(_PREQUANTIZE_DEFAULT_FQNS)
    logger.info("fsdp_prequantize whitelist = user (%d patterns)", len(user_fqns))
    resolved = tuple(user_fqns)
    return dict.fromkeys(_PREQUANTIZE_DEFAULT_FQNS, resolved)


def _recipe_converters(
    recipe: str,
    *,
    model_type: _ModelType,
    enable_sparse_attention_quantization: bool = False,
    enable_mxfp4_qat: bool,
    dst_type_max: float,
    enable_fsdp_prequantize: bool,
    model_compile_enabled: bool,
    li_quantization: LIQuantization | None = None,
    li_kernel_config: DeepSeekV4LightningIndexerKernelConfig | None = None,
    kv_norm_quantization: KVNormQuantizationConfig | None = None,
    prequantize_fqns_by_scope: dict[str, tuple[str, ...]] | None = None,
    model_config=None,
    fsdp_prequantize_dtype: Literal["fp8", "mxfp4"] = "fp8",
    enable_hif8_save_quant_codes: bool = False,
) -> list[QuantizationConverter.Config]:
    converters: list[QuantizationConverter.Config]
    try:
        filters = _CONFIG_FILTERS_BY_MODEL_TYPE[model_type]
    except KeyError as exc:
        raise ValueError(f"unsupported quantization model type {model_type!r}") from exc
    dense_filter = filters["dense"]
    if model_type == "kimi_k3" and (
        recipe != "all_mxfp8"
        or enable_sparse_attention_quantization
        or li_quantization is not None
        or (kv_norm_quantization is not None and kv_norm_quantization.format is not None)
    ):
        raise ValueError("Kimi K3 supports the all_mxfp8 expert recipe without DeepSeek-specific quantization options")

    if recipe == "all_mxfp8":
        converters = [
            _quantization_converter(
                _mxfp8_param_swap(),
                any_config_filter(dense_filter, filters["routed_expert"]),
                model_compile_enabled=model_compile_enabled,
            )
        ]

    elif recipe == "all_hif8":
        converters = [
            _quantization_converter(
                _hif8_param_swap(enable_hif8_save_quant_codes),
                any_config_filter(dense_filter, filters["routed_expert"]),
                model_compile_enabled=model_compile_enabled,
            )
        ]

    else:
        routed_config = _block_fp8_param_swap(
            enable_mxfp4_qat=enable_mxfp4_qat,
            dst_type_max=dst_type_max,
            enable_fsdp_prequantize=enable_fsdp_prequantize,
            fsdp_prequantize_dtype=fsdp_prequantize_dtype,
        )
        if recipe == "mix":
            dense_config = _mxfp8_param_swap()
        elif recipe == "all_block_fp8":
            dense_config = _block_fp8_param_swap(
                enable_fsdp_prequantize=enable_fsdp_prequantize,
                # Dense weights retain Block FP8; MXFP4 QAT is expert-only.
                fsdp_prequantize_dtype="fp8",
            )
        else:
            raise ValueError(f"recipe must be one of {_SUPPORTED_RECIPES}, got {recipe!r}")

        # A scope only participates in the whitelist policy when its param
        # swap actually pre-quantizes (e.g. "mix" runs dense as plain MX:
        # passing the dense whitelist there would make every dense pattern a
        # zero-match error in _check_prequantize_policy).
        def _scope_fqns(config: ParamSwapConfig, scope: str) -> tuple[str, ...] | None:
            if prequantize_fqns_by_scope is None:
                return None
            if not _is_prequantizing_block_mx(config.weight_config):
                return None
            return prequantize_fqns_by_scope.get(scope)

        converters = [
            _quantization_converter(
                dense_config,
                dense_filter,
                model_compile_enabled=model_compile_enabled,
                prequantize_fqns=_scope_fqns(dense_config, "dense"),
            ),
            _quantization_converter(
                routed_config,
                filters["routed_expert"],
                model_compile_enabled=model_compile_enabled,
                prequantize_fqns=_scope_fqns(routed_config, "routed_expert"),
            ),
        ]

    if li_quantization is not None:
        if model_type == "v41" and not enable_sparse_attention_quantization:
            # The quantized selector reads the entry the quantized sparse attention builds: its
            # schedules are that backend's, and the unquantized attention's entry carries a
            # different kernel's.  One without the other is a mismatch, not a degraded mode.
            raise ValueError(
                "DeepSeek-V4.1 QLI/QSLI quantization requires "
                "--extension.quantization.enable-sparse-attention-quantization: the quantized "
                "selector launches the quantized sparse attention's schedules."
            )
        li_config: DeepSeekV4QuantLightningIndexerConfig | DeepSeekV41QuantLightningIndexerConfig
        if model_type == "v41":
            if li_quantization != "mxfp4":
                raise ValueError("DeepSeek-V4.1 QLI/QSLI supports only li_quantization='mxfp4'")
            li_config = DeepSeekV41QuantLightningIndexerConfig()
        else:
            li_config = _quant_deepseek_v4_lightning_indexer(
                li_kernel_config or DeepSeekV4LightningIndexerKernelConfig(),
                li_quantization=li_quantization,
                dst_type_max=dst_type_max,
            )
        converters.append(
            _quantization_converter(
                li_config,
                filters["lightning_indexer"],
                model_compile_enabled=model_compile_enabled,
            )
        )
        if model_type == "v4":
            converters.append(
                _quantization_converter(
                    None,
                    filters["lightning_indexer_metadata"],
                    model_compile_enabled=model_compile_enabled,
                    replacement_config_type=_DeepSeekV4QuantizedLightningIndexerMetadataAdapter.Config,
                    replacement_kwargs={
                        "quant_mode": cast("DeepSeekV4QuantLightningIndexerConfig", li_config).quant_mode,
                    },
                )
            )
    if enable_sparse_attention_quantization:
        if "sparse_attention" in filters:
            converters.append(
                _quantization_converter(
                    DeepSeekV41QuantCompressorConfig(),
                    filters["compressor"],
                    model_compile_enabled=model_compile_enabled,
                )
            )
            # The quantized attention is its own backend: its config carries the numbers its
            # metadata is sized with and builds that metadata itself, so nothing here relies on
            # the unquantized override being installed.
            v41_geometry = _v41_attention_numbers(model_config)
            converters.append(
                _quantization_converter(
                    None,
                    filters["sparse_attention"],
                    model_compile_enabled=model_compile_enabled,
                    replacement_config_type=QuantV41SparseAttention.Config,
                    replacement_kwargs={
                        **v41_geometry,
                        "_torchao_npu_config": DeepSeekV41QuantSparseAttentionConfig(),
                    },
                )
            )
        else:
            logger.warning(
                "Sparse attention quantization is enabled, but model type %s has no "
                "sparse_attention filter; skipping the sparse attention replacement.",
                model_type,
            )

    if kv_norm_quantization is not None and kv_norm_quantization.format is not None:
        converters.append(
            KVNormFakeQuantConverter.Config(
                base_config=MXQuantizeConfig(
                    elem_dtype=torch.float8_e4m3fn, block_size=kv_norm_quantization.block_size
                ),
                target_fqns=tuple(kv_norm_quantization.fqns),
                model_compile_enabled=model_compile_enabled,
            )
        )
    return converters


def _get_deepseek_v4_li_kernel_config(model_config) -> DeepSeekV4LightningIndexerKernelConfig:
    metadata_matches = [
        match
        for match in model_config.traverse(DeepSeekV4LightningIndexerMetadata.Config)
        if _DSV4_CONFIG_FILTERS["lightning_indexer_metadata"](match[1], match[0])
    ]
    li_matches = [
        match
        for match in model_config.traverse(DeepSeekV4LightningIndexer.Config)
        if _DSV4_CONFIG_FILTERS["lightning_indexer"](match[1], match[0])
    ]
    if len(metadata_matches) != 1:
        raise ValueError("Quantized LI requires one lightning_indexer_metadata config")
    if not li_matches:
        raise ValueError("Quantized LI requires a DeepSeek-V4 LightningIndexer config")

    metadata = metadata_matches[0][1]
    for fqn, li_config, _, _ in li_matches:
        if li_config.li_kernel_config != metadata.li_kernel_config or li_config.index_topk != metadata.index_topk:
            raise ValueError(f"LightningIndexer at {fqn!r} disagrees with its metadata config")
    return metadata.li_kernel_config


def apply_quantization_converter(
    model_spec: ModelSpec | None,
    quantization_config: QuantizationExtensionConfig,
    *,
    model_compile_enabled: bool,
) -> ModelSpec:
    """Apply the selected TorchAO-NPU recipe to an existing BF16 model spec.

    This runs during ``TrainerEx`` construction and before base ``Trainer``
    initialization.  The regular model registry therefore remains responsible
    only for constructing the high-precision config tree, while this function
    performs the optional low-precision config replacement.
    """

    if model_spec is None:
        raise ValueError("TorchAO-NPU quantization requires model_spec to be configured")
    # Recipe-independent, so it is installed here rather than in
    # _hif8_param_swap: these are block ops, not quantized ops, and MXFP8 wants
    # them saved just as much as HiF8 does. Off by default -- at level 0
    # nothing is patched and the partition is the one we ship today.
    _save_block_ops_under_selective_ac(quantization_config.save_block_ops_level)
    if not quantization_config.enable_quantized_training:
        return model_spec
    quantization_config.validate()

    model_config = model_spec.model
    model_type = _model_type_for_spec(model_spec)

    li_kernel_config = (
        _get_deepseek_v4_li_kernel_config(model_spec.model)
        if model_type == "v4" and quantization_config.li_quantization is not None
        else None
    )
    prequantize_fqns_by_scope = _resolve_prequantize_fqns(
        quantization_config.fsdp_prequantize_fqns,
        enabled=quantization_config.enable_fsdp_prequantize,
    )
    converter_configs = _recipe_converters(
        quantization_config.recipe,
        model_type=model_type,
        enable_sparse_attention_quantization=quantization_config.enable_sparse_attention_quantization,
        enable_mxfp4_qat=quantization_config.enable_mxfp4_qat,
        dst_type_max=quantization_config.dst_type_max,
        enable_fsdp_prequantize=quantization_config.enable_fsdp_prequantize,
        fsdp_prequantize_dtype=quantization_config.fsdp_prequantize_dtype,
        model_compile_enabled=model_compile_enabled,
        li_quantization=quantization_config.li_quantization,
        li_kernel_config=li_kernel_config,
        kv_norm_quantization=quantization_config.kv_norm_quantization,
        prequantize_fqns_by_scope=prequantize_fqns_by_scope,
        enable_hif8_save_quant_codes=quantization_config.enable_hif8_save_quant_codes,
        model_config=model_config,
    )
    validate_converter_order(converter_configs)

    built_converters: list[object] = []
    for converter_config in converter_configs:
        converter = converter_config.build()
        built_converters.append(converter)
        model_config = converter.convert(model_config)

    # Aggregate the per-node whitelist decisions. Only patterns actually handed
    # to a pre-quantizing scope participate (e.g. "mix" runs dense as plain MX,
    # so dense patterns are not passed there and must not be reported as
    # unmatched). Zero-match handling is source-dependent: user whitelists fail
    # fast on typos, the recipe default tolerates absent optional modules.
    # Shard-alignment validation depends on the runtime FSDP sharding and is
    # left to the fail-fast guard in the wrapper tensor.
    effective_fqns = tuple(
        dict.fromkeys(
            fqn
            for converter_config in converter_configs
            for fqn in (getattr(converter_config, "prequantize_fqns", None) or ())
        )
    )
    if quantization_config.enable_fsdp_prequantize and not effective_fqns:
        # No scope of the recipe pre-quantizes (e.g. "all_mxfp8" is pure MX),
        # so the master switch and the whitelist would both be silently inert.
        raise ValueError(
            f"enable_fsdp_prequantize=True but recipe {quantization_config.recipe!r} pre-quantizes no "
            "weight scope; use all_block_fp8 (or mix, which pre-quantizes routed experts only) "
            "or disable enable_fsdp_prequantize."
        )
    _check_prequantize_policy(
        built_converters,
        effective_fqns,
        strict=quantization_config.fsdp_prequantize_fqns is not None,
    )

    logger.info(
        "Applied TorchAO-NPU recipe=%s, sparse_attention_quantization=%s, mxfp4_qat=%s, "
        "li_quantization=%s, kv_norm_quantization=%s, kv_norm_block_size=%s, "
        "dst_type_max=%s, enable_fsdp_prequantize=%s, fsdp_prequantize_dtype=%s",
        quantization_config.recipe,
        quantization_config.enable_sparse_attention_quantization,
        quantization_config.enable_mxfp4_qat,
        quantization_config.li_quantization,
        quantization_config.kv_norm_quantization.format,
        quantization_config.kv_norm_quantization.block_size,
        quantization_config.dst_type_max,
        quantization_config.enable_fsdp_prequantize,
        quantization_config.fsdp_prequantize_dtype,
    )
    return replace(model_spec, model=model_config)


__all__ = [
    "ConfigFilterFn",
    "NpuQuantizeConverter",
    "NpuQuantizedAscGroupedExpertsModule",
    "any_config_filter",
    "apply_quantization_converter",
    "match_config_fqn_suffix",
]
