# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Metadata extension and LI provider seams for model-built attention metadata.

The model owns its per-batch metadata construction (``build_attention_masks``
on the Decoder — including its own context-parallel handling).  The
``LightningIndexerMetadata`` provider owns ratio-4 LI metadata, while
``MetadataExtension`` owns the remaining backend-specific metadata.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Annotated, Any, cast

import tyro
from torch.utils._pytree import GetAttrKey, KeyEntry, register_pytree_node
from torchtitan.config.configurable import Configurable


class SparseAttentionMetadata(Configurable):
    """Select the sparse-attention metadata implementation independently."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        extension: Annotated[type[Configurable] | None, tyro.conf.Suppress] = None

    def __init__(self, config: Config):
        del config


class CompressorMetadata(Configurable):
    """Build compressor metadata only when explicitly selected."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        callback: Annotated[Callable[[Any], None] | None, tyro.conf.Suppress] = None

    def __init__(self, config: Config):
        self.callback = config.callback

    def __call__(self, metadata) -> None:
        if self.callback is not None:
            self.callback(metadata)


@dataclass(kw_only=True, slots=True)
class LightningIndexerKernelConfig:
    """LI kernel contract shared by metadata and kernel callers."""

    layout_q: str = "TND"
    layout_k: str = "TND"
    mask_mode: int = 3
    cmp_ratio: int = 4


class MetadataExtension(Configurable):
    """The default identity post-process of the built attention masks."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        sparse_attn: SparseAttentionMetadata.Config = field(default_factory=SparseAttentionMetadata.Config)
        compressor: CompressorMetadata.Config = field(default_factory=CompressorMetadata.Config)
        window_size: int = 0
        """The sliding-window size (model-config constant), consumed by the
        vendor extension's kernel metadata fills."""
        num_heads: int | None = None
        head_dim: int | None = None
        index_n_heads: int | None = None
        index_head_dim: int | None = None
        index_topk: int | None = None
        """Static sparse-attention geometry consumed by vendor metadata
        extensions when the selected path requires it."""
        compress_ratios: tuple[int, ...] = ()
        """The stack's per-layer compression ratios (model-config constant).
        Vendor extensions that precompute per-ratio kernel metadata fill it
        for exactly these, window-only layers (ratio 0) included; empty means
        the ratios are read off the metadata's own frames instead."""
        li_kernel_config: LightningIndexerKernelConfig = field(
            default_factory=LightningIndexerKernelConfig,
        )
        """Shared LI operator contract available to backend extensions."""

        def build(self, **kwargs):
            from torchtitan.config import derive

            selected: Configurable.Config = self
            if self.sparse_attn.extension is not None:
                selected = derive(self, self.sparse_attn.extension.Config)
            return Configurable.Config.build(selected, **kwargs)

    def __init__(self, config: Config):
        self.config = config
        self._compressor_metadata = config.compressor.build()

    def prepare_compressor_metadata(self, metadata) -> None:
        self._compressor_metadata(metadata)

    def __call__(self, metadata):
        self.prepare_compressor_metadata(metadata)
        return metadata


class LightningIndexerMetadata(Configurable):
    """Independent metadata provider for the LightningIndexer kernel.

    The provider is separate from ``MetadataExtension`` because BF16 and
    quantized LI kernels use different opaque metadata contracts. SMLA,
    SMLAG, and SLIG metadata remains owned by ``MetadataExtension``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        window_size: int = 0
        index_n_heads: int | None = None
        index_head_dim: int | None = None
        index_topk: int | None = None
        li_kernel_config: LightningIndexerKernelConfig = field(
            default_factory=LightningIndexerKernelConfig,
        )
        """LI layout, mask, and compression-ratio contract for the provider."""

    def __init__(self, config: Config):
        self.config = config

    def __call__(self, metadata):
        return metadata


def register_pytree_node_for_dataclass(cls: type) -> None:
    """Register a kw-only dataclass as a pytree node (idempotent).

    The graph_trainer's ``minimal_fx_tracer`` requires every ``attention_masks``
    leaf to be a tensor/primitive; an unregistered dataclass is rejected as a
    single non-primitive leaf.  Shared by the DSV4 and V4.1 metadata stacks.
    """
    from torch.utils._pytree import SUPPORTED_NODES

    if cls in SUPPORTED_NODES:
        return
    # Eager-only caches must not add batch-dependent primitive leaves to a graph.
    field_names = [f.name for f in fields(cls) if f.metadata.get("pytree", True)]

    def flatten(obj):
        return [getattr(obj, name) for name in field_names], None

    def flatten_with_keys(obj):
        # ``GetAttrKey`` is structurally assignable to the ``KeyEntry`` protocol
        # at runtime, but pyrefly treats list as invariant, so cast the list.
        keys = cast(
            "list[tuple[KeyEntry, Any]]",
            [(GetAttrKey(name), getattr(obj, name)) for name in field_names],
        )
        return keys, None

    def unflatten(values, context):
        return cls(**dict(zip(field_names, values, strict=True)))

    register_pytree_node(
        cls,
        flatten,
        unflatten,
        flatten_with_keys_fn=flatten_with_keys,
        serialized_type_name=f"{cls.__module__}.{cls.__name__}",
    )
