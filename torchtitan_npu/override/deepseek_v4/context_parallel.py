# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""BF16 document context parallelism selected by a TorchTitan model-config transform."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Literal, cast

import spmd_types as spmd
import torch
from torch.distributed._functional_collectives import all_to_all_single
from torchtitan.config.transform import ModelConfigTransform, convert_config_type, transform_model_config_
from torchtitan.distributed.context_parallel import cp_all_gather_tensors, get_cp_input_seq_len, shard_tensors
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.base.module import Module, ModuleList
from torchtitan.models.common.attention import create_varlen_metadata_for_document
from torchtitan.models.common.attention.cp_attention import CPInnerAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.models.deepseek_v4.model import DeepSeekV4Model as TorchTitanV4Model

from torchtitan_npu.models.deepseek_v4.compressor import Compressor as V4Compressor
from torchtitan_npu.models.deepseek_v4.metadata import DeepSeekV4Metadata, build_v4_metadata
from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model
from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.models.deepseek_v4_1.compressor import Compressor as V41Compressor
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Metadata, DeepSeekV41Model

from .cp_metadata import DocumentCPLoadBalancer, DocumentCPMetadata, DocumentCPPlan


def cp_group():
    group = spmd_mesh_group("cp")
    if group is None:
        raise RuntimeError("DeepSeek document CP requires an active CP mesh.")
    return group


def exchange(tensor, send, recv):
    return all_to_all_single(tensor.contiguous(), output_split_sizes=recv, input_split_sizes=send, group=cp_group())


def gather(tensor, reduce_dtype):
    return None if tensor is None else cp_all_gather_tensors((tensor,), reduce_dtype=reduce_dtype)[0]


def slice_part(tensor, part):
    return None if tensor is None else tensor[part]


@dataclass(frozen=True, kw_only=True, slots=True)
class PackedAttention:
    row: DeepSeekV4Metadata | DeepSeekV41Metadata
    window_size: int
    head_tail: bool
    attach_records: Callable[..., Any]


class CPAttention(CPInnerAttention):
    """Keep Q local, exchange the SWA halo, and gather compressed keys."""

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config):
        kernel: Any
        head_tail: bool = False
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"

        @property
        def attention_metadata_key(self):
            return self.kernel.attention_metadata_key

        @property
        def window_size(self):
            return self.kernel.window_size

        @property
        def index_topk(self):
            return self.kernel.index_topk

        @property
        def aux_loss(self):
            return self.kernel.aux_loss

        def build_attention_metadata(self, positions, **kwargs) -> Any:
            from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import (
                AscV41SparseAttention,
                build_fused_metadata,
            )

            from .sparse_attn.ascendc import AscCompressedSparseInnerAttention, build_aicpu_metadata

            kernel = self.kernel
            if isinstance(kernel, AscCompressedSparseInnerAttention.Config):
                documents = create_varlen_metadata_for_document(positions, **kwargs)
                with spmd.no_typecheck():
                    cu = torch.unique_consecutive(documents.cu_seq_q)
                row = build_v4_metadata(positions, kernel.compress_ratio, cu_seq_q=cu, build_slots=False)
                attach_records = partial(build_aicpu_metadata, geometry=kernel.geometry)
            elif isinstance(kernel, AscV41SparseAttention.Config):
                row = CompressedSparseInnerAttention2.Config.build_attention_metadata(kernel, positions, **kwargs)
                attach_records = partial(build_fused_metadata, ratio=row.ratio, geometry=kernel.geometry)
            else:
                raise ValueError("DeepSeek document CP requires an AscendC sparse attention backend.")
            return PackedAttention(
                row=row, window_size=kernel.window_size, head_tail=self.head_tail, attach_records=attach_records
            )

    def __init__(self, config: Config):
        super().__init__()
        self.inner = config.kernel.build()
        self.reduce_dtype = getattr(torch, config.reduce_dtype)

    @property
    def aux_loss(self):
        return self.inner.aux_loss

    @staticmethod
    def prepare_cp_metadata(attention_metadata: PackedAttention, *, permutation, plan=None):
        if permutation is None:
            raise ValueError("DeepSeek document CP requires a document permutation.")
        row = attention_metadata.row
        if plan is None:
            group = cp_group()
            plan = DocumentCPPlan.build(
                permutation,
                row.q.cu_seqlens,
                cp_size=group.size(),
                cp_rank=group.rank(),
                head_tail=attention_metadata.head_tail,
            )

        def build_entry(**frames):
            return attention_metadata.attach_records(replace(row, **frames))

        return plan.prepare(
            row,
            window_size=attention_metadata.window_size,
            overlap=isinstance(row, DeepSeekV4Metadata) and row.ratio == 4,
            build_entry=build_entry,
        )

    def forward(
        self,
        q,
        swa_k,
        attn_sink,
        attention_metadata: DocumentCPMetadata,
        *,
        cmp_k=None,
        topk_indices=None,
        topk_scores=None,
        aux_loss_denominator=None,
        padding_mask_T=None,
    ):
        keys = gather(cmp_k, self.reduce_dtype)
        route = attention_metadata.window
        received = exchange(swa_k.index_select(0, route.send_index), route.send_split, route.recv_split)
        available = torch.cat((swa_k, received))

        def attend(part, window_index, entry):
            return self.inner(
                q[part],
                available.index_select(0, window_index),
                attn_sink,
                attention_metadata=entry,
                cmp_k=keys,
                topk_indices=slice_part(topk_indices, part),
                topk_scores=slice_part(topk_scores, part),
                padding_mask_T=slice_part(padding_mask_T, part),
                aux_loss_denominator=aux_loss_denominator,
            )

        outputs = [
            attend(part, window_index, entry)
            for part, window_index, entry in zip(
                attention_metadata.query_parts(q.size(0)), route.key_indices, attention_metadata.entries, strict=True
            )
        ]
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


class CPIndexer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        kernel: Any
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"

        @property
        def candidate_topk_blocks(self):
            return self.kernel.candidate_topk_blocks

        @property
        def candidate_block_size(self):
            return self.kernel.candidate_block_size

    def __init__(self, config: Config):
        super().__init__()
        self.inner = config.kernel.build()
        self.reduce_dtype = getattr(torch, config.reduce_dtype)

    @property
    def return_scores(self):
        return self.inner.return_scores

    @return_scores.setter
    def return_scores(self, enabled):
        self.inner.return_scores = enabled

    def forward(self, idx_q, idx_k, idx_w, attention_metadata: DocumentCPMetadata, **kwargs):
        keys = gather(idx_k, self.reduce_dtype)

        def select(part, entry):
            return self.inner(
                idx_q[part],
                keys,
                idx_w[part],
                attention_metadata=entry,
                **{name: slice_part(value, part) for name, value in kwargs.items()},
            )

        outputs = [
            select(part, entry)
            for part, entry in zip(
                attention_metadata.query_parts(idx_q.size(0)), attention_metadata.entries, strict=True
            )
        ]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(None if head is None else torch.cat((head, tail)) for head, tail in zip(*outputs, strict=True))


def compressor_input_layout(x, metadata: DocumentCPMetadata, *, overlap: bool = False):
    """Restore the flat compressor slab and, for V4 CSA, its overlap block.

    Returns ``(slab, positions, compressor_cu_seqlens, trim)``: the contiguous token interval
    this rank pools, the matching document-relative positions, the slab-local document
    boundaries, and how many pooled rows the overlap carry leaves to cut off the front.
    """
    flat = metadata.flat
    send = x if flat.send_index is None else x.index_select(0, flat.send_index)
    slab = exchange(send, flat.send_split, flat.recv_split).index_select(0, flat.recv_index)
    positions = metadata.position_k
    trim = 0
    if overlap:
        group = cp_group()
        rank, size, ratio = group.rank(), group.size(), metadata.entries[0].ratio
        send_split = [ratio if peer == rank + 1 else 0 for peer in range(size)]
        recv_split = [ratio if peer == rank - 1 else 0 for peer in range(size)]
        carry = exchange(slab[-ratio:][: sum(send_split)], send_split, recv_split)
        slab = torch.cat((carry, slab))
        positions = torch.cat((positions.new_zeros(carry.size(0)), positions))
        trim = carry.size(0) // ratio
    return slab, positions, metadata.compressor_cu_seqlens, trim


class CPCompressor(V4Compressor):
    """Pool the contiguous key slab, with a one-block carry for CSA."""

    @dataclass(kw_only=True, slots=True)
    class Config(V4Compressor.Config):
        pass

    def forward(self, x, positions, cu_seqlens=None, *, attention_metadata=None):
        if not isinstance(attention_metadata, DocumentCPMetadata):
            return super().forward(x, positions, cu_seqlens, attention_metadata=attention_metadata)
        slab, positions, cu_seqlens, trim = compressor_input_layout(x, attention_metadata, overlap=self.overlap)
        return super().forward(slab, positions, cu_seqlens)[trim:]


class ContextParallelCompressor41(V41Compressor):
    """V4.1 compressor that restores its flat input before pooling the local slab."""

    @dataclass(kw_only=True, slots=True)
    class Config(V41Compressor.Config):
        pass

    def forward(self, x_TD, positions_T, cmp_k=None, *, attention_metadata=None):
        if self.is_source and attention_metadata is not None:
            x_TD, positions_T, _, _ = compressor_input_layout(x_TD, attention_metadata)
        return super().forward(x_TD, positions_T, cmp_k, attention_metadata=attention_metadata)


def engram_history(tokens: torch.Tensor, history_size: int) -> torch.Tensor:
    """Collect each token's predecessors before applying the shared document permutation."""
    rows = torch.arange(tokens.size(0), device=tokens.device)
    distances = torch.arange(1, history_size + 1, device=tokens.device)
    return tokens[(rows[:, None] - distances).clamp_min(0)]


def _shard_documents(input_dict, input_shardings, parallelism):
    # v0.4.0 discovers a balancer from the first full-attention backend and keys
    # its metadata by backend type. These sparse stacks share a document grid
    # across compression geometries. The rest of input sharding is upstream.
    metadata = input_dict.get("attention_metadata")
    if not metadata:
        raise ValueError("DeepSeek document CP requires packed input positions.")
    packed = next(iter(metadata.values()))
    group = cp_group()
    balancer = parallelism.context_parallel_load_balancer.build(
        seq_len=get_cp_input_seq_len(input_dict, input_shardings=input_shardings),
        attention_metadata=packed,
    )
    permutation = balancer.generate_permutation()
    plan = DocumentCPPlan.build(
        permutation,
        packed.row.q.cu_seqlens,
        cp_size=group.size(),
        cp_rank=group.rank(),
        head_tail=balancer.head_tail,
        boundaries=balancer.bounds,
    )
    input_dict["attention_metadata"] = {
        key: CPAttention.prepare_cp_metadata(entry, permutation=permutation, plan=plan)
        for key, entry in metadata.items()
    }
    return shard_tensors(input_dict, input_shardings=input_shardings, permutation=permutation)


class _V4DecoderWithCP(TorchTitanV4Model):
    def __init__(self, config):
        # The pinned upstream constructor rejects CP before building its layers.
        Decoder.__init__(self, config)
        self.hc_mult = config.hc_mult
        self.n_mtp_layers = config.n_mtp_layers
        self.compress_ratios = list(config.compress_ratios)[: config.n_layers]
        self.n_main_layers = config.n_layers
        self.hc_head = config.hc_head.build()
        self.mtp_layers = ModuleList(layer.build() for layer in config.mtp_layers or ())


class _V4DocumentCPModel(DeepSeekV4Model, _V4DecoderWithCP):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4Model.Config):
        pass

    def _cp_shard(self, input_dict, input_shardings, parallelism_context, parallelism):
        return _shard_documents(input_dict, input_shardings, parallelism)


class _V41DocumentCPModel(DeepSeekV41Model):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV41Model.Config):
        pass

    def _cp_shard(self, input_dict, input_shardings, parallelism_context, parallelism):
        if self.engram_history_size:
            input_dict["engram_token_history"] = engram_history(input_dict["input"], self.engram_history_size)
            input_shardings["engram_token_history"] = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        return _shard_documents(input_dict, input_shardings, parallelism)


@dataclass(kw_only=True, slots=True)
class DeepSeekContextParallelTransform(ModelConfigTransform):
    """Install CP backends before TorchTitan validates the trainer config."""

    head_tail: bool = False

    def transform(self, model: Module.Config) -> Module.Config:
        if type(model) is DeepSeekV4Model.Config:
            decoder = cast("DeepSeekV4Model.Config", convert_config_type(model, _V4DocumentCPModel))
            is_v4 = True
        elif type(model) is DeepSeekV41Model.Config:
            decoder = cast("DeepSeekV41Model.Config", convert_config_type(model, _V41DocumentCPModel))
            is_v4 = False
        else:
            raise ValueError("DeepSeek document CP requires an eager V4 or V4.1 text model.")

        if is_v4:
            for _, compressor, parent, name in decoder.traverse(V4Compressor.Config):
                if not isinstance(compressor, CPCompressor.Config):
                    assert parent is not None and isinstance(name, str)
                    setattr(parent, name, convert_config_type(compressor, CPCompressor))

        for layer in (*decoder.layers, *(getattr(decoder, "mtp_layers", None) or ())):
            attention = layer.attention
            if not is_v4 and attention.compressor.is_source:
                if attention.compress_ratio == 4:
                    raise NotImplementedError("V4.1 document CP does not support ratio-4 overlap pooling.")
                attention.compressor = convert_config_type(attention.compressor, ContextParallelCompressor41)

            attention.inner_attention = CPAttention.Config(
                kernel=replace(attention.inner_attention, sharding_config=None), head_tail=self.head_tail
            )
            indexer = attention.indexer
            if indexer is not None and indexer.inner_indexer is not None:
                indexer.inner_indexer = CPIndexer.Config(kernel=replace(indexer.inner_indexer, sharding_config=None))
        return decoder


def configure_context_parallel[ConfigT: Module.Config](model: ConfigT, parallelism) -> ConfigT:
    if parallelism.context_parallel_degree == 1:
        return model
    if parallelism.tensor_parallel_degree != 1 or parallelism.pipeline_parallel_degree != 1:
        raise ValueError("DeepSeek document CP requires TP=PP=1.")
    balancer = parallelism.context_parallel_load_balancer
    if balancer is None:
        balancer = DocumentCPLoadBalancer.Config()
        parallelism.context_parallel_load_balancer = balancer
    elif not isinstance(balancer, DocumentCPLoadBalancer.Config):
        raise ValueError("DeepSeek CP requires DocumentCPLoadBalancer.Config.")
    transformed = transform_model_config_(model, [DeepSeekContextParallelTransform(head_tail=balancer.head_tail)])
    return cast("ConfigT", transformed)
