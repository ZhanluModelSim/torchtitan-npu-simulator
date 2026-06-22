# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as F
from einops import rearrange
from scipy.linalg import hadamard
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.common.rope import (
    _reshape_for_broadcast_complex as reshape_for_broadcast_complex,
)
from torchtitan.models.deepseek_v3.model import (
    Attention as DeepSeekV3Attention,
)
from torchtitan.models.deepseek_v3.model import (
    DeepSeekV3TransformerBlock,
)
from torchtitan.protocols.module import Module, ModuleDict

from torchtitan_npu.models.common.dsa_indexer_loss import (
    DSAIndexerLoss,
    DSAIndexerLossAutoScaler,
    DSAIndexerLossLoggingHelper,
)
from torchtitan_npu.models.deepseek_v3.model import DeepSeekV3ModelNpu

logger = logging.getLogger()


_Linear = Module.from_nn_module(nn.Linear)
_LayerNorm = Module.from_nn_module(nn.LayerNorm)


class DeepSeekV32ModelNpu(DeepSeekV3ModelNpu):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3ModelNpu.Config):
        # Model-level v32 fields. Per-attention architecture knobs (index_*,
        # enable_mla_absorb) live on ``Attention.Config`` defined later in this
        # module. ``enable_indexer_loss`` stays here because it gates trainer
        # -side logging (see ``torchtitan_npu.train``), not per-layer compute.
        enable_indexer_loss: bool = True
        save_format: str = "dcp"
        save_expert_format: str | None = None
        hf_save_dir: str | None = None
        save_patch_enabled: bool = False
        moe_impl: str = "standard"
        num_mtp_modules: int = 0

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            DeepSeekV3ModelNpu.Config.update_from_config(self, trainer_config=trainer_config, **kwargs)
            if not hasattr(trainer_config.training, "num_mtp_modules"):
                return
            new_mtp = trainer_config.training.num_mtp_modules
            if new_mtp == self.num_mtp_modules:
                return
            if new_mtp < 0:
                raise ValueError(f"num_mtp_modules must be >= 0, got {new_mtp}")
            # Registry currently always builds with ``num_mtp_modules=0``
            # (see ``_make_dsv32_model_config``). Refuse asymmetric updates
            # so the layer list stays consistent with ``num_mtp_modules``.
            if self.num_mtp_modules != 0:
                raise ValueError(
                    f"num_mtp_modules update not supported: registry built "
                    f"{self.num_mtp_modules} MTP layers but trainer requests "
                    f"{new_mtp}. Either match the registry value or set the "
                    f"registry default to 0."
                )
            # Late import: ``__init__.py`` imports this module, so importing it
            # at module top would create a cycle.
            from torchtitan_npu.models.deepseek_v32 import _extend_dsv32_layers_with_mtp

            n_dense_layers = sum(1 for l in self.layers if l.feed_forward is not None)
            new_layers = _extend_dsv32_layers_with_mtp(self.layers, n_dense_layers, new_mtp)
            ref_attn = self.layers[0].attention
            for layer_cfg in new_layers:
                layer_cfg.attention.rope_max_seq_len = ref_attn.rope_max_seq_len
                layer_cfg.attention.rope_factor = ref_attn.rope_factor
                layer_cfg.attention.rope_original_seq_len = ref_attn.rope_original_seq_len
            self.layers.extend(new_layers)
            self.num_mtp_modules = new_mtp

    def __init__(self, config: "DeepSeekV32ModelNpu.Config"):
        nn.Module.__init__(self)
        self.config = config
        self.tok_embeddings = config.tok_embeddings.build()
        self.rope = config.rope.build()
        self.register_buffer("freqs_cis", self.rope.cache, persistent=False)
        self.norm = config.norm.build()
        self.output = config.output.build()

        n_total = len(config.layers)
        n_main = n_total - config.num_mtp_modules
        self.layers = ModuleDict()
        for layer_id, layer_cfg in enumerate(config.layers):
            if layer_id < n_main:
                self.layers[str(layer_id)] = TransformerBlockV32(layer_cfg, layer_id, n_total)
            else:
                self.layers[str(layer_id)] = MTPModule(layer_cfg, layer_id, n_total)
        self.num_mtp_modules = config.num_mtp_modules
        self.n_main_layers = n_main
        # [TODO] refactor it . ``model_args`` is the legacy attribute name read by
        # parallelize.py state_dict_adapter.py and downstream tools. We alias it to
        # the Config object; new callers should access fields via the Config tree
        # (e.g. ``model_args.layers[0].attention.<...>``).
        self.model_args = config

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        """Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            attention_masks: Optional masks for flex/varlen attention.
            positions: Position ids for CP/TP; set by Trainer when context_parallel is enabled
                (see torchtitan v0.2.2 prepare_context_parallel_input). Defaults to None.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """
        residual = None
        seq_len = tokens.shape[1]
        seq_len -= self.num_mtp_modules
        h = self.tok_embeddings(tokens[:, :seq_len]) if self.tok_embeddings is not None else tokens[:, :seq_len]
        # Main model calculate
        for layer_id, layer in enumerate(self.layers.values()):
            if layer_id < self.n_main_layers:
                h, residual = layer(h, residual, self.freqs_cis, attention_masks, positions)
            else:
                break
        if residual is not None:
            h = h + residual
        prev_embed = h
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h.float()) if self.output is not None else h
        if self.num_mtp_modules <= 0:
            return output
        else:
            output_list = [None] * (1 + self.num_mtp_modules)
            output_list[0] = output  # pyrefly: ignore [unsupported-operation]
        # MTP module calculate
        for mtp_layer_id in range(self.num_mtp_modules):
            token_offset_id = mtp_layer_id + 1
            token_end_idx = token_offset_id + seq_len
            token_offset = tokens[:, token_offset_id:token_end_idx]
            input_offset = self.tok_embeddings(  # pyrefly: ignore [not-callable]
                token_offset
            )
            layer_id = mtp_layer_id + self.n_main_layers
            h, residual = self.layers[str(layer_id)](
                input_offset, prev_embed, self.freqs_cis, attention_masks, positions
            )
            if residual is not None:
                h = h + residual
            prev_embed = h
            h = self.norm(h) if self.norm is not None else h
            output = self.output(h.float()) if self.output is not None else h
            output_list[mtp_layer_id + 1] = output
        return output_list


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
    interleaved: bool = True,
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.
    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.
        interleaved: If False, use non-interleaved layout (Indexer path).
        positions: Global position ids when sequence is sharded (e.g. CP); optional.
    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast_complex(freqs_cis, x, positions)
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)


def hadamard_transform_ref(x, scale=1.0):
    """
    Eager implementation of the Hadamard transform
    Args:
        x:(torch.Tensor): input tensor
    """

    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(
        x,
        torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device),
    )
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    hadamard_transform = hadamard_transform_ref
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


def bf16_index(q: torch.Tensor, weight: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Perform index score using BF16 precision.
    """
    query = rearrange(q, "b s h d -> b h s d").to(torch.float32)
    key = rearrange(k, "b s h d -> b h d s").to(torch.float32)
    p = torch.matmul(query, key)
    relu_out = torch.nn.functional.relu(p)

    weight_out = relu_out * weight.permute(0, 2, 1, 3)

    reduce_out = torch.sum(weight_out, dim=1)
    return reduce_out


class Indexer(Module):
    def __init__(self, config: "Attention.Config"):
        super().__init__()
        self.dim: int = config.dim
        self.n_heads: int = config.index_n_heads
        self.head_dim: int = config.index_head_dim
        self.rope_head_dim: int = config.qk_rope_head_dim
        self.index_topk: int = config.index_topk
        self.q_lora_rank: int = config.q_lora_rank
        self.wq_b = _Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = _Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = _LayerNorm(self.head_dim)
        # weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32 for convenient.
        self.weights_proj = _Linear(self.dim, self.n_heads, dtype=torch.float32, bias=False)
        self.softmax_scale = self.head_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
        positions: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # rope in indexer is not interleaved
        q_pe = apply_rotary_emb(q_pe, freqs_cis, positions, interleaved=False)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        # rope in indexer is not interleaved
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, positions, interleaved=False).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1).unsqueeze(2)
        q = rotate_activation(q)
        k = rotate_activation(k)
        weights = self.weights_proj(x) * self.n_heads**-0.5
        weights = weights * self.softmax_scale
        return q, weights, k, end_pos


def get_attn_scores(
    query,
    key,
    attention_mask,
    attn_scale,
):
    """aggregate the main attention scores"""
    num_head_q = query.shape[1]
    num_head_k = key.shape[1]
    if num_head_q != num_head_k and num_head_k != 1:
        raise NotImplementedError(
            f"Only support num_head_q == num_head_k or num_head_k == 1. Current {num_head_q=}, {num_head_k=}."
        )

    attn = (query @ key.transpose(-1, -2)) * attn_scale

    if attention_mask is not None:
        attn.masked_fill_(attention_mask, float("-inf"))

    attn = F.softmax(attn, dim=-1, dtype=torch.float32)
    attn = attn.sum(dim=1)
    return attn


class DSV32_SDPA(Module):  # noqa: N801
    """Wrapper around `F.scaled_dot_product_attention` to make it CP compatible.

    This wrapper is needed because `F.scaled_dot_product_attention` is not
    a torch.nn.Module, and thus cannot be applied with _ContextParallel.
    We need to wrap it into a torch.nn.Module.

    Note:
        The forward function must have q, k, v as the first three arguments to be
        compatible with _ContextParallel.
    """

    sdpa_backends: ClassVar[list[SDPBackend]] = []

    def __init__(self, config: "Attention.Config") -> None:
        super().__init__()
        self.config = config
        self.compute_dsa_indexer_loss = DSAIndexerLoss.Config().build()
        if not self.sdpa_backends:
            # pyrefly: ignore [read-only]
            self.sdpa_backends = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        scale: float | None = None,
        q_indexer: torch.Tensor | None = None,
        k_indexer: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        end_pos: torch.Tensor | None = None,
        index_topk: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = q.size(0)

        if q_indexer is None:
            with sdpa_kernel(self.sdpa_backends, set_priority=True):
                output = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    scale=scale,
                    is_causal=(attn_mask is None),
                )
            return q.new_zeros(1), output

        # prepare attention_mask for sparse attention according to lightning_indexer score
        index_score = bf16_index(
            q_indexer.contiguous(),
            # pyrefly: ignore [missing-attribute]
            weights.unsqueeze(-1),
            # pyrefly: ignore [missing-attribute]
            k_indexer.contiguous(),
        )

        seqlen = index_score.size(1)
        if q.size(2) != seqlen:
            q = q.narrow(2, 0, seqlen)
        if k.size(2) != seqlen:
            k = k.narrow(2, 0, seqlen)
        if v.size(2) != seqlen:
            v = v.narrow(2, 0, seqlen)
        if attn_mask is None:
            attn_mask = torch.where(
                torch.triu(
                    torch.ones((bsz, seqlen, seqlen), dtype=q.dtype, device=q.device),
                    diagonal=1,
                )
                == 1,
                float("-inf"),
                0.0,
            )
        index_score += attn_mask
        # pyrefly: ignore [bad-argument-type]
        topk_score, topk_indices = index_score.topk(min(index_topk, end_pos), dim=-1)
        query_positions = torch.arange(seqlen, device=topk_indices.device).unsqueeze(0).unsqueeze(-1)
        valid_positions = topk_indices <= query_positions
        # scatter_/gather reject -1; invalid slots use seqlen-1, a causal-masked
        # future token, so the scattered 0 is absorbed (gather path zeros them below).
        topk_indices = torch.where(valid_positions, topk_indices, torch.full_like(topk_indices, seqlen - 1))
        index_mask = torch.full((bsz, seqlen, seqlen), float("-inf"), device=q.device).scatter_(-1, topk_indices, 0)
        attention_masks = attn_mask + index_mask
        attention_masks = torch.isinf(attention_masks) & (attention_masks < 0)
        attention_masks = attention_masks.unsqueeze(1)

        # compute sparse attention
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            attn_mask = ~attention_masks
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=scale, is_causal=False)

        # compute sparse lightning_indexer loss
        q_det = q.detach()
        k_det = k.detach()
        main_attn_dist = get_attn_scores(q_det, k_det, attention_masks, scale)
        selected_main_attn_dist = torch.gather(main_attn_dist, dim=-1, index=topk_indices)
        # Zero out sentinel-slot reads so the indexer KL loss does not depend on the
        # implicit softmax(-inf)=0 invariant of main_attn_dist[..., seqlen-1].
        selected_main_attn_dist = selected_main_attn_dist.masked_fill(~valid_positions, 0.0)
        loss = self.compute_dsa_indexer_loss(
            selected_main_attn_dist,
            topk_score,
            topk_indices,
            1.0,
        )
        return loss, output


DSASparseAttention = DSV32_SDPA


class PreAttention(Module):
    """
    Multi-head attention (MLA) module.
    """

    def __init__(self, config: "Attention.Config"):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.enable_mla_absorb = config.enable_mla_absorb

        # v32 always uses LoRA query projection (q_lora_rank > 0).
        assert config.wq_a is not None and config.wq_b is not None, (
            "DeepSeek V3.2 requires q_lora_rank > 0; Attention.Config must provide wq_a/wq_b"
        )
        self.wq_a = config.wq_a.build()
        self.q_norm = config.q_norm.build()
        self.wq_b = config.wq_b.build()
        self.wkv_a = config.wkv_a.build()
        self.kv_norm = config.kv_norm.build()
        self.wkv_b = config.wkv_b.build()
        self.softmax_scale = self.qk_head_dim**-0.5

        if config.rope_max_seq_len > config.rope_original_seq_len:
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
        self.indexer = Indexer(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        layer_id,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()

        # Query projection
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, -1, self.qk_head_dim)  # (bsz, seqlen, n_heads, qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis, positions=positions)
        q = torch.cat([q_nope, q_pe], dim=-1)

        # Key-value projection
        kv = self.wkv_a(x)  # (bsz, seqlen, kv_lora_rank + qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, positions=positions)  # (bsz, seqlen, 1, qk_rope_head_dim)

        if not self.enable_mla_absorb:
            kv = self.wkv_b(self.kv_norm(kv))  # (bsz, seqlen, n_heads * (qk_nope_head_dim + v_head_dim))
            kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

            q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
            k = k.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
            v = v.transpose(1, 2)  # (bsz, n_heads, seqlen, v_head_dim)

            q_indexer, weights, k_indexer, end_pos = self.indexer(
                x.detach(), qr.detach(), 0, freqs_cis, attention_masks, positions
            )
            return (
                q,
                k,
                v,
                attention_masks,
                self.softmax_scale,
                q_indexer,
                k_indexer,
                weights,
                end_pos,
                self.indexer.index_topk,
                None,
            )
        else:
            kv = self.kv_norm(kv)
            wkv_b_weight = self.wkv_b.weight.reshape(-1, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
            w_uk = wkv_b_weight[:, : self.qk_nope_head_dim, :]
            w_uv = wkv_b_weight[:, self.qk_nope_head_dim :, :]
            w_uv_t = w_uv.permute(0, 2, 1).contiguous()

            q_nope = torch.einsum("bshq,hqr->bshr", q_nope, w_uk)
            k_nope = kv.unsqueeze(2)
            v = kv.unsqueeze(2)

            k = torch.cat([k_nope, k_pe], dim=-1)
            q = torch.cat([q_nope, q_pe], dim=-1)

            q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
            k = k.transpose(1, 2)  # (bsz, 1, seqlen, qk_head_dim)
            v = v.transpose(1, 2)  # (bsz, 1, seqlen, v_head_dim)

            q_indexer, weights, k_indexer, end_pos = self.indexer(
                x.detach(), qr.detach(), 0, freqs_cis, attention_masks, positions
            )

            return (
                q,
                k,
                v,
                attention_masks,
                self.softmax_scale,
                q_indexer,
                k_indexer,
                weights,
                end_pos,
                self.indexer.index_topk,
                w_uv_t,
            )


class PostAttention(Module):
    def __init__(self, config: "Attention.Config", num_total_layers: int):
        super().__init__()
        self.enable_mla_absorb = config.enable_mla_absorb
        self.wo = config.wo.build()
        # Used by DSAIndexerLossLoggingHelper.save_loss_to_tracker to size the
        # per-layer loss vector. ``num_total_layers`` is dense layers + MTP modules.
        self.n_layers = num_total_layers

    def forward(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        w_uv_t: torch.Tensor,
        loss: torch.Tensor,
        layer_id: int,
    ):
        bsz, seqlen, _ = x.size()
        if self.enable_mla_absorb:
            output = torch.einsum("bhsr,hrv->bhsv", output, w_uv_t)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous()  # (bsz, seqlen, n_heads, v_head_dim)
        output = output.view(bsz, seqlen, -1)  # (bsz, seqlen, n_heads * v_head_dim)
        output = self.wo(output)
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_id, self.n_layers)
        output = DSAIndexerLossAutoScaler.apply(output, loss)
        return output  # (bsz, seqlen, dim)


class Attention(Module):
    """DeepSeek V3.2 three-stage attention (Pre + sparse SDPA + Post).

    Wraps an MLA-style ``PreAttention`` + the ``DSV32_SDPA`` sparse kernel +
    ``PostAttention`` (which projects with ``wo`` and emits the indexer
    auxiliary loss). Not a subclass of v3 ``Attention`` because the forward
    path is fundamentally different.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3Attention.Config):
        # v32-specific lightning indexer + MLA-absorb knobs.
        index_n_heads: int = 64
        index_head_dim: int = 128
        index_topk: int = 2048
        enable_mla_absorb: bool = True

    def __init__(self, config: Config, num_total_layers: int):
        super().__init__()
        self.pre_attention = PreAttention(config)
        self.inner_attention = DSASparseAttention(config)
        self.post_attention = PostAttention(config, num_total_layers)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        layer_id,
        positions: torch.Tensor | None = None,
    ):
        (
            q,
            k,
            v,
            attention_masks,
            softmax_scale,
            q_indexer,
            k_indexer,
            weights,
            end_pos,
            index_topk,
            w_uv_t,
        ) = self.pre_attention(x, freqs_cis, attention_masks, layer_id, positions)
        loss, output = self.inner_attention(
            q,
            k,
            v,
            attn_mask=attention_masks,
            scale=softmax_scale,
            q_indexer=q_indexer,
            k_indexer=k_indexer,
            weights=weights,
            end_pos=end_pos,
            index_topk=index_topk,
        )
        final_output = self.post_attention(x, output, w_uv_t, loss, layer_id)
        return final_output


class TransformerBlockV32(DeepSeekV3TransformerBlock):
    """
    Transformer block with attention and feed-forward layers.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3TransformerBlock.Config):
        # Tighten ``attention`` typing to the v32 sub-Config so trainer-side
        # validation catches misconfigured layers early.
        attention: "Attention.Config"  # pyrefly: ignore [bad-override]

    def __init__(self, config: Config, layer_id: int, num_total_layers: int):
        # Skip ``DeepSeekV3TransformerBlock.__init__`` because v32 builds its
        # own three-stage Attention; reuse the rest of the block's standard
        # build for attention_norm / ffn_norm / moe / feed_forward.
        nn.Module.__init__(self)
        self.layer_id = layer_id
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        assert isinstance(config.attention, Attention.Config), (
            "TransformerBlockV32 requires attention to be Attention.Config"
        )
        self.attention = Attention(config.attention, num_total_layers)
        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            assert config.moe is not None
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

    # pyrefly: ignore [bad-param-name-override]
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        if residual is not None:
            x = x + residual
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, freqs_cis, attention_masks, self.layer_id, positions)
        x = x + residual
        residual = x
        x = self.ffn_norm(x)
        x = self.moe(x) if self.moe_enabled else self.feed_forward(x)
        return x, residual


class MTPModule(TransformerBlockV32):
    """
    MTP block with linear projection and transformerblock layers.
    """

    def __init__(
        self,
        config: "TransformerBlockV32.Config",
        layer_id: int,
        num_total_layers: int,
    ):
        super().__init__(config, layer_id, num_total_layers)
        self._dim = config.attention.dim
        # enorm/hnorm normalize the model-dim stream like attention_norm, so
        # reuse its config (same normalized_shape/eps/param_init).
        self.enorm = config.attention_norm.build()
        self.hnorm = config.attention_norm.build()
        self.eh_proj = _Linear(2 * self._dim, self._dim, bias=False)

    # pyrefly: ignore [bad-param-name-override]
    def forward(
        self,
        input_offset: torch.Tensor,
        prev_embed: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Transformer block.

        Args:
            input_offset (torch.Tensor): Input tensor of original offset token embedding (batch_size, seq_len, dim).
            prev_embed (torch.Tensor): Predicted tokens of previous layer  (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        input_offset = self.enorm(input_offset)
        prev_embed = self.hnorm(prev_embed)
        h = torch.cat([input_offset, prev_embed], dim=-1)
        h = self.eh_proj(h)
        h, residual = self.attention_norm(h), h
        h = self.attention(h, freqs_cis, attention_masks, self.layer_id, positions)
        h = h + residual
        residual = h
        h = self.ffn_norm(h)
        h = self.moe(h) if self.moe_enabled else self.feed_forward(h)
        return h, residual

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        # ``Module.init_states`` recurses into Module children, so enorm/hnorm
        # are initialized via their config param_init. eh_proj only needs a
        # custom init here (the framework default does not match).
        super().init_states(buffer_device=buffer_device)
        final_out_std = self._dim**-0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            self.eh_proj.weight,  # pyrefly: ignore [bad-argument-type]
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )
