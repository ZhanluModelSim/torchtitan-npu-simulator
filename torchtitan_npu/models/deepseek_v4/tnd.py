# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from itertools import count
from types import SimpleNamespace
from typing import Any, NamedTuple

import torch
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.models.common import VarlenAttention

_SMLA_ATTENTION_MASK_CACHE_IDS = count()


class DeepSeekV4SMLAAttentionMasks(NamedTuple):
    cu_seqlens_q: torch.Tensor
    cu_seqlens_ori_kv: torch.Tensor
    cu_seqlens_cmp_kv: dict[int, torch.Tensor]
    cmp_residual_k: dict[int, torch.Tensor]
    block_starts_by_ratio: dict[int, torch.Tensor]
    max_seqlen_q: int
    max_seqlen_cmp_kv: dict[int, int]
    batch_size: int
    seq_len: int
    cache_id: int = -1


def lengths_to_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (
            lengths.new_zeros((1,)),
            torch.cumsum(lengths, dim=0, dtype=lengths.dtype),
        )
    )


def _residual_cmp_ratios(model_args: Any) -> tuple[int, ...]:
    cmp_ratios = (*model_args.compress_ratios, model_args.mtp_layer_compress_ratio)
    return tuple(sorted({ratio for ratio in cmp_ratios if ratio > 1}))


class _SMLARequestLayout(NamedTuple):
    flat_positions: torch.Tensor
    starts: torch.Tensor
    ends: torch.Tensor
    lengths: torch.Tensor
    seq_len: int


def _request_layout_from_positions(
    positions: torch.Tensor,
    *,
    num_mtp_modules: int = 0,
) -> _SMLARequestLayout:
    if positions.dim() != 2:
        raise ValueError("DeepSeek-V4 SMLA positions must be a [B, S] tensor.")

    seq_len = positions.shape[1] - num_mtp_modules
    if seq_len <= 0:
        raise ValueError(
            f"DeepSeek-V4 SMLA positions have no main-sequence tokens after removing {num_mtp_modules} MTP token(s)."
        )

    main_positions = positions[:, :seq_len]
    valid_positions = main_positions.ge(0)
    flat_valid = valid_positions.reshape(-1)
    invalid_seen = torch.cumsum((~flat_valid).to(torch.int32), dim=0) > 0
    if bool(torch.any(invalid_seen & flat_valid).item()):
        raise ValueError("DeepSeek-V4 SMLA valid positions must be contiguous before padding in row-major order.")
    flat_positions = main_positions.reshape(-1)[flat_valid]
    if flat_positions.numel() == 0:
        raise ValueError("DeepSeek-V4 SMLA positions contain no valid tokens.")

    starts = torch.nonzero(flat_positions.eq(0), as_tuple=False).flatten()
    if starts.numel() == 0 or int(starts[0].item()) != 0:
        starts = torch.cat((starts.new_zeros((1,)), starts))

    ends = torch.cat((starts[1:], starts.new_tensor((flat_positions.numel(),))))
    lengths = (ends - starts).to(dtype=torch.int32)
    return _SMLARequestLayout(flat_positions, starts, ends, lengths, seq_len)


def _tnd_compressed_block_starts(flat_positions: torch.Tensor, ratio: int) -> torch.Tensor:
    token_indices = torch.arange(flat_positions.numel(), device=flat_positions.device)
    end_indices = token_indices + ratio - 1
    end_in_range = end_indices < flat_positions.numel()
    clamped_end_indices = end_indices.clamp_max(flat_positions.numel() - 1)
    end_positions = flat_positions[clamped_end_indices]
    block_starts = flat_positions.remainder(ratio).eq(0) & end_in_range & end_positions.eq(flat_positions + ratio - 1)
    return torch.nonzero(block_starts, as_tuple=False).flatten()


def build_smla_attention_masks(
    positions: torch.Tensor,
    model_args: Any,
    batch_size: int | None = None,
    *,
    use_tnd_metadata: bool | None = None,
) -> DeepSeekV4SMLAAttentionMasks:
    positions_are_tnd = positions.dim() == 1
    if positions.dim() == 1:
        positions = positions.unsqueeze(0)

    position_batch_size, total_seq_len = positions.shape
    if batch_size is None:
        batch_size = position_batch_size
    elif position_batch_size == 1 and batch_size != 1:
        positions = positions.expand(batch_size, total_seq_len)
    elif position_batch_size != batch_size:
        raise ValueError(
            f"DeepSeek-V4 SMLA positions batch size ({position_batch_size}) "
            f"does not match input batch size ({batch_size})."
        )

    seq_len = total_seq_len if positions_are_tnd else total_seq_len - model_args.num_mtp_modules
    device = positions.device
    residual_cmp_ratios = _residual_cmp_ratios(model_args)
    if use_tnd_metadata is None:
        use_tnd_metadata = getattr(model_args, "use_global_tnd", True)
    if use_tnd_metadata:
        request_layout = _request_layout_from_positions(
            positions,
            num_mtp_modules=0 if positions_are_tnd else model_args.num_mtp_modules,
        )
        actual_seq_q = request_layout.lengths.to(device=device)
        block_starts_by_ratio = {
            ratio: _tnd_compressed_block_starts(request_layout.flat_positions, ratio).contiguous()
            for ratio in residual_cmp_ratios
        }
        cmp_lengths = {
            ratio: (
                torch.searchsorted(block_starts, request_layout.ends)
                - torch.searchsorted(block_starts, request_layout.starts)
            )
            .to(dtype=torch.int32)
            .to(device=device)
            for ratio, block_starts in block_starts_by_ratio.items()
        }
        batch_size = int(actual_seq_q.numel())
        seq_len = request_layout.seq_len
    else:
        valid_positions = positions[:, :seq_len].ge(0)
        actual_seq_q = valid_positions.sum(dim=1, dtype=torch.int32).to(device=device)
        cmp_lengths = {ratio: torch.div(actual_seq_q, ratio, rounding_mode="floor") for ratio in residual_cmp_ratios}
        block_starts_by_ratio = {}
    actual_seq_ori_kv = actual_seq_q.clone()
    return DeepSeekV4SMLAAttentionMasks(
        cu_seqlens_q=lengths_to_cu_seqlens(actual_seq_q),
        cu_seqlens_ori_kv=lengths_to_cu_seqlens(actual_seq_ori_kv),
        cu_seqlens_cmp_kv={ratio: lengths_to_cu_seqlens(lengths) for ratio, lengths in cmp_lengths.items()},
        cmp_residual_k={ratio: actual_seq_ori_kv - ratio * cmp_lengths[ratio] for ratio in residual_cmp_ratios},
        block_starts_by_ratio=block_starts_by_ratio,
        max_seqlen_q=int(actual_seq_q.max().item()) if actual_seq_q.numel() > 0 else 0,
        max_seqlen_cmp_kv={
            ratio: int(lengths.max().item()) if lengths.numel() > 0 else 0 for ratio, lengths in cmp_lengths.items()
        },
        batch_size=batch_size,
        seq_len=seq_len,
        cache_id=next(_SMLA_ATTENTION_MASK_CACHE_IDS),
    )


def smla_get_attention_masks(
    self,
    *args,
    positions: torch.Tensor | None = None,
    input_batch: torch.Tensor | None = None,
    tokenizer: Any | None = None,
    extra_inputs: dict[str, torch.Tensor] | None = None,
    **kwargs,
) -> DeepSeekV4SMLAAttentionMasks:
    del tokenizer, kwargs
    if args:
        if len(args) == 1 and positions is None and input_batch is None:
            positions = args[0]
        else:
            if input_batch is None:
                input_batch = args[0]
            if len(args) >= 3 and extra_inputs is None:
                extra_inputs = args[2]
    if positions is None and extra_inputs is not None:
        positions = extra_inputs.get("positions")
    if positions is None:
        raise ValueError("DeepSeek-V4 SMLA attention masks require dataloader-provided positions.")
    batch_size = input_batch.shape[0] if input_batch is not None and input_batch.dim() > 1 else None
    return build_smla_attention_masks(positions, self.model_args, batch_size)


def smla_layers(self):
    num_layers = self.n_layers + self.num_mtp_modules
    if not getattr(self, "use_global_tnd", False):
        return range(num_layers)
    inner_attention = VarlenAttention.Config()
    return tuple(
        SimpleNamespace(attention=SimpleNamespace(inner_attention=inner_attention, mask_type="causal"))
        for _ in range(num_layers)
    )


def smla_attn_type(self) -> str:
    return "varlen" if getattr(self, "use_global_tnd", False) else "sdpa"


def _compact_valid_tnd_tensor(
    tensor: torch.Tensor,
    valid_tokens: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if tensor.dim() not in (1, 2):
        raise RuntimeError(f"DeepSeek-V4 global TND expects {name} to be a 1D or 2D tensor.")
    if tensor.dim() != valid_tokens.dim() or tensor.shape != valid_tokens.shape:
        raise RuntimeError(
            f"DeepSeek-V4 global TND expects {name} shape {tuple(tensor.shape)} "
            f"to match positions shape {tuple(valid_tokens.shape)}."
        )
    return tensor[valid_tokens]


def smla_global_tnd_post_dataloading_process(
    input_dict: Any,
    labels: torch.Tensor,
    model_args: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    if not isinstance(input_dict, dict) or "input" not in input_dict:
        raise RuntimeError("DeepSeek-V4 global TND requires input_dict with an 'input' tensor.")

    inputs = input_dict["input"]
    positions = input_dict.get("positions")
    if not isinstance(inputs, torch.Tensor):
        raise RuntimeError("DeepSeek-V4 global TND requires input_dict['input'] to be a tensor.")
    if not isinstance(labels, torch.Tensor):
        raise RuntimeError("DeepSeek-V4 global TND requires labels to be a tensor.")
    if not isinstance(positions, torch.Tensor):
        raise RuntimeError("DeepSeek-V4 global TND requires dataloader-provided positions.")
    if inputs.shape != labels.shape or inputs.shape != positions.shape:
        raise RuntimeError(
            "DeepSeek-V4 global TND expects input, labels, and positions to share the same packed shape."
        )

    num_mtp_modules = int(getattr(model_args, "num_mtp_modules", 0))
    main_seq_len = inputs.shape[-1] - num_mtp_modules
    if main_seq_len <= 0:
        raise RuntimeError("DeepSeek-V4 global TND positions contain no main-sequence tokens.")

    main_positions = positions[..., :main_seq_len]
    valid_tokens = main_positions.ge(0)
    inputs_main = inputs[..., :main_seq_len]
    labels_main = labels[..., :main_seq_len]
    inputs_tnd = _compact_valid_tnd_tensor(inputs_main, valid_tokens, "input")
    positions_tnd = _compact_valid_tnd_tensor(main_positions, valid_tokens, "positions").to(dtype=torch.int32)
    if positions_tnd.numel() == 0:
        raise RuntimeError("DeepSeek-V4 global TND positions contain no valid tokens.")

    if num_mtp_modules > 0:
        request_ids = torch.cumsum(positions.eq(0).to(torch.int32), dim=-1)
        main_request_ids = request_ids[..., :main_seq_len]
        mtp_inputs = []
        mtp_labels = [labels_main[valid_tokens]]
        for mtp_idx in range(num_mtp_modules):
            offset = mtp_idx + 1
            shift_end = offset + main_seq_len
            shifted_positions = positions[..., offset:shift_end]
            shifted_request_ids = request_ids[..., offset:shift_end]
            valid_shift = (
                valid_tokens
                & shifted_positions.ge(0)
                & shifted_request_ids.eq(main_request_ids)
                & shifted_positions.eq(main_positions + offset)
            )
            shifted_inputs = inputs[..., offset:shift_end]
            shifted_labels = labels[..., offset:shift_end]
            # The dataloader pairs each label with the input at the same index
            # before packing, so request/position continuity protects both.
            shifted_inputs = torch.where(valid_shift, shifted_inputs, inputs_main)
            shifted_labels = shifted_labels.masked_fill(~valid_shift, IGNORE_INDEX)
            mtp_inputs.append(shifted_inputs[valid_tokens])
            mtp_labels.append(shifted_labels[valid_tokens])
        labels_tnd = torch.stack(mtp_labels, dim=0)
    else:
        mtp_inputs = []
        labels_tnd = _compact_valid_tnd_tensor(labels_main, valid_tokens, "labels")

    attention_masks = build_smla_attention_masks(positions_tnd, model_args)
    extra_inputs = {k: v for k, v in input_dict.items() if k not in ("input", "positions")}
    extra_kwargs: dict[str, Any] = {
        "positions": positions_tnd,
        "attention_masks": attention_masks,
    }
    if mtp_inputs:
        extra_kwargs["mtp_inputs"] = torch.stack(mtp_inputs, dim=0)
    return inputs_tnd, labels_tnd, extra_inputs, extra_kwargs
