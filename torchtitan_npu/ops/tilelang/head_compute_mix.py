# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch_npu

if TYPE_CHECKING:
    import tilelang  # type: ignore[import-untyped] # noqa: TC004
    from tilelang import language as T  # type: ignore[import-untyped] # noqa: TC004

logger = logging.getLogger(__name__)

# Global kernel caches
_fwd_kernel_cache: dict[tuple[int, float, int], Any] = {}
_bwd_kernel_cache: dict[tuple[int, int, int], Any] = {}

# Constants for TileLang execution
VEC_NUM = 2
_RESHAPE_FACTOR = 4
_FWD_TOKEN_BLOCK_SIZE = 512 // _RESHAPE_FACTOR
_BWD_TOKEN_BLOCK_SIZE = 128

_tilelang_initialized = False
_FWD_PASS_CONFIGS = None
_BWD_PASS_CONFIGS = None


def _ensure_tilelang() -> None:
    """Helper function to lazily import TileLang and initialize configs on first use."""
    global _tilelang_initialized, _FWD_PASS_CONFIGS, _BWD_PASS_CONFIGS, tilelang, T
    if not _tilelang_initialized:
        try:
            import tilelang  # type: ignore[import-untyped]
            from tilelang import language as T  # type: ignore[import-untyped]

            globals()["tilelang"] = tilelang
            globals()["T"] = T

            _FWD_PASS_CONFIGS = {
                tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
                tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
                tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            }

            _BWD_PASS_CONFIGS = {
                tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
                tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
                tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
            }

            tilelang.cache.clear_cache()
            _tilelang_initialized = True
        except ImportError as e:
            raise RuntimeError(
                "Missing tilelang-ascend dependency. Please install 'tilelang-ascend' for NPU usage. "
                "https://github.com/tile-ai/tilelang-ascend#installation."
            ) from e


def _pad_if_needed(tensor, actual_len, padded_len, dim_size, device):
    if padded_len > actual_len:
        pad_tensor = torch.zeros((padded_len - actual_len, dim_size), dtype=tensor.dtype, device=device)
        return torch.cat([tensor, pad_tensor], dim=0)
    return tensor


def _reshape_for_tilelang(x_proj_2d: torch.Tensor, mhc_mult: int) -> tuple[torch.Tensor, int, int]:
    """Flatten (B*S, mhc_mult) -> pad tokens to a multiple of _RESHAPE_FACTOR -> (N, mhc_mult*_RESHAPE_FACTOR).

    Returns (reshaped, num_tokens_padded, reshape_mhc_mult) where num_tokens_padded
    is a multiple of _RESHAPE_FACTOR and >= B*S. Padding is zero-filled so the
    extra rows do not affect the sigmoid/mix result that is sliced back.
    """
    num_tokens = x_proj_2d.shape[0]
    num_tokens_padded = (num_tokens + _RESHAPE_FACTOR - 1) // _RESHAPE_FACTOR * _RESHAPE_FACTOR
    reshape_mhc_mult = mhc_mult * _RESHAPE_FACTOR
    padded = _pad_if_needed(x_proj_2d, num_tokens, num_tokens_padded, mhc_mult, x_proj_2d.device)
    # padded: (num_tokens_padded, mhc_mult). View as (N, _RESHAPE_FACTOR, mhc_mult) and
    # fold the _RESHAPE_FACTOR axis into the column axis -> (N, mhc_mult * _RESHAPE_FACTOR).
    reshaped = padded.reshape(num_tokens_padded // _RESHAPE_FACTOR, _RESHAPE_FACTOR, mhc_mult)
    reshaped = reshaped.reshape(num_tokens_padded // _RESHAPE_FACTOR, reshape_mhc_mult).contiguous()
    return reshaped, num_tokens_padded, reshape_mhc_mult


def _get_fwd_kernel(mhc_mult: int, mhc_pre_eps: float, reshape_factor: int) -> Any:
    _ensure_tilelang()
    reshape_mhc_mult = mhc_mult * reshape_factor
    key = (reshape_mhc_mult, mhc_pre_eps, reshape_factor)
    if key not in _fwd_kernel_cache:

        @tilelang.jit(pass_configs=_FWD_PASS_CONFIGS)
        def _mhc_head_compute_mix_fwd(
            mhc_mult: int,
            mhc_pre_eps: float,
            reshape_factor: int = 1,
            token_block_size: int = 512,
        ) -> tilelang.JITKernel:
            num_tokens = T.symbolic("num_tokens")
            dtype = "float32"
            grid_size = T.ceildiv(num_tokens, token_block_size)
            pad_mhc_mult = T.ceildiv(mhc_mult, 8) * 8
            sub_block_tokens = token_block_size // VEC_NUM
            orig_mhc_mult = mhc_mult // reshape_factor if reshape_factor > 1 else mhc_mult

            @T.prim_func
            def mhc_head_compute_mix_fwd_kernel(
                input_mix: T.Tensor[(num_tokens, mhc_mult), dtype],
                mhc_scale: T.Tensor[(1,), dtype],
                mhc_base: T.Tensor[(orig_mhc_mult,), dtype],
                output_mix: T.Tensor[(num_tokens, mhc_mult), dtype],
            ) -> None:
                with T.Kernel(grid_size, is_npu=True) as (cid, vid):
                    row_start = cid * token_block_size + vid * sub_block_tokens
                    in_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
                    out_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
                    bcast_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
                    scale_ub = T.alloc_ub((1,), dtype)
                    base_ub = T.alloc_ub((pad_mhc_mult,), dtype)

                    T.set_flag("mte3", "mte2", 0)
                    T.wait_flag("mte3", "mte2", 0)
                    T.copy(
                        input_mix[row_start : row_start + sub_block_tokens, 0:mhc_mult],
                        in_ub[0:sub_block_tokens, 0:mhc_mult],
                    )
                    T.copy(mhc_scale[0:1], scale_ub)
                    T.copy(mhc_base[0:orig_mhc_mult], base_ub[0:orig_mhc_mult])

                    for r in range(1, reshape_factor):
                        for j in range(orig_mhc_mult):
                            base_ub[r * orig_mhc_mult + j] = base_ub[j]

                    T.tile.broadcast(bcast_ub, base_ub, axis=0)
                    T.tile.axpy(bcast_ub, in_ub, scale_ub[0])

                    T.tile.sigmoid(out_ub, bcast_ub)
                    T.tile.add(out_ub, out_ub, mhc_pre_eps)

                    T.set_flag("v", "mte3", 0)
                    T.wait_flag("v", "mte3", 0)
                    T.copy(
                        out_ub[0:sub_block_tokens, 0:mhc_mult],
                        output_mix[row_start : row_start + sub_block_tokens, 0:mhc_mult],
                    )
                    T.set_flag("mte3", "mte2", 0)
                    T.wait_flag("mte3", "mte2", 0)

            return mhc_head_compute_mix_fwd_kernel

        _fwd_kernel_cache[key] = _mhc_head_compute_mix_fwd(
            reshape_mhc_mult,
            mhc_pre_eps,
            reshape_factor=reshape_factor,
            token_block_size=_FWD_TOKEN_BLOCK_SIZE,
        )

    return _fwd_kernel_cache[key]


def _get_bwd_kernel(mhc_mult: int, partial_size: int, reshape_factor: int) -> Any:
    _ensure_tilelang()
    reshape_mhc_mult = mhc_mult * reshape_factor
    key = (reshape_mhc_mult, partial_size, reshape_factor)
    if key not in _bwd_kernel_cache:

        @tilelang.jit(pass_configs=_BWD_PASS_CONFIGS)
        def _mhc_head_compute_mix_bwd(
            mhc_mult: int,
            reshape_factor: int = 1,
            token_block_size: int = 128,
            partial_size: int = 128,
        ) -> tilelang.JITKernel:
            num_tokens = T.symbolic("num_tokens")
            dtype = "float32"
            pad_mhc_mult = T.ceildiv(mhc_mult, 8) * 8
            sub_block_tokens = token_block_size // VEC_NUM
            grid_size = T.ceildiv(num_tokens, token_block_size)
            orig_mhc_mult = mhc_mult // reshape_factor if reshape_factor > 1 else mhc_mult

            @T.prim_func
            def mhc_head_compute_mix_bwd_kernel(
                output_mix_grad: T.Tensor[(num_tokens, mhc_mult), dtype],
                input_mix: T.Tensor[(num_tokens, mhc_mult), dtype],
                mhc_scale: T.Tensor[(1,), dtype],
                mhc_base: T.Tensor[(orig_mhc_mult,), dtype],
                input_mix_grad: T.Tensor[(num_tokens, mhc_mult), dtype],
                mhc_scale_grad_partial: T.Tensor[(partial_size, mhc_mult), dtype],
                mhc_base_grad_partial: T.Tensor[(partial_size, mhc_mult), dtype],
            ) -> None:
                with T.Kernel(grid_size, is_npu=True) as (cid, vid):
                    row_start = cid * token_block_size + vid * sub_block_tokens

                    base_grad_ub = T.alloc_ub((1, pad_mhc_mult), dtype)
                    reduce_col_ub = T.alloc_ub((1, pad_mhc_mult), dtype)
                    scale_grad_ub = T.alloc_ub((1, pad_mhc_mult), dtype)

                    in_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
                    buf_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
                    val_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
                    sig_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
                    bcast_ub = T.alloc_ub((sub_block_tokens, pad_mhc_mult), dtype)
                    scale_ub = T.alloc_ub((1,), dtype)
                    base_ub = T.alloc_ub((pad_mhc_mult,), dtype)

                    T.tile.fill(base_grad_ub, 0.0)
                    T.tile.fill(scale_grad_ub, 0.0)

                    T.set_flag("mte3", "mte2", 0)
                    T.wait_flag("mte3", "mte2", 0)
                    T.copy(
                        input_mix[row_start : row_start + sub_block_tokens, 0:mhc_mult],
                        in_ub[0:sub_block_tokens, 0:mhc_mult],
                    )
                    T.copy(
                        output_mix_grad[row_start : row_start + sub_block_tokens, 0:mhc_mult],
                        buf_ub[0:sub_block_tokens, 0:mhc_mult],
                    )
                    T.copy(mhc_scale[0:1], scale_ub)
                    T.copy(mhc_base[0:orig_mhc_mult], base_ub[0:orig_mhc_mult])

                    T.set_flag("mte2", "v", 0)
                    T.wait_flag("mte2", "v", 0)

                    for r in range(1, reshape_factor):
                        for j in range(orig_mhc_mult):
                            base_ub[r * orig_mhc_mult + j] = base_ub[j]

                    T.tile.broadcast(bcast_ub, base_ub, axis=0)
                    T.tile.axpy(bcast_ub, in_ub, scale_ub[0])

                    T.tile.sigmoid(sig_ub, bcast_ub)

                    T.tile.fill(val_ub, 1.0)
                    T.tile.sub(val_ub, val_ub, sig_ub)
                    T.tile.mul(sig_ub, sig_ub, val_ub)

                    T.tile.mul(sig_ub, sig_ub, buf_ub)

                    T.reduce_sum(sig_ub, reduce_col_ub, dim=0)
                    T.tile.add(base_grad_ub, base_grad_ub, reduce_col_ub)

                    T.tile.mul(buf_ub, sig_ub, scale_ub[0])

                    T.tile.mul(bcast_ub, sig_ub, in_ub)
                    T.reduce_sum(bcast_ub, reduce_col_ub, dim=0)
                    T.tile.add(scale_grad_ub, scale_grad_ub, reduce_col_ub)

                    T.set_flag("v", "mte3", 0)
                    T.wait_flag("v", "mte3", 0)
                    partial_idx = cid * VEC_NUM + vid
                    T.copy(base_grad_ub[0, 0:mhc_mult], mhc_base_grad_partial[partial_idx, 0:mhc_mult])
                    T.copy(scale_grad_ub[0, 0:mhc_mult], mhc_scale_grad_partial[partial_idx, 0:mhc_mult])
                    T.copy(
                        buf_ub[0:sub_block_tokens, 0:mhc_mult],
                        input_mix_grad[row_start : row_start + sub_block_tokens, 0:mhc_mult],
                    )
                    T.set_flag("mte3", "mte2", 0)
                    T.wait_flag("mte3", "mte2", 0)

            return mhc_head_compute_mix_bwd_kernel

        _bwd_kernel_cache[key] = _mhc_head_compute_mix_bwd(
            reshape_mhc_mult,
            reshape_factor=reshape_factor,
            token_block_size=_BWD_TOKEN_BLOCK_SIZE,
            partial_size=partial_size,
        )

    return _bwd_kernel_cache[key]


def _hc_pre_bmm_forward(pre: torch.Tensor, x_unflatten: torch.Tensor) -> torch.Tensor:
    return torch.sum(pre.unsqueeze(-1) * x_unflatten, dim=-2)


def _hc_pre_bmm_backward(
    pre: torch.Tensor, x_unflatten: torch.Tensor, grad_y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_y_expanded = grad_y.unsqueeze(-2)
    grad_pre = torch.sum(grad_y_expanded * x_unflatten, dim=-1)
    grad_x_direct = (grad_y_expanded * pre.unsqueeze(-1)).reshape(x_unflatten.shape[0], x_unflatten.shape[1], -1)
    return grad_pre, grad_x_direct


@torch.library.custom_op("torchtitan_npu::mhc_head_compute_mix_fwd", mutates_args=())
def mhc_head_compute_mix_fwd_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_gamma: torch.Tensor | None,
    mhc_use_gamma: bool,
    norm_eps: float,
    hc_eps: float,
    num_stream: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    B, S, nD = x.shape
    dtype = x.dtype

    x_float = x.float().clone()
    weight_f = weight.float().t().clone()
    hc_scale_f = hc_scale.float().contiguous().clone()
    hc_base_f = hc_base.float().contiguous().clone()

    x_flat = x_float.reshape(-1, nD).clone()
    norm_gamma_f = (
        torch.ones(nD, device=x.device, dtype=torch.float32)
        if not mhc_use_gamma or norm_gamma is None
        else norm_gamma.float().clone()
    )

    x_norm_flat, rstd = torch_npu.npu_rms_norm(x_flat, gamma=norm_gamma_f, epsilon=norm_eps)
    x_norm_flat = x_norm_flat.clone()
    rstd = rstd.clone()

    x_norm_mat = x_norm_flat.reshape(B, S, nD)
    x_proj = torch.matmul(x_norm_mat, weight_f)

    mhc_mult = num_stream

    # x_proj: (B, S, mhc_mult). Flatten to (B*S, mhc_mult) then fold into
    # (N, mhc_mult * _RESHAPE_FACTOR), padding tokens up to a multiple of
    # _RESHAPE_FACTOR so B*S need not be divisible by _RESHAPE_FACTOR (e.g.
    # 3D TND inputs where B=T, S=1 and T % 4 != 0).
    x_proj_2d = x_proj.reshape(B * S, mhc_mult).contiguous()
    x_proj_reshaped, num_tokens_padded, reshape_mhc_mult = _reshape_for_tilelang(x_proj_2d, mhc_mult)
    reshaped_num_tokens = num_tokens_padded // _RESHAPE_FACTOR
    # Use safe integer arithmetic instead of math ceil to avoid graph break in torch compile
    fwd_padded_tokens = (
        (reshaped_num_tokens + _FWD_TOKEN_BLOCK_SIZE - 1) // _FWD_TOKEN_BLOCK_SIZE * _FWD_TOKEN_BLOCK_SIZE
    )
    x_proj_input = _pad_if_needed(x_proj_reshaped, reshaped_num_tokens, fwd_padded_tokens, reshape_mhc_mult, x.device)
    output_mix = torch.empty(fwd_padded_tokens, reshape_mhc_mult, dtype=torch.float32, device=x.device)
    fwd_kernel = _get_fwd_kernel(mhc_mult, hc_eps, _RESHAPE_FACTOR)
    fwd_kernel(x_proj_input, hc_scale_f, hc_base_f, output_mix)
    # Slice back to the real rows and reshape to (B, S, mhc_mult). Only the
    # first B*S rows are real; the padded rows are discarded.
    pre = output_mix[:reshaped_num_tokens].reshape(num_tokens_padded, mhc_mult)[: B * S].reshape(B, S, mhc_mult).clone()

    x_unflatten = x_float.unflatten(dim=-1, sizes=(num_stream, -1)).clone()
    y = _hc_pre_bmm_forward(pre, x_unflatten).to(dtype).clone()

    x_proj_flat = x_proj.reshape(B * S, mhc_mult).contiguous().clone()

    return y, x_flat, x_norm_flat, rstd, x_proj_flat, weight_f, hc_scale_f, hc_base_f, pre, x_unflatten


@mhc_head_compute_mix_fwd_op.register_fake
def _mhc_head_compute_mix_fwd_fake(
    x, weight, hc_scale, hc_base, norm_gamma, mhc_use_gamma, norm_eps, hc_eps, num_stream
):
    B, S, nD = x.shape
    c_per_stream = nD // num_stream
    y = torch.empty((B, S, c_per_stream), dtype=x.dtype, device=x.device)
    x_flat = torch.empty((B * S, nD), dtype=torch.float32, device=x.device)
    x_norm_flat = torch.empty((B * S, nD), dtype=torch.float32, device=x.device)
    rstd = torch.empty((B * S, 1), dtype=torch.float32, device=x.device)
    x_proj_flat = torch.empty((B * S, num_stream), dtype=torch.float32, device=x.device)

    # Exact match for real path layout: weight.float().t().clone()
    weight_f = weight.float().t().clone()

    hc_scale_f = hc_scale.float().contiguous().clone()
    hc_base_f = hc_base.float().contiguous().clone()
    pre = torch.empty((B, S, num_stream), dtype=torch.float32, device=x.device)
    x_unflatten = torch.empty((B, S, num_stream, c_per_stream), dtype=torch.float32, device=x.device)
    return y, x_flat, x_norm_flat, rstd, x_proj_flat, weight_f, hc_scale_f, hc_base_f, pre, x_unflatten


@torch.library.custom_op("torchtitan_npu::mhc_head_compute_mix_bwd", mutates_args=())
def mhc_head_compute_mix_bwd_op(
    grad_y: torch.Tensor,
    x_flat: torch.Tensor,
    x_norm_flat: torch.Tensor,
    rstd: torch.Tensor,
    x_proj_flat: torch.Tensor,
    weight_f: torch.Tensor,
    hc_scale_f: torch.Tensor,
    hc_base_f: torch.Tensor,
    pre: torch.Tensor,
    x_unflatten: torch.Tensor,
    norm_gamma: torch.Tensor | None,
    mhc_use_gamma: bool,
    norm_eps: float,
    hc_eps: float,
    num_stream: int,
    needs_weight_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B = pre.shape[0]
    S = pre.shape[1]
    nD = x_flat.shape[-1]
    mhc_mult = num_stream

    grad_pre, grad_x_direct = _hc_pre_bmm_backward(pre, x_unflatten, grad_y)

    # grad_pre / x_proj: (B, S, mhc_mult). Fold into (N, mhc_mult*_RESHAPE_FACTOR)
    # with token padding so B*S need not be divisible by _RESHAPE_FACTOR.
    grad_pre_2d = grad_pre.reshape(B * S, mhc_mult).contiguous().float()
    x_proj_2d = x_proj_flat.reshape(B * S, mhc_mult).contiguous()
    grad_pre_reshaped, num_tokens_padded, reshape_mhc_mult = _reshape_for_tilelang(grad_pre_2d, mhc_mult)
    x_proj_reshaped, _, _ = _reshape_for_tilelang(x_proj_2d, mhc_mult)
    reshaped_num_tokens = num_tokens_padded // _RESHAPE_FACTOR
    # Use safe integer arithmetic instead of math ceil to avoid graph break in torch compile
    bwd_grid_size = (reshaped_num_tokens + _BWD_TOKEN_BLOCK_SIZE - 1) // _BWD_TOKEN_BLOCK_SIZE
    bwd_padded_tokens = bwd_grid_size * _BWD_TOKEN_BLOCK_SIZE
    partial_size = bwd_grid_size * VEC_NUM
    device = x_flat.device
    grad_pre_input = _pad_if_needed(grad_pre_reshaped, reshaped_num_tokens, bwd_padded_tokens, reshape_mhc_mult, device)
    x_proj_input = _pad_if_needed(x_proj_reshaped, reshaped_num_tokens, bwd_padded_tokens, reshape_mhc_mult, device)
    input_mix_grad = torch.empty(bwd_padded_tokens, reshape_mhc_mult, dtype=torch.float32, device=device)
    scale_grad_partial = torch.zeros(partial_size, reshape_mhc_mult, dtype=torch.float32, device=device)
    base_grad_partial = torch.zeros(partial_size, reshape_mhc_mult, dtype=torch.float32, device=device)
    bwd_kernel = _get_bwd_kernel(mhc_mult, partial_size, _RESHAPE_FACTOR)
    bwd_kernel(
        grad_pre_input,
        x_proj_input,
        hc_scale_f,
        hc_base_f,
        input_mix_grad,
        scale_grad_partial,
        base_grad_partial,
    )
    # Unfold (N, mhc_mult*_RESHAPE_FACTOR) -> (num_tokens_padded, mhc_mult),
    # take the real B*S rows, reshape to (B, S, mhc_mult).
    grad_x_proj = (
        input_mix_grad[:reshaped_num_tokens].reshape(num_tokens_padded, mhc_mult)[: B * S].reshape(B, S, mhc_mult)
    )
    grad_hc_scale = scale_grad_partial[:, :reshape_mhc_mult].sum().reshape(1).clone()
    grad_hc_base = (
        base_grad_partial[:, :reshape_mhc_mult].sum(dim=0).reshape(_RESHAPE_FACTOR, mhc_mult).sum(dim=0).clone()
    )

    if needs_weight_grad:
        grad_weight = torch.matmul(x_norm_flat.t(), grad_x_proj.reshape(-1, mhc_mult)).t().clone()
    else:
        grad_weight = torch.empty((0,), device=x_flat.device)

    grad_x_norm_mat = torch.matmul(grad_x_proj, weight_f.t())

    norm_gamma_f = (
        torch.ones(nD, device=x_flat.device, dtype=torch.float32)
        if not mhc_use_gamma or norm_gamma is None
        else norm_gamma.float()
    )

    grad_x_rms_flat, grad_gamma = torch_npu.npu_rms_norm_backward(
        grad_x_norm_mat.view(-1, nD), x_flat, norm_gamma_f, rstd
    )
    grad_x_rms_flat = grad_x_rms_flat.clone()
    grad_gamma = grad_gamma.clone() if mhc_use_gamma else torch.empty((0,), device=x_flat.device)

    grad_x_rms = grad_x_rms_flat.view(B, S, nD)
    grad_x = (grad_x_direct.view(B, S, nD) + grad_x_rms).clone()

    return grad_x, grad_weight, grad_hc_scale, grad_hc_base, grad_gamma


@mhc_head_compute_mix_bwd_op.register_fake
def _mhc_head_compute_mix_bwd_fake(
    grad_y,
    x_flat,
    x_norm_flat,
    rstd,
    x_proj_flat,
    weight_f,
    hc_scale_f,
    hc_base_f,
    pre,
    x_unflatten,
    norm_gamma,
    mhc_use_gamma,
    norm_eps,
    hc_eps,
    num_stream,
    needs_weight_grad,
):
    B = pre.shape[0]
    S = pre.shape[1]
    nD = x_flat.shape[-1]

    # Explicitly set output dtypes to float32 to strictly match real backward kernel
    grad_x = torch.empty((B, S, nD), dtype=torch.float32, device=grad_y.device)

    # Exact match for real path grad_weight layout and float32 dtype
    grad_weight = (
        torch.empty((nD, num_stream), dtype=torch.float32, device=grad_y.device).t().clone()
        if needs_weight_grad
        else torch.empty((0,), dtype=torch.float32, device=grad_y.device)
    )
    grad_hc_scale = torch.empty_like(hc_scale_f, dtype=torch.float32)
    grad_hc_base = torch.empty_like(hc_base_f, dtype=torch.float32)
    grad_gamma = (
        torch.empty_like(norm_gamma, dtype=torch.float32)
        if mhc_use_gamma
        else torch.empty((0,), dtype=torch.float32, device=grad_y.device)
    )

    return grad_x, grad_weight, grad_hc_scale, grad_hc_base, grad_gamma


# Setup context for autograd registration
def _mhc_head_compute_mix_fwd_setup_context(ctx, inputs, output):
    (
        x,
        _weight,
        _hc_scale,
        _hc_base,
        norm_gamma,
        mhc_use_gamma,
        norm_eps,
        hc_eps,
        num_stream,
    ) = inputs
    (
        _y,
        x_flat,
        x_norm_flat,
        rstd,
        x_proj_flat,
        weight_f,
        hc_scale_f,
        hc_base_f,
        pre,
        x_unflatten,
    ) = output

    gamma_was_none = norm_gamma is None
    if norm_gamma is None:
        norm_gamma = torch.empty(0, device=x.device)

    ctx.save_for_backward(
        x_flat,
        x_norm_flat,
        rstd,
        x_proj_flat,
        weight_f,
        hc_scale_f,
        hc_base_f,
        pre,
        x_unflatten,
        norm_gamma,
    )
    ctx.mhc_use_gamma = mhc_use_gamma
    # Remember whether the caller actually passed a gamma tensor. When
    # norm_gamma was None (unit gamma) we must return None for that input's
    # gradient even if mhc_use_gamma is True, because a None input cannot
    # receive a Tensor gradient.
    ctx.gamma_was_none = gamma_was_none
    ctx.norm_eps = norm_eps
    ctx.hc_eps = hc_eps
    ctx.num_stream = num_stream


# Backward function for autograd registration
def _mhc_head_compute_mix_fwd_backward(
    ctx,
    grad_y,
    grad_x_flat,
    grad_x_norm_flat,
    grad_rstd,
    grad_x_proj_flat,
    grad_weight_f,
    grad_hc_scale_f,
    grad_hc_base_f,
    grad_pre,
    grad_x_unflatten,
):
    (
        x_flat,
        x_norm_flat,
        rstd,
        x_proj_flat,
        weight_f,
        hc_scale_f,
        hc_base_f,
        pre,
        x_unflatten,
        norm_gamma,
    ) = ctx.saved_tensors

    needs_weight_grad = ctx.needs_input_grad[1]

    grad_x, grad_weight, grad_hc_scale, grad_hc_base, grad_gamma = mhc_head_compute_mix_bwd_op(
        grad_y,
        x_flat,
        x_norm_flat,
        rstd,
        x_proj_flat,
        weight_f,
        hc_scale_f,
        hc_base_f,
        pre,
        x_unflatten,
        norm_gamma,
        ctx.mhc_use_gamma,
        ctx.norm_eps,
        ctx.hc_eps,
        ctx.num_stream,
        needs_weight_grad,
    )

    return (
        grad_x,
        grad_weight if needs_weight_grad else None,
        grad_hc_scale,
        grad_hc_base,
        grad_gamma if (ctx.mhc_use_gamma and not ctx.gamma_was_none) else None,
        None,
        None,
        None,
        None,
    )


# Register autograd for the forward operator
mhc_head_compute_mix_fwd_op.register_autograd(
    _mhc_head_compute_mix_fwd_backward,
    setup_context=_mhc_head_compute_mix_fwd_setup_context,
)


# User-facing API wrapper returning only the primary output tensor y
def mhc_head_compute_mix_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_gamma: torch.Tensor | None,
    mhc_use_gamma: bool = True,
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    num_stream: int = 4,
) -> torch.Tensor:
    y, *_ = mhc_head_compute_mix_fwd_op(
        x,
        weight,
        hc_scale,
        hc_base,
        norm_gamma,
        mhc_use_gamma,
        norm_eps,
        hc_eps,
        num_stream,
    )
    return y
