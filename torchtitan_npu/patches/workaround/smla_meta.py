"""Fix the CANN ``sparse_flash_mla`` meta kernel so ``torch.compile`` can trace it.

CANN 9.1.0 sizes the last ``softmax_lse`` dimension with ``/`` instead of ``//``,
so the meta kernel hands ``torch.empty`` a float and FakeTensor tracing dies with
``TypeError: type must be tuple of ints, but got float``. Eager never runs the
meta kernel, so only compiled runs hit it.

``apply()`` runs from ``torchtitan_npu.ops.cann_transformer``: the meta kernel is
registered while ``cann_ops_transformer`` is imported, so the override has to come
after that import rather than at package import time. Drop this module once CANN
ships the floor division.
"""

import torch
from torchtitan.tools.logging import logger

_NAMESPACE = "cann_ops_transformer"
_OP_NAME = "sparse_flash_mla"

# Registrations die with their Library, so keep it alive for the process.
_library: torch.library.Library | None = None


def _sparse_flash_mla_meta(
    q,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_ori_kv=None,
    cu_seqlens_cmp_kv=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    cmp_residual_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    softmax_scale=1.0,
    cmp_ratio=1,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="BSND",
    topk_value_mode=1,
    return_softmax_lse=False,
):
    attn_out = torch.empty(q.shape, dtype=q.dtype, device="meta")
    if not return_softmax_lse:
        # The kernel still needs a valid tensor here, just not a sized one.
        return attn_out, torch.empty([], dtype=torch.float32, device="meta")

    assert ori_kv is not None
    if layout_q == "BSND":
        kv_heads = ori_kv.shape[2]
        lse_shape = [q.shape[0], kv_heads, q.shape[1], q.shape[2] // kv_heads]
    else:  # TND
        kv_heads = ori_kv.shape[1]
        lse_shape = [kv_heads, q.shape[0], q.shape[1] // kv_heads]
    return attn_out, torch.empty(lse_shape, dtype=torch.float32, device="meta")


def apply() -> None:
    """Override the CANN meta kernel. Safe to call more than once."""

    global _library
    if _library is not None:
        return

    library = torch.library.Library(_NAMESPACE, "FRAGMENT")
    library.impl(_OP_NAME, _sparse_flash_mla_meta, "Meta")
    _library = library
    logger.info(
        "[WORKAROUND] cann_ops_transformer::sparse_flash_mla Meta kernel -> "
        "torchtitan_npu (integer softmax_lse shape)"
    )
