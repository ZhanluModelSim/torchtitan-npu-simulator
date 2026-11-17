import pytest
import torch

from tests.unit_tests.models.mtp_test_utils import build_cpu_model
from torchtitan_npu.override.deepseek_v4_1.sparse_attn import ascendc


@pytest.mark.parametrize(
    "checkpointed",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not torch.npu.is_available(),
                reason=(
                    "checkpoint recompute enters NPU autocast, which lazy-initializes "
                    "the NPU runtime; CPU-only CI runners have no device"
                ),
            ),
        ),
    ],
)
def test_smla_carries_the_raw_teacher_to_slikg(monkeypatch, v41_metadata, checkpointed):
    """SMLAG's by-product reaches ``topk_scores``' gradient unchanged.

    The teacher is the only thing that trains the indexer, and it travels on
    ``topk_scores`` because SMLAG's backward runs after the indexer's forward.  So the
    gradient this port produces must be the kernel's ``cmp_softmax_l1_norm`` verbatim --
    no loss scaling, no ``Z * Y`` term -- because SLIKG applies ``dI = Z * Y - p`` itself
    and would read anything else as the teacher.
    """
    meta = torch.zeros(1, dtype=torch.int32)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_metadata", lambda *a, **k: meta)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad_metadata", lambda *a, **k: meta)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla",
        lambda q, **k: (q.clone(), torch.zeros(1, q.shape[0], q.shape[1])),
    )

    mass = [0.1, 0.3, 9.0]

    # What an unused slot gets from the real op: junk.  This is measured, not assumed --
    # on a 950PR the op fills every ``-1`` slot of ``cmp_softmax_l1_norm`` with the same
    # non-zero marginal it reports for the valid slots, which is why the port has to clean
    # them (see the SMLAG override).  A fake that wrote zero here would make the port's
    # masking untestable, and did: nothing below noticed when the mask was removed.
    junk = 7.0

    def backward(q, *args, **kwargs):
        # What a real SMLAG reports.  Shape and dtype are the op's own: it allocates this
        # output as ``cmp_sparse_indices.new_empty(cmp_sparse_indices.shape)`` in FP32 and
        # its tiling asserts every dimension matches that input.  So it comes back in the
        # selection's layout, and a slot the selection does not use (``-1``) -- including
        # every slot of a row whose whole selection is padding -- carries no usable mass.
        indices = kwargs["cmp_sparse_indices"]
        p = torch.zeros(indices.shape, dtype=torch.float32)
        valid = indices >= 0
        p[..., 1] = torch.where(valid[..., 1], mass[1], junk)
        p[..., 2] = torch.where(valid[..., 2], mass[2], junk)
        return (
            torch.zeros_like(q),
            torch.zeros_like(kwargs["ori_kv"]),
            torch.zeros_like(kwargs["cmp_kv"]),
            torch.zeros_like(kwargs["sinks"]),
            None,
            p,
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", backward)
    config = ascendc.AscV41SparseAttention.Config(
        window_size=2,
        softmax_scale=0.5,
        compress_ratio=1,
        num_heads=1,
        head_dim=4,
        index_topk=3,
        index_num_heads=1,
        index_head_dim=4,
    )
    attn = build_cpu_model(config)
    # Two documents of two tokens: whole pooling groups, which is the only shape the backend's
    # frames are built for.  The metadata comes from this layer's own config, which is what a run
    # does: the records are sized by the config's geometry, and the lookup compares them back.
    metadata = v41_metadata([0, 2, 4], port=config)
    # The real carrier is FP32: the fused selector emits the kernel's FP32 relaxed scores,
    # which is what lets the port forward SMLAG's FP32 mass without converting it.  An
    # unused slot holds whatever the kernel wrote there (``-inf`` from the reference
    # selector); nothing may depend on it, because SLIKG never reads a ``-1`` position.
    scores = torch.tensor([[[float("-inf"), 1.0, 0.0]] * 4], dtype=torch.float32, requires_grad=True)
    indices = torch.tensor([[[-1, 0, 1]] * 4])
    indices[:, 3] = -1
    q = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16, requires_grad=True)

    def consume(q, scores):
        return attn(
            q,
            q.squeeze(2),
            torch.zeros(1),
            metadata,
            cmp_k=q.squeeze(2),
            topk_indices=indices,
            topk_scores=scores,
        )

    if checkpointed:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(consume, q, scores, use_reentrant=False)
    else:
        out = consume(q, scores)
    # The selection is shared across a layer group, so a second consumer drives the same
    # edge.  Both outputs are zeroed before the backward, so the scores see only the
    # teacher edge -- and it is delivered once per consumer's backward, not per forward.
    out2 = consume(q, scores)
    ((out.sum() + out2.sum()) * 0).backward()

    valid = (indices >= 0).reshape(4, 3)
    grad = scores.grad.reshape(4, 3)
    # The carrier is FP32 and SMLAG's mass is FP32, and the op returns it in the
    # selection's own layout, so the port forwards it with no conversion and no
    # reshaping at all -- the value that lands here is the kernel's tensor.
    assert scores.dtype == torch.float32
    assert scores.grad.dtype == scores.dtype
    # Every ``-1`` slot comes back zero however much junk the kernel put there: the port
    # masks on the selection, because SLIKG's ``ReduceSumVf`` sums every slot and a padded
    # one carrying mass would enter ``dI = Z * Y - p``.
    assert grad[0, 0] == 0
    assert not valid[3].any()
    assert torch.equal(grad[3], torch.zeros(3))
    assert (grad[:3][valid[:3]] != 0).all()
    # Each consumer's backward delivers the mass again, hence the doubling.
    assert torch.equal(grad[0], torch.tensor([0.0, mass[1] * 2, mass[2] * 2]))


@pytest.mark.parametrize(
    "ratio,training,with_pool",
    [
        (0, True, False),  # window-only in training: no selection, so no teacher
        (1, False, True),  # eval: no indexer gradient to feed, so no teacher
        (2, False, True),  # same, on the compressed path
    ],
)
def test_attention_without_a_teacher_edge(monkeypatch, v41_metadata, ratio, training, with_pool):
    """No carrier reaches the kernel unless this layer is training with a compressed pool.

    A compressed layer always receives its pool -- the port pairs ``cmp_k`` with
    ``topk_indices`` -- so "no pool" only arises for a window-only layer; the other axis
    is ``training``, which suppresses the teacher because there is no indexer gradient to
    feed.
    """
    captured = []

    def kernel(*args):
        captured.append(args)
        return args[0].clone(), torch.zeros(1, 4, 1)

    monkeypatch.setattr(ascendc._SparseMLA, "apply", kernel)
    config = ascendc.AscV41SparseAttention.Config(
        window_size=2,
        softmax_scale=0.5,
        compress_ratio=ratio,
        num_heads=1,
        head_dim=4,
        index_topk=3,
        index_num_heads=1,
        index_head_dim=4,
    )
    attention = build_cpu_model(config)
    attention.train(training)
    q = torch.zeros(1, 4, 1, 4, dtype=torch.bfloat16)
    # The port pairs the pool and its selection: one without the other is a contract
    # violation, so the no-pool case carries neither.
    indices = torch.zeros(1, 4, 1, dtype=torch.long) if with_pool else None
    shared = torch.zeros(1, 4 // ratio, 4, dtype=torch.bfloat16) if with_pool else None
    out = attention(
        q,
        q.squeeze(2),
        torch.zeros(1),
        v41_metadata([0, 4], port=config),
        cmp_k=shared,
        topk_indices=indices,
        topk_scores=torch.ones(1, 4, 1, requires_grad=True),
    )
    torch.testing.assert_close(out, q)
    # The apply args are (q, swa_k, cmp_k, topk_indices, sinks, attention_metadata,
    # softmax_scale, ratio, window_size, topk_scores, wants_teacher): the teacher travels
    # as the carrier's gradient, so no carrier means this layer carries none.  The boundary
    # tensors are no longer arguments -- they are read off the mask at position 5.
    assert captured[0][9] is None
    assert captured[0][10] is False
