"""CPU coverage for CP receive-row gradients and selective checkpointing."""

import functools

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from torchtitan_npu.models.deepseek_v4.token_dispatcher import (
    CPTokenDispatcher,
    ExchangePlan,
    WindowPlan,
    _GatherReceivedRows,
)


@pytest.fixture
def identity_exchange():
    # Model only identity transport; real multi-rank communication is covered
    # by the existing Gloo tests and the separate NPU training validation.
    with torch.library._scoped_library("cp_received_rows_test", "FRAGMENT") as lib:
        lib.define("exchange(Tensor x, SymInt[] splits) -> Tensor")
        lib.impl("exchange", lambda x, splits: x.clone(), "CPU")
        torch.library.register_fake(
            "cp_received_rows_test::exchange",
            lambda x, splits: x.new_empty((sum(splits), x.shape[1])),
            lib=lib,
        )
        torch.library.register_autograd("cp_received_rows_test::exchange", lambda ctx, grad: (grad, None), lib=lib)
        try:
            yield
        finally:
            torch._dynamo.reset()


class _IdentityDispatcher(CPTokenDispatcher):
    def _all_to_all(self, x, in_splits, out_splits):
        return torch.ops.cp_received_rows_test.exchange.default(x, out_splits)


def _policy(ctx, op, *args, **kwargs):
    return (
        CheckpointPolicy.MUST_SAVE
        if op == torch.ops.cp_received_rows_test.exchange.default
        else CheckpointPolicy.PREFER_RECOMPUTE
    )


def _compile_backends():
    """Run CPU-safe graph capture in CI; exercise Inductor only on an NPU runner."""
    if hasattr(torch, "npu") and torch.npu.is_available():
        return ["aot_eager", "inductor"]
    return ["aot_eager"]


@pytest.mark.parametrize("backend", _compile_backends())
def test_cp_sac_compile_dynamic_backward(identity_exchange, backend):
    dispatcher = _IdentityDispatcher(CPTokenDispatcher.Config())

    def run(x, plan):
        return checkpoint(
            lambda value: dispatcher.gather(value, plan).sin().square(),
            x,
            use_reentrant=False,
            context_fn=functools.partial(create_selective_checkpoint_contexts, _policy),
        )

    compiled = torch.compile(run, backend=backend, fullgraph=True, dynamic=True)
    generator = torch.Generator().manual_seed(42)
    for splits in ([1, 0, 2, 1], [2, 1, 0, 3], [0, 0, 0, 0], [1, 1, 1, 1]):
        count = sum(splits)
        send = torch.arange(count) % 3  # Duplicate destinations must accumulate gradients.
        recv = torch.arange(count - 1, -1, -1)
        gather = torch.cat((torch.arange(8), 8 + torch.arange(count)))
        plan = WindowPlan(
            exchange=ExchangePlan(
                send_indices=send, send_splits=list(splits), recv_splits=list(splits), recv_offsets=recv
            ),
            gather_indices=gather,
            cu_seqlens_ori_kv=torch.tensor([0, 8 + count], dtype=torch.int32),
        )
        x = torch.randn(1, 8, 4, generator=generator, requires_grad=True)
        reference = x.detach().clone().requires_grad_()
        local = reference.flatten(0, 1)
        received = local[send]  # Identity exchange, followed by the original indexing path.
        expected = torch.cat([local, received[recv]], dim=0)[gather].unsqueeze(0).sin().square()
        actual = compiled(x, plan)
        expected.sum().backward()
        actual.sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x.grad, reference.grad)


def test_received_rows_gradcheck():
    rows = torch.randn(5, 2, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(42), requires_grad=True)
    indices = torch.tensor([4, 4, 0, 2, 1])

    def gather(value):
        return _GatherReceivedRows.apply(value, indices)

    assert torch.autograd.gradcheck(gather, (rows,))
    assert torch.autograd.gradgradcheck(gather, (rows,))
