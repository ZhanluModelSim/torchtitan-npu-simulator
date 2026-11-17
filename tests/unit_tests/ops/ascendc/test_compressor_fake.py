# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise the CANN fake boundary without loading CANN or an NPU kernel."""

import sys
import types

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode, ShapeEnv

from torchtitan_npu.ops.ascendc.compressor import _compressor_forward_fake, load_compressor


@pytest.mark.parametrize("ratio", [4, 128])
def test_compressor_fake_matches_target_parameter_order(ratio):
    with FakeTensorMode():
        x, weight = torch.empty((8192, 8)), torch.empty((32, 8))
        cache, ape = torch.empty((1, 8, 64)), torch.empty((ratio, 32))
        table = torch.empty((10, 135), dtype=torch.int32)
        cu = torch.empty((11,), dtype=torch.int32)
        lengths = torch.empty((10,), dtype=torch.int32)
        outputs = _compressor_forward_fake(x, weight, weight, cache, ape, ratio, table, cu, lengths, None, 2, 1, True)
        assert outputs[0].shape == (min(8192, 8192 // ratio + 10), 16)
        assert outputs[1].shape == (*outputs[0].shape[:1], 2 * ratio, 16)


@pytest.mark.parametrize("symbolic_tokens,symbolic_documents", [(True, True), (True, False), (False, True)])
def test_compressor_fake_uses_input_symbols(symbolic_tokens, symbolic_documents):
    env = ShapeEnv()
    tokens = env.create_unbacked_symint() if symbolic_tokens else 8192
    documents = env.create_unbacked_symint() if symbolic_documents else 11
    with FakeTensorMode(shape_env=env):
        for size in (tokens, documents):
            if isinstance(size, torch.SymInt):
                torch._constrain_as_size(size, min=2)
        x = torch.empty((tokens, 8))
        weight = torch.empty((8, 8))
        cu = torch.empty((documents,), dtype=torch.int32)
        capacity = _compressor_forward_fake(x, weight, weight, x, x, cu_seqlens=cu, cmp_ratio=4)[0].shape[0]
    assert isinstance(capacity, torch.SymInt)
    assert capacity.node.shape_env is env
    input_symbols = set()
    for size in (tokens, documents):
        if isinstance(size, torch.SymInt):
            input_symbols.update(size.node.expr.free_symbols)
    assert capacity.node.expr.free_symbols <= input_symbols
    # Equivalent Min and Abs expressions need not have the same SymPy tree.
    # Check both sides of the capacity cap, including the equality boundary.
    for token_value, document_value in ((2, 2), (8, 2), (8, 7), (8, 8), (8192, 11), (8192, 8193)):
        bindings = {}
        if isinstance(tokens, torch.SymInt):
            bindings[tokens.node.expr] = token_value
        else:
            token_value = tokens
        if isinstance(documents, torch.SymInt):
            bindings[documents.node.expr] = document_value
        else:
            document_value = documents
        rows = int(capacity.node.expr.subs(bindings))
        assert rows == min(token_value, token_value // 4 + document_value - 1)


@pytest.mark.parametrize("ratio,coff", [(4, 2), (128, 1)])
@pytest.mark.parametrize("shape,batch", [((16, 8), 2), ((3, 8), 3), ((0, 8), 0), ((2, 17, 8), 2)])
def test_compressor_fake_shapes_and_dtypes(ratio, coff, shape, batch):
    with FakeTensorMode():
        x = torch.empty(shape, dtype=torch.bfloat16)
        weight = torch.empty((coff * 16, 8))
        outputs = _compressor_forward_fake(
            x, weight, weight, torch.empty((1, 8, 4 * 16)), torch.empty((ratio, coff * 16)),
            cmp_ratio=ratio, cu_seqlens=torch.empty((batch + 1,), dtype=torch.int32), coff=coff,
        )
        if len(shape) == 3:
            prefix = (shape[0], (shape[1] + ratio - 1) // ratio)
        else:
            prefix = (min(shape[0], shape[0] // ratio + batch),)
        assert outputs[0].shape == (*prefix, 16)
        assert outputs[0].dtype == torch.bfloat16
        for saved in outputs[1:]:
            assert saved.shape == (*prefix, coff * ratio, 16)
            assert saved.dtype == torch.float32


@pytest.fixture(scope="module")
def cann_boundary():
    @torch.library.custom_op("test_compressor_compat::forward", mutates_args=())
    def forward(x: torch.Tensor, cu: torch.Tensor, ratio: int) -> torch.Tensor:
        rows = min(x.shape[0], x.shape[0] // ratio + cu.shape[0] - 1)
        return x.sum().expand(rows, x.shape[1]).clone()

    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return torch.ones_like(x) * grad.sum(), None, None

    forward.register_autograd(backward, setup_context=setup_context)
    return forward


@pytest.mark.parametrize("ratio", [4, 128])
def test_compressor_fake_under_make_fx_with_unbacked_sizes(cann_boundary, ratio):
    def adapted(x, cu, ratio):
        weight = x.new_empty((8, 8))
        return _compressor_forward_fake(x, weight, weight, x, x, ratio, None, cu, None, None)[0]

    cann_boundary.register_fake(adapted)
    env = ShapeEnv()
    tokens, documents = env.create_unbacked_symint(), env.create_unbacked_symint()
    with FakeTensorMode(shape_env=env):
        torch._constrain_as_size(tokens, min=1)
        torch._constrain_as_size(documents, min=2)
        x = torch.empty((tokens, 8))
        cu = torch.empty((documents,), dtype=torch.int32)
        graph = make_fx(lambda x, cu: cann_boundary(x, cu, ratio), tracing_mode="symbolic")(x, cu)
    output = next(node for node in graph.graph.nodes if node.op == "output").args[0]
    assert isinstance(output.meta["val"].shape[0], torch.SymInt)
    assert all("abs" not in str(node.target) for node in graph.graph.nodes if node.op == "call_function")


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_loader_replaces_unbacked_fake_and_preserves_autograd(monkeypatch, cann_boundary, ratio, checkpointed):
    # Match the failing Python min in CANN's fake, not a traced Python kernel.
    def original_fake(x, cu, ratio):
        return x.new_empty((min(x.shape[0], x.shape[0] // ratio + cu.shape[0] - 1), x.shape[1]))

    cann_boundary.register_fake(original_fake)

    def symbolic_call():
        env = ShapeEnv()
        tokens, documents = env.create_unbacked_symint(), env.create_unbacked_symint()
        with FakeTensorMode(shape_env=env):
            torch._constrain_as_size(tokens, min=1)
            torch._constrain_as_size(documents, min=2)
            return cann_boundary(torch.empty((tokens, 8)), torch.empty((documents,), dtype=torch.int32), ratio)

    with pytest.raises(GuardOnDataDependentSymNode):
        symbolic_call()

    # Adapt the small CPU test schema to CANN's real fake signature. Registration
    # still passes through the production loader and PyTorch dispatcher.
    register_fake = torch.library.register_fake

    def register(op, fake):
        assert fake is _compressor_forward_fake
        def adapted(x, cu, ratio):
            weight = x.new_empty((x.shape[1], x.shape[1]))
            return fake(x=x, wkv=weight, wgate=weight, state_cache=x, ape=x, cmp_ratio=ratio, cu_seqlens=cu)[0]

        return register_fake(cann_boundary._opoverload, adapted)

    monkeypatch.setattr(torch.library, "register_fake", register)
    namespace = types.SimpleNamespace(_compressor_forward=torch.ops.test_compressor_compat.forward)
    monkeypatch.setattr(torch.ops, "cann_ops_transformer", namespace)
    module = types.ModuleType("cann_ops_transformer.ops.compressor")
    module.compressor = cann_boundary
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert load_compressor() is cann_boundary
    assert load_compressor() is cann_boundary  # Construction for another layer is safe.
    symbolic_call()

    from torch.utils.checkpoint import checkpoint

    def run(x, cu, ratio):
        if checkpointed:
            return checkpoint(cann_boundary, x, cu, ratio, use_reentrant=False)
        return cann_boundary(x, cu, ratio)

    compiled = torch.compile(run, backend="aot_eager", fullgraph=True, dynamic=True)
    for tokens, batch in [(16, 2), (8, 8)]:
        x = torch.randn((tokens, 8), requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        cu = torch.zeros(batch + 1, dtype=torch.int32)
        setattr(x, "_dynamo_unbacked_indices", {0})  # noqa: B010
        setattr(cu, "_dynamo_unbacked_indices", {0})  # noqa: B010
        actual = compiled(x, cu, ratio)
        expected = cann_boundary(reference_x, cu, ratio)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(x.grad, reference_x.grad)
