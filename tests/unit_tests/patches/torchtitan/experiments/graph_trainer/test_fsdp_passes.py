"""Communication deduplication must preserve quantized consumers and gradients."""

import operator

import torchtitan_npu  # noqa: F401
import torch
from torch import fx
from torchtitan.experiments.graph_trainer import fsdp_passes, passes
from torchtitan.experiments.graph_trainer.fsdp_patterns import find_fsdp_unshard_outputs


def _quantized_components(weight):
    return weight * 0.5, weight.sum(dim=-1), weight.sum(dim=-2)


class _CollectiveInterpreter(fx.Interpreter):
    def call_function(self, target, args, kwargs):
        if target == torch.ops._c10d_functional.all_gather_into_tensor.default:
            return torch.cat([args[0]] * args[1])
        if target == torch.ops._c10d_functional.wait_tensor.default:
            return args[0]
        return super().call_function(target, args, kwargs)


def _graph():
    graph = fx.Graph()
    weight = graph.placeholder("weight")
    gathered = []
    for _ in range(2):
        cast = graph.call_function(torch.ops.aten._to_copy.default, (weight,), {"dtype": torch.float32})
        gather = graph.call_function(torch.ops._c10d_functional.all_gather_into_tensor.default, (cast, 2, "efsdp"))
        gathered.append(graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (gather,)))
    raw = graph.call_function(torch.ops.aten.sum.default, (gathered[0],))
    quant = graph.call_function(_quantized_components, (gathered[1],))
    qdata = graph.call_function(operator.getitem, (quant, 0))
    qdata = graph.call_function(torch.ops.aten.permute.default, (qdata, [0, 2, 1]))
    scale1 = graph.call_function(operator.getitem, (quant, 1))
    scale2 = graph.call_function(operator.getitem, (quant, 2))
    graph.output((raw, qdata, scale1, scale2))
    return fx.GraphModule({}, graph), weight, gathered[0], quant


def test_live_prepared_chain_replaces_dead_raw_chain():
    assert passes.deduplicate_fsdp_unshard_chains_pass is fsdp_passes.deduplicate_fsdp_unshard_chains_pass
    graph, weight, dead_wait, quant = _graph()
    output = next(node for node in graph.graph.nodes if node.op == "output")
    output.args = (output.args[0][1:],)
    graph.graph.eliminate_dead_code()
    graph.recompile()
    # The structural matcher follows the live wait through the unary quantizer.
    assert find_fsdp_unshard_outputs(weight) == (dead_wait, quant)
    assert not dead_wait.users
    data = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).requires_grad_()
    expected = _CollectiveInterpreter(graph).run(data)
    passes.deduplicate_fsdp_unshard_chains_pass(graph)
    assert find_fsdp_unshard_outputs(weight) == (quant,)
    assert dead_wait not in graph.graph.nodes
    expected_grad = torch.autograd.grad(sum(value.sum() for value in expected), data)[0]
    actual = _CollectiveInterpreter(graph).run(data)
    actual_grad = torch.autograd.grad(sum(value.sum() for value in actual), data)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert (
        sum(node.target == torch.ops._c10d_functional.all_gather_into_tensor.default for node in graph.graph.nodes) == 1
    )
