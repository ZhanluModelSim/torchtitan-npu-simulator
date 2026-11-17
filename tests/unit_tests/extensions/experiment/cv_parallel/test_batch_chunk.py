# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU checks for symbolic CV batch chunking and isolated calibration."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torchtitan.config import ConfigManager
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer, run_traced

from torchtitan_npu.extensions.experiment.cv_parallel import batch_chunk, symbolic_meta
from torchtitan_npu.extensions.experiment.cv_parallel.batch_chunk import _make_run_candidate


def test_mx_meta_family_keeps_flattened_batch_symbolic():
    shape_env = ShapeEnv()
    batch = shape_env.create_unbacked_symint()
    activation = torch.empty((4096 * batch, 7168), device="meta")

    q1, scale1, q2, scale2 = symbolic_meta._symbolic_dynamic_mx_quant_with_dual_axis_meta(
        activation,
        dst_type=292,
        scale_alg=1,
    )

    assert q1.shape == q2.shape == activation.shape
    assert q1.dtype == q2.dtype == torch.float8_e4m3fn
    assert scale1.shape == (4096 * batch, 112, 2)
    assert scale2.shape == (64 * batch, 7168, 2)
    assert scale2.shape[0].node.hint is None

    q, scale = symbolic_meta._symbolic_dynamic_mx_quant_meta(
        activation,
        axis=0,
        dst_type=292,
        block_size=32,
        scale_alg=1,
    )
    block_q, block_scale1, block_scale2 = symbolic_meta._symbolic_dynamic_block_mx_quant_meta(
        activation,
        dst_type=292,
        scale_alg=1,
    )

    assert q.shape == block_q.shape == activation.shape
    assert scale.shape == (64 * batch, 7168, 2)
    assert block_scale1.shape == scale1.shape
    assert block_scale2.shape == scale2.shape


def test_symbolic_meta_override_is_scoped_and_destroyed_on_error(monkeypatch):
    events = []

    class FakeLibrary:
        def __init__(self, *args):
            events.append(("create", args))

        def impl(self, op_name, function, *, allow_override):
            events.append(("impl", op_name, function, allow_override))

        def _destroy(self):
            events.append(("destroy",))

    monkeypatch.setattr(symbolic_meta, "_vendor_mx_supports_unbacked", lambda _: False)
    monkeypatch.setattr(torch.library, "Library", FakeLibrary)

    with symbolic_meta.batch_chunk_symbolic_meta_context(enabled=False):
        assert not events

    with (
        pytest.raises(RuntimeError, match="trace failed"),
        symbolic_meta.batch_chunk_symbolic_meta_context(enabled=True),
    ):
        raise RuntimeError("trace failed")

    assert [event[0] for event in events] == ["create", "impl", "impl", "impl", "destroy"]


def test_calibration_replays_restore_buffers_and_preserve_inputs_and_gradients():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor([[0.2, 0.4], [0.6, 0.8]]))
    model.register_buffer("counter", torch.tensor([3.0]))
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def step(x):
        model.counter.add_(1)
        x.add_(0.25)
        loss = (x @ model.weight).square().sum()
        return loss, *torch.autograd.grad(loss, (model.weight,))

    traced = minimal_fx_tracer(step, module=model)(inputs.clone())
    model.counter.fill_(3)
    model.weight.grad = torch.full_like(model.weight, 0.5)
    original_grad = model.weight.grad
    reference_weight = model.weight.detach().clone().requires_grad_()
    expected_loss = ((inputs + 0.25) @ reference_weight).square().sum()
    expected_gradient = torch.autograd.grad(expected_loss, reference_weight)[0]
    runner = _make_run_candidate(
        {
            "module": model,
            "traced_result": traced,
            "args": (inputs,),
            "train_context": nullcontext,
        }
    )

    for _ in range(2):
        runner.prepare()
        try:
            actual_loss, actual_gradient = runner(traced.gm)
            torch.testing.assert_close(model.counter, torch.tensor([4.0]), rtol=0, atol=0)
        finally:
            runner.finalize()

        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        torch.testing.assert_close(model.counter, torch.tensor([3.0]), rtol=0, atol=0)
        torch.testing.assert_close(inputs, torch.tensor([[1.0, 2.0], [3.0, 4.0]]), rtol=0, atol=0)
        torch.testing.assert_close(model.weight, reference_weight, rtol=0, atol=0)
        assert model.weight.grad is original_grad
        torch.testing.assert_close(original_grad, torch.full((2, 2), 0.5), rtol=0, atol=0)


def test_minimum_batch_preparation_traces_native_loss_and_gradients():
    # Import after the NPU package patches Trainer.Config.
    from torchtitan.experiments.graph_trainer.registry import POST_INIT_HOOKS
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_npu.models.deepseek_v4",
            "--config",
            "graph_trainer_deepseek_v4_debugmodel",
            "--hf-assets-path",
            "tests/assets/deepseek_v3",
            "--training.local-batch-size",
            "2",
            "--compile.ep-overlap.enabled",
            "--compile.ep-overlap.chunk-dim",
            "batch",
            "--compile.ep-overlap.strategy",
            "graph",
            "--compile.ep-overlap.module-fqn",
            "layers.*",
            "--compile.pass-pipeline",
            "cv_parallel",
        ]
    )
    config.loss.loss_fn.global_vocab_size = 16
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        head = torch.nn.Linear(4, 16, bias=False)
        hidden = torch.randn(config.training.local_batch_size, 16, 4, requires_grad=True)
        labels = torch.randint(16, (config.training.local_batch_size, 16))
    loss_fn = config.loss.build()
    loss_fn.set_lm_head(head)
    inputs = (hidden, labels, torch.tensor(labels.numel()))

    def step(x, target, valid_tokens):
        head.zero_grad(set_to_none=True)
        loss, _ = loss_fn(x, target, valid_tokens)
        return loss, *torch.autograd.grad(loss, (x, head.weight))

    expected = step(*inputs)
    trainer = SimpleNamespace(config=config, model_parts=[head], _make_fx_forward_backward_step=lambda *args: None)
    trainer._prepare_trace_inputs = lambda args, kwargs: GraphTrainer._prepare_trace_inputs(trainer, args, kwargs)
    POST_INIT_HOOKS[config.compile.pass_pipeline](trainer)
    trainer._prepare_trace_inputs(inputs, {})
    traced = minimal_fx_tracer(step, module=head)(*inputs)
    actual = run_traced(traced, module=head)(*inputs)

    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    # The chunk pass still needs the batch symbol to derive its B//2 shapes.
    fake_hidden = next(
        n.meta["val"] for n in traced.gm.graph.nodes if n.op == "placeholder" and n.meta["val"].ndim == 3
    )
    assert isinstance(fake_hidden.shape[0], torch.SymInt)


def test_runtime_context_is_instance_local_and_preserves_training_gradients(monkeypatch):
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
    from torchtitan.experiments.graph_trainer.registry import PASS_PIPELINE_REGISTRY, POST_INIT_HOOKS
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

    from torchtitan_npu.extensions.experiment.cv_parallel.batch_chunk import _RUNTIME_CONTEXT

    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.2, -0.3]]))
    inputs = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    labels = torch.tensor([[0.7], [-0.5]])
    valid_tokens = torch.tensor(2)
    expected_error = inputs @ model.weight.detach().T - labels
    expected_loss = expected_error.square().sum() / 2
    expected_gradient = expected_error.T @ inputs
    observed_contexts = []
    symbolic_meta_contexts = []

    @contextmanager
    def record_symbolic_meta_context(*, enabled):
        symbolic_meta_contexts.append(enabled)
        yield

    def inspect_pipeline(traced_result, config, *, parallel_dims):
        context = _RUNTIME_CONTEXT.get()()
        assert context["traced_result"] is traced_result
        assert context["module"] is model
        assert context["args"][0] is inputs
        observed_contexts.append(context)
        return []

    monkeypatch.setitem(PASS_PIPELINE_REGISTRY, "cv_parallel", inspect_pipeline)
    monkeypatch.setattr(batch_chunk, "batch_chunk_symbolic_meta_context", record_symbolic_meta_context)
    for experimental in (False, True, False):
        # Exercise the real CPU trace/replay without NPU Trainer initialization.
        trainer = object.__new__(GraphTrainer)
        compile_config = GraphTrainerCompileConfig(enable_passes=experimental, pass_pipeline="cv_parallel")
        compile_config.ep_overlap.enabled = experimental
        trainer.config = SimpleNamespace(compile=compile_config)
        trainer.parallel_dims = None
        trainer.model_parts = [model]
        trainer._traced_step = None
        trainer.train_context = nullcontext
        trainer.loss_fn = lambda pred, target, global_valid_tokens: (pred - target).square().sum() / global_valid_tokens
        if experimental:
            POST_INIT_HOOKS[trainer.config.compile.pass_pipeline](trainer)
        else:
            assert "_make_fx_forward_backward_step" not in vars(trainer)
            assert "_prepare_trace_inputs" not in vars(trainer)
        model.zero_grad(set_to_none=True)

        loss = trainer._make_fx_forward_backward_step(model, inputs, labels, valid_tokens, list(model.parameters()), {})

        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(model.weight.grad, expected_gradient, rtol=0, atol=0)
        assert _RUNTIME_CONTEXT.get() is None
    assert len(observed_contexts) == 1
    assert symbolic_meta_contexts == [True]
