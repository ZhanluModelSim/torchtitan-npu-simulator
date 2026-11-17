# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recovery contracts using upstream train_step and actual NPU routers, without distributed setup."""

from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchtitan_npu.extensions.torchtitan.trainer import Trainer, TrainerEx
from torchtitan_npu.extensions.experiment.anticipatory_routing.cache import RoutingMode
from torchtitan_npu.extensions.experiment.anticipatory_routing.config import AnticipatoryRoutingConfig

from torchtitan_npu.extensions.experiment.anticipatory_routing.router import build_routing_cache
from torchtitan_npu.extensions.experiment.anticipatory_routing.schedule import AnticipatorySchedule, Phase
from torchtitan_npu.extensions.experiment.anticipatory_routing.router import AnticipatoryHashRouter


class Scheduler(torch.optim.lr_scheduler.StepLR):
    def get_metrics(self):
        return {"lr": self.get_last_lr()[0]}


class Loader:
    def __init__(self, size=40):
        self.position = 0
        self.reads = []
        self.size = size

    def __iter__(self):
        for i in range(self.position, self.size):
            self.position = i + 1
            self.reads.append(i)
            yield [{"input": torch.tensor([[[i + 1.0, 0.0]]])}, torch.tensor([[i]])]

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        router = AnticipatoryHashRouter.__new__(AnticipatoryHashRouter)
        nn.Module.__init__(router)
        router.gate = nn.Identity()
        router.hash = False
        router.sorted_topk = False
        router.score_func = "sigmoid"
        router.num_experts, router.top_k = 2, 1
        router.num_expert_groups = None
        router.route_norm, router.route_scale = False, 1.0
        router._debug_force_load_balance = False
        self.router = router
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("counter", torch.tensor(0.0))


    def forward(self, x):
        self.counter.add_(1)
        self.router(x)
        return x * self.weight


def make_trainer(monkeypatch, *, delay=2, active=3, size=40, steps=10, device="cpu", compile_backend=None):
    trainer = object.__new__(TrainerEx)
    trainer.device = torch.device(device)
    trainer.step, trainer.ntokens_seen = 0, 0
    trainer.model_parts = [Model().to(trainer.device)]
    trainer.dataloader = Loader(size)
    config = AnticipatoryRoutingConfig(
        enable=True, delay_steps=delay, active_steps=active
    )
    trainer.config = SimpleNamespace(
        anticipatory=config, training=SimpleNamespace(steps=steps, disable_cuda_graphs=True, max_norm=100.0), checkpoint=SimpleNamespace(interval=2, keep_latest_k=0)
    )
    trainer.gradient_accumulation_steps, trainer.num_pipeline_parallel_microbatches = 2, 1
    trainer.parallel_dims = SimpleNamespace(
        pp_enabled=False, dp_enabled=False, ep_enabled=False, dp_cp_enabled=False, get_optional_mesh=lambda name: None
    )
    trainer.metrics_processor = SimpleNamespace(
        ntokens_since_last_log=0, data_loading_times=[], should_log=lambda step: False
    )
    trainer.train_context = nullcontext
    trainer._sdc = SimpleNamespace(finalize_sdc_step=lambda: None)
    # Avoid torch_npu's automatic optimizer backend probe in CPU-only CI.
    trainer.optimizers = torch.optim.SGD(
        trainer.model_parts[0].parameters(), lr=0.001, momentum=0.9, foreach=False, fused=False
    )
    trainer.lr_schedulers = Scheduler(trainer.optimizers, step_size=1, gamma=0.9)
    saves = []

    def save(curr_step, last_step=False):
        saves.append((curr_step, last_step))
        return True

    trainer.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None, save=save, saves=saves)

    def process(inputs, labels):
        trainer.ntokens_seen += labels.numel()
        return inputs["input"], labels, {}

    trainer.post_dataloading_process = process
    if compile_backend is not None:
        trainer.config.compile = SimpleNamespace(enable=True, components=["model"], backend=compile_backend)
        trainer.model_parts[0].compile(backend=compile_backend, fullgraph=True)
    trainer.anticipatory_schedule = AnticipatorySchedule(trainer)
    cache = trainer.anticipatory_schedule.cache
    trained = []

    def forward_backward(self, *, input_dict, labels, global_valid_tokens):
        index = int(labels.item())
        trained.append((self.step, index, cache.mode))
        inputs, _, _ = self.post_dataloading_process(input_dict, labels)
        loss = self.model_parts[0](inputs).square().sum() / global_valid_tokens
        loss.backward()
        return loss

    monkeypatch.setattr(Trainer, "forward_backward_step", forward_backward)
    return trainer, trained


def run_step(trainer):
    trainer.step += 1
    trainer.train_step(trainer.batch_generator(trainer.dataloader))


def snapshot(trainer):
    return deepcopy({
        "model": trainer.model_parts[0].state_dict(),
        "optimizer": trainer.optimizers.state_dict(),
        "scheduler": trainer.lr_schedulers.state_dict(),
        "data": trainer.dataloader.state_dict(),
        "step": trainer.step,
        "tokens": trainer.ntokens_seen,
    })


def install_checkpoint(trainer, saved):
    """In-memory checkpoint boundary; exercise the real engine rollback path."""
    def load(*, step):
        assert step == saved["step"]
        trainer.model_parts[0].load_state_dict(saved["model"])
        trainer.optimizers.load_state_dict(saved["optimizer"])
        trainer.lr_schedulers.load_state_dict(saved["scheduler"])
        trainer.dataloader.load_state_dict(saved["data"])
        trainer.step = saved["step"]
        trainer.ntokens_seen = saved["tokens"]
        return True

    trainer.checkpointer.load = load
    trainer.checkpointer.maybe_wait_for_saving = lambda: None


def test_rollback_restores_training_state_and_restarts_iterator(monkeypatch):
    trainer, trained = make_trainer(monkeypatch)
    stream = trainer.batch_generator(trainer.dataloader)
    for _ in range(2):
        run_step(trainer)
    saved = snapshot(trainer)
    assert saved["optimizer"]["state"]  # Momentum must actually have been created.
    assert saved["scheduler"]["last_epoch"] == 2
    install_checkpoint(trainer, saved)

    # Establish the exact next update from the saved state.
    run_step(trainer)
    expected_next = snapshot(trainer)
    run_step(trainer)
    old_iterator = stream.iterator
    assert trainer.dataloader.position == 8
    assert trainer.lr_schedulers.last_epoch == 4
    assert not torch.equal(trainer.model_parts[0].weight, saved["model"]["weight"])

    trainer.anticipatory_schedule.engine.rollback_to(2)
    torch.testing.assert_close(snapshot(trainer), saved, rtol=0, atol=0)
    assert all(p.grad is None for p in trainer.model_parts[0].parameters())
    assert stream.iterator is None
    assert trainer.batch_generator(trainer.dataloader) is stream

    # The abandoned iterator has its own cursor: without reset it would yield 8.
    run_step(trainer)
    assert stream.iterator is not old_iterator
    assert [index for _, index, _ in trained[-2:]] == [4, 5]
    torch.testing.assert_close(snapshot(trainer), expected_next, rtol=0, atol=0)


def test_multiple_microbatches_replay_their_own_layer_routes(monkeypatch):
    trainer, _ = make_trainer(monkeypatch)
    model = trainer.model_parts[0]
    model.second_router = deepcopy(model.router)
    cache = build_routing_cache([model], index_store_dtype="auto")
    trainer.anticipatory_schedule.cache = cache
    routers = [model.router, model.second_router]
    try:
        cache.begin_initialization()
        for router in routers:
            router(torch.ones(3, 2))
    finally:
        cache.reset()
    slots, recorded = [], []
    # Each (microbatch, layer) gets a distinct ordered token-to-expert pattern.
    patterns = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
    for microbatch in range(2):
        slot, expected = {}, []
        try:
            cache.begin_capture(slot)
            for layer, router in enumerate(routers):
                ids = torch.tensor(patterns[2 * microbatch + layer])
                logits = torch.nn.functional.one_hot(ids, num_classes=2).float() * 5
                _, actual, _ = router(logits)
                torch.testing.assert_close(actual.reshape(-1), ids, rtol=0, atol=0)
                expected.append(actual.detach().clone())
            cache.save_captured_routes()
        finally:
            cache.reset()
        assert set(slot) == {router._anticipatory_key for router in routers}
        slots.append(slot)
        recorded.append(expected)

    consumed = []
    def forward_backward(self, *, input_dict, labels, global_valid_tokens):
        microbatch = int(labels.item())
        for layer, router in enumerate(routers):
            expected = recorded[microbatch][layer]
            # Live top-k deliberately disagrees at EVERY token.
            live_ids = 1 - expected.reshape(-1)
            logits = (torch.nn.functional.one_hot(live_ids, 2).float() * 5).requires_grad_()
            scores, actual, _ = router(logits)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # Consume the returned IDs, rather than comparing cache to itself.
            experts = torch.tensor([2.0, 7.0])
            torch.testing.assert_close(experts[actual], experts[expected], rtol=0, atol=0)
            scores.sum().backward()
            selected_grad = logits.grad.gather(-1, expected)
            unselected_grad = logits.grad.gather(-1, 1 - expected)
            assert torch.all(selected_grad != 0)
            assert torch.all(unselected_grad == 0)
            consumed.append((microbatch, layer))
        return torch.tensor(1.0)

    monkeypatch.setattr(Trainer, "forward_backward_step", forward_backward)
    engine = trainer.anticipatory_schedule.engine
    engine.begin_step()
    engine.in_step = True
    try:
        cache.begin_training(slots)
        for microbatch in range(2):
            # The real TrainerEx wrapper must select the right slot.
            trainer.forward_backward_step(
                input_dict={}, labels=torch.tensor(microbatch), global_valid_tokens=torch.tensor(1)
            )
    finally:
        cache.reset()
        engine.in_step = False
    assert consumed == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert engine._microbatch_index == 2
    assert cache.mode is RoutingMode.OFF
    assert cache.slot is None and cache.step_slots is None


def test_spike_recovery_warmup_active_drain_and_normal(monkeypatch):
    trainer, trained = make_trainer(monkeypatch, active=2, steps=8)
    engine, schedule = trainer.anticipatory_schedule.engine, trainer.anticipatory_schedule
    saves = trainer.checkpointer.saves
    assert trainer.checkpointer.save(curr_step=0)
    observations = []

    def observe_spike(step, loss):
        observations.append(step)
        return 4 if step == 4 and observations.count(4) == 1 else None

    # Inject a detector verdict to exercise recovery without a manual-trigger config.
    monkeypatch.setattr(schedule.detector, "observe", observe_spike)
    router = trainer.model_parts[0].router
    recorded_ids, replayed_ids = [], []

    def change_live_choice(module, args):
        if schedule.cache.mode is RoutingMode.REPLAY:
            # Recorded inputs favor expert 0; live replay scores favor expert 1.
            return (args[0].flip(-1),)

    def observe_routes(module, args, output):
        ids = output[1].detach().clone()
        if schedule.cache.mode is RoutingMode.CAPTURE:
            recorded_ids.append(ids)
        elif schedule.cache.mode is RoutingMode.REPLAY:
            expected = recorded_ids[len(replayed_ids)]
            torch.testing.assert_close(ids, expected, rtol=0, atol=0)
            assert torch.all(ids == 0)
            replayed_ids.append(ids)

    gate_hook = router.gate.register_forward_pre_hook(change_live_choice)
    router_hook = router.register_forward_hook(observe_routes)
    monkeypatch.setattr(engine, "find_rollback_target", lambda onset: 2 if onset == 4 else None)
    updates = []
    original_step = trainer.optimizers.step

    def count_update(*args, **kwargs):
        updates.append(trainer.step)
        return original_step(*args, **kwargs)

    # Install after scheduler construction; preserve its optimizer-step marker.
    count_update._wrapped_by_lr_sched = True
    monkeypatch.setattr(trainer.optimizers, "step", count_update)
    for _ in range(2):
        run_step(trainer)
    saved = snapshot(trainer)
    install_checkpoint(trainer, saved)
    run_step(trainer)
    run_step(trainer)

    # WARMUP is inline: no optimizer/scheduler/token/buffer advancement.
    assert updates == [1, 2, 3, 4]
    after_warmup = snapshot(trainer)
    assert after_warmup.pop("data") == {"position": 8}
    restored = deepcopy(saved)
    restored.pop("data")
    torch.testing.assert_close(after_warmup, restored, rtol=0, atol=0)
    assert all(module.training for part in trainer.model_parts for module in part.modules())
    assert schedule.phase is Phase.ACTIVE
    assert len(schedule._queue) == 2
    assert [[int(labels.item()) for _, labels in entry.microbatches]
            for entry in schedule._queue] == [[4, 5], [6, 7]]
    assert all(entry.slots and len(entry.slots) == 2 for entry in schedule._queue)
    for entry in schedule._queue:
        for inputs, labels in entry.microbatches:
            assert inputs["input"].device.type == labels.device.type == "cpu"
        for slot in entry.slots:
            assert all(ids.device == trainer.device for ids in slot.values())
    assert engine.suppress_checkpoint_saves
    assert not trainer.checkpointer.save(curr_step=trainer.step)

    phases, lengths, positions, suppressed = [], [], [], []
    while trainer.step < 8:
        run_step(trainer)
        phases.append(schedule.phase)
        lengths.append(len(schedule._queue))
        positions.append(trainer.dataloader.position)
        suppressed.append(engine.suppress_checkpoint_saves)
        assert trainer.checkpointer.save(trainer.step) is (not engine.suppress_checkpoint_saves)

    assert phases == [Phase.ACTIVE, Phase.DRAIN, Phase.DRAIN, Phase.NORMAL, Phase.NORMAL, Phase.NORMAL]
    assert lengths == [2, 2, 1, 0, 0, 0]
    assert positions == [10, 12, 12, 12, 14, 16]
    assert suppressed == [True, True, True, False, False, False]
    assert [index for _, index, _ in trained] == list(range(8)) + list(range(4, 16))
    assert [mode for _, _, mode in trained[8:]] == [RoutingMode.REPLAY] * 8 + [RoutingMode.OFF] * 4
    assert trainer.dataloader.reads == list(range(8)) + list(range(4, 16))
    assert updates == [1, 2, 3, 4, 3, 4, 5, 6, 7, 8]
    assert schedule.detector._completed_updates == len(updates)
    assert trainer.lr_schedulers.last_epoch == 8
    assert trainer.ntokens_seen == 16
    assert trainer.model_parts[0].counter.item() == 16
    assert schedule._num_rollbacks == 1
    assert observations == [1, 2, 3, 4, 6, 7, 8]
    assert not schedule._queue
    assert schedule.cache.mode is RoutingMode.OFF
    assert schedule.cache.slot is None and schedule.cache.step_slots is None
    assert not engine.in_step and not engine.failed
    assert saves == [(0, False), (6, False), (7, False), (8, False)]
    assert len(recorded_ids) == len(replayed_ids) == 8
    gate_hook.remove()
    router_hook.remove()
    assert "train" not in TrainerEx.__dict__


def test_replay_failure_cleans_state_and_blocks_checkpoint(monkeypatch):
    trainer, _ = make_trainer(monkeypatch)
    schedule = trainer.anticipatory_schedule
    schedule._enter_anticipatory(schedule.data)
    failure = RuntimeError("injected replay failure")

    def fail_forward(module, args):
        if schedule.cache.mode is RoutingMode.REPLAY:
            assert schedule.cache.slot is not None
            raise failure

    handle = trainer.model_parts[0].register_forward_pre_hook(fail_forward)
    try:
        with pytest.raises(RuntimeError) as caught:
            run_step(trainer)
    finally:
        handle.remove()
    assert caught.value is failure
    assert schedule.engine.failed and not schedule.engine.in_step
    assert not schedule._queue
    assert schedule.cache.mode is RoutingMode.OFF
    assert schedule.cache.slot is None and schedule.cache.step_slots is None
    assert trainer.checkpointer.save(trainer.step, last_step=True) is False
    assert trainer.checkpointer.saves == []


@pytest.mark.parametrize("stage", ["initialization", "capture", "snapshot"])
def test_recording_failure_resets_routing_before_next_forward(monkeypatch, stage):
    trainer, _ = make_trainer(monkeypatch)
    schedule = trainer.anticipatory_schedule
    engine, cache = schedule.engine, schedule.cache
    model = trainer.model_parts[0]
    batch = ({"input": torch.ones(1, 1, 2)}, torch.zeros(1))
    if stage != "initialization":
        engine.initialize_routing(cache, batch)
    before = deepcopy(model.state_dict())
    failure = RuntimeError("injected recording failure")

    def fail(*args, **kwargs):
        raise failure

    handle = None
    if stage == "snapshot":
        monkeypatch.setattr(cache, "capture", fail)
    else:
        handle = model.register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError) as caught:
            if stage == "initialization":
                engine.initialize_routing(cache, batch)
            else:
                schedule._prefetch_and_capture(schedule.data)
    finally:
        if handle is not None:
            handle.remove()
    assert caught.value is failure
    assert cache.mode is RoutingMode.OFF
    assert cache.slot is None and cache.step_slots is None
    assert not cache.replay_enabled.item() and not schedule._queue
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    # OFF must also reach the router: a different token count must not read or
    # write the fixed-shape runtime buffers after the failed recording.
    inputs = torch.ones(1, 3, 2)
    torch.testing.assert_close(model(inputs), inputs, rtol=0, atol=0)


def test_rollback_target_skips_checkpoint_without_optimizer(monkeypatch, tmp_path):
    from torchtitan.components.checkpoint import MODEL, OPTIMIZER
    from torchtitan_npu.extensions.experiment.anticipatory_routing import engine as engine_module

    trainer, _ = make_trainer(monkeypatch)
    engine = trainer.anticipatory_schedule.engine
    trainer.checkpointer.folder = str(tmp_path)
    trainer.checkpointer.states = {MODEL: trainer.model_parts[0]}
    trainer.checkpointer.maybe_wait_for_saving = lambda: None
    for step in (2, 4):
        (tmp_path / f"step-{step}").mkdir()
    complete = dict.fromkeys(trainer.model_parts[0].state_dict())
    complete.update({f"{key}.leaf": None for key in engine._required_states()})
    incomplete = {key: value for key, value in complete.items() if not key.startswith(OPTIMIZER + ".")}

    def reader(path):
        keys = incomplete if str(path).endswith("step-4") else complete
        return SimpleNamespace(read_metadata=lambda: SimpleNamespace(state_dict_metadata=keys))

    monkeypatch.setattr(engine_module.dcp, "FileSystemReader", reader)
    assert engine.find_rollback_target(6) == 2


def test_batch_generator_rejects_foreign_loader(monkeypatch):
    trainer, _ = make_trainer(monkeypatch)
    with pytest.raises(ValueError, match="resumable dataloader"):
        trainer.batch_generator(Loader())


def test_replay_rejects_broadcastable_wrong_shape(monkeypatch):
    trainer, _ = make_trainer(monkeypatch)
    schedule = trainer.anticipatory_schedule
    router = trainer.model_parts[0].router
    try:
        schedule.cache.begin_initialization()
        router(torch.ones(3, 2))
    finally:
        schedule.cache.reset()
    try:
        schedule.cache.begin_training([{router._anticipatory_key: torch.zeros(1, 1, dtype=torch.int16)}])
        with pytest.raises(ValueError, match="Cached route shape"):
            schedule.cache.select(0)
    finally:
        schedule.cache.reset()


@pytest.mark.parametrize("backend", [None, "aot_eager"], ids=["eager", "compiled"])
@pytest.mark.parametrize("capture_grad_enabled", [False, True], ids=["capture-no-grad", "capture-grad"])
@pytest.mark.parametrize("training", [True, False], ids=["train-mode", "eval-mode"])
def test_routing_operation_controls_capture_and_replay(monkeypatch, backend, capture_grad_enabled, training):
    """Explicit routing operations take precedence over local autograd/module modes."""
    torch._dynamo.reset()
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 8)
    router = Model().router
    router.train(training)
    cache = build_routing_cache([router], index_store_dtype="auto")
    recorded_logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])
    expected = torch.tensor([[0], [1], [0]])
    live_logits = recorded_logits.flip(-1)
    try:
        cache.begin_initialization()
        router(recorded_logits)
    finally:
        cache.reset()
    if backend is not None:
        router.compile(backend=backend, fullgraph=True)

    try:
        slot = {}
        try:
            cache.begin_capture(slot)
            with torch.set_grad_enabled(capture_grad_enabled):
                _, actual, _ = router(recorded_logits)
            cache.save_captured_routes()
        finally:
            cache.reset()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(slot[router._anticipatory_key], expected.to(torch.int16), rtol=0, atol=0)
        buffer = router._anticipatory_replay_buffer
        recording = buffer.capture_indices.clone()

        try:
            cache.begin_training([slot])
            with torch.set_grad_enabled(not capture_grad_enabled):
                cache.select(0)
                _, actual, _ = router(live_logits)
        finally:
            cache.reset()
        # The live top-k is the opposite at every token, including under no_grad.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(buffer.capture_indices, recording, rtol=0, atol=0)

        try:
            cache.begin_training(None)
            with torch.no_grad():
                _, actual, _ = router(live_logits)
        finally:
            cache.reset()
        torch.testing.assert_close(actual, 1 - expected, rtol=0, atol=0)
        torch.testing.assert_close(buffer.capture_indices, recording, rtol=0, atol=0)
        # After reset, validation may use a different token count.
        _, actual, _ = router(live_logits[:1])
        torch.testing.assert_close(actual, (1 - expected)[:1], rtol=0, atol=0)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("compile_every", [1, 2], ids=["all-compiled", "mixed-eager-compiled"])
@pytest.mark.parametrize("full_ac", [False, True])
def test_compiled_routes_share_graphs_across_layers_modes_and_microbatches(
    monkeypatch, full_ac, compile_every, *, device="cpu", backend_name="aot_eager"
):
    """Forty distinct router keys must not require forty compiled variants."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch._dynamo.backends.registry import lookup_backend
    from torch.utils.checkpoint import DefaultDeviceType

    from torchtitan_npu.extensions.experiment.anticipatory_routing.engine import AnticipatoryTrainingEngine

    torch._dynamo.reset()
    if torch.device(device).type == "cpu":
        # CPU-only checkpoint inputs must not inherit the registered NPU backend.
        monkeypatch.setattr(DefaultDeviceType, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 8)
    monkeypatch.setattr(torch._dynamo.config, "skip_fwd_side_effects_in_bwd_under_checkpoint", True)
    original_router = Model().router.to(device)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = deepcopy(original_router)
            self.weight = nn.Parameter(torch.full((1, 1, 1), 0.01))
            self.register_buffer("counter", torch.zeros(()))

        def forward(self, x):
            self.counter.add_(1)
            scores, ids, _ = self.router(x)
            # Both expert choice and live gate gradients affect the result.
            # Concatenation keeps the two-feature residual layout without a
            # broadcast reduction in backward (older NPU lowering lacks strict_sum).
            routed = self.weight * scores * (ids + 1)
            return x + torch.cat((routed, routed * 2), dim=-1)

    class ManyBlocks(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                checkpoint_wrapper(Block()) if full_ac else Block() for _ in range(40)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = ManyBlocks().to(device)
    reference = deepcopy(model)
    cache = build_routing_cache([model], index_store_dtype="auto", device=torch.device(device))
    reference_cache = build_routing_cache([reference], index_store_dtype="auto")
    checkpoint_keys = set(model.state_dict())
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        # Use AOTAutograd for FullAC: running a raw FX checkpoint graph eagerly
        # does not functionalize the in-place counter used by this model.
        return lookup_backend(backend_name)(graph, inputs)

    for index, layer in enumerate(model.layers):
        if index % compile_every == 0:
            layer.compile(backend=backend, fullgraph=True)

    def make_engine(module):
        engine = object.__new__(AnticipatoryTrainingEngine)
        engine.trainer = SimpleNamespace(
            model_parts=[module], device=torch.device(device), ntokens_seen=0,
            train_context=nullcontext,
            post_dataloading_process=lambda inputs, labels: (inputs["input"], labels, {}),
        )
        return engine

    def update(module, routing, inputs, slots=None, index=0):
        module.zero_grad(set_to_none=True)
        inputs = inputs.detach().clone().requires_grad_()
        make_engine(module).initialize_routing(routing, ({"input": inputs}, torch.zeros(1)))
        try:
            routing.begin_training(slots)
            routing.select(index)
            output = module(inputs)
            output.sum().backward()
        finally:
            routing.reset()
        return output.detach(), inputs.grad

    def compare_update(inputs, slots=None, index=0):
        tolerance = {"rtol": 1e-5, "atol": 1e-6} if backend_name == "inductor" else {"rtol": 0, "atol": 0}
        torch.testing.assert_close(
            update(model, cache, inputs, slots, index),
            update(reference, reference_cache, inputs, slots, index), **tolerance,
        )
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad, **tolerance)

    def capture(inputs):
        # Exercise compiled recording, then snapshot IDs outside the graph.
        engine = make_engine(model)
        before = deepcopy(model.state_dict())
        entries = [layer._compiled_call_impl for layer in model.layers]
        slot = {}
        try:
            cache.begin_capture(slot)
            engine.forward_only_microbatch(({"input": inputs}, torch.zeros(1)))
            cache.save_captured_routes()
        finally:
            cache.reset()
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
        assert [layer._compiled_call_impl for layer in model.layers] == entries
        expected = {}
        try:
            reference_cache.begin_capture(expected)
            make_engine(reference).forward_only_microbatch(({"input": inputs}, torch.zeros(1)))
            reference_cache.save_captured_routes()
        finally:
            reference_cache.reset()
        torch.testing.assert_close(list(slot.values()), list(expected.values()), rtol=0, atol=0)
        assert len(slot) == 40
        assert all(ids.dtype == torch.int16 for ids in slot.values())
        assert all(slot[key].data_ptr() != buffer.capture_indices.data_ptr()
                   for key, buffer in cache.replay_buffers.items())
        return slot

    try:
        first = torch.tensor([[[2.0, -2.0]]], device=device)
        second = first.flip(-1)
        before = deepcopy(model.state_dict())
        make_engine(model).initialize_routing(cache, ({"input": first}, torch.zeros(1)))
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
        assert not graphs  # Shape initialization is isolated, eager and runs only once.
        compare_update(first)
        graph_count = len(graphs)
        assert 0 < graph_count < 8
        compare_update(second)
        assert len(graphs) == graph_count
        pointers = {key: buffer.indices.data_ptr() for key, buffer in cache.replay_buffers.items()}

        slots = [capture(first), capture(second)]
        assert graph_count < len(graphs) < 8
        assert any("capture_indices" in str(graph.graph) for graph in graphs[graph_count:])
        graph_count = len(graphs)
        saved_slots = deepcopy(slots)
        for _ in range(3):
            capture(second)
            capture(first)
        assert len(graphs) == graph_count
        torch.testing.assert_close(slots, saved_slots, rtol=0, atol=0)
        # Opposite live scores prove that replay consumes the recorded IDs.
        for _ in range(2):
            compare_update(second, slots, 0)
            compare_update(first, slots, 1)
            compare_update(first)
            assert len(graphs) == graph_count
            assert not cache.replay_enabled.item()

        # Loading a checkpoint updates weights/counters without replacing runtime tensors.
        saved = deepcopy(model.state_dict())
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1)
        model.load_state_dict(saved)
        reference.load_state_dict(saved)
        compare_update(second, slots, 0)
        capture(second)
        assert len(graphs) == graph_count
        assert torch._dynamo.config.recompile_limit == 8
        assert set(model.state_dict()) == checkpoint_keys
        assert pointers == {key: buffer.indices.data_ptr() for key, buffer in cache.replay_buffers.items()}
    finally:
        torch._dynamo.reset()


def test_replay_validates_slots_before_model_forward():
    model = Model()
    cache = build_routing_cache([model], index_store_dtype="auto", device=torch.device("cpu"))
    try:
        cache.begin_initialization()
        model(torch.ones(1, 3, 2))
    finally:
        cache.reset()
    try:
        cache.begin_training(None)
        model(torch.ones(1, 3, 2)).sum().backward()
    finally:
        cache.reset()
    key = model.router._anticipatory_key
    for slot, error, message in [
        ({}, KeyError, "No cached routing indices"),
        ({key: torch.zeros(1, 1, dtype=torch.int16)}, ValueError, "Cached route shape"),
    ]:
        with pytest.raises(error, match=message):
            try:
                cache.begin_training([slot])
                cache.select(0)
            finally:
                cache.reset()
        assert cache.mode is RoutingMode.OFF
        assert not cache.replay_enabled.item()
        assert cache.slot is None and cache.step_slots is None


def test_capture_compile_failure_preserves_upstream_entry_and_state():
    from torchtitan_npu.extensions.experiment.anticipatory_routing.engine import AnticipatoryTrainingEngine

    torch._dynamo.reset()
    model = Model()
    cache = build_routing_cache([model], index_store_dtype="auto", device=torch.device("cpu"))

    def fail_backend(graph, inputs):
        raise RuntimeError("capture compiler failed")

    model.compile(backend=fail_backend, fullgraph=True)
    original = model._compiled_call_impl
    original_forward = model.forward
    had_forward = "forward" in model.__dict__

    engine = object.__new__(AnticipatoryTrainingEngine)
    engine.trainer = SimpleNamespace(
        model_parts=[model], device=torch.device("cpu"), ntokens_seen=7,
        train_context=nullcontext,
        post_dataloading_process=lambda inputs, labels: (inputs["input"], labels, {}),
    )
    batch = ({"input": torch.ones(1, 3, 2)}, torch.zeros(1))
    before = deepcopy(model.state_dict())
    rng = torch.get_rng_state().clone()
    slot = {}
    try:
        engine.initialize_routing(cache, batch)
        with pytest.raises(RuntimeError, match="capture compiler failed"):
            try:
                cache.begin_capture(slot)
                engine.forward_only_microbatch(batch)
                cache.save_captured_routes()
            finally:
                cache.reset()
        assert model._compiled_call_impl is original
        assert model.forward == original_forward
        assert ("forward" in model.__dict__) == had_forward
        assert cache.mode is RoutingMode.OFF and cache.slot is None
        assert slot == {}
        assert engine.trainer.ntokens_seen == 7
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    finally:
        torch._dynamo.reset()


def test_compiled_capture_preserves_hooks_and_rng():
    from torchtitan_npu.extensions.experiment.anticipatory_routing.engine import AnticipatoryTrainingEngine

    class RandomModel(Model):
        def forward(self, x):
            return super().forward(x + torch.rand_like(x) * 0.01)

    def flip_inputs(module, args):
        return (args[0].flip(-1),)

    torch._dynamo.reset()
    model = RandomModel()
    hook = model.register_forward_pre_hook(flip_inputs)
    cache = build_routing_cache([model], index_store_dtype="auto", device=torch.device("cpu"))
    model.compile(backend="aot_eager", fullgraph=True)
    engine = object.__new__(AnticipatoryTrainingEngine)
    engine.trainer = SimpleNamespace(
        model_parts=[model], device=torch.device("cpu"), ntokens_seen=7,
        train_context=nullcontext,
        post_dataloading_process=lambda inputs, labels: (inputs["input"], labels, {}),
    )
    batch = ({"input": torch.tensor([[[2.0, -2.0]]])}, torch.zeros(1))
    rng = torch.get_rng_state().clone()
    before = deepcopy(model.state_dict())
    try:
        engine.initialize_routing(cache, batch)
        slot = {}
        try:
            cache.begin_capture(slot)
            engine.forward_only_microbatch(batch)
            cache.save_captured_routes()
        finally:
            cache.reset()
        # The hook flips expert preference before the compiled body executes.
        assert torch.all(slot[model.router._anticipatory_key] == 1)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
        assert engine.trainer.ntokens_seen == 7 and model.training
    finally:
        hook.remove()
        torch._dynamo.reset()


def test_compiled_spike_recovery_runs_capture_and_replay_without_recompiling(monkeypatch):
    from torch._dynamo.backends.registry import lookup_backend

    torch._dynamo.reset()
    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 8)
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return lookup_backend("aot_eager")(graph, inputs)

    try:
        trainer, trained = make_trainer(monkeypatch, active=2, steps=8, compile_backend=backend)
        schedule = trainer.anticipatory_schedule
        engine = schedule.engine
        for _ in range(2):
            run_step(trainer)
        saved = snapshot(trainer)
        install_checkpoint(trainer, saved)
        monkeypatch.setattr(engine, "find_rollback_target", lambda onset: 2)
        monkeypatch.setattr(schedule.detector, "observe", lambda step, loss: 4 if step == 4 else None)
        run_step(trainer)
        run_step(trainer)
        assert schedule.phase is Phase.ACTIVE
        after = snapshot(trainer)
        after.pop("data")
        expected = deepcopy(saved)
        expected.pop("data")
        torch.testing.assert_close(after, expected, rtol=0, atol=0)
        assert all(torch.all(ids == 0) for entry in schedule._queue for slot in entry.slots for ids in slot.values())
        assert any("capture_indices" in str(graph.graph) for graph in graphs)
        graph_count = len(graphs)
        assert 1 < graph_count < 8
        # Once recovery has started, do not inject another synthetic verdict.
        monkeypatch.setattr(schedule.detector, "observe", lambda step, loss: None)
        phases = []
        while trainer.step < 8:
            run_step(trainer)
            phases.append(schedule.phase)
        assert phases == [Phase.ACTIVE, Phase.DRAIN, Phase.DRAIN, Phase.NORMAL, Phase.NORMAL, Phase.NORMAL]
        assert len(graphs) == graph_count
        assert schedule._num_rollbacks == 1 and not schedule._queue
        assert trainer.ntokens_seen == 16 and trainer.model_parts[0].counter.item() == 16
        assert [index for _, index, _ in trained] == list(range(8)) + list(range(4, 16))
        assert torch._dynamo.config.recompile_limit == 8
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("full_ac", [False, True])
@pytest.mark.parametrize("instance_forward", [False, True])
def test_compiled_capture_keeps_disabled_hooks_outside_graph(monkeypatch, full_ac, instance_forward):
    """FSDP-style disabled pre/post hooks must run eagerly around capture."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
    from torch.distributed.fsdp._fully_shard._fsdp_common import _dynamo_disable as disable_hook

    from torchtitan_npu.extensions.experiment.anticipatory_routing.engine import AnticipatoryTrainingEngine

    # Use the actual FSDP decorator: its module-level Dynamo skip rules differ
    # from those of a wrapper defined in a user test file.
    events = []

    @disable_hook
    def pre_hook(module, args, kwargs):
        assert not torch.compiler.is_compiling()
        events.append("pre")
        return (args[0].flip(-1),), kwargs

    @disable_hook
    def post_hook(module, args, output):
        assert not torch.compiler.is_compiling()
        events.append("post")
        return output

    torch._dynamo.reset()
    monkeypatch.setattr(torch._dynamo.config, "skip_fwd_side_effects_in_bwd_under_checkpoint", True)
    model = checkpoint_wrapper(Model()) if full_ac else Model()
    if instance_forward:
        model.forward = model.forward
    original_forward = model.forward
    model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    model.register_forward_hook(post_hook)
    cache = build_routing_cache([model], index_store_dtype="auto", device=torch.device("cpu"))
    model.compile(backend="aot_eager", fullgraph=True)
    original_compiled = model._compiled_call_impl
    engine = object.__new__(AnticipatoryTrainingEngine)
    engine.trainer = SimpleNamespace(
        model_parts=[model], device=torch.device("cpu"), ntokens_seen=7,
        train_context=nullcontext,
        post_dataloading_process=lambda inputs, labels: (inputs["input"], labels, {}),
    )
    batch = ({"input": torch.tensor([[[2.0, -2.0]]])}, torch.zeros(1))
    before = deepcopy(model.state_dict())
    try:
        engine.initialize_routing(cache, batch)
        events.clear()
        for _ in range(2):
            slot = {}
            try:
                cache.begin_capture(slot)
                engine.forward_only_microbatch(batch)
                cache.save_captured_routes()
            finally:
                cache.reset()
            assert all(torch.all(ids == 1) for ids in slot.values())
            assert model._compiled_call_impl is original_compiled
            assert model.forward == original_forward
            assert ("forward" in model.__dict__) == instance_forward
            torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
        assert events == ["pre", "post", "pre", "post"]
        assert engine.trainer.ntokens_seen == 7 and model.training
        # The same upstream entry must still execute hooks and backward.
        model(batch[0]["input"]).sum().backward()
        assert events[-2:] == ["pre", "post"]
        assert all(parameter.grad is not None for parameter in model.parameters())
    finally:
        torch._dynamo.reset()
