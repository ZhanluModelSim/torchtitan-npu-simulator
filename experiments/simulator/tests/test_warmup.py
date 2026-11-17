# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from torchtitan_simulator.meta_env import _pp_context
from torchtitan_simulator.runtime import meta_runtime
from torchtitan_simulator.step import run_simulation_step
from torchtitan_simulator.warmup import warmup_training_iteration


@pytest.mark.parametrize("failure", [None, "backward", "clip", "optimizer"])
def test_warmup_executes_full_ga_and_cleans_gradient_and_context(failure):
    parameter = torch.nn.Parameter(torch.ones(2, device="meta"))
    optimizer = torch.optim.AdamW([parameter], foreach=False)
    calls = []
    previous = dict(_pp_context)

    def forward_backward():
        calls.append(("backward", _pp_context["mb_idx"]))
        parameter.sum().backward()
        if failure == "backward":
            raise RuntimeError("failed backward")

    def clip():
        calls.append(("clip", None))
        if failure == "clip":
            raise RuntimeError("failed clip")

    def update():
        calls.append(("optimizer", None))
        if failure == "optimizer":
            raise RuntimeError("failed optimizer")
        optimizer.step()

    with meta_runtime():
        kwargs = dict(
            gradient_accumulation=3,
            forward_backward=forward_backward,
            clip_gradients=clip,
            optimizer_step=update,
            zero_grad=optimizer.zero_grad,
        )
        if failure:
            with pytest.raises(RuntimeError, match=f"failed {failure}"):
                warmup_training_iteration(**kwargs)
        else:
            warmup_training_iteration(**kwargs)

    assert parameter.grad is None
    assert _pp_context == previous
    if failure is None:
        assert calls == [("backward", 0), ("backward", 1), ("backward", 2), ("clip", None), ("optimizer", None)]
        assert optimizer.state[parameter]["step"].item() == 1


def test_capture_after_one_or_two_warmups_has_same_graph_and_unique_resident_state():
    captures = []
    for warmup_count in (1, 2):
        model = torch.nn.Linear(3, 2, bias=False, device="meta")
        optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
        inputs = {"input": torch.ones(4, 3, device="meta")}
        dims = SimpleNamespace(
            world_size=1, pp=1, dp_replicate=1, dp_shard=1, cp=1, tp=1, ep=1, pp_enabled=False, dp_enabled=False
        )

        def forward_backward(*, input_dict=inputs, labels=None, global_valid_tokens=8, model=model):
            model(input_dict["input"]).sum().div(global_valid_tokens).backward()

        with meta_runtime():
            for _ in range(warmup_count):
                warmup_training_iteration(
                    gradient_accumulation=2,
                    forward_backward=forward_backward,
                    clip_gradients=lambda: None,
                    optimizer_step=optimizer.step,
                    zero_grad=optimizer.zero_grad,
                )
            state_storage = {
                int(value.untyped_storage()._cdata)
                for state in optimizer.state.values()
                for value in state.values()
                if value.device.type == "meta"
            }
            workload = run_simulation_step(
                model_parts=[model],
                parallel_dims=dims,
                forward_backward_step=forward_backward,
                input_dict=inputs,
                labels=torch.zeros(4, device="meta"),
                optimizer_step=optimizer.step,
                optimizer_zero_grad=optimizer.zero_grad,
                lr_scheduler_step=lambda: None,
                local_batch_size=4,
                seq_len=1,
                global_valid_tokens=8,
                gradient_accumulation=2,
                memory_device_type="meta",
                print_memory_summary=False,
            )
        memory = workload.iteration.schedule.annotations["memory_plan"]
        assert memory.optimizer_state_bytes == 48
        assert memory.optimizer_state_storage_count == len(state_storage) == 2
        state_lifetimes = [item for item in memory.tensor_lifetimes if item.kind == "optimizer_state"]
        assert len(state_lifetimes) == len({item.storage_key for item in state_lifetimes}) == 2
        assert all(item.birth_seq == -1 for item in state_lifetimes)
        assert sum(item.num_bytes for item in state_lifetimes) == 48
        assert memory.steady_state_forward_peak_active_bytes == memory.forward_peak_active_bytes
        names = [event.raw_op_type for event in memory.raw_events]
        assert not any("zeros_like" in name for name in names)
        captures.append((names, memory.steady_state_peak_active_bytes))

    assert captures[0] == captures[1]
