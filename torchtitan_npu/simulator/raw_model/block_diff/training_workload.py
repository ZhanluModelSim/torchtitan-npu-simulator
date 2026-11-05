# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capture actual CPU/meta training operators of the portable block-diffusion core.

Forward, loss backward and an optional real AdamW update are recorded through
TorchDispatchMode. SDPA's math backend is selected explicitly so the capture
contains its actual PyTorch decomposition, including its autograd operations.
This is a portable reference trace, NOT native TorchTitan simulator IR and NOT
an NPU kernel, HBM, peak-live-storage or latency estimate. Backend fusion and
lowering are a separate integration step.

Tensor objects are retained until capture finishes to make identity/alias IDs
stable. Consequently, captured tensor liveness must not be interpreted as normal
execution liveness. Parameter, materialized gradient and materialized optimizer
state bytes are reported separately after the step; no backward multiplier or
saved-activation peak estimate is substituted for execution.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode

if TYPE_CHECKING or __package__:
    from .model import BlockDiffusionModel, ModelConfig, RoutingPlan, TrainingSpec
else:
    from model import BlockDiffusionModel, ModelConfig, RoutingPlan, TrainingSpec


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value).removeprefix("torch.")
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    return value


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _flatten_tensors(item)]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _flatten_tensors(item)]
    return []


@dataclass(frozen=True)
class TensorInfo:
    tensor_id: int
    storage_id: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: str
    device: str
    storage_nbytes: int


@dataclass(frozen=True)
class OperatorRecord:
    sequence: int
    phase: str
    op: str
    inputs: tuple[TensorInfo, ...]
    outputs: tuple[TensorInfo, ...]
    known_matrix_flops: int | None
    mutated_inputs: tuple[int, ...]
    attributes: dict[str, Any]


@dataclass(frozen=True)
class TrainingTrace:
    model_config: ModelConfig
    training_spec: TrainingSpec
    batch_size: int
    seq_len: int
    num_masked_tokens_per_request: int
    device: str
    optimizer: str
    optimizer_config: dict[str, Any]
    records: tuple[OperatorRecord, ...]
    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    parameter_tensors: tuple[TensorInfo, ...]
    gradient_tensors: tuple[TensorInfo, ...]
    optimizer_state_tensors: tuple[TensorInfo, ...]
    loss: float | None
    valid_tokens: int
    torch_version: str
    attention_backend: str = "pytorch_sdpa_math"

    def to_dict(self) -> dict[str, Any]:
        result = _serialize(self)
        result.update(
            schema_version=1,
            scope="actual_single_rank_training_reference_trace",
            mask_pattern="last num_masked_tokens_per_request positions; input mask ID 0; unshifted targets",
            matrix_flop_convention="multiply/add = 2 for observed mm, bmm and addmm only",
            unknown_operation_costs="null matrix FLOPs: no inferred NPU cost",
            memory_scope="distinct parameter/gradient/optimizer storage after the step; no peak or HBM estimate",
            tensor_identity_scope="capture-local IDs; shared storage_id expresses aliasing",
        )
        result["phases"] = {
            phase: {
                "num_operators": sum(record.phase == phase for record in self.records),
                "known_matrix_flops": sum(
                    record.known_matrix_flops or 0 for record in self.records if record.phase == phase
                ),
                "num_operators_without_matrix_flop_model": sum(
                    record.known_matrix_flops is None for record in self.records if record.phase == phase
                ),
            }
            for phase in ("forward", "backward", "optimizer")
        }
        return result


class _TrainingCapture(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.phase = "forward"
        self.records: list[OperatorRecord] = []
        self._tensor_ids: dict[int, int] = {}
        self._storage_ids: dict[int, int] = {}
        self._retained_tensors: list[torch.Tensor] = []

    def tensor_info(self, tensor: torch.Tensor) -> TensorInfo:
        identity = id(tensor)
        if identity not in self._tensor_ids:
            self._tensor_ids[identity] = len(self._tensor_ids)
            self._retained_tensors.append(tensor)
        storage = tensor.untyped_storage()
        storage_identity = storage._cdata
        if storage_identity not in self._storage_ids:
            self._storage_ids[storage_identity] = len(self._storage_ids)
        return TensorInfo(
            tensor_id=self._tensor_ids[identity],
            storage_id=self._storage_ids[storage_identity],
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            storage_offset=int(tensor.storage_offset()),
            dtype=str(tensor.dtype).removeprefix("torch."),
            device=str(tensor.device),
            storage_nbytes=storage.nbytes(),
        )

    def _attribute(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return {"tensor_id": self.tensor_info(value).tensor_id}
        if isinstance(value, (tuple, list)):
            return [self._attribute(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._attribute(item) for key, item in value.items()}
        if isinstance(value, (torch.dtype, torch.device, torch.layout, torch.memory_format)):
            return str(value).removeprefix("torch.")
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return str(value)

    def __torch_dispatch__(self, func, types, args: tuple[Any, ...] = (), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        input_tensors = _flatten_tensors((args, kwargs))
        inputs = tuple(self.tensor_info(tensor) for tensor in input_tensors)
        attributes, mutated = {}, []
        for index, argument in enumerate(func._schema.arguments):
            supplied = index < len(args) or argument.name in kwargs
            if not supplied:
                continue
            value = args[index] if index < len(args) else kwargs[argument.name]
            attributes[argument.name] = self._attribute(value)
            if argument.alias_info is not None and argument.alias_info.is_write:
                mutated.extend(self.tensor_info(tensor).tensor_id for tensor in _flatten_tensors(value))
        result = func(*args, **kwargs)
        output_tensors = _flatten_tensors(result)
        outputs = tuple(self.tensor_info(tensor) for tensor in output_tensors)
        known_matrix_flops = None
        name = str(func)
        if name in ("aten.mm.default", "aten.bmm.default"):
            known_matrix_flops = 2 * output_tensors[0].numel() * args[0].shape[-1]
        elif name == "aten.addmm.default":
            known_matrix_flops = 2 * output_tensors[0].numel() * args[1].shape[-1]
        self.records.append(
            OperatorRecord(
                sequence=len(self.records),
                phase=self.phase,
                op=name,
                inputs=inputs,
                outputs=outputs,
                known_matrix_flops=known_matrix_flops,
                mutated_inputs=tuple(dict.fromkeys(mutated)),
                attributes=attributes,
            )
        )
        return result


def _storage_bytes(tensors: tuple[TensorInfo, ...]) -> int:
    return sum({tensor.storage_id: tensor.storage_nbytes for tensor in tensors}.values())


def _positive_integer(name: str, value: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def capture_training_step(
    config: ModelConfig,
    spec: TrainingSpec,
    *,
    batch_size: int,
    seq_len: int,
    num_masked_tokens_per_request: int,
    device: str = "meta",
    optimizer: str = "none",
) -> TrainingTrace:
    """Execute and capture one actual single-rank forward/backward/update.

    CPU uses deterministic initialized weights and suffix corruption; meta keeps
    the same shapes and graph without numerical values. MoE uses a balanced,
    explicit routing replay in both modes, so inactive experts remain inactive.
    AdamW uses PyTorch's non-foreach, non-fused implementation (lr=1e-3); state
    bytes are inspected after its actual first update, including scalar steps.
    Inputs/initialization and optimizer construction are outside captured phases.
    """
    _positive_integer("batch_size", batch_size)
    _positive_integer("seq_len", seq_len)
    _positive_integer("num_masked_tokens_per_request", num_masked_tokens_per_request, 0)
    target_device = torch.device(device)
    if target_device.type not in ("cpu", "meta"):
        raise ValueError("training reference capture supports CPU and meta only")
    if optimizer not in ("none", "adamw"):
        raise ValueError("optimizer must be none or adamw")
    if config.vocab_size < 2:
        raise ValueError("deterministic corruption requires vocab_size >= 2")
    ignore_in_targets = 1 <= spec.ignore_index < config.vocab_size
    num_target_ids = config.vocab_size - 1 - int(ignore_in_targets)
    if num_target_ids == 0:
        raise ValueError("deterministic corruption needs a non-mask, non-ignored vocabulary token")
    maximum_masked = spec.block_size if spec.attention_policy == "prefix_causal_canvas_bidirectional" else seq_len
    if num_masked_tokens_per_request > maximum_masked:
        raise ValueError("masked-token count exceeds the policy's eligible positions")
    if seq_len < spec.block_size or seq_len % spec.block_size:
        raise ValueError("seq_len must be a positive multiple of TrainingSpec.block_size")

    # Isolate the deterministic reference initialization from the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        with torch.device(target_device):
            model = BlockDiffusionModel(config)
        model.train()
        clean = torch.arange(batch_size * seq_len, device=target_device, dtype=torch.long).reshape(batch_size, seq_len)
        clean = clean.remainder(num_target_ids) + 1
        if ignore_in_targets:
            clean = torch.where(clean >= spec.ignore_index, clean + 1, clean)
        positions = torch.arange(seq_len, device=target_device)
        corrupted = positions >= seq_len - num_masked_tokens_per_request
        input_ids = torch.where(corrupted, 0, clean)
        labels = torch.where(corrupted, clean, spec.ignore_index)
        routing_plan = None
        if config.num_experts:
            routing_plan = RoutingPlan.balanced(batch_size * seq_len, config.num_experts, config.num_experts_per_tok)
        optimizer_instance = None
        optimizer_config = {}
        if optimizer == "adamw":
            optimizer_config = {
                "lr": 1e-3,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
                "foreach": False,
                "maximize": False,
                "capturable": False,
                "differentiable": False,
                "fused": False,
            }
            optimizer_instance = torch.optim.AdamW(model.parameters(), **optimizer_config)

        capture = _TrainingCapture()
        parameter_tensors = tuple(capture.tensor_info(parameter) for parameter in model.parameters())
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH), capture:
            output = model.forward_training(input_ids, labels, spec=spec, routing_plan=routing_plan)
            capture.phase = "backward"
            output.loss.backward()
            if optimizer_instance is not None:
                capture.phase = "optimizer"
                try:
                    optimizer_instance.step()
                except (NotImplementedError, RuntimeError) as error:
                    if target_device.type == "meta":
                        raise RuntimeError(
                            "this PyTorch version cannot execute AdamW on meta; use optimizer='none' or device='cpu'"
                        ) from error
                    raise
        gradient_tensors = tuple(
            capture.tensor_info(parameter.grad) for parameter in model.parameters() if parameter.grad is not None
        )
        optimizer_state_tensors = ()
        if optimizer_instance is not None:
            optimizer_state_tensors = tuple(
                capture.tensor_info(tensor)
                for state in optimizer_instance.state.values()
                for tensor in _flatten_tensors(state)
            )
        loss = float(output.loss.detach()) if target_device.type == "cpu" else None
        valid_tokens = (
            int(output.valid_tokens) if target_device.type == "cpu" else batch_size * num_masked_tokens_per_request
        )
        return TrainingTrace(
            model_config=config,
            training_spec=spec,
            batch_size=batch_size,
            seq_len=seq_len,
            num_masked_tokens_per_request=num_masked_tokens_per_request,
            device=str(target_device),
            optimizer=optimizer,
            optimizer_config=optimizer_config,
            records=tuple(capture.records),
            parameter_bytes=_storage_bytes(parameter_tensors),
            gradient_bytes=_storage_bytes(gradient_tensors),
            optimizer_state_bytes=_storage_bytes(optimizer_state_tensors),
            parameter_tensors=parameter_tensors,
            gradient_tensors=gradient_tensors,
            optimizer_state_tensors=optimizer_state_tensors,
            loss=loss,
            valid_tokens=valid_tokens,
            torch_version=torch.__version__,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "meta"), default="meta")
    parser.add_argument("--optimizer", choices=("none", "adamw"), default="none")
    parser.add_argument(
        "--attention-policy", choices=("block_causal", "prefix_causal_canvas_bidirectional"), default="block_causal"
    )
    parser.add_argument("--output", help="Write full JSON trace; otherwise print a compact summary")
    args = parser.parse_args()
    trace = capture_training_step(
        ModelConfig(),
        TrainingSpec(block_size=4, attention_policy=args.attention_policy),
        batch_size=1,
        seq_len=8,
        num_masked_tokens_per_request=4,
        device=args.device,
        optimizer=args.optimizer,
    )
    data = trace.to_dict()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(data, output, indent=2)
            output.write("\n")
    else:
        print(
            json.dumps(
                {
                    key: data[key]
                    for key in (
                        "device",
                        "optimizer",
                        "loss",
                        "valid_tokens",
                        "parameter_bytes",
                        "gradient_bytes",
                        "optimizer_state_bytes",
                        "phases",
                    )
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
