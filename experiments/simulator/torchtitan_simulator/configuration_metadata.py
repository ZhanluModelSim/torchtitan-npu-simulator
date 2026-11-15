# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read-only configuration metadata derived from supported native factories."""

from dataclasses import asdict, fields
from importlib import import_module

from torchtitan_simulator.config import PRESET_FACTORIES, SimulationConfig
from torchtitan_simulator.model_overrides import ModelOverrides, MoEOverrides, describe_geometry

_GROUPS = {
    "input": ("seq_len", "local_batch_size", "global_batch_size"),
    "precision": (
        "training_dtype",
        "mixed_precision_param",
        "mixed_precision_reduce",
        "enable_quantized_training",
        "quantization_recipe",
    ),
    "parallel": (
        "world_size",
        "logical_rank",
        "data_parallel_replicate_degree",
        "data_parallel_shard_degree",
        "tensor_parallel_degree",
        "context_parallel_degree",
        "expert_parallel_degree",
        "pipeline_parallel_degree",
    ),
    "memory": (
        "activation_checkpoint_mode",
        "memory_parameter_storage_dtype",
        "memory_offload_ac_saved_tensors",
        "memory_offload_optimizer_states",
    ),
    "execution": ("optimizer_name", "muon_ns_steps"),
}


def _field(path: list[str], annotation: str, default: object, group: str) -> dict[str, object]:
    kind = next(
        (
            kind
            for name, kind in (("bool", "boolean"), ("int", "integer"), ("float", "float"), ("str", "string"))
            if annotation.startswith(name)
        ),
        "literal",
    )
    choices: list[dict[str, object]] = []
    if annotation.startswith("Literal["):
        # Literal alternatives belong to the native dataclass annotation, not a second validation registry.
        import ast

        expression = ast.parse(annotation.split(" | ")[0], mode="eval").body
        if isinstance(expression, ast.Subscript):
            values = expression.slice.elts if isinstance(expression.slice, ast.Tuple) else [expression.slice]
            choices = [{"label": str(ast.literal_eval(value)), "value": ast.literal_eval(value)} for value in values]
    return {
        "path": path,
        "label": path[-1],
        "group": group,
        "kind": kind,
        "default": default,
        "choices": choices,
        "help": "Inherited from the native preset when absent.",
    }


def describe_configuration() -> dict[str, object]:
    """Describe built-in presets without model construction, overrides or capture.

    Runtime/precision/input values survive model selection. MTP count is model
    structure and resets to the simulator's native default along with geometry.
    Arbitrary user factories are intentionally never imported by discovery.
    """
    simulation = SimulationConfig()
    declarations = {item.name: item for item in fields(SimulationConfig)}
    common = [
        _field([name], str(declarations[name].type), getattr(simulation, name), group)
        for group, names in _GROUPS.items()
        for name in names
        if getattr(simulation, name) is not None
    ]
    models = []
    for name, target in PRESET_FACTORIES.items():
        module, factory = target.rsplit(".", 1)
        trainer = getattr(import_module(module), factory)(num_mtp_layers=simulation.num_mtp_modules)
        identity = [{"path": ["config"], "value": name}]
        model_fields = [_field(["num_mtp_modules"], "int", simulation.num_mtp_modules, "model")]
        for group, names in _GROUPS.items():
            for key in names:
                if getattr(simulation, key) is not None:
                    continue
                native_key = {"training_dtype": "dtype"}.get(key, key)
                native_default = getattr(trainer.training, native_key, None)
                if key == "muon_ns_steps":
                    native_default = getattr(trainer.optimizer, key, None)
                if key == "optimizer_name":
                    from torchtitan_simulator.config import _optimizer_selection

                    native_default = _optimizer_selection(simulation, trainer)
                model_fields.append(_field([key], str(declarations[key].type), native_default, group))
        limitations = []
        if trainer.model_spec.name == "deepseek_v4":
            geometry = asdict(describe_geometry(trainer, num_mtp_modules=simulation.num_mtp_modules))
            aliases = {"o_groups": "n_groups", "hc_sinkhorn_iters": "sinkhorn_iters"}
            for prefix, cls in (([], ModelOverrides), (["moe_args"], MoEOverrides)):
                for item in fields(cls):
                    if item.name == "moe_args":
                        continue
                    value = geometry[aliases.get(item.name, item.name)]
                    if isinstance(value, tuple):
                        value = list(value)
                    model_fields.append(_field(["model_overrides", *prefix, item.name], str(item.type), value, "model"))
        else:
            limitations.append(
                "Flat model_overrides are supported only for DeepSeek-V4; this preset retains native geometry."
            )
        models.append(
            {
                "id": name,
                "label": name,
                "identity": identity,
                "assignments": [*identity, {"path": ["num_mtp_modules"], "value": simulation.num_mtp_modules}],
                "fields": model_fields,
                "limitations": limitations,
            }
        )
    return {
        "model_reset_paths": [["config"], ["model_overrides"], ["num_mtp_modules"]],
        "fields": common,
        "models": models,
    }
