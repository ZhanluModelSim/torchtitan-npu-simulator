# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read-only configuration metadata derived from supported native factories."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, fields
from importlib import import_module
from typing import cast

from torchtitan_simulator.config import (
    ACTIVATION_CHECKPOINT_MODES,
    PRECISION_CHOICES,
    PRESET_FACTORIES,
    TARGET_NPU_DEVICE_TYPES,
    ConfigurationCheckError,
    SimulationConfig,
    prepare_capture_config,
)
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
    "execution": ("optimizer_name", "muon_ns_steps", "enable_sla2_fusion", "target_npu_device_type"),
    "capture": ("selective_ac_save_ops", "enable_fsdp_allgather_fp8", "enable_ep_dispatch_fp8"),
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
    closed = {
        **PRECISION_CHOICES,
        "activation_checkpoint_mode": ACTIVATION_CHECKPOINT_MODES,
        "target_npu_device_type": TARGET_NPU_DEVICE_TYPES,
    }
    if len(path) == 1 and path[0] in closed:
        choices = [{"label": value, "value": value} for value in closed[path[0]]]
    elif kind == "boolean":
        choices = [{"label": str(value), "value": value} for value in (False, True)]
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
    return _describe_presets(tuple(PRESET_FACTORIES))


def _describe_presets(presets: Sequence[str]) -> dict:
    simulation = SimulationConfig()
    declarations = {item.name: item for item in fields(SimulationConfig)}
    common = [
        _field([name], str(declarations[name].type), getattr(simulation, name), group)
        for group, names in _GROUPS.items()
        for name in names
        if getattr(simulation, name) is not None
    ]
    models = []
    for name in presets:
        target = PRESET_FACTORIES[name]
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


def _rejection(check: str, reason: str, path: Sequence[str] | None = None) -> dict[str, object]:
    return {"valid": False, "reason": reason, "failed_check": check, "failed_path": list(path) if path else None}


def _check_draft(simulation: SimulationConfig) -> tuple[dict, dict[str, int] | None]:
    # Catalog discovery may only import producer-owned factories and their declared overrides.
    if simulation.config not in PRESET_FACTORIES:
        return _rejection(
            "registered_preset", "Training field discovery requires a registered preset", ["config"]
        ), None
    if simulation.override_imports:
        return _rejection(
            "override_imports",
            "Training field discovery does not import explicit override modules",
            ["override_imports"],
        ), None
    try:
        prepared = prepare_capture_config(simulation)
    except ConfigurationCheckError as exc:
        return _rejection(exc.check, str(exc), [exc.field_name] if exc.field_name else None), None
    except (ValueError, TypeError, AssertionError, NotImplementedError) as exc:
        return _rejection("native_configuration", str(exc) or type(exc).__name__), None
    return {"valid": True, "reason": None}, prepared.batch_configuration


def _draft_value(simulation: SimulationConfig, path: Sequence[str], value: object) -> SimulationConfig:
    draft = deepcopy(simulation)
    owner: object = draft
    if path[0] == "model_overrides":
        if draft.model_overrides is None:
            draft.model_overrides = ModelOverrides()
        owner = draft.model_overrides.moe_args if len(path) == 3 else draft.model_overrides
        if path[-1] == "compress_ratios" and isinstance(value, list):
            value = tuple(value)
    setattr(owner, path[-1], value)
    return draft


def describe_configuration_field(
    simulation: SimulationConfig, path: Sequence[str], *, candidates: Sequence[object] = ()
) -> dict[str, object]:
    """Probe one field at a time against the unchanged draft, without capture.

    Closed fields use their declared alternatives; callers may supply values for
    open ranges such as DP and batch size. A rejected current draft remains inspectable.
    """
    if simulation.config not in PRESET_FACTORIES:
        raise ValueError("Training field discovery requires a registered preset")
    catalog = _describe_presets([simulation.config])
    entries = [*catalog["fields"], *catalog["models"][0]["fields"]]
    if list(path) == ["config"]:
        declared = _field(["config"], "str", SimulationConfig().config, "model")
        declared["choices"] = [{"value": key, "label": key} for key in PRESET_FACTORIES]
    else:
        declared = next((item for item in entries if item["path"] == list(path)), None)
        if declared is None:
            raise ValueError(f"Unknown training catalog field for {simulation.config}: {'.'.join(path)}")
    current = simulation
    for key in path:
        current = getattr(current, key, None)
    validation, batch = _check_draft(simulation)
    values = list(candidates)
    if not values:
        values = [item["value"] for item in cast("list[dict[str, object]]", declared["choices"])]
        if not values:
            hint = current if current is not None else declared["default"]
            if current is None and batch is not None and len(path) == 1 and path[0] in batch:
                hint = batch[path[0]]
            if hint is not None and not (path[-1] == "global_batch_size" and hint == -1):
                values = [hint]
    choices = []
    for value in values:
        if type(value) is type(current) and value == current:
            status = validation
        else:
            kind = declared["kind"]
            if (
                (kind == "integer" and type(value) is not int)
                or (kind == "boolean" and type(value) is not bool)
                or (kind == "string" and not isinstance(value, str))
                or (kind == "float" and (isinstance(value, bool) or not isinstance(value, (float, int))))
            ):
                status = _rejection("candidate_type", f"{'.'.join(path)} expects {kind}", path)
            else:
                status, _ = _check_draft(_draft_value(simulation, path, value))
        choice = {"value": value, "label": str(value), **status}
        choice["available"] = choice.pop("valid")
        choices.append(choice)
    return {
        "field": {**declared, "choices": choices, "choice_scope": "current_request_preflight", "requested": current},
        "validation": validation,
        "batch_configuration": batch,
    }
