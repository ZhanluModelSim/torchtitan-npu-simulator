# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native DeepSeek V4 model/loss/optimizer capture with explicit correctness gates."""

import logging
from collections import Counter

import torch

from torchtitan_simulator.capture.fsdp_residency import install_fsdp_residency_hooks
from torchtitan_simulator.capture.op_mapping import communication_primitive
from torchtitan_simulator.compat.checkpoint import logical_named_parameters
from torchtitan_simulator.compat.distributed import logical_world
from torchtitan_simulator.compat.routing import install_balanced_dispatch
from torchtitan_simulator.config import build_meta_model, prepare_config
from torchtitan_simulator.meta_env import compute_block
from torchtitan_simulator.metadata import to_meta_metadata
from torchtitan_simulator.runtime import meta_runtime
from torchtitan_simulator.step import run_simulation_step


def _gradient_manifest(model):
    manifest = {}
    missing = []
    for name, parameter in logical_named_parameters(model):
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        if gradient.shape != parameter.shape or gradient.dtype != parameter.dtype:
            raise RuntimeError(f"Gradient metadata differs from parameter {name}")
        if getattr(gradient, "placements", ()) != getattr(parameter, "placements", ()):
            raise RuntimeError(f"Gradient placement differs from parameter {name}")
        gradient_mesh = getattr(gradient, "device_mesh", None)
        parameter_mesh = getattr(parameter, "device_mesh", None)
        # Native reductions can return equivalent mesh views. Validate
        # topology and actual process groups, rather than Python identity.
        if gradient_mesh is not parameter_mesh and (
            gradient_mesh != parameter_mesh
            or gradient_mesh is None
            or parameter_mesh is None
            or gradient_mesh.mesh_dim_names != parameter_mesh.mesh_dim_names
            or any(
                gradient_mesh.get_group(axis) is not parameter_mesh.get_group(axis)
                for axis in range(parameter_mesh.ndim)
            )
        ):
            raise RuntimeError(f"Gradient mesh topology/groups differ from parameter {name}")
        local = getattr(gradient, "_local_tensor", gradient)
        parameter_local = getattr(parameter, "_local_tensor", parameter)
        if local.shape != parameter_local.shape:
            raise RuntimeError(f"Local gradient shape differs from parameter {name}")
        manifest[name] = {
            "shape": tuple(gradient.shape),
            "local_shape": tuple(local.shape),
            "dtype": str(gradient.dtype),
            "placements": str(getattr(gradient, "placements", ())),
        }
    if missing:
        raise RuntimeError(f"Expected parameter gradients are missing: {missing}")
    return manifest


def _buffer_signature(model):
    signature = {}
    for name, value in model.named_buffers():
        tensor = getattr(value, "_local_tensor", value)
        storage = tensor.untyped_storage()
        signature[name] = (tuple(tensor.shape), tensor.dtype, tensor.device, int(storage._cdata), storage.nbytes())
    return signature


def simulate(simulation):
    prepared = prepare_config(simulation)
    if prepared.trainer.model_spec.name == "deepseek_v4" and prepared.trainer.training.dtype != "float32":
        raise ValueError(
            "DeepSeek V4 capture requires training.dtype=float32: native FSDP expects uniform original "
            "parameter dtype, but base bfloat16 mixes with explicit float32 attn_sink/ape parameters. "
            "Use mixed_precision_param=bfloat16 for native BF16 compute, or memory_parameter_storage_dtype "
            "for a storage-only estimate. Build-only can still inspect the native bfloat16 model."
        )
    if prepared.trainer.model_spec.name == "kimi_k3" and prepared.trainer.training.dtype != "float32":
        raise ValueError(
            "Kimi K3 capture requires training.dtype=float32: native FSDP expects uniform original parameter dtype, "
            "but the model defines A_log/dt_bias in float32. Use mixed_precision_param=bfloat16 for BF16 compute, "
            "or memory_parameter_storage_dtype for a storage-only estimate. Build-only can inspect bfloat16."
        )
    simulation = prepared.simulation
    config = prepared.trainer
    with meta_runtime() as runtime:
        runtime.configure_model(prepared)
        with logical_world(prepared, runtime, rank=simulation.logical_rank) as dims:
            model = build_meta_model(prepared)
            quantized_parameters = []
            if simulation.enable_quantized_training:
                from torchao_npu.wrapper_tensors.mx_wrapper_tensor import MXTrainingWeightWrapperTensor

                quantized_parameters = [
                    name
                    for name, parameter in logical_named_parameters(model)
                    if isinstance(parameter, MXTrainingWeightWrapperTensor)
                ]
                if not quantized_parameters:
                    raise RuntimeError("Native all_mxfp8 recipe selected no MX training weights")
            original_shapes = {name: tuple(parameter.shape) for name, parameter in logical_named_parameters(model)}
            model = config.model_spec.parallelize_fn(
                model,
                parallel_dims=dims,
                training=config.training,
                parallelism=config.parallelism,
                compile_config=config.compile,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
            )
            parameter_layouts = {}
            for name, parameter in logical_named_parameters(model):
                local = getattr(parameter, "_local_tensor", parameter)
                mesh = getattr(parameter, "device_mesh", None)
                parameter_layouts[name] = {
                    "original_shape": original_shapes[name],
                    "local_shape": tuple(local.shape),
                    "dtype": str(parameter.dtype),
                    "storage_bytes": local.numel() * local.element_size(),
                    "mesh_dim_names": tuple(mesh.mesh_dim_names) if mesh is not None else (),
                    "mesh_shape": tuple(mesh.shape) if mesh is not None else (),
                    "placements": str(getattr(parameter, "placements", ())),
                }
            # torch_npu may install its FSDP layer on first fully_shard. Observe
            # the final layer; restoring this scope also removes the observer.
            install_fsdp_residency_hooks(runtime.patches)
            install_balanced_dispatch(runtime.patches, model)
            loss_fn = config.loss.build(compile_config=config.compile)
            loss_fn.set_lm_head(model.lm_head)
            model._skip_lm_head = True
            optimizers = config.optimizer.build(model_parts=[model])
            if config.model_spec.post_optimizer_build_fn is not None:
                config.model_spec.post_optimizer_build_fn(optimizers, [model], dims)
            scheduler = config.lr_scheduler.build(optimizers=optimizers, training_steps=config.training.steps)

            seq_len = simulation.seq_len
            batch_size = config.training.local_batch_size
            batch_degree = dims.dp_replicate * dims.dp_shard
            global_batch_size = config.training.global_batch_size
            if global_batch_size < 0:
                global_batch_size = batch_size * batch_degree
            if global_batch_size < 1 or global_batch_size % (batch_size * batch_degree):
                raise ValueError("Global batch size must be a positive multiple of local batch size * DP degree")
            gradient_accumulation = global_batch_size // (batch_size * batch_degree)
            source = torch.arange(batch_size * (seq_len + 1)).reshape(batch_size, seq_len + 1)
            source %= config.model_spec.model.vocab_size
            tokens, labels = source[:, :-1].contiguous(), source[:, 1:].contiguous()
            valid_tokens = int((labels != -100).sum().item())
            positions = torch.arange(seq_len).expand(batch_size, -1).contiguous()
            cp_mesh = dims.get_optional_mesh("cp")
            if cp_mesh is not None:
                from torch.distributed.device_mesh import DeviceMesh

                # Plan construction is pure host math. Retain the same group,
                # coordinates and native CP planner while keeping its scalar
                # geometry readable before transferring metadata to Meta.
                cp_mesh = DeviceMesh.from_group(
                    cp_mesh.get_group(), "cpu", mesh=cp_mesh.mesh, mesh_dim_names=cp_mesh.mesh_dim_names
                )
            tokens, labels, kwargs = model.build_attention_masks(
                tokens,
                labels,
                {"positions": positions},
                cp_mesh=cp_mesh,
                load_balancer_type=config.parallelism.context_parallel_load_balancer,
            )
            inputs = {"tokens": tokens.to("meta"), **to_meta_metadata(kwargs)}
            labels = labels.to("meta")
            gradient_manifest = {}
            from torchtitan.distributed.utils import clip_grad_norm_, get_spmd_context

            train_context = get_spmd_context(parallel_dims=dims, spmd_typechecking=config.debug.spmd_typechecking)

            def forward_backward(*, input_dict, labels, global_valid_tokens):
                with train_context():
                    with compute_block("F"):
                        predictions = model(**input_dict)
                        loss, _ = loss_fn(predictions, labels, global_valid_tokens, positions=input_dict["positions"])
                    with compute_block("B"):
                        loss.backward()
                gradient_manifest.update(_gradient_manifest(model))
                return loss

            buffers_before = _buffer_signature(model)
            workload = run_simulation_step(
                model_parts=[model],
                parallel_dims=dims,
                forward_backward_step=forward_backward,
                input_dict=inputs,
                labels=labels,
                optimizer_step=optimizers.step,
                optimizer_zero_grad=optimizers.zero_grad,
                lr_scheduler_step=scheduler.step,
                clip_grad_norm=lambda: clip_grad_norm_(
                    model.parameters(),
                    config.training.max_norm,
                    foreach=False,
                    pp_mesh=None,
                    ep_enabled=dims.ep_enabled,
                ),
                local_batch_size=batch_size,
                seq_len=seq_len,
                global_valid_tokens=float(valid_tokens * batch_degree * gradient_accumulation),
                gradient_accumulation=gradient_accumulation,
                memory_parameter_storage_dtype=simulation.memory_parameter_storage_dtype,
                memory_device_type="meta",
                memory_offload_ac_saved_tensors=simulation.memory_offload_ac_saved_tensors,
                scoped_backward_phases=True,
                rank=simulation.logical_rank,
                capture_process_rank=0,
            )
            if _buffer_signature(model) != buffers_before:
                raise RuntimeError(
                    "Registered buffers changed storage/layout during capture; explicit residency modeling required"
                )
            annotations = workload.iteration.schedule.annotations
            if simulation.selective_ac_save_ops and {"comm", "full"} & set(simulation.selective_ac_save_ops):
                memory = annotations["memory_plan"]
                cached_storages = {
                    slot.storage_key for slot in memory.autograd_saved_tensors if slot.source == "selective_cache"
                }
                matched = any(
                    communication_primitive(event.raw_op_type) is not None
                    and any(ref.storage_key in cached_storages for ref in event.outputs)
                    for event in memory.raw_events
                )
                if not matched:
                    message = (
                        "Native comm SAC save-set matched no dispatcher-visible collective outputs; "
                        "eager SPMD exchanges follow native recomputation behavior."
                    )
                    logging.getLogger(__name__).warning(message)
                    prepared.adaptations.append(message)
            annotations["batch_configuration"] = {
                "global_batch_size": global_batch_size,
                "local_batch_size": batch_size,
                "data_parallel_degree": batch_degree,
                "gradient_accumulation": gradient_accumulation,
                "captured_microbatches": 1,
                "global_valid_tokens": valid_tokens * batch_degree * gradient_accumulation,
                "memory_scope": "one microbatch and one optimizer update; no accumulated multi-MB replay",
            }
            annotations["capture_coordinates"] = {
                "logical_rank": simulation.logical_rank,
                "logical_world_size": simulation.world_size,
                "capture_workers": 1,
                "capture_process_rank": 0,
                "target_device_type": simulation.target_npu_device_type,
                "pipeline_stage": 0,
                "compute_device": "meta",
                "metadata_planning_device": "cpu",
            }
            annotations["parameter_layouts"] = parameter_layouts
            from dataclasses import asdict

            annotations["model_configuration"] = {
                "config_factory": prepared.config_factory,
                "preset": simulation.config,
                "overrides": asdict(simulation.model_overrides) if simulation.model_overrides is not None else None,
                "dim": config.model_spec.model.dim,
                "vocab_size": config.model_spec.model.vocab_size,
                "num_layers": config.model_spec.model.n_layers,
                "num_mtp_layers": len(getattr(config.model_spec.model, "mtp_layers", ())),
            }
            annotations["quantization_configuration"] = {
                "enabled": simulation.enable_quantized_training,
                "recipe": simulation.quantization_recipe if simulation.enable_quantized_training else None,
                "selection_source": "native TorchAO-NPU recipe; simulator FQN overrides unavailable",
                "quantized_parameters": quantized_parameters,
            }
            annotations["gradient_validation"] = gradient_manifest
            annotations["runtime_capabilities"] = dict(runtime.capabilities)
            annotations["simulation_assumptions"] = prepared.adaptations
            annotations["input_source"] = "synthetic single-document positions and valid next-token labels"
            annotations["loss_execution"] = (
                "F owns native model/loss forward including interleaved chunked-head backward; "
                "B owns decoder backward. Per-op phase and execution_kind retain actual autograd execution."
            )
            annotations["cost_unknown_counts"] = dict(
                Counter(
                    node.annotations["raw_op_type"]
                    for step in workload.step_templates.values()
                    for node in step.nodes.values()
                    if node.annotations.get("cost_unknown")
                )
            )
    from torchtitan_simulator.validation import validate_workload

    workload.iteration.schedule.annotations["capture_validation"] = validate_workload(workload)
    workload.iteration.schedule.annotations["precision_configuration"] = {
        name: getattr(prepared.trainer.training, name)
        for name in ("dtype", "mixed_precision_param", "mixed_precision_reduce")
    }
    return workload


def export_result(workload, simulation):
    """Write memory artifacts only; an empty format list has no filesystem effects."""
    from torchtitan_simulator.memory.export import export_memory_plan

    unknown = set(simulation.output_formats) - {"mem"}
    if unknown:
        raise ValueError(f"Unsupported output formats: {sorted(unknown)}; only mem is supported")
    if "mem" not in simulation.output_formats:
        return
    memory = workload.iteration.schedule.annotations.get("memory_plan")
    if memory is None:
        raise ValueError("Cannot export mem without a captured memory plan")
    export_memory_plan(memory, simulation.output_dir)
