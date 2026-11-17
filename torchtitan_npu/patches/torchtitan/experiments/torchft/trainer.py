# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4534
# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4575
# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4663
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport TorchFT config overrides, chunked-loss setup, and replica loss averaging.

Remove this module after the TorchTitan dependency includes the PR.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import torch
import torchtitan.experiments.torchft.trainer
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.components.loss import IGNORE_INDEX, ChunkedLossWrapper
from torchtitan.config import TORCH_DTYPE_MAP, apply_overrides
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.cudagraph import wrap_with_cuda_graph
from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer
from torchtitan.tools import utils
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torchtitan.experiments.torchft.trainer import FaultTolerantTrainer
    from torchtitan.protocols import BaseModel


@record
def patched_init(self, config: FaultTolerantTrainer.Config):
    torch._C._log_api_usage_once("torchtitan.train")

    self.config = config
    assert config.model_spec is not None, "model_spec must be set before creating Trainer"
    model_spec = config.model_spec

    device_module, device_type = utils.device_module, utils.device_type
    # pyrefly: ignore [read-only]
    self.device = utils.get_local_device()
    # Device has to be set before creating TorchFT manager.
    device_module.set_device(self.device)

    # init distributed and build meshes (FT override handles ft_manager creation)
    self.parallel_dims = parallel_dims = self.init_distributed()

    # Logging needs to happen after distributed initialized
    config.maybe_log()

    if parallel_dims.dp_enabled:
        batch_mesh = parallel_dims.get_mesh("batch")
        batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
    else:
        batch_degree, batch_rank = 1, 0

    # FT addition: adjust dp info via ft_manager
    batch_degree, batch_rank = self.ft_manager.get_dp_info(batch_degree, batch_rank)

    # take control of garbage collection to avoid stragglers
    self.gc_handler = utils.GarbageCollection(gc_freq=config.training.gc_freq, debug=config.training.gc_debug)

    # Set random seed, and maybe enable deterministic mode
    # (mainly for debugging, expect perf loss).
    dist_utils.set_determinism(
        parallel_dims,
        self.device,
        config.debug,
        distinct_seed_mesh_dims=["pp"],
    )

    # build tokenizer
    self.tokenizer = (
        config.tokenizer.build(tokenizer_path=config.hf_assets_path) if config.tokenizer is not None else None
    )

    # build dataloader
    dataloader_batch_size = (
        config.parallelism.pipeline_parallel_microbatch_size
        if parallel_dims.pp_enabled
        else config.training.local_batch_size
    )
    self.dataloader = config.dataloader.build(
        dp_world_size=batch_degree,
        dp_rank=batch_rank,
        tokenizer=self.tokenizer,
        seq_len=config.training.seq_len,
        local_batch_size=dataloader_batch_size,
    )

    # build model (using meta init)
    model_config = model_spec.model
    # set the model args from training job configs
    model_config.update_from_config(
        config=config,
    )
    self.model_config = model_config

    # Apply overrides after model config updates, before building the model.
    if config.override.imports:
        apply_overrides(config.override, config)
    # Overrides can change fields checked during config construction.
    config.__post_init__()

    logger.info(
        f"Building {model_spec.name} {model_spec.flavor} "
        f"with {json.dumps(model_config.to_dict(), indent=2, ensure_ascii=False)}"
    )
    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
    ):
        model = model_config.build()

    # Verify all submodules satisfy the Module protocol
    model.verify_module_protocol()

    # metrics logging (FT addition: ft_enable, ft_replica_id)
    self.metrics_processor = config.metrics.build(
        parallel_dims=parallel_dims,
        dump_folder=config.dump_folder,
        pp_schedule=config.parallelism.pipeline_parallel_schedule,
        ft_enable=config.fault_tolerance.enable,
        ft_replica_id=config.fault_tolerance.replica_id,
        config_dict=config.to_dict(),
    )
    color = self.metrics_processor.color

    # calculate model size and flops per token
    (
        model_param_count,
        self.metrics_processor.num_flops_per_token,
    ) = model_config.get_nparams_and_flops(model, config.training.seq_len)

    logger.info(
        f"{color.blue}Model {model_spec.name} {model_spec.flavor} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
    )

    # move sharded model to CPU/GPU and initialize weights via DTensor
    buffer_device: torch.device | None
    if config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
        buffer_device = None
    elif config.training.enable_cpu_offload:
        init_device = "cpu"
        buffer_device = torch.device(device_type)
    else:
        init_device = device_type
        buffer_device = None

    self.loss_fn = config.loss.build(
        compile_config=config.compile,
    )

    # verify batch sizes
    global_batch_size = config.training.global_batch_size
    if global_batch_size < 0:
        # This global batch size results in 1 gradient accumulation
        # step.
        global_batch_size = config.training.local_batch_size * batch_degree
    assert global_batch_size > 0
    assert global_batch_size % (config.training.local_batch_size * batch_degree) == 0, (
        f"global batch size must be multiple of local batch size times "
        f"data-parallel degree ({global_batch_size} "
        f"% ({config.training.local_batch_size} * {batch_degree}) != 0)"
    )

    # calculate gradient accumulation steps
    self.gradient_accumulation_steps = global_batch_size // (config.training.local_batch_size * batch_degree)
    assert self.gradient_accumulation_steps > 0
    self.num_pipeline_parallel_microbatches = (
        config.training.local_batch_size // config.parallelism.pipeline_parallel_microbatch_size
        if parallel_dims.pp_enabled
        else 1
    )

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        from torchtitan.components.metrics import ensure_pp_loss_visible

        if not model_spec.pipelining_fn:
            raise RuntimeError(f"Pipeline Parallel is enabled but {model_spec.name} does not support pipelining")

        # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
        (
            self.pp_schedule,
            self.model_parts,
            self.pp_has_first_stage,
            self.pp_has_last_stage,
        ) = model_spec.pipelining_fn(
            model,
            parallel_dims=parallel_dims,
            training=config.training,
            parallelism=config.parallelism,
            compile_config=config.compile,
            ac_config=config.activation_checkpoint,
            dump_folder=config.dump_folder,
            device=self.device,
            model_config=model_config,
            parallelize_fn=model_spec.parallelize_fn,
            loss_fn=self.loss_fn,
        )
        # when PP is enabled, `model` obj is no longer used after this point,
        # model_parts is used instead
        del model

        for m in self.model_parts:
            m.to_empty(device=init_device)
            with torch.no_grad():
                cast("BaseModel", m).init_states(buffer_device=buffer_device)
            m.train()

        # confirm that user will be able to view loss metrics on the console
        ensure_pp_loss_visible(
            parallel_dims=parallel_dims,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            color=color,
        )
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        model = model_spec.parallelize_fn(
            model,
            parallel_dims=parallel_dims,
            training=config.training,
            parallelism=config.parallelism,
            compile_config=config.compile,
            ac_config=config.activation_checkpoint,
            dump_folder=config.dump_folder,
        )

        model.to_empty(device=init_device)
        with torch.no_grad():
            cast("BaseModel", model).init_states(buffer_device=buffer_device)
        model.train()

        self.model_parts = [model]

    # Set lm_head reference for ChunkedLossWrapper after model construction.
    self._configure_chunked_loss()

    # FT addition: set all reduce hook
    self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = self.metrics_processor.device_memory_monitor
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    # FT addition: pass ft_manager for TorchFTOptimizersContainer
    if isinstance(config.optimizer, TorchFTOptimizersContainer.Config):
        self.optimizers = config.optimizer.build(model_parts=self.model_parts, ft_manager=self.ft_manager)
    else:
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
    if model_spec.post_optimizer_build_fn is not None:
        model_spec.post_optimizer_build_fn(self.optimizers, self.model_parts, parallel_dims)
    self.lr_schedulers = config.lr_scheduler.build(
        optimizers=self.optimizers,
        training_steps=config.training.steps,
    )
    self.metrics_processor.optimizers = self.optimizers
    self.metrics_processor.model_parts = self.model_parts

    # Initialize trainer states that will be saved in checkpoint.
    # These attributes must be initialized before checkpoint loading.
    self.step = 0
    self.ntokens_seen = 0

    # FT addition: pass ft_manager to CheckpointManager
    self.checkpointer = config.checkpoint.build(
        dataloader=self.dataloader,
        model_parts=self.model_parts,
        optimizers=self.optimizers,
        lr_schedulers=self.lr_schedulers,
        states={"train_state": self},
        sd_adapter=(
            model_spec.state_dict_adapter(model_config, config.hf_assets_path)
            if model_spec.state_dict_adapter
            else None
        ),
        base_folder=config.dump_folder,
        ft_manager=self.ft_manager,
    )

    self.train_context = dist_utils.get_spmd_context(
        parallel_dims=parallel_dims,
    )
    self.fwd_bwd_fn = self._forward_backward_body
    if not config.training.disable_cuda_graphs:
        self.fwd_bwd_fn = wrap_with_cuda_graph(self.fwd_bwd_fn)

    # Build validator if validation is configured
    if config.validator.enable:
        pp_schedule, pp_has_first_stage, pp_has_last_stage = (
            (
                self.pp_schedule,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            )
            if parallel_dims.pp_enabled
            else (None, None, None)
        )

        self.validator = config.validator.build(
            parallelism=config.parallelism,
            job_config=config,
            dp_world_size=batch_degree,
            dp_rank=batch_rank,
            tokenizer=self.tokenizer,
            parallel_dims=parallel_dims,
            loss_fn=self.loss_fn,
            validation_context=self.train_context,
            metrics_processor=self.metrics_processor,
            seq_len=config.training.seq_len,
            local_batch_size=config.training.local_batch_size,
            pp_schedule=pp_schedule,
            pp_has_first_stage=pp_has_first_stage,
            pp_has_last_stage=pp_has_last_stage,
        )

    logger.info(
        "Trainer is initialized with "
        f"local batch size {config.training.local_batch_size}, "
        f"global batch size {global_batch_size}, "
        f"gradient accumulation steps {self.gradient_accumulation_steps}, "
        f"sequence length {config.training.seq_len}, "
        f"total steps {config.training.steps} "
        f"(warmup {config.lr_scheduler.warmup_steps})"
    )


def _configure_chunked_loss(self) -> None:
    # Non-PP: single model part always has lm_head.
    # PP: only the last stage has lm_head; non-last stages skip this.
    if isinstance(self.loss_fn, ChunkedLossWrapper):
        if self.parallel_dims.pp_enabled:
            if self.pp_has_last_stage:
                lm_head = self.model_parts[-1].lm_head
                assert lm_head is not None, "Last PP stage must have lm_head for ChunkedLossWrapper"
                self.loss_fn.set_lm_head(
                    lm_head  # pyrefly: ignore[bad-argument-type]
                )
                self.model_parts[-1]._skip_lm_head = True  # pyrefly: ignore[bad-argument-type]
        else:
            assert len(self.model_parts) == 1
            lm_head = self.model_parts[0].lm_head
            assert lm_head is not None, "Model must have lm_head for ChunkedLossWrapper"
            self.loss_fn.set_lm_head(lm_head)  # pyrefly: ignore[bad-argument-type]
            self.model_parts[0]._skip_lm_head = True  # pyrefly: ignore[bad-argument-type]


def patched_train_step(self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]):
    self.optimizers.zero_grad(set_to_none=self.config.training.disable_cuda_graphs)
    # Save the current step learning rate for logging
    lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]
    should_log = self.metrics_processor.should_log(self.step)

    # Keep these variables local to shorten the code as these are
    # the major variables that are used in the training loop.
    parallel_dims = self.parallel_dims
    # All groups form one optimizer step; each group feeds one fwd-bwd call.
    microbatch_groups: list[list[tuple[dict[str, torch.Tensor], torch.Tensor]]] = []
    local_valid_tokens = torch.tensor(0, dtype=torch.int64)
    for _ in range(self.gradient_accumulation_steps):
        microbatches = []
        for _ in range(self.num_pipeline_parallel_microbatches):
            input_dict, labels = next(data_iterator)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()
            microbatches.append((input_dict, labels))
        microbatch_groups.append(microbatches)

    # Keep the global token count on device so loss normalization does not
    # introduce a CPU synchronization in the training path.
    global_valid_tokens = local_valid_tokens.to(self.device)
    if parallel_dims.dp_enabled:
        batch_mesh = parallel_dims.get_mesh("batch")
        global_valid_tokens = dist_utils.dist_sum_tensor(global_valid_tokens, batch_mesh)

    accumulated_loss: torch.Tensor | None = None
    for microbatches in microbatch_groups:
        input_dict_mbs = []
        label_mbs = []
        for input_dict, labels in microbatches:
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.to(self.device)
            input_dict_mbs.append(input_dict)
            label_mbs.append(labels.to(self.device))

        if parallel_dims.pp_enabled:
            fwd_bwd_input_dict = input_dict_mbs
            fwd_bwd_labels = label_mbs
        else:
            assert len(input_dict_mbs) == len(label_mbs) == 1
            fwd_bwd_input_dict = input_dict_mbs[0]
            fwd_bwd_labels = label_mbs[0]

        loss = self.forward_backward_step(
            input_dict=fwd_bwd_input_dict,
            labels=fwd_bwd_labels,
            global_valid_tokens=global_valid_tokens,
        )
        if should_log:
            loss = loss.detach()
            if accumulated_loss is None:
                # Take ownership before the next replay overwrites the
                # graph-owned output. Later losses accumulate in place.
                accumulated_loss = loss.clone()
            else:
                accumulated_loss.add_(loss)

    grad_norm = dist_utils.clip_grad_norm_(
        [p for m in self.model_parts for p in m.parameters()],
        self.config.training.max_norm,
        foreach=True,
        pp_mesh=parallel_dims.get_optional_mesh("pp"),
        ep_enabled=parallel_dims.ep_enabled,
    )
    self.checkpointer.maybe_wait_for_staging()
    self.optimizers.step()
    self.lr_schedulers.step()

    # log metrics
    if not should_log:
        return

    assert accumulated_loss is not None

    if parallel_dims.dp_cp_enabled:
        # FT addition: use ft_manager.loss_sync_pg for extra process group
        ft_pg = self.ft_manager.loss_sync_pg
        loss_mesh = parallel_dims.get_optional_mesh("loss")

        # For global_avg_loss, we want the average loss across all ranks:
        # accumulated_loss = local_loss_sum / global_valid_tokens
        # global_avg_loss = sum(local_loss_sum) / global_valid_tokens
        #                 = sum(accumulated_loss)
        #
        # For global_max_loss, we want the max of local average losses across ranks:
        # local_avg_loss = local_loss_sum / local_valid_tokens
        #                = (accumulated_loss * global_valid_tokens) / local_valid_tokens
        # global_max_loss = max(local_avg_loss)
        local_avg_loss = accumulated_loss * global_valid_tokens / local_valid_tokens
        global_avg_loss, global_max_loss, global_ntokens_seen = (
            dist_utils.dist_sum(accumulated_loss, loss_mesh, ft_pg),
            dist_utils.dist_max(local_avg_loss, loss_mesh, ft_pg),
            dist_utils.dist_sum(
                torch.tensor(self.ntokens_seen, dtype=torch.int64, device=self.device),
                loss_mesh,
                ft_pg,
            ),
        )
        # ft_pg is None in semi-sync training.
        if ft_pg is not None:
            # Avoid artificial jumps in logged loss when replicas leave or rejoin.
            global_avg_loss /= ft_pg.size()
    else:
        global_avg_loss = global_max_loss = accumulated_loss.item()
        global_ntokens_seen = self.ntokens_seen

    extra_metrics = {
        "n_tokens_seen": global_ntokens_seen,
        "lr": lr,
    }
    self.metrics_processor.log(
        self.step,
        global_avg_loss,
        global_max_loss,
        grad_norm.item(),
        extra_metrics=extra_metrics,
    )


def apply() -> None:
    torchtitan.experiments.torchft.trainer.FaultTolerantTrainer.__init__ = patched_init
    torchtitan.experiments.torchft.trainer.FaultTolerantTrainer._configure_chunked_loss = _configure_chunked_loss
    torchtitan.experiments.torchft.trainer.FaultTolerantTrainer.train_step = patched_train_step


apply()
