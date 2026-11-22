# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA training, selective freezing and PEFT export on real NPUs."""

import json
import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor
from torchtitan.components.optim import AdamW, DistMuon
from torchtitan.config import override
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms

from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.deepseek_v4.lora import deepseek_v4_lora_transform, is_lora_parameter
from torchtitan_npu.models.deepseek_v4.peft import DeepSeekV4PEFTCheckpointManager
from torchtitan_npu.override.common.rope import WorkaroundComplexRoPE, workaround
from torchtitan_npu.patches.torchtitan.models.common.rope import SplitComplexRoPEConfig
from torchtitan_npu_recipes.models.deepseek_v4 import _dsv4_muon_profile
from torchtitan_npu_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel

logger = logging.getLogger(__name__)


@override(target=SplitComplexRoPEConfig)
def lora_rope_workaround(cfg: SplitComplexRoPEConfig) -> WorkaroundComplexRoPE.Config:
    # LoRA wraps the config in a frozen subclass, so the exact-type override skips it.
    # This RoPE module has buffers only.
    return workaround(cfg)


def _cpu_tensor(tensor):
    tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    # Training wrappers own ordinary floating-point master data.
    if hasattr(tensor, "to_tensor"):
        tensor = tensor.to_tensor()
    return tensor.detach().cpu().clone()


class _LoRAAssertionsTrainer(TrainerEx):
    @dataclass(kw_only=True, slots=True)
    class Config(TrainerEx.Config):
        pass

    def __init__(self, config):
        if config.checkpointer.load_step > 0:
            # Compare with the uninterrupted artifacts without overwriting them.
            config.checkpointer.load_only = True
        super().__init__(config)

    def train_step(self, data_iterator):
        engine = self.engine
        model = engine.model_parts[0]
        self._check_quantized_parameters(model)
        frozen = {name: _cpu_tensor(p) for name, p in model.named_parameters() if not p.requires_grad}
        biases = {name: _cpu_tensor(b) for name, b in model.named_buffers() if name.endswith("expert_bias_E")}
        adapter = model.layers["0"].attention.wq_a.lora_b.weight
        before = _cpu_tensor(adapter)
        adapter_a = model.layers["0"].attention.wq_a.lora_a.weight
        before_a = _cpu_tensor(adapter_a)
        expert_b = model.layers["0"].feed_forward.routed_experts.w13.lora_b.weight
        before_expert_b = _cpu_tensor(expert_b.full_tensor() if isinstance(expert_b, DTensor) else expert_b)
        assert biases
        super().train_step(data_iterator)
        for name, expected in frozen.items():
            parameter = model.get_parameter(name)
            assert parameter.grad is None, name
            torch.testing.assert_close(_cpu_tensor(parameter), expected, rtol=0, atol=0)
        for name, expected in biases.items():
            torch.testing.assert_close(_cpu_tensor(model.get_buffer(name)), expected, rtol=0, atol=0)
        assert not torch.equal(_cpu_tensor(adapter), before)
        current = _cpu_tensor(expert_b.full_tensor() if isinstance(expert_b, DTensor) else expert_b)
        assert current.shape[1] == 2
        for gate_up in range(2):
            assert not torch.equal(current[:, gate_up], before_expert_b[:, gate_up])
        if adapter_a.requires_grad and torch.count_nonzero(before):
            assert not torch.equal(_cpu_tensor(adapter_a), before_a)

    def train(self):
        super().train()
        engine = self.engine
        checkpointer = engine.checkpointer
        state = {
            name: (value.full_tensor() if isinstance(value, DTensor) else value).detach().cpu()
            for name, value in engine.model_parts[0].state_dict().items()
            if is_lora_parameter(name)
        }
        expected = checkpointer.sd_adapter.to_peft(state)
        checkpoint = Path(checkpointer.folder) / f"step-{engine.num_completed_steps}"
        exported = load_file(checkpoint / "adapter_model.safetensors")
        assert exported.keys() == expected.keys()
        for name, value in expected.items():
            torch.testing.assert_close(exported[name], value.to(checkpointer.export_dtype), rtol=0, atol=0)
        with (checkpoint / "adapter_config.json").open(encoding="utf-8") as source:
            metadata = json.load(source)
        options = self.config.model.layers[0].attention.wq_a
        assert metadata["r"] == options.rank
        assert metadata["lora_alpha"] == options.alpha
        target_parameters = set(metadata["target_parameters"])
        exported_targets = set()
        for name in exported:
            target = name.removeprefix("base_model.model.").rsplit(".lora_", 1)[0]
            # PEFT nests the gate/up parameter adapter inside the down adapter.
            if target.endswith(".mlp.experts.base_layer"):
                target = target.removesuffix(".base_layer") + ".gate_up_proj"
            elif target.endswith(".mlp.experts"):
                target += ".down_proj"
            else:
                target += ".weight"
            exported_targets.add(target)
        assert target_parameters == exported_targets
        logger.info(
            "PEFT export verified: %d adapter tensors, rank=%d, alpha=%s",
            len(exported),
            metadata["r"],
            metadata["lora_alpha"],
        )

    def _check_quantized_parameters(self, model):
        if not self.config.extension.quantization.enable_quantized_training:
            return
        from torchao_npu.wrapper_tensors import BaseTrainingWeightWrapperTensor

        parameters = {
            name: value.to_local() if isinstance(value, DTensor) else value for name, value in model.named_parameters()
        }
        adapters = [value for name, value in parameters.items() if is_lora_parameter(name)]
        wrapped = [
            value
            for name, value in parameters.items()
            if not is_lora_parameter(name) and isinstance(value, BaseTrainingWeightWrapperTensor)
        ]
        assert wrapped and all(not value.requires_grad for value in wrapped)
        assert adapters and all(
            value.is_floating_point() and value.requires_grad and not isinstance(value, BaseTrainingWeightWrapperTensor)
            for value in adapters
        )


class LoRATrainer(_LoRAAssertionsTrainer):
    def train_step(self, data_iterator):
        engine = self.engine
        checkpoint = self.config.checkpointer
        step = engine.num_completed_steps + 1
        if checkpoint.save_training_state and checkpoint.load_step > 0 and step == checkpoint.load_step + 1:
            expected = torch.load(
                self._optimizer_snapshot_path(checkpoint.load_step),
                map_location="cpu",
                weights_only=True,
            )
            torch.testing.assert_close(self._optimizer_snapshot(), expected, rtol=0, atol=0)
        if any(isinstance(optimizer, DistMuon) for optimizer in engine.optim.optimizers.optimizers):
            expert_b = engine.model_parts[0].layers["0"].feed_forward.routed_experts.w13.lora_b.weight
            muon_parameters = {
                id(p)
                for optimizer in engine.optim.optimizers.optimizers
                if isinstance(optimizer, DistMuon)
                for group in optimizer.param_groups
                for p in group["params"]
            }
            assert id(expert_b) in muon_parameters
        super().train_step(data_iterator)

        if checkpoint.save_training_state and not checkpoint.load_only and step % checkpoint.interval == 0:
            state = self._optimizer_snapshot()
            state_fields = (
                ("momentum_buffer", "exp_avg")
                if any(isinstance(optimizer, DistMuon) for optimizer in engine.optim.optimizers.optimizers)
                else ("exp_avg",)
            )
            for field in state_fields:
                values = [value for name, value in state.items() if field in name]
                assert values and any(torch.count_nonzero(value) for value in values), field
            torch.save(state, self._optimizer_snapshot_path(step))

    def _optimizer_snapshot(self):
        return {
            name: _cpu_tensor(value)
            for name, value in self.engine.optim.optimizers.state_dict().items()
            if isinstance(value, torch.Tensor)
        }

    def _optimizer_snapshot_path(self, step):
        return Path(self.config.dump_folder) / f"optimizer_state_step{step}_rank{dist.get_rank()}.pt"


class V4LoRATrainer(_LoRAAssertionsTrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(_LoRAAssertionsTrainer.Config):
        expected_optimizer: Literal["Muon", "AdamW"]

    def __init__(self, config):
        super().__init__(config)
        self._check_optimizer()

    def _check_optimizer(self):
        expected = self.config.expected_optimizer
        engine = self.engine
        optimizers = engine.optim.optimizers.optimizers
        muon = [optimizer for optimizer in optimizers if isinstance(optimizer, DistMuon)]
        assert bool(muon) == (expected == "Muon"), f"Expected {expected}, got {optimizers}"
        assert any(isinstance(optimizer, AdamW) for optimizer in optimizers)
        assert all(isinstance(optimizer, (DistMuon, AdamW)) for optimizer in optimizers)
        if muon:
            owned = set()
            for optimizer in muon:
                for group in optimizer.param_groups:
                    owned.update(map(id, group["params"]))
            experts = {
                name: id(p)
                for name, p in engine.model_parts[0].named_parameters()
                if p.requires_grad and ".routed_experts." in name and is_lora_parameter(name)
            }
            assert experts
            missing = [name for name, identity in experts.items() if identity not in owned]
            assert not missing, f"DistMuon does not own routed expert adapters: {missing}"
        logger.info("LoRA optimizer verified: %s", expected)

    def train_step(self, data_iterator):
        engine = self.engine
        checkpoint = self.config.checkpointer
        step = engine.num_completed_steps + 1
        if checkpoint.save_training_state and checkpoint.load_step > 0 and step == checkpoint.load_step + 1:
            self._verify_training_snapshot(checkpoint.load_step, self._training_snapshot())
        super().train_step(data_iterator)

        if checkpoint.save_training_state and step >= checkpoint.interval:
            snapshot = self._training_snapshot()
            path = self._training_snapshot_path(step)
            if checkpoint.load_step > 0:
                self._verify_training_snapshot(step, snapshot)
            else:
                if step % checkpoint.interval == 0:
                    for field in ("momentum_buffer", "exp_avg"):
                        values = [value for name, value in snapshot["optimizer"].items() if field in name]
                        assert values and any(torch.count_nonzero(value) for value in values), field
                # Each rank owns its local shards; only the uninterrupted phase writes.
                with path.open("xb") as stream:
                    torch.save(snapshot, stream)

    def _training_snapshot(self):
        optimizer = self.engine.optim.optimizers.state_dict()
        adapters = {
            name: _cpu_tensor(value)
            for name, value in self.engine.model_parts[0].named_parameters()
            if is_lora_parameter(name)
        }
        assert adapters and optimizer
        return {
            "adapters": adapters,
            "optimizer": {
                name: _cpu_tensor(value) if isinstance(value, torch.Tensor) else value
                for name, value in optimizer.items()
            },
            "scheduler": self.engine.optim.lr_schedulers.state_dict(),
            "learning_rates": self.engine.optim.lr_schedulers.get_host_lrs_per_scheduler(),
        }

    def _training_snapshot_path(self, step):
        return Path(self.config.dump_folder) / f"training_state_step{step}_rank{dist.get_rank()}.pt"

    def _verify_training_snapshot(self, step, snapshot):
        expected = torch.load(self._training_snapshot_path(step), map_location="cpu", weights_only=True)
        torch.testing.assert_close(snapshot["adapters"], expected["adapters"], rtol=0, atol=0)
        assert snapshot["scheduler"] == expected["scheduler"]
        assert snapshot["learning_rates"] == expected["learning_rates"]
        optimizer = snapshot["optimizer"]
        baseline = expected["optimizer"]
        assert optimizer.keys() == baseline.keys()
        for name, value in optimizer.items():
            reference = baseline[name]
            assert type(value) is type(reference), name
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(
                    value, reference, rtol=0, atol=0, msg=lambda detail, name=name: f"{name}: {detail}"
                )
            else:
                assert value == reference, name
        logger.info("Exact LoRA training state verified: step=%d, rank=%d", step, dist.get_rank())


def _lora_training(expected_optimizer: Literal["Muon", "AdamW"], *, resume=False):
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        expert_parallel_degree=2,
        fsdp_reshard_after_forward="always",
    )
    config = apply_transforms(deepseek_v4_debugmodel(parallelism=parallelism), [deepseek_v4_lora_transform()])
    if expected_optimizer == "Muon":
        profile = _dsv4_muon_profile(config.model)
        config.optim.optimizer.optimizers = [
            DistMuon.Config(
                pattern=profile.muon_pattern,
                compute_sharding_by_fqn=profile.compute_sharding_by_fqn,
                bucket_configs=profile.bucket_configs,
                lr=1e-5,
                ns_steps=10,
                adjust_lr_fn="match_rms_adamw",
            ),
            AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-8, fused=True, foreach=False),
        ]
    else:
        config.optim.optimizer.optimizers = [AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-6)]
    config.training.steps = 4 if resume else 2
    config.training.num_tokens_per_train_step = 4096
    config.metrics.enable_tensorboard = True
    config.metrics.save_tb_folder = f"tb_phase_{os.environ.get('TT_NPU_INTEGRATION_PHASE', '0')}"
    config.checkpointer = DeepSeekV4PEFTCheckpointManager.Config(
        interval=2 if resume else 10000,
        keep_latest_k=0,
        periodic_save_adapter_only=False,
        save_training_state=resume,
    )
    if resume:
        config.debug.seed = 42
        config.debug.deterministic = True
    return V4LoRATrainer.Config(
        expected_optimizer=expected_optimizer,
        **{field.name: getattr(config, field.name) for field in fields(config) if field.init},
    )


def deepseek_v4_lora_training():
    return _lora_training("Muon")


def deepseek_v4_lora_resume():
    return _lora_training("Muon", resume=True)


def deepseek_v4_lora_quantized_training():
    config = _lora_training("AdamW")
    config.extension.quantization.enable_quantized_training = True
    config.extension.quantization.recipe = "all_block_fp8"
    config.extension.quantization.enable_fsdp_prequantize = False
    return config
