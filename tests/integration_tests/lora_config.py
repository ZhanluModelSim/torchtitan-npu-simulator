# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA training, selective freezing and PEFT export on real NPUs."""

import json
import logging
from dataclasses import dataclass, fields, replace
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor
from torchtitan.components.optim import DistMuon
from torchtitan.config import override
from torchtitan.config.transform import apply_transforms

from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.deepseek_v4.lora import deepseek_v4_lora_transform, is_lora_parameter
from torchtitan_npu.models.deepseek_v4.muon import _dsv4_muon_profile
from torchtitan_npu.models.deepseek_v4.peft import DeepSeekV4PEFTCheckpointManager
from torchtitan_npu.override.common.rope import WorkaroundComplexRoPE, workaround
from torchtitan_npu.patches.torchtitan.models.common.rope import SplitComplexRoPEConfig
from torchtitan_npu_recipes.models.deepseek_v4 import deepseek_v4_debugmodel

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


class LoRATrainer(TrainerEx):
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
        checkpoint = self.config.checkpointer
        step = engine.num_completed_steps + 1
        if checkpoint.save_training_state and checkpoint.load_step > 0 and step == checkpoint.load_step + 1:
            expected = torch.load(
                self._optimizer_snapshot_path(checkpoint.load_step),
                map_location="cpu",
                weights_only=True,
            )
            torch.testing.assert_close(self._optimizer_snapshot(), expected, rtol=0, atol=0)
        model = engine.model_parts[0]
        self._check_quantized_parameters(model)
        frozen = {name: _cpu_tensor(p) for name, p in model.named_parameters() if not p.requires_grad}
        biases = {name: _cpu_tensor(b) for name, b in model.named_buffers() if name.endswith("expert_bias_E")}
        adapter = model.layers["0"].attention.wq_a.lora_b.weight
        before = _cpu_tensor(adapter)
        adapter_a = model.layers["0"].attention.wq_a.lora_a.weight
        before_a = _cpu_tensor(adapter_a)
        expert_b = model.layers["0"].moe.routed_experts.w13.lora_b.weight
        before_expert_b = _cpu_tensor(expert_b.full_tensor() if isinstance(expert_b, DTensor) else expert_b)
        if self.config.optim.optimizer.name == "Muon":
            muon_parameters = {
                id(p)
                for optimizer in engine.optim.optimizers.optimizers
                if isinstance(optimizer, DistMuon)
                for group in optimizer.param_groups
                for p in group["params"]
            }
            assert id(expert_b) in muon_parameters
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

        if checkpoint.save_training_state and not checkpoint.load_only and step % checkpoint.interval == 0:
            state = self._optimizer_snapshot()
            state_fields = (
                ("momentum_buffer", "exp_avg")
                if self.config.optim.optimizer.name == "Muon"
                else ("exp_avg",)
            )
            for field in state_fields:
                values = [value for name, value in state.items() if field in name]
                assert values and any(torch.count_nonzero(value) for value in values), field
            torch.save(state, self._optimizer_snapshot_path(step))

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
        checkpoint = Path(checkpointer._create_checkpoint_id(engine.num_completed_steps))
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

    def _optimizer_snapshot(self):
        return {
            name: _cpu_tensor(value)
            for name, value in self.engine.optim.optimizers.state_dict().items()
            if isinstance(value, torch.Tensor)
        }

    def _optimizer_snapshot_path(self, step):
        return Path(self.config.dump_folder) / f"optimizer_state_step{step}_rank{dist.get_rank()}.pt"

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


def deepseek_v4_lora_training():
    config = apply_transforms(deepseek_v4_debugmodel(), [deepseek_v4_lora_transform()])
    config.optim.optimizer = replace(config.optim.optimizer, _muon_profile=_dsv4_muon_profile(config.model))
    config.checkpointer = DeepSeekV4PEFTCheckpointManager.Config(interval=10000, periodic_save_adapter_only=False)
    return LoRATrainer.Config(
        **{field.name: getattr(config, field.name) for field in fields(config) if field.init},
    )


def deepseek_v4_lora_quantized_training():
    config = deepseek_v4_lora_training()
    config.extension.quantization.enable_quantized_training = True
    config.extension.quantization.recipe = "all_block_fp8"
    config.extension.quantization.enable_fsdp_prequantize = False
    return config
