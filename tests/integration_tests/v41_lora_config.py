# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torchtitan.components.checkpoint_utils import canonical_fqn
from torchtitan.config import derive
from torchtitan.tools.logging import logger

from tests.integration_tests.lora_config import LoRATrainer, _cpu_tensor
from torchtitan_npu.models.deepseek_v4.lora import is_lora_parameter
from torchtitan_npu.models.deepseek_v4_1.config_registry import (
    DeepSeekV41Trainer,
    _make_lora_trainer_config,
    deepseek_v4_1_debugmodel_lora,
)
from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable
from torchtitan_npu.models.deepseek_v4_1.vision.data import build_image_token_layout


class V41LoRATrainer(LoRATrainer, DeepSeekV41Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV41Trainer.Config):
        check_resume_state: bool = False

    def __init__(self, config):
        if config.checkpoint.periodic_save_adapter_only:
            config.checkpoint.initial_load_path = str(Path(config.dump_folder) / "fixed_base")
        super().__init__(config)
        if config.checkpoint.periodic_save_adapter_only:
            base = Path(self.checkpointer.initial_load_path)
            if not (base / ".metadata").exists():
                # All ranks save the initialized, sharded base once; resumed runs reuse it.
                dcp.save(
                    {name: value for name, value in self.model_parts[0].state_dict().items() if not is_lora_parameter(name)},
                    checkpoint_id=base,
                )

    def train(self):
        super().train()
        if self.checkpointer.periodic_save_adapter_only:
            path = Path(self.checkpointer.folder) / f"step-{self.config.checkpoint.interval}"
            keys = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
            model = self.model_parts[0]
            assert any(is_lora_parameter(name) for name in keys)
            assert any(name.startswith("optimizer.") for name in keys)
            buffers = {canonical_fqn(name) for name, _ in model.named_buffers()}
            frozen = {name for name in model.state_dict() if not is_lora_parameter(name) and name not in buffers}
            assert any(".engram." in name for name in frozen)
            assert not frozen.intersection(keys)

    def _check_resume_state(self, step):
        optimizer = self.optimizers.state_dict()
        tensors = {
            "adapters": {
                name: _cpu_tensor(value) for name, value in self.model_parts[0].named_parameters()
                if is_lora_parameter(name)
            },
            "optimizer": {name: _cpu_tensor(value) for name, value in optimizer.items()
                          if isinstance(value, torch.Tensor)},
        }
        assert tensors["adapters"] and tensors["optimizer"]
        metadata = {name: value for name, value in optimizer.items() if not isinstance(value, torch.Tensor)}
        path = Path(self.config.dump_folder) / f"resume_state_step{step}_rank{dist.get_rank()}.pt"
        if (self.config.checkpoint.load_step or 0) > 0:
            expected = torch.load(path, map_location="cpu", weights_only=True)
            torch.testing.assert_close(tensors, expected["tensors"], rtol=0, atol=0)
            assert metadata == expected["metadata"]
            logger.info(f"V4.1 exact adapter/optimizer resume state match: step={step}, rank={dist.get_rank()}")
        else:
            # Only the uninterrupted phase writes the comparison baseline.
            with path.open("xb") as stream:
                torch.save({"tensors": tensors, "metadata": metadata}, stream)

    def train_step(self, data_iterator):
        checkpoint = self.config.checkpoint
        load_step = checkpoint.load_step or 0
        if self.config.check_resume_state and load_step > 0 and self.step == load_step + 1:
            self._check_resume_state(load_step)
        tables = [m for m in self.model_parts[0].modules() if isinstance(m, HostEngramTable)]
        assert len(tables) == 2
        super().train_step(data_iterator)
        assert all(table.pending_sparse_grad() is None for table in tables)
        assert all(not table.weight.requires_grad for table in tables)
        if self.config.check_resume_state and self.step >= checkpoint.interval:
            self._check_resume_state(self.step)


def deepseek_v4_1_lora_training():
    config = deepseek_v4_1_debugmodel_lora()
    config.checkpoint.periodic_save_adapter_only = False
    return derive(config, V41LoRATrainer.Config)


def deepseek_v4_1_lora_quantized_training():
    config = deepseek_v4_1_lora_training()
    config.extension.quantization.enable_quantized_training = True
    config.extension.quantization.recipe = "all_block_fp8"
    config.extension.quantization.enable_fsdp_prequantize = False
    return config


def deepseek_v4_1_lora_adapter_only_training():
    config = deepseek_v4_1_lora_training()
    config.check_resume_state = True
    config.checkpoint.periodic_save_adapter_only = True
    return config


class V41MultimodalLoRATrainer(V41LoRATrainer):
    @dataclass(kw_only=True, slots=True)
    class Config(V41LoRATrainer.Config):
        pass

    def train_step(self, data_iterator):
        model = self.model_parts[0]
        assert model.vision_encoder is not None
        image_forwards = []

        def observe_vision(module, args, output):
            assert args[0].shape[-2:] == (9, 588)
            assert torch.count_nonzero(args[0]) > 0
            image_forwards.append(output.shape)

        def image_batches():
            for inputs, labels in data_iterator:
                inputs = dict(inputs)
                tokens = inputs["input"].clone()
                labels = labels.clone()
                ids, types, indices = build_image_token_layout(
                    [(3, 3)], span_start=2, vocab_size=self.config.model_spec.model.vocab_size,
                )
                image_span_end = ids.numel()
                tokens[:, 2:image_span_end] = ids[2:].to(tokens.device)
                labels[:, : ids.numel()] = -100
                token_types = torch.full_like(tokens, -1)
                feature_indices = torch.full_like(tokens, -1)
                token_types[:, : types.numel()] = types.to(tokens.device)
                feature_indices[:, : indices.numel()] = indices.to(tokens.device)
                inputs.update(
                    input=tokens,
                    positions=torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0),
                    pixel_values=torch.linspace(-1, 1, 9 * 588, device=tokens.device).reshape(1, 9, 588),
                    image_grid=torch.tensor([[3, 3]], device=tokens.device),
                    image_feature_indices=feature_indices,
                    token_types=token_types,
                )
                yield inputs, labels

        handle = model.vision_encoder.register_forward_hook(observe_vision)
        try:
            super().train_step(image_batches())
            assert image_forwards
        finally:
            handle.remove()


def deepseek_v4_1_lora_multimodal_training():
    config = _make_lora_trainer_config("deepseek_v4_1_debugmodel")
    # C4 supplies text/labels; the trainer adds a deterministic image-protocol fixture.
    config.dataloader = deepseek_v4_1_debugmodel_lora().dataloader
    config.checkpoint.periodic_save_adapter_only = False
    return derive(config, V41MultimodalLoRATrainer.Config)
