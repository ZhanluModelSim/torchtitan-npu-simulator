# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen example configuration selection through the native recipe loader."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file, save_file
from torchtitan.components.data import HuggingFaceStreamingSource
from torchtitan.config import apply_overrides
from torchtitan.config.loader import ConfigLoader

import torchtitan_npu  # noqa: F401
from torchtitan_npu.models.qwen3_5 import NPUQwen35Model
from torchtitan_npu_recipes.models import qwen3_5
from torchtitan_npu_recipes.models.qwen3_5 import _text_sft_config


@pytest.fixture
def qwen_resources(monkeypatch, tmp_path):
    for name in (
        "DATASET",
        "DATASET_PATH",
        "QWEN_DATASET_PATH",
        "HF_ASSETS_PATH",
        "OVERRIDE_IMPORTS",
        "ENABLE_NPU_MOE_DISPATCHER",
        "WORLD_SIZE",
        "NGPU",
    ):
        monkeypatch.delenv(name, raising=False)
    upstream = tmp_path / "upstream"
    (upstream / "tests/assets/tokenizer").mkdir(parents=True)
    monkeypatch.setenv("TORCHTITAN_REPO", str(upstream))
    return upstream


@pytest.mark.cpu
@pytest.mark.parametrize(
    "recipe,microbatch,step_tokens,ep,dp",
    [
        ("qwen3_5_debugmodel_1p_cpt_512", 512, 512, 1, -1),
        ("qwen3_5_debugmodel_moe_8p_cpt_512", 1024, -1, 8, 8),
    ],
    ids=["dense-1p", "moe-8p"],
)
def test_loader_selects_qwen_example_training_and_resources(recipe, microbatch, step_tokens, ep, dp, qwen_resources):
    config = ConfigLoader().load(["--module", "torchtitan_npu_recipes.tests.models.qwen3_5", "--config", recipe])

    assert isinstance(config.model, NPUQwen35Model.Config)
    assert config.model.max_context_length == config.training.max_context_length == 512
    assert config.training.num_tokens_per_microbatch_per_dp_rank == microbatch
    assert config.training.num_tokens_per_train_step == step_tokens
    assert config.training.steps == 10
    assert config.training.disable_cuda_graphs
    assert config.debug.print_config
    assert config.optim.optimizer.optimizers[0].lr == 5e-3
    assert config.parallelism.expert_parallel_degree == ep
    assert config.parallelism.data_parallel_shard_degree == dp
    assert config.parallelism.data_parallel_replicate_degree == 1
    assert config.parallelism.tensor_parallel_degree == 1
    assert config.parallelism.pipeline_parallel_degree == 1
    assert config.parallelism.context_parallel_degree == 1
    assert config.hf_assets_path == str(qwen_resources / "tests/assets/tokenizer")
    assert isinstance(config.dataloader.dataset.source, HuggingFaceStreamingSource.Config)
    assert config.dataloader.dataset.source.path == str(qwen_resources / "tests/assets/cc12m_test")
    assert config.override.imports == ["torchtitan_npu.override.qwen3_5.gated_delta.npu"]


@pytest.mark.cpu
@pytest.mark.parametrize("size_variable", ["NGPU", "WORLD_SIZE"])
def test_moe_recipe_replicates_complete_eight_rank_groups(size_variable, monkeypatch, qwen_resources):
    monkeypatch.setenv("NGPU", "8")
    monkeypatch.setenv(size_variable, "16")
    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan_npu_recipes.tests.models.qwen3_5",
            "--config",
            "qwen3_5_debugmodel_moe_8p_cpt_512",
        ]
    )

    assert config.parallelism.data_parallel_replicate_degree == 2
    assert config.parallelism.data_parallel_shard_degree == 8
    assert config.parallelism.expert_parallel_degree == 8


@pytest.mark.cpu
def test_moe_recipe_rejects_incomplete_eight_rank_groups(monkeypatch, qwen_resources):
    monkeypatch.setenv("WORLD_SIZE", "12")
    with pytest.raises(ValueError, match="positive multiple of 8"):
        ConfigLoader().load(
            [
                "--module",
                "torchtitan_npu_recipes.tests.models.qwen3_5",
                "--config",
                "qwen3_5_debugmodel_moe_8p_cpt_512",
            ]
        )


@pytest.mark.cpu
def test_loader_honors_qwen_asset_paths_and_dispatcher_selection(monkeypatch, tmp_path, qwen_resources):
    assets = tmp_path / "model-assets"
    assets.mkdir()
    monkeypatch.setenv("HF_ASSETS_PATH", str(assets))
    monkeypatch.setenv("DATASET_PATH", str(tmp_path / "samples"))
    monkeypatch.setenv("ENABLE_NPU_MOE_DISPATCHER", "1")

    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan_npu_recipes.tests.models.qwen3_5",
            "--config",
            "qwen3_5_debugmodel_moe_8p_cpt_512",
        ]
    )

    assert config.hf_assets_path == str(assets)
    assert config.dataloader.dataset.source.path == str(tmp_path / "samples")
    assert config.override.imports == [
        "torchtitan_npu.override.qwen3_5.gated_delta.npu",
        "torchtitan_npu.override.common.token_dispatcher.asc",
    ]


@pytest.mark.cpu
def test_legacy_qwen_override_environment_requires_explicit_configuration(monkeypatch, qwen_resources):
    monkeypatch.setenv("OVERRIDE_IMPORTS", "")
    with pytest.raises(ValueError, match=r"config\.override\.imports.*--override"):
        ConfigLoader().load(
            [
                "--module",
                "torchtitan_npu_recipes.tests.models.qwen3_5",
                "--config",
                "qwen3_5_debugmodel_1p_cpt_512",
            ]
        )


def test_qwen_moe_checkpoint_preserves_single_router_update_and_gradients(monkeypatch):
    from copy import deepcopy

    from torch import nn
    from torch.utils.checkpoint import DefaultDeviceType
    from torchtitan.models.common.moe import MoE

    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")

    class CountingMoE(MoE):
        def __init__(self):
            nn.Module.__init__(self)
            self.routed_experts = nn.Module()
            self.routed_experts.w13 = nn.Linear(4, 8)
            self.routed_experts.w2 = nn.Linear(8, 4)
            self.shared_experts = nn.Linear(4, 4)
            self.register_buffer("routing_calls", torch.zeros((), dtype=torch.long))

        def forward(self, inputs):
            self.routing_calls.add_(1)
            hidden = torch.nn.functional.silu(self.routed_experts.w13(inputs))
            return self.routed_experts.w2(hidden) + self.shared_experts(inputs)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Linear(4, 4)
            self.feed_forward = CountingMoE()

        def forward(self, inputs):
            hidden = inputs + self.attention(inputs)
            return hidden + self.feed_forward(hidden)

    reference = Block()
    candidate = deepcopy(reference)
    container = nn.Module()
    container.layers = nn.ModuleList([candidate])
    qwen3_5.QwenFullActivationRematConfig().apply(container)
    inputs = torch.randn(3, 4, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    expected = reference(reference_inputs)
    actual = candidate(inputs)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    assert reference.feed_forward.routing_calls.item() == candidate.feed_forward.routing_calls.item() == 1
    assert candidate.state_dict().keys() == reference.state_dict().keys()
    for expected_parameter, actual_parameter in zip(reference.parameters(), candidate.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


@pytest.mark.parametrize(
    "preset,length,cp",
    [
        ("run_qwen3_5_4k", 64, 1),
        ("run_qwen3_5_32k_cp8", 32768, 8),
        ("run_qwen3_5_64k_cp8", 65536, 8),
        ("run_qwen3_5_128k_cp8", 131072, 8),
        ("qwen3_5_27b_sft_64k_cp8", 65536, 8),
    ],
)
@pytest.mark.parametrize("launcher", ["examples/qwen3_5/run_qwen_sft.sh"])
def test_sft_launcher_emits_only_native_cli(preset, length, cp, launcher, monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[3]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "capture.json"
    fake = bin_dir / "torchrun"
    fake.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, pathlib, sys
args = sys.argv[1:]
pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps({
    'args': args[args.index('-m') + 2:],
    'env': {k: v for k, v in os.environ.items() if k.startswith('QWEN_') or k == 'SEQ_LEN'},
}))
"""
    )
    fake.chmod(0o755)
    set_env = tmp_path / "set_env.sh"
    set_env.write_text("# CPU argv rendering only\n")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"messages":[{"role":"user","content":"question"},{"role":"assistant","content":"answer"}]}\n')
    monkeypatch.setenv("HF_ASSETS_PATH", str(tmp_path))
    monkeypatch.setenv("DATA_FILES", str(dataset))
    env = os.environ | {
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "CAPTURE": str(capture),
        "ASCEND_SET_ENV_PATH": str(set_env),
        "SEQ_LEN": "64",
        "NGPU": "16",
        "DUMP_FOLDER": str(tmp_path / "output"),
    }
    selector = ["--config", preset] if cp > 1 else []
    subprocess.run(
        ["bash", str(root / launcher), *selector, "--output-dir", str(tmp_path / "output")],
        cwd=root,
        env=env,
        check=True,
    )
    rendered = json.loads(capture.read_text())
    for name, value in rendered["env"].items():
        monkeypatch.setenv(name, value)
    config = ConfigLoader().load(rendered["args"])
    apply_overrides(config.override, config)
    from torchtitan_npu.models.qwen3_5.chat_processor import QwenSFTChatProcessor
    from torchtitan_npu.models.qwen3_5.dataloader import SFTGrainDataLoader

    assert isinstance(config.dataloader, SFTGrainDataLoader.Config)
    assert isinstance(config.dataloader.dataset.dataset.processor, QwenSFTChatProcessor.Config)
    assert config.training.max_context_length == length
    assert config.parallelism.context_parallel_degree == cp
    assert config.parallelism.data_parallel_shard_degree == (2 if cp > 1 else -1)
    assert config.model.text_only and config.checkpointer.initial_load_in_hf
    assert config.dump_folder == str(tmp_path / "output")
    assert config.dataloader.max_num_documents == (512 if length >= 32768 else None)
    from torchtitan_npu.override.qwen3_5.gated_delta import AscInnerGDN, TritonGatedDeltaKernel

    gdn_configs = [inner for _, inner, _, _ in config.model.traverse(AscInnerGDN.Config)]
    assert gdn_configs and all(isinstance(inner.kernel, TritonGatedDeltaKernel.Config) for inner in gdn_configs)
    from torchtitan_npu.override.qwen3_5.gated_delta import QwenSFTGatedRMSNorm

    assert len(list(config.model.traverse(QwenSFTGatedRMSNorm.Config))) == len(gdn_configs)


@pytest.mark.parametrize("trial", [False, True])
def test_external_python_recipe_uses_native_config_loader(tmp_path, monkeypatch, trial):
    from torchtitan_npu_recipes.qwen_sft import config_file_args, sft_launch_args

    folder = tmp_path / "my configs"
    folder.mkdir()
    (folder / "qwen_team_settings.py").write_text("STEPS = 9\n")
    path = folder / "qwen.recipe.py"
    path.write_text(
        "from qwen_team_settings import STEPS\n"
        "from torchtitan_npu_recipes.qwen_sft import run_qwen3_5_4k\n"
        "def train():\n"
        "    config = run_qwen3_5_4k()\n"
        "    config.training.steps = STEPS\n"
        "    return config\n"
    )
    monkeypatch.setenv("HF_ASSETS_PATH", str(tmp_path))
    monkeypatch.setenv("NGPU", "1")
    args = [
        "--module",
        "torchtitan_npu_recipes.qwen_sft",
        "--config",
        "run_qwen3_5_4k",
        f"--module={path}",
        "--config",
        "train",
        "--output-dir",
        str(tmp_path / "train"),
    ]
    if trial:
        args += ["--smoke-test"]
    with config_file_args(args) as selected, sft_launch_args(selected) as native:
        config = ConfigLoader().load(native)
    assert config.training.steps == (1 if trial else 9)
    assert config.dump_folder == str(tmp_path / "train")
    assert config.checkpointer.load_only is trial


@pytest.fixture(params=("model", "model.language_model"), ids=("text-checkpoint", "conditional-checkpoint"))
def recipe_and_prefix(request, monkeypatch, tmp_path):
    prefix = request.param
    keys = (
        f"{prefix}.embed_tokens.weight",
        f"{prefix}.layers.0.linear_attn.in_proj_qkv.weight",
        f"{prefix}.layers.0.linear_attn.conv1d.weight",
        "lm_head.weight",
        "model.visual.patch_embed.proj.weight",
        "mtp.layers.0.weight",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": dict.fromkeys(keys, "model-00001-of-00001.safetensors"),
            }
        )
    )
    save_file(
        {
            "model.visual.patch_embed.proj.weight": torch.full((2, 3), 0.25),
            "mtp.layers.0.weight": torch.full((2, 3), 0.5),
        },
        tmp_path / "model-00001-of-00001.safetensors",
    )
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"prompt":"sum?","response":"two"}\n')
    monkeypatch.setenv("HF_ASSETS_PATH", str(tmp_path))
    monkeypatch.setenv("DATA_FILES", str(dataset))
    recipe = _text_sft_config(flavor="debugmodel", length=512, cp=2, dp_shard=2, steps=2, load_checkpoint=True)
    adapter = recipe.model._owner.state_dict_adapter_cls(recipe.model, str(tmp_path / "tokenizer-only"))
    adapter.prepare_hf_load(str(tmp_path))
    return recipe, adapter, prefix


def test_text_recipe_imports_source_prefix_and_gdn_slices(recipe_and_prefix, tmp_path):
    recipe, adapter, prefix = recipe_and_prefix
    gdn = recipe.model.layers[0].attention
    q_channels, v_channels = gdn.in_proj_q.out_features, gdn.in_proj_v.out_features
    rows = 2 * q_channels + v_channels
    qkv = torch.arange(rows * recipe.model.dim, dtype=torch.float32).reshape(rows, recipe.model.dim)
    embedding = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    actual = adapter.from_hf(
        {
            f"{prefix}.embed_tokens.weight": embedding,
            f"{prefix}.layers.0.linear_attn.in_proj_qkv.weight": qkv,
            "lm_head.weight": torch.ones(2, 3),
        }
    )

    torch.testing.assert_close(actual["tok_embeddings.weight"], embedding, rtol=0, atol=0)
    key_end = 2 * q_channels
    for name, expected in (
        ("q", qkv[:q_channels]),
        ("k", qkv[q_channels:key_end]),
        ("v", qkv[key_end:]),
    ):
        torch.testing.assert_close(actual[f"layers.0.attention.in_proj_{name}.weight"], expected, rtol=0, atol=0)
    vision, mtp = "model.visual.patch_embed.proj.weight", "mtp.layers.0.weight"
    restored = type(adapter)(recipe.model, str(tmp_path / "different-tokenizer"))
    dcp.save({"source": adapter}, checkpoint_id=tmp_path / "dcp")
    dcp.load({"source": restored}, checkpoint_id=tmp_path / "dcp")
    assert restored.state_dict() == adapter.state_dict()
    load_targets = adapter.to_hf(actual)
    assert vision not in load_targets and mtp not in load_targets
    with restored.export_hf():
        exported = restored.to_hf(actual)
    torch.testing.assert_close(exported[vision], torch.full((2, 3), 0.25))
    torch.testing.assert_close(exported[mtp], torch.full((2, 3), 0.5))


@pytest.mark.parametrize("layout", ["vision", "stacked", "separate"])
def test_qwen_offline_export_uses_dcp_layout(layout, monkeypatch, tmp_path):
    from torchtitan_npu.models import qwen3_5 as model_module
    from torchtitan_npu.tools.checkpoint_conversion.convert_to_hf import convert_to_hf

    build_config = model_module.checkpoint_model_config

    def compact(flavor, **kwargs):
        config = build_config(flavor, **kwargs)
        config.vocab_size = config.tok_embeddings.num_embeddings = config.lm_head.out_features = 32
        config.layers = config.layers[:1]
        return config

    monkeypatch.setattr(model_module, "checkpoint_model_config", compact)
    config = compact("debugmodel", has_vision=layout == "vision", separate_gate_up=layout == "separate")
    with torch.device("cpu"):
        model = config.build()
    state = {key: torch.zeros_like(value) for key, value in model.state_dict().items()}
    state["tok_embeddings.weight"].fill_(0.125)
    vision_key = "model.visual.patch_embed.proj.weight"
    source = tmp_path / "original_hf"
    source.mkdir()
    hf_assets = tmp_path / "different_assets"
    if layout == "vision":
        state["vision_encoder.patch_embed.weight"].fill_(0.25)
        adapter = type(model).state_dict_adapter_cls(config, None)
        expected_index = {
            key: "model-00002-of-00002.safetensors" if key == vision_key else "model-00001-of-00002.safetensors"
            for key in adapter.to_hf(state)
        }
        hf_assets.mkdir()
        reference_index = {**expected_index, "mtp.fc.weight": expected_index[vision_key]}
        (hf_assets / "model.safetensors.index.json").write_text(json.dumps({"weight_map": reference_index}))
    else:
        save_file(
            {
                "model.language_model.embed_tokens.weight": state["tok_embeddings.weight"],
                vision_key: torch.full((2, 3), 0.5),
            },
            source / "model.safetensors",
        )
        adapter = type(model).state_dict_adapter_cls(config, None)
        adapter.prepare_hf_load(str(source))
        state["qwen_hf_source"] = adapter
    dcp.save(state, checkpoint_id=tmp_path / "input")

    convert_to_hf(
        SimpleNamespace(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            model_name="qwen3_5",
            model_flavor="debugmodel",
            hf_assets_path=hf_assets,
            export_dtype="float32",
            ema_output=None,
            read_threads=1,
        )
    )
    exported = {}
    for shard in (tmp_path / "output").glob("*.safetensors"):
        exported.update(load_file(shard))
    torch.testing.assert_close(exported["model.language_model.embed_tokens.weight"], state["tok_embeddings.weight"])
    if layout == "vision":
        index = json.loads((tmp_path / "output" / "model.safetensors.index.json").read_text())
        assert set(exported) == set(expected_index)
        assert index["weight_map"] == expected_index
        torch.testing.assert_close(exported[vision_key].flatten(1), state["vision_encoder.patch_embed.weight"])
    else:
        torch.testing.assert_close(exported[vision_key], torch.full((2, 3), 0.5))
