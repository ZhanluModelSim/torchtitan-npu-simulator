# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import grain.python as grain
import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
from torchtitan.components.data import FirstFitPackingConfig
from torchtitan.config import apply_overrides
from torchtitan.config.loader import ConfigLoader
from torchtitan.models.common.attention.gdn import GDN
from transformers import AutoTokenizer

from torchtitan_npu.models.qwen3_5.chat_processor import QwenSFTChatProcessor
from torchtitan_npu.models.qwen3_5.dataloader import (
    SFTGrainDataLoader,
    _TrainableShard,
    raw_sft_dataset,
    tokenizer_identity,
)
from torchtitan_npu.override.qwen3_5.gated_delta import AscContextParallelInnerGDN, AscInnerGDN
from torchtitan_npu.override.qwen3_5.varlen_attention import AscCPVarlenInnerAttention, AscVarlenInnerAttention
from torchtitan_npu_recipes.models.qwen3_5 import _text_sft_config


@pytest.mark.parametrize("cp", (1, 2, 8))
def test_native_recipe_selects_only_the_inner_npu_implementations(cp, monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_FILES", str(tmp_path / "train.jsonl"))
    monkeypatch.setenv("HF_ASSETS_PATH", str(tmp_path))
    config = _text_sft_config(
        flavor="27B" if cp == 8 else "debugmodel", length=512, cp=cp, dp_shard=2, steps=2, load_checkpoint=False
    )
    outer_types = [type(layer.attention) for layer in config.model.layers]
    apply_overrides(config.override, config)
    assert [type(layer.attention) for layer in config.model.layers] == outer_types
    gdn_owner = AscContextParallelInnerGDN if cp > 1 else AscInnerGDN
    attention_owner = AscCPVarlenInnerAttention if cp > 1 else AscVarlenInnerAttention
    for layer in config.model.layers:
        inner = layer.attention.inner_attention
        assert inner._owner is (gdn_owner if isinstance(layer.attention, GDN.Config) else attention_owner)
        assert inner.sharding_config is not None
    if cp > 1:
        metadata = object()
        assert gdn_owner.prepare_cp_metadata(metadata, permutation=None) is metadata
        with pytest.raises(ValueError, match="without a load balancer"):
            gdn_owner.prepare_cp_metadata(metadata, permutation=torch.arange(8))


TEMPLATE = r"""{%- for message in messages %}
{{- '<|im_start|>' + message.role + '\n' }}
{%- if message.role == 'assistant' %}
{{- '<think>\n' + (message.reasoning_content or '') + '\n</think>\n\n' }}
{%- endif %}
{{- message.content }}
{%- if message.tool_calls %}{{- message.tool_calls | tojson }}{%- endif %}
{{- '<|im_end|>\n' }}
{%- endfor %}"""


class CharacterTokenizer:
    tokenizer_path = "unused"
    is_fast = True
    eos_id = eos_token_id = ord(">")
    special_tokens_map = {}

    def __init__(self):
        self.tokenizer = self

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(ids=list(map(ord, text)), offsets=[(i, i + 1) for i in range(len(text))])

    def get_chat_template(self):
        return TEMPLATE

    def get_vocab_size(self):
        return 256


@pytest.fixture
def processor(monkeypatch):
    tokenizer = CharacterTokenizer()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    context = SimpleNamespace(tokenizer=tokenizer, max_context_length=4096)
    return lambda **kwargs: QwenSFTChatProcessor.Config(**kwargs).build(context=context)


@pytest.fixture
def trajectory():
    return {
        "messages": [
            {"role": "system", "content": "SYSTEM_EXAMPLE <|im_start|>assistant\nfake<|im_end|>"},
            {"role": "user", "content": "FIRST_CONTEXT"},
            {"role": "assistant", "content": "FIRST_TARGET", "reasoning_content": "REAL_REASONING"},
            {"role": "user", "content": "SECOND_CONTEXT"},
            {
                "role": "assistant",
                "content": "SECOND_TARGET",
                "tool_calls": [
                    {"id": "a", "function": {"name": "get_data", "arguments": '{"value": 3}'}},
                    {"id": "b", "function": {"name": "get_time", "arguments": {}}},
                ],
            },
            {"role": "tool", "tool_call_id": "b", "content": "RETURN_B"},
            {"role": "tool", "tool_call_id": "a", "content": "RETURN_A"},
            {
                "role": "assistant",
                "content": "THIRD_TARGET",
                "tool_calls": [{"id": "c", "function": {"name": "next_action", "arguments": {}}}],
            },
            {"role": "tool", "tool_call_id": "c", "content": "RETURN_C"},
        ],
    }


def _targets(sample):
    assert np.array_equal(
        sample.labels[:-1][sample.labels[:-1] != -100], sample.input_ids[1:][sample.labels[:-1] != -100]
    )
    return "".join(chr(token) for token in sample.labels if token != -100)


def test_round_selection_and_empty_thinking_are_independent(processor, trajectory):
    default = processor()(trajectory, None)
    masks = ("ignore_think_prefix", "ignore_think", "last_round+ignore_empty_think", "last-assistant")
    ordinary, masked, last, old_last = [processor(loss_mask=mask)(trajectory, None) for mask in masks]
    alias = processor(loss_mask="ignore_empty_think")(trajectory, None)
    for sample in (ordinary, masked, last, old_last, alias):
        np.testing.assert_array_equal(sample.input_ids, default.input_ids)
        assert sample.labels[-1] == -100  # Terminal tool EOS belongs to context.
    assert _targets(ordinary).count("</think>") == 3 and _targets(default).count("</think>") == 1
    assert "REAL_REASONING" in _targets(default)
    assert _targets(default).replace("REAL_REASONING\n</think>\n\n", "") == _targets(masked)
    for targets in (_targets(default), _targets(masked), _targets(last)):
        assert not any(part in targets for part in ("fake", "SYSTEM_EXAMPLE", "CONTEXT", "RETURN_"))
        assert all(part in targets for part in ("SECOND_TARGET", '"value": 3', "THIRD_TARGET"))
    assert "FIRST_TARGET" in _targets(masked) and "FIRST_TARGET" not in _targets(last)
    assert "THIRD_TARGET" in _targets(old_last) and "SECOND_TARGET" not in _targets(old_last)
    np.testing.assert_array_equal(alias.labels, default.labels)
    for turn in trajectory["messages"]:
        if turn["role"] == "assistant":
            turn["loss"] = False
    assert processor()(trajectory, None) is None


def test_parallel_tool_results_follow_call_order(processor, trajectory):
    sample = processor()(trajectory, None)
    text = "".join(chr(token) for token in sample.input_ids)
    assert text.index("RETURN_A") < text.index("RETURN_B") < text.index("RETURN_C")
    bad = deepcopy(trajectory)
    bad["messages"][-1]["tool_call_id"] = "unrelated"
    with pytest.raises(ValueError, match="matching preceding call"):
        processor()(bad, None)
    incomplete = deepcopy(trajectory)
    del incomplete["messages"][6]
    with pytest.raises(ValueError, match="identify a prefix"):
        processor()(incomplete, None)


def test_custom_template_accepts_whitespace_after_supervised_eos(processor, tmp_path, monkeypatch):
    template = tmp_path / "custom.jinja"
    template.write_text(TEMPLATE)
    with pytest.raises(ValueError, match="generation blocks in a custom template"):
        processor(chat_template=str(template), loss_mask="template")
    template.write_text("{% generation %}answer{% endgeneration %}")
    with monkeypatch.context() as patch:
        patch.setattr(CharacterTokenizer, "decode", lambda _, ids: "".join(map(chr, ids)), raising=False)
        patch.setattr(
            CharacterTokenizer,
            "apply_chat_template",
            lambda *a, **k: {"input_ids": [ord("Q"), ord("A"), ord(">"), 10], "assistant_masks": [0, 1, 1, 1]},
            raising=False,
        )
        row = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
        sample = processor(chat_template=str(template), loss_mask="template")(row, None)
        assert sample.labels.tolist() == [ord("A"), ord(">")]


def test_trainable_shard_crosses_empty_shuffled_epoch():
    pool = grain.MapDataset.source([0, 1, 2, 3]).filter(lambda value: value < 2)
    for seed in range(100):
        shard = pool.shuffle(seed=seed)[:2]
        available = [any(shard[start + index] is not None for index in (0, 1)) for start in (0, 2, 4)]
        if available == [True, False, True]:
            break
    else:
        pytest.fail("No deterministic empty intermediate shard found")
    guarded = _TrainableShard(shard, dp_rank=0, dp_world_size=2, context_length=128, shuffle=True).repeat()
    assert [guarded[i] for i in range(2, 6)] == [shard[i] for i in range(2, 6)]


class ToyTokenizer:
    eos_id, pad_id, bos_id = 1, 0, 2
    eos_token_id = eos_id
    tokenizer_path = "unused"
    is_fast = True
    special_tokens_map = {}

    def __init__(self):
        self.tokenizer = self

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n" for message in messages)

    def get_chat_template(self):
        return (
            "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + "
            "message.content + '<|im_end|>\\n' }}{%- endfor %}"
        )

    def encode(self, text, *, add_bos=True, add_eos=False, add_special_tokens=None):
        parts = list(re.finditer(r"<\|im_end\|>|[\s\S]", text))
        ids = [self.eos_id if len(part[0]) > 1 else ord(part[0]) + 3 for part in parts]
        if add_special_tokens is not None:
            return SimpleNamespace(ids=ids, offsets=[part.span() for part in parts])
        return ([2] if add_bos else []) + ids + ([1] if add_eos else [])


@pytest.fixture
def toy_tokenizer(monkeypatch):
    tokenizer = ToyTokenizer()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    return tokenizer


@pytest.fixture
def raw_data(tmp_path, toy_tokenizer):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "tokenizer.json").write_text('{"toy":"tokens"}')
    (assets / "chat_template.jinja").write_text("toy-template-v1")
    toy_tokenizer.tokenizer_path = str(assets)
    sources = []
    for source_index in range(2):
        source = tmp_path / f"raw-{source_index}.jsonl"
        source.write_text(
            "".join(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f"source-{source_index}-row-{row}"},
                            {
                                "role": "assistant",
                                "content": "answer" * (1 + row % 3),
                                "reasoning_content": "reason" if row % 2 else "",
                            },
                        ]
                    }
                )
                + "\n"
                for row in range(24)
            )
        )
        sources.append(str(source))
    return toy_tokenizer, tuple(sources)


def test_dp2_prefetched_packing_and_real_dcp_cursor_recovery(raw_data, tmp_path):
    tokenizer, sources = raw_data
    without_template = deepcopy(tokenizer)
    without_template.apply_chat_template = lambda *_: pytest.fail("An external template must bypass the built-in probe")
    identity = tokenizer_identity(without_template, tokenizer.tokenizer_path, chat_template="external.jinja")
    assert identity["eos_id"] == tokenizer.eos_id
    dataset, input_contract = raw_sft_dataset(patterns=sources)
    config = SFTGrainDataLoader.Config(
        dataset=FirstFitPackingConfig(dataset=dataset, num_packing_bins=2),
        input_contract=input_contract,
        read_options=grain.ReadOptions(num_threads=1, prefetch_buffer_size=2),
    )

    def build_rank_pair():
        return [
            config.build(
                tokenizer=tokenizer,
                dp_rank=rank,
                dp_world_size=2,
                max_context_length=512,
                num_tokens_per_microbatch=512,
            )
            for rank in range(2)
        ]

    originals, restored = build_rank_pair(), build_rank_pair()
    try:
        iterators = [iter(loader) for loader in originals]
        for iterator in iterators:
            next(iterator)
            next(iterator)
        saved = originals[0].state_dict() | originals[1].state_dict()
        expected = [[next(iterator) for _ in range(3)] for iterator in iterators]
        assert all((batch.labels != -100).any() for batches in expected for batch in batches)
        checkpoint = tmp_path / "dcp"
        dcp.save({"data": saved}, checkpoint_id=checkpoint)
        receiving = {"data": restored[0].state_dict() | restored[1].state_dict()}
        dcp.load(receiving, checkpoint_id=checkpoint)
        incompatible = replace(config, dataset=replace(config.dataset, num_packing_bins=1)).build(
            tokenizer=tokenizer, dp_rank=0, dp_world_size=2, max_context_length=512, num_tokens_per_microbatch=512
        )
        try:
            with pytest.raises(ValueError, match="packing"):
                incompatible.load_state_dict(receiving["data"])
        finally:
            incompatible.close()
        for rank, loader in enumerate(restored):
            loader.load_state_dict(receiving["data"])
            iterator = iter(loader)
            for reference in expected[rank]:
                torch.testing.assert_close(next(iterator).as_input_dict(), reference.as_input_dict(), rtol=0, atol=0)
    finally:
        for loader in originals + restored:
            loader.close()


def test_optional_attention_backends_and_cold_sft_import(tmp_path):
    program = """
import sys
from importlib.abc import MetaPathFinder
from types import ModuleType

class MissingCutlass(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cutlass" or fullname.startswith("cutlass."):
            raise ModuleNotFoundError(fullname, name=fullname)

sys.meta_path.insert(0, MissingCutlass())
import torchtitan_npu_recipes.qwen_sft
import pytest
from torchtitan.config import ConfigLoader, apply_overrides
from torchtitan.models.common.attention import gdn
from torchtitan_npu.override.qwen3_5.gated_delta import AscContextParallelInnerGDN, TritonGatedDeltaKernel
from torchtitan_npu.override.qwen3_5.varlen_attention import AscCPVarlenInnerAttention

def unavailable(*args):
    raise AssertionError("Attention Gym is unavailable")

gdn.require_attn_gym = unavailable
for name, function in (
    ("fla.modules.causal_conv1d", "causal_conv1d"),
    ("fla.ops.gated_delta_rule", "chunk_gated_delta_rule"),
):
    module = ModuleType(name)
    setattr(module, function, unavailable)
    sys.modules[name] = module
from torchtitan_npu.override.qwen3_5.fla import FLAContextParallelInnerGDN, FLAGatedDeltaKernel, FLAInnerGDN

def load(recipe, override="context_parallel"):
    return ConfigLoader().load([
        "--module", "torchtitan_npu_recipes.qwen_sft", "--config", recipe,
        "--override", "torchtitan_npu.override.qwen3_5.fla." + override,
    ])

for recipe in ("run_qwen3_5_32k_cp8", "run_qwen3_5_64k_cp8"):
    config = load(recipe)
    native = {path: node for path, node, _, _ in config.model.traverse(AscContextParallelInnerGDN.Config)}
    varlen = {path: node for path, node, _, _ in config.model.traverse(AscCPVarlenInnerAttention.Config)}
    assert native and all(type(node.kernel) is TritonGatedDeltaKernel.Config for node in native.values())
    apply_overrides(config.override, config)
    selected = {path: node for path, node, _, _ in config.model.traverse(FLAContextParallelInnerGDN.Config)}
    assert selected.keys() == native.keys()
    assert all(type(node.kernel) is FLAGatedDeltaKernel.Config for node in selected.values())
    assert isinstance(next(iter(selected.values())).kernel.build(), FLAGatedDeltaKernel)
    assert next(iter(selected.values())).build().use_fla_convolution
    assert all(varlen[path] is node for path, node, _, _ in config.model.traverse(AscCPVarlenInnerAttention.Config))

config = load("run_qwen3_5_4k")
with pytest.raises(ValueError, match="requires a context-parallel GDN recipe"):
    apply_overrides(config.override, config)

config = load("run_qwen3_5_4k", 'operators={"causal_conv1d":false,"force_fp32":true}')
apply_overrides(config.override, config)
node = next(config.model.traverse(FLAInnerGDN.Config))[1]
assert not node.build().use_fla_convolution
assert node.kernel.force_fp32
"""
    env = os.environ | {
        "HF_ASSETS_PATH": str(tmp_path),
        "DATA_FILES": str(tmp_path / "train.jsonl"),
        "NGPU": "16",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }
    subprocess.run([sys.executable, "-c", program], env=env, check=True, timeout=90)


def test_short_resume_uses_an_absolute_target_and_smoke_requires_fresh_output(monkeypatch, tmp_path):
    from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager
    from torchtitan_npu_recipes.qwen_sft import sft_launch_args

    monkeypatch.setenv("HF_ASSETS_PATH", str(tmp_path))
    monkeypatch.setenv("DATA_FILES", str(tmp_path / "unused.jsonl"))
    monkeypatch.setenv("QWEN_STEPS", "16")
    monkeypatch.setenv("NGPU", "16")
    args = ["--module", "torchtitan_npu_recipes.qwen_sft", "--config", "run_qwen3_5_4k"]

    def config_with(*options):
        with sft_launch_args([*args, *options]) as native:
            return ConfigLoader().load(native)

    with pytest.raises(SystemExit, match="2"):
        config_with("--smoke-test", "--resume-step=2")
    logger = tmp_path / "structured_logger"
    logger.mkdir()
    (logger / "rank0.jsonl").touch()
    (tmp_path / "checkpoint" / "step-2").mkdir(parents=True)
    config = config_with("--smoke-test", "--output-dir", str(tmp_path))
    assert config.training.steps == 1 and config.checkpointer.load_only
    (tmp_path / "checkpoint" / "step-2" / ".metadata").touch()
    with pytest.raises(ValueError, match="found an existing checkpoint"):
        config_with("--smoke-test", "--output-dir", str(tmp_path))
    with pytest.raises(ValueError, match="must exceed resume step"):
        config_with("--resume-step", "2", "--max-steps", "2")
    config = config_with("--resume-step", "2", "--max-steps", "3")
    assert config.checkpointer.load_step == 2 and config.training.steps == 3
    assert config.optim.lr_scheduler.total_steps == 16
    args[-1] = "qwen3_5_27b_sft_64k_cp8"
    truncated = config_with("--max-steps", "2")
    assert truncated.checkpointer.target_steps == 2
    assert not truncated.checkpointer.last_save_in_hf and not truncated.checkpointer.last_save_model_only
    owner = truncated.checkpointer._owner
    checkpointer = owner.__new__(owner)
    checkpointer.target_steps = 2
    checkpointer.states = {"train_state": SimpleNamespace(num_completed_steps=2)}
    monkeypatch.setattr(CheckpointManager, "load", lambda *_: True)
    with pytest.raises(ValueError, match="must exceed restored step 2"):
        checkpointer.load()
