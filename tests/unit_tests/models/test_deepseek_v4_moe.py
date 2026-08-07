# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU behavior tests for DeepSeek-V4 routing and the MoE wrapper."""

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
NPU_ROOT = REPO_ROOT / "torchtitan_npu"


@dataclass(kw_only=True, slots=True)
class _GateConfig:
    in_features: int
    out_features: int
    bias: bool = False

    def build(self):
        return torch.nn.Linear(
            self.in_features, self.out_features, bias=self.bias
        )


class _TokenChoiceTopKRouter(torch.nn.Module):
    @dataclass(kw_only=True, slots=True)
    class Config:
        num_experts: int
        gate: _GateConfig
        top_k: int = 1
        score_func: str = "sigmoid"
        route_norm: bool = False
        route_scale: float = 1.0
        _debug_force_load_balance: bool = False

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.gate = config.gate.build()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self._debug_force_load_balance = config._debug_force_load_balance

    def _debug_force_load_balance_routing(self, scores):
        batch, length, _ = scores.shape
        selected = (
            torch.arange(batch * length * self.top_k)
            .reshape(batch, length, self.top_k)
            .remainder(self.num_experts)
        )
        return selected, scores.gather(-1, selected)


class _MoE(torch.nn.Module):
    @dataclass(kw_only=True, slots=True)
    class Config:
        pass


class _DeepEPTokenDispatcher:
    pass


class _TransformerBlock(torch.nn.Module):
    @dataclass(kw_only=True, slots=True)
    class Config:
        attention: object
        attention_norm: object
        ffn_norm: object
        feed_forward: object | None = None
        moe: object | None = None

    def __init__(self) -> None:
        super().__init__()
        self._param_init = {}


class _Decoder(torch.nn.Module):
    @dataclass(kw_only=True, slots=True)
    class Config:
        pass


class _VarlenAttention:
    @dataclass(kw_only=True, slots=True)
    class Config:
        pass


class _BaseMaskHandler:
    @dataclass(kw_only=True, slots=True)
    class Config:
        def build(self):
            return _BaseMaskHandler()


class _Component:
    @dataclass(kw_only=True, slots=True)
    class Config:
        pass


class _Builds:
    def __init__(self, module) -> None:
        self.module = module

    def build(self):
        return self.module


class _PassthroughAttention(torch.nn.Module):
    def forward(self, x, attention_masks=None, positions=None):
        del attention_masks, positions
        return x


class _PassthroughHcPre(torch.nn.Module):
    def forward(self, x, hc_fn, hc_scale, hc_base):
        del hc_fn, hc_scale, hc_base
        return x, None, None


class _PassthroughHcPost(torch.nn.Module):
    def forward(self, x, residual, post, comb):
        del residual, post, comb
        return x


class _RecordingBlockMoE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.received_input_ids = None

    def forward(self, x, *, input_ids):
        self.received_input_ids = input_ids
        return x


def _install_package(monkeypatch: pytest.MonkeyPatch, name: str, path: Path) -> None:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    monkeypatch.setitem(sys.modules, name, package)


def _load_module(monkeypatch: pytest.MonkeyPatch, name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _install_torchtitan_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "torchtitan",
        "torchtitan.models",
        "torchtitan.models.common",
    ):
        _install_package(monkeypatch, name, REPO_ROOT)

    moe = types.ModuleType("torchtitan.models.common.moe")
    moe.MoE = _MoE
    moe.TokenChoiceTopKRouter = _TokenChoiceTopKRouter
    monkeypatch.setitem(sys.modules, "torchtitan.models.common.moe", moe)

    token_dispatcher = types.ModuleType(
        "torchtitan.models.common.token_dispatcher"
    )
    token_dispatcher.DeepEPTokenDispatcher = _DeepEPTokenDispatcher
    monkeypatch.setitem(
        sys.modules,
        "torchtitan.models.common.token_dispatcher",
        token_dispatcher,
    )


def _install_model_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    attention = types.ModuleType("torchtitan.models.common.attention")
    attention.AttentionMasksType = object
    attention.VarlenAttention = _VarlenAttention
    monkeypatch.setitem(sys.modules, "torchtitan.models.common.attention", attention)

    decoder = types.ModuleType("torchtitan.models.common.decoder")
    decoder.Decoder = _Decoder
    decoder.TransformerBlock = _TransformerBlock
    monkeypatch.setitem(sys.modules, "torchtitan.models.common.decoder", decoder)

    mask_handler = types.ModuleType(
        "torchtitan_npu.patches.torchtitan.models.common.mask_handler"
    )
    mask_handler.BaseMaskHandler = _BaseMaskHandler
    monkeypatch.setitem(
        sys.modules,
        "torchtitan_npu.patches.torchtitan.models.common.mask_handler",
        mask_handler,
    )

    mhc = types.ModuleType("torchtitan_npu.models.deepseek_v4.mhc")
    mhc.HcHead = _Component
    mhc.HcPost = _Component
    mhc.HcPre = _Component
    monkeypatch.setitem(sys.modules, "torchtitan_npu.models.deepseek_v4.mhc", mhc)

    packed = types.ModuleType("torchtitan_npu.models.deepseek_v4.packed")
    packed.build_dsv4_packed_metadata = lambda *args, **kwargs: (args, kwargs)
    monkeypatch.setitem(
        sys.modules, "torchtitan_npu.models.deepseek_v4.packed", packed
    )


@pytest.fixture(scope="module")
def moe_module():
    monkeypatch = pytest.MonkeyPatch()
    _install_torchtitan_stubs(monkeypatch)
    package_paths = {
        "torchtitan_npu": NPU_ROOT,
        "torchtitan_npu.models": NPU_ROOT / "models",
        "torchtitan_npu.models.deepseek_v4": NPU_ROOT / "models" / "deepseek_v4",
        "torchtitan_npu.patches": NPU_ROOT / "patches",
        "torchtitan_npu.patches.torchtitan": NPU_ROOT / "patches" / "torchtitan",
        "torchtitan_npu.patches.torchtitan.models": NPU_ROOT / "patches" / "torchtitan" / "models",
        "torchtitan_npu.patches.torchtitan.models.common": (
            NPU_ROOT / "patches" / "torchtitan" / "models" / "common"
        ),
    }
    for name, path in package_paths.items():
        _install_package(monkeypatch, name, path)

    _load_module(
        monkeypatch,
        "torchtitan_npu.patches.torchtitan.models.common.moe",
        NPU_ROOT / "patches" / "torchtitan" / "models" / "common" / "moe.py",
    )
    module = _load_module(
        monkeypatch,
        "torchtitan_npu.models.deepseek_v4.moe",
        NPU_ROOT / "models" / "deepseek_v4" / "moe.py",
    )
    yield module
    monkeypatch.undo()


@pytest.fixture(scope="module")
def model_module(moe_module):
    del moe_module
    monkeypatch = pytest.MonkeyPatch()
    _install_model_stubs(monkeypatch)
    module = _load_module(
        monkeypatch,
        "torchtitan_npu.models.deepseek_v4._test_model_runtime",
        NPU_ROOT / "models" / "deepseek_v4" / "model.py",
    )
    yield module
    monkeypatch.undo()


def _build_router(moe_module, *, hash_layer: bool = True):
    config = moe_module.DeepSeekV4Router.Config(
        num_experts=4,
        gate=_GateConfig(in_features=3, out_features=4, bias=False),
        top_k=2,
        score_func="sigmoid",
        route_norm=False,
        route_scale=1.0,
        vocab_size=6,
        n_hash_layers=1,
        layer_id=0 if hash_layer else 1,
    )
    return moe_module.DeepSeekV4Router(config)


def _build_transformer_block(model_module, *, moe=None, feed_forward=None):
    config = model_module.DeepSeekV4TransformerBlock.Config(
        dim=2,
        attention=_Builds(_PassthroughAttention()),
        attention_norm=_Builds(torch.nn.Identity()),
        ffn_norm=_Builds(torch.nn.Identity()),
        feed_forward=None if feed_forward is None else _Builds(feed_forward),
        moe=None if moe is None else _Builds(moe),
        hc_pre=_Builds(_PassthroughHcPre()),
        hc_post=_Builds(_PassthroughHcPost()),
    )
    return model_module.DeepSeekV4TransformerBlock(config)


@pytest.mark.cpu
def test_hash_routing_table_is_valid_and_persisted(moe_module):
    with pytest.raises(ValueError, match="top_k .* must be <= num_experts"):
        moe_module._build_hash_routing_table(4, 2, 3)

    router = _build_router(moe_module)

    assert router.tid2eid.shape == (6, 2)
    assert router.tid2eid.dtype == torch.int64
    assert torch.all((router.tid2eid >= 0) & (router.tid2eid < 4))
    assert torch.all(router.tid2eid[:, 0] != router.tid2eid[:, 1])
    assert torch.equal(router.state_dict()["tid2eid"], router.tid2eid)


@pytest.mark.cpu
def test_transformer_block_exposes_moe_marker_used_by_fsdp(model_module):
    moe = _RecordingBlockMoE()
    moe_block = _build_transformer_block(model_module, moe=moe)
    dense_block = _build_transformer_block(
        model_module, feed_forward=torch.nn.Identity()
    )

    assert moe_block.moe_enabled is True
    assert dense_block.moe_enabled is False

    x = torch.randn(1, 2, 4, 2)
    input_ids = torch.tensor([[0, 1]])
    output = moe_block(x, input_ids, attention_masks=None)

    assert moe.received_input_ids is input_ids
    torch.testing.assert_close(output, x)


@pytest.mark.cpu
def test_hash_router_requires_input_ids_and_returns_selected_scores(moe_module):
    router = _build_router(moe_module)
    routing_table = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]], dtype=torch.int64
    )
    with torch.no_grad():
        router.gate.weight.zero_()
        router.tid2eid.copy_(routing_table)

    x = torch.zeros(1, 3, 3)
    input_ids = torch.tensor([[0, 3, 5]])
    with pytest.raises(ValueError, match="input_ids is required"):
        router(x)

    top_scores, selected_experts, scores = router(x, input_ids=input_ids)

    expected_experts = routing_table[input_ids]
    assert torch.equal(selected_experts, expected_experts)
    torch.testing.assert_close(scores, torch.full((1, 3, 4), 0.5))
    torch.testing.assert_close(top_scores, scores.gather(-1, expected_experts))


class _RecordingRouter(torch.nn.Module):
    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.received_bias = None
        self.received_input_ids = None

    def forward(self, x, expert_bias, *, input_ids):
        self.received_bias = expert_bias
        self.received_input_ids = input_ids
        selected = input_ids.remainder(self.num_experts).unsqueeze(-1)
        scores = x.new_zeros((*x.shape[:2], self.num_experts))
        top_scores = x.new_ones((*x.shape[:2], 1))
        return top_scores, selected, scores


class _RecordingRoutedExperts(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_dispatcher = types.SimpleNamespace(sp_size=1)
        self.received_counts = None
        self.received_num_local_tokens = None

    def forward(
        self,
        x,
        topk_scores,
        topk_expert_ids,
        tokens_per_expert,
        *,
        num_local_tokens_after_seq_dim_padding,
    ):
        self.received_counts = tokens_per_expert
        self.received_num_local_tokens = num_local_tokens_after_seq_dim_padding
        return x + 1


@pytest.mark.cpu
def test_moe_forwards_router_kwargs_and_accumulates_expert_counts(moe_module):
    moe = moe_module.DeepSeekV4MoE.__new__(moe_module.DeepSeekV4MoE)
    torch.nn.Module.__init__(moe)
    moe.seq_dim_tp_sharded = False
    moe.router = _RecordingRouter(num_experts=3)
    moe.routed_experts = _RecordingRoutedExperts()
    moe.shared_experts = None
    moe.register_buffer("expert_bias_E", torch.tensor([0.1, 0.2, 0.3]))
    moe.register_buffer(
        "tokens_per_expert_E", torch.zeros(3), persistent=False
    )

    x = torch.randn(1, 4, 2)
    input_ids = torch.tensor([[0, 1, 1, 2]])
    output = moe(x, input_ids=input_ids)

    assert moe.router.received_input_ids is input_ids
    assert moe.router.received_bias is moe.expert_bias_E
    assert torch.equal(moe.tokens_per_expert_E, torch.tensor([1.0, 2.0, 1.0]))
    assert torch.equal(
        moe.routed_experts.received_counts, torch.tensor([1, 2, 1])
    )
    assert moe.routed_experts.received_num_local_tokens == 4
    torch.testing.assert_close(output, x + 1)
