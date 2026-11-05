# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Same-weight parity against real TorchTitan modules, without global NPU patches.

The namespace fixture bypasses only plugin package initialization. Numerical
attention, decoder, MoE, normalization, RoPE and state adapters are real source
modules. These CPU tests do not establish NPU or distributed integration.
"""

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from legacy_state_dict import from_legacy_state_dict, to_legacy_state_dict
from raw_reference import MODEL_CONTRACT_VERSION, BlockDiffusionModel, ModelConfig, TrainingSpec


@pytest.fixture(scope="module")
def native_api():
    try:
        importlib.import_module("torchtitan.models.common")
    except ImportError as error:
        pytest.skip(f"real TorchTitan CPU dependencies unavailable: {error}")
    root = Path(__file__).resolve().parents[4]
    initial_modules = set(sys.modules)
    patch = pytest.MonkeyPatch()
    if "torchtitan_npu" not in sys.modules:
        namespace = types.ModuleType("torchtitan_npu")
        namespace.__path__ = [str(root / "torchtitan_npu")]
        namespace.__spec__ = importlib.machinery.ModuleSpec("torchtitan_npu", loader=None, is_package=True)
        patch.setitem(sys.modules, "torchtitan_npu", namespace)
    native = importlib.import_module("torchtitan_npu.models.block_diffusion")
    attention = importlib.import_module("torchtitan_npu.models.block_diffusion.attention")
    state = importlib.import_module("torchtitan_npu.models.block_diffusion.state_dict_adapter")
    model = importlib.import_module("torchtitan_npu.models.block_diffusion.model")
    yield SimpleNamespace(registry=native, attention=attention, state=state, model=model)
    for name in tuple(sys.modules):
        if name.startswith("torchtitan_npu.") and name not in initial_modules:
            sys.modules.pop(name, None)
    patch.undo()


def raw_config(moe=False):
    return ModelConfig(
        vocab_size=17,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=12,
        num_experts=4 if moe else 0,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        num_shared_experts=1 if moe else 0,
        dtype=torch.float32,
    )


def native_config(api, cfg, policy="block_causal", mode="exact"):
    return api.registry.make_block_diffusion_config(
        vocab_size=cfg.vocab_size,
        dim=cfg.hidden_size,
        n_layers=cfg.num_hidden_layers,
        n_heads=cfg.num_attention_heads,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        num_experts=cfg.num_experts,
        moe_intermediate_size=cfg.moe_intermediate_size,
        num_experts_per_tok=cfg.num_experts_per_tok,
        block_size=2,
        attention_compute_alpha=1.0,
        mask_token_id=0,
        max_seq_len=32,
        use_grouped_mm=False,
        attention_mode=mode,
        attention_policy=policy,
    )


def visibility(length, block, policy):
    return torch.tensor(
        [
            [
                key // block <= query // block
                if policy == "block_causal"
                else (key <= query if query < length - block else True)
                for key in range(length)
            ]
            for query in range(length)
        ],
        dtype=torch.bool,
    )


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("policy", ["block_causal", "prefix_causal_canvas_bidirectional"])
def test_native_exact_matches_raw_weights_logits_loss_and_gradients(native_api, moe, policy):
    from torchtitan.components.loss import cross_entropy_loss

    torch.manual_seed(515)
    cfg = raw_config(moe)
    raw = BlockDiffusionModel(cfg)
    native_cfg = native_config(native_api, cfg, policy)
    native = native_cfg.build()
    native.init_states(buffer_device=torch.device("cpu"))
    adapter = native_api.state.BlockDiffusionStateDictAdapter(native_cfg)
    converted = adapter.from_hf(to_legacy_state_dict(raw.state_dict()))
    loaded = native.load_state_dict(converted, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    assert native_cfg.raw_contract_version == MODEL_CONTRACT_VERSION == 1
    assert native_cfg.rope.backend == "cos_sin"
    assert native_cfg.rope.scaling == "none"
    if moe:
        assert all(layer.moe.expert_bias is None for layer in native.layers.values())
    tokens = torch.tensor([[1, 0, 3, 4, 0, 6], [2, 3, 0, 4, 5, 0]])
    labels = torch.tensor([[-100, 2, -100, -100, 5, -100], [-100, -100, 4, -100, -100, 6]])
    raw_result = raw.forward_training(tokens, labels, spec=TrainingSpec(block_size=2, attention_policy=policy))
    native_logits = native(tokens)
    native_loss_sum = cross_entropy_loss(native_logits, labels)
    native_loss = native_loss_sum / (labels != -100).sum()
    torch.testing.assert_close(native_logits, raw_result.logits, atol=4e-6, rtol=3e-5)
    torch.testing.assert_close(native_loss_sum, raw_result.loss_sum, atol=4e-6, rtol=2e-5)
    torch.testing.assert_close(native_loss, raw_result.loss, atol=2e-6, rtol=2e-5)
    native_loss.backward()
    raw_result.loss.backward()
    native_gradients = {
        name: parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
        for name, parameter in native.named_parameters()
    }
    mapped_gradients = from_legacy_state_dict(adapter.to_hf(native_gradients))
    assert set(mapped_gradients) == {name for name, _ in raw.named_parameters()}
    for name, parameter in raw.named_parameters():
        expected = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
        torch.testing.assert_close(mapped_gradients[name], expected, atol=5e-6, rtol=5e-4, msg=name)


@pytest.mark.parametrize("policy", ["block_causal", "prefix_causal_canvas_bidirectional"])
def test_native_exact_attention_matches_full_mask_oracle(native_api, policy):
    attention = native_api.attention.ScaledCausalSDPA.Config(
        attention_mode="exact",
        attention_policy=policy,
        block_size=2,
        compute_alpha=1.0,
    ).build()
    q = torch.randn(2, 6, 4, 4, requires_grad=True)
    k = torch.randn(2, 6, 2, 4, requires_grad=True)
    v = torch.randn(2, 6, 2, 4, requires_grad=True)
    actual = attention(q, k, v, enable_gqa=True)
    mask = visibility(6, 2, policy)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask,
        enable_gqa=True,
    ).transpose(1, 2)
    gradient = torch.randn_like(actual)
    result_grads = torch.autograd.grad(actual, (q, k, v), gradient, retain_graph=True)
    reference_grads = torch.autograd.grad(expected, (q, k, v), gradient)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for result, reference in zip(result_grads, reference_grads, strict=False):
        torch.testing.assert_close(result, reference, atol=3e-6, rtol=3e-5)


def test_compute_proxy_preserves_old_causal_attention_and_rope(native_api):
    attention = native_api.attention.ScaledCausalSDPA.Config(compute_alpha=0.25).build()
    q, k, v = [torch.randn(1, 6, 2, 4) for _ in range(3)]
    actual = attention(q, k, v)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    old = native_api.registry.model_registry("dense_debug").model
    exact = native_api.registry.model_registry("dense_debug_exact").model
    assert old.attention_mode == "compute_proxy"
    assert old.rope.backend == "complex" and old.rope.scaling == "llama"
    assert exact.attention_mode == "exact"
    assert exact.rope.backend == "cos_sin" and exact.rope.scaling == "none"


def test_exact_configuration_rejects_alpha_scaling_and_unsupported_cp(native_api):
    with pytest.raises(ValueError):
        native_api.attention.ScaledCausalSDPA.Config(attention_mode="exact", block_size=2, compute_alpha=0.75)
    cfg = native_config(native_api, raw_config())
    trainer = SimpleNamespace(
        training=SimpleNamespace(seq_len=6),
        parallelism=SimpleNamespace(
            tensor_parallel_degree=1,
            context_parallel_degree=2,
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
        ),
    )
    with pytest.raises(ValueError, match=r"[Cc][Pp]|context.parallel"):
        cfg.update_from_config(trainer_config=trainer)


def test_canonical_raw_directory_contains_only_contract_files():
    root = Path(__file__).resolve().parents[4]
    raw = root / "torchtitan_npu/simulator/raw_model/block_diff"
    files = {path.name for path in raw.iterdir() if path.is_file()}
    assert files == {"README.md", "model.py"}


@pytest.mark.parametrize("conflict", ["proxy_to_exact", "rope_backend", "rope_scaling"])
def test_exact_builder_rejects_incoherent_top_level_overrides(native_api, conflict):
    if conflict == "proxy_to_exact":
        cfg = native_api.registry.model_registry("debug").model
        cfg.attention_mode = "exact"
        cfg.attention_compute_alpha = 1.0
    else:
        cfg = native_config(native_api, raw_config(moe=True))
        if conflict == "rope_backend":
            cfg.rope.backend = "complex"
        else:
            cfg.rope.scaling = "llama"
    with pytest.raises(ValueError):
        cfg.build()


def test_exact_dataset_positions_restart_per_window(native_api, monkeypatch):
    from datasets import Dataset
    from torchtitan.hf_datasets import text_datasets

    data = importlib.import_module("torchtitan_npu.models.block_diffusion.data")
    samples = Dataset.from_dict({"text": ["1 2 3 4 5 6 7 8"]})
    monkeypatch.setattr(
        text_datasets, "_validate_dataset", lambda *_: ("local", lambda _: samples, lambda sample: sample["text"])
    )
    tokenizer = SimpleNamespace(encode=lambda text, **_: [int(token) for token in text.split()])
    kwargs = dict(
        dataset_name="local",
        dataset_path=None,
        tokenizer=tokenizer,
        seq_len=4,
        dp_rank=0,
        dp_world_size=1,
        infinite=False,
        block_size=2,
        mask_token_id=0,
        corruption_seed=12,
        min_mask_ratio=0.5,
        max_mask_ratio=0.5,
    )
    window = iter(data.BlockDiffusionDataset(**kwargs, position_policy="window"))
    document = iter(data.BlockDiffusionDataset(**kwargs, position_policy="document"))
    for _ in range(2):
        window_input, _labels = next(window)
        assert window_input["positions"].tolist() == [0, 1, 2, 3]
    next(document)
    document_input, _labels = next(document)
    assert document_input["positions"].tolist() == [2, 3, 0, 1]


@pytest.fixture(scope="module")
def npu_modules(native_api):
    """Load real converters, substituting only unavailable hardware entry points."""
    root = Path(__file__).resolve().parents[4]
    patch = pytest.MonkeyPatch()
    if "torchtitan_npu.converters" not in sys.modules:
        namespace = types.ModuleType("torchtitan_npu.converters")
        namespace.__path__ = [str(root / "torchtitan_npu/converters")]
        namespace.__spec__ = importlib.machinery.ModuleSpec("torchtitan_npu.converters", loader=None, is_package=True)
        patch.setitem(sys.modules, "torchtitan_npu.converters", namespace)
    hardware = sys.modules.get("torch_npu")
    if hardware is None:
        hardware = types.ModuleType("torch_npu")
        hardware.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
        hardware.npu_fusion_attention = None
        hardware.npu_fusion_attention_grad = None
        patch.setitem(sys.modules, "torch_npu", hardware)
    from torch.distributed.tensor import parallel

    original_parallelize = parallel.parallelize_module
    kernel = importlib.import_module("torchtitan_npu.converters.kernels.block_diffusion_attention")
    shim = importlib.import_module("torchtitan_npu.simulator.hardware_shims.block_diffusion_attention")
    capture = importlib.import_module("torchtitan_npu.simulator.capture.dispatch_capture")
    yield SimpleNamespace(kernel=kernel, shim=shim, capture=capture, hardware=kernel.torch_npu)
    parallel.parallelize_module = original_parallelize
    patch.undo()


def _hardware_oracle(q, k, v, *, head_num, sparse_mode, scale):
    batch, query_length, query_width = q.shape
    key_length = k.shape[1]
    head_dim = query_width // head_num
    kv_heads = k.shape[-1] // head_dim
    return (
        F.scaled_dot_product_attention(
            q.reshape(batch, query_length, head_num, head_dim).transpose(1, 2),
            k.reshape(batch, key_length, kv_heads, head_dim).transpose(1, 2),
            v.reshape(batch, key_length, kv_heads, head_dim).transpose(1, 2),
            is_causal=sparse_mode == 2,
            enable_gqa=head_num != kv_heads,
            scale=scale,
        )
        .transpose(1, 2)
        .reshape(batch, query_length, query_width)
    )


@pytest.mark.parametrize(
    "mode,policy,segments",
    [
        ("exact", "block_causal", ((2, 2, False), (2, 4, False), (2, 6, False))),
        ("exact", "prefix_causal_canvas_bidirectional", ((4, 4, True), (2, 6, False))),
        ("compute_proxy", "block_causal", ((6, 6, True),)),
    ],
)
def test_npu_converter_boundary_and_gradients_match_visibility(
    native_api, npu_modules, monkeypatch, mode, policy, segments
):
    forward_calls, backward_calls = [], []

    def fusion(q, k, v, **options):
        forward_calls.append((q.shape[1], k.shape[1], options))
        result = _hardware_oracle(
            q, k, v, head_num=options["head_num"], sparse_mode=options["sparse_mode"], scale=options["scale"]
        )
        statistics = torch.zeros((q.shape[0], options["head_num"], q.shape[1], 8))
        return result, statistics, statistics, torch.empty(0), 0, 0, 0

    def fusion_grad(q, k, v, gradient, **options):
        backward_calls.append((q.shape[1], k.shape[1], options))
        with torch.enable_grad():
            local = [tensor.detach().clone().requires_grad_() for tensor in (q, k, v)]
            result = _hardware_oracle(
                *local, head_num=options["head_num"], sparse_mode=options["sparse_mode"], scale=options["scale_value"]
            )
            grads = torch.autograd.grad(result, local, gradient)
        return *grads, None, None

    monkeypatch.setattr(npu_modules.hardware, "npu_fusion_attention", fusion)
    monkeypatch.setattr(npu_modules.hardware, "npu_fusion_attention_grad", fusion_grad)
    holder = torch.nn.Module()
    holder.register_parameter("weight", torch.nn.Parameter(torch.ones(1)))
    holder.attention = native_api.attention.ScaledCausalSDPA.Config(
        attention_mode=mode,
        attention_policy=policy,
        block_size=2,
        compute_alpha=1.0 if mode == "exact" else 0.75,
    ).build()
    npu_modules.kernel.NPUBlockDiffusionAttentionConverter(SimpleNamespace(name="block_diffusion")).convert(holder)
    q = torch.randn(1, 6, 4, 4, requires_grad=True)
    k = torch.randn(1, 6, 2, 4, requires_grad=True)
    v = torch.randn(1, 6, 2, 4, requires_grad=True)
    output = holder.attention(q, k, v, enable_gqa=True)
    mask = visibility(6, 2, policy) if mode == "exact" else torch.ones((6, 6), dtype=torch.bool).tril()
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, enable_gqa=True
    ).transpose(1, 2)
    upstream = torch.randn_like(output)
    actual_grads = torch.autograd.grad(output, (q, k, v), upstream, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (q, k, v), upstream)
    torch.testing.assert_close(output, expected, atol=3e-6, rtol=2e-5)
    for result, reference in zip(actual_grads, expected_grads, strict=False):
        torch.testing.assert_close(result, reference, atol=4e-6, rtol=4e-5)
    assert [(query, key, options["sparse_mode"] == 2) for query, key, options in forward_calls] == list(segments)
    assert len(backward_calls) == len(segments)
    for _query, _key, options in forward_calls + backward_calls:
        causal = options["sparse_mode"] == 2
        assert options["input_layout"] == "BSH"
        assert options["head_num"] == 4
        assert options["next_tockens"] == (0 if causal else 2_147_483_647)
        assert (options["atten_mask"] is not None) is causal
        assert options["keep_prob"] == 1.0
    assert holder.attention.attention_mode == mode
    assert holder.attention.attention_policy == policy


@pytest.mark.parametrize(
    "policy,physical_keys,logical_pairs",
    [
        ("block_causal", (2, 4, 6), 24),
        ("prefix_causal_canvas_bidirectional", (4, 6), 22),
    ],
)
@pytest.mark.parametrize("kv_heads", [2, 4])
def test_sim_exact_ledger_preserves_segments_and_backward_shapes(
    native_api, npu_modules, policy, physical_keys, logical_pairs, kv_heads
):
    holder = torch.nn.Module()
    holder.attention = native_api.attention.ScaledCausalSDPA.Config(
        attention_mode="exact",
        attention_policy=policy,
        block_size=2,
        compute_alpha=1.0,
    ).build()
    npu_modules.shim.SimBlockDiffusionAttentionConverter(SimpleNamespace(name="block_diffusion")).convert(holder)
    q = torch.empty((1, 6, 4, 4), device="meta", requires_grad=True)
    k = torch.empty((1, 6, kv_heads, 4), device="meta", requires_grad=True)
    v = torch.empty((1, 6, kv_heads, 4), device="meta", requires_grad=True)
    phase = {"value": "forward"}
    capture = npu_modules.capture.OpDispatchCapture(phase_provider=lambda: phase["value"])
    with capture:
        output = holder.attention(q, k, v, enable_gqa=True)
        phase["value"] = "backward"
        output.sum().backward()
    nodes = [
        node
        for node in capture.build_nodes().values()
        if node.attrs.get("decomposed_from") == "npu.npu_fusion_attention.default"
    ]
    forward = [node for node in nodes if node.annotations["comp_type"] == "F"]
    backward = [node for node in nodes if node.annotations["comp_type"] == "B"]
    core_forward = [node for node in forward if node.op_type in ("matmul", "softmax")]
    assert len(core_forward) == 3 * len(physical_keys)
    for node in nodes:
        if node.op_type != "matmul":
            continue
        left, right = node.inputs[:2]
        expected_batch = tuple(torch.broadcast_shapes(left.shape[:-2], right.shape[:-2]))
        expected_shape = (*expected_batch, left.shape[-2], right.shape[-1])
        assert left.shape[-1] == right.shape[-2]
        assert tuple(node.outputs[0].shape) == expected_shape, (node.op_id, node.annotations["comp_type"])
    core_backward = [node for node in backward if node.op_type in ("matmul", "softmax")]
    assert len(core_backward) == 5 * len(physical_keys)
    head_reductions = [node for node in backward if node.attrs.get("reduction") == "gqa_query_head_groups"]
    assert len(head_reductions) == (2 * len(physical_keys) if kv_heads == 2 else 0)
    for offset in range(0, len(core_backward), 5):
        # dV and dK must compute all query-head contributions before reducing GQA groups.
        for node in (core_backward[offset], core_backward[offset + 4]):
            assert node.outputs[0].shape[0] == 4
            assert node.outputs[0].shape[1] == node.attrs["kv_seq_len"]
            assert node.flops == 2 * 4 * node.attrs["q_seq_len"] * node.attrs["kv_seq_len"] * 4
    for node in head_reductions:
        assert node.inputs[0].shape[1:3] == (2, 2)
        assert node.outputs[0].shape[1] == 2
    qk_nodes = core_forward[::3]
    assert tuple(node.attrs["effective_kv_seq_len"] for node in qk_nodes) == physical_keys
    assert sum(node.attrs["visible_pairs_per_head"] for node in qk_nodes) == logical_pairs
    assert all(node.attrs["attention_mode"] == "exact" for node in nodes)
    for node in qk_nodes:
        assert node.inputs[1].shape[-1] == node.attrs["kv_seq_len"]
        assert node.outputs[0].shape[-1] == node.attrs["kv_seq_len"]
    assert output.shape == q.shape
    assert all(tensor.grad is not None and tensor.grad.shape == tensor.shape for tensor in (q, k, v))


def test_sim_compute_proxy_keeps_existing_dimension_scaling(native_api, npu_modules):
    attention = npu_modules.shim.SimBlockDiffusionAttention(
        native_api.attention.ScaledCausalSDPA.Config(compute_alpha=0.75)
    )
    q = torch.empty((1, 8, 2, 4), device="meta", requires_grad=True)
    capture = npu_modules.capture.OpDispatchCapture()
    with capture:
        attention(q, q, q)
    nodes = [
        node
        for node in capture.build_nodes().values()
        if node.attrs.get("decomposed_from") == "npu.npu_fusion_attention.default"
    ]
    assert len(nodes) == 3
    assert nodes[0].attrs["effective_kv_seq_len"] == 3
    assert nodes[0].attrs["compute_alpha"] == 0.75
    assert nodes[0].attrs["attention_mode"] == "compute_proxy"


@pytest.mark.parametrize("moe", [False, True])
def test_native_mfu_metric_is_per_token_and_sequence_only_affects_attention(native_api, moe):
    cfg = native_config(native_api, raw_config(moe))
    model = cfg.build()
    actual_parameters, six_token_metric = cfg.get_nparams_and_flops(model, seq_len=6)
    _, twelve_token_metric = cfg.get_nparams_and_flops(model, seq_len=12)
    assert actual_parameters == sum(parameter.numel() for parameter in model.parameters())
    dim, layers, head_dim = cfg.dim, cfg.n_layers, cfg.head_dim
    attention_parameters = layers * 2 * dim * head_dim * (cfg.n_heads + cfg.n_kv_heads)
    norms = (2 * layers + 1) * dim
    output = cfg.vocab_size * dim
    if moe:
        router = layers * dim * cfg.num_experts
        active_experts = layers * 3 * dim * cfg.moe_intermediate_size * cfg.num_experts_per_tok
        shared = layers * 3 * dim * cfg.intermediate_size
        active_nonembedding = output + norms + attention_parameters + router + active_experts + shared
    else:
        active_nonembedding = output + norms + attention_parameters + layers * 3 * dim * cfg.intermediate_size
    # TorchTitan's documented MFU convention counts 6 per active non-embedding
    # parameter and a dense attention term; it is not the exact segmented ledger.
    expected = 6 * active_nonembedding + 12 * layers * cfg.n_heads * head_dim * 6
    assert six_token_metric == expected
    assert twelve_token_metric - six_token_metric == 12 * layers * cfg.n_heads * head_dim * 6


@pytest.mark.parametrize("moe", [False, True])
def test_native_mfu_tied_weights_reduce_residency_not_output_compute(native_api, moe):
    untied_cfg = native_config(native_api, raw_config(moe))
    tied_cfg = native_config(native_api, raw_config(moe))
    tied_cfg.enable_weight_tying = True
    untied = untied_cfg.build()
    tied = tied_cfg.build()
    plain_parameters, plain_flops = untied_cfg.get_nparams_and_flops(untied, seq_len=6)
    tied_parameters, tied_flops = tied_cfg.get_nparams_and_flops(tied, seq_len=6)
    assert plain_parameters - tied_parameters == tied_cfg.vocab_size * tied_cfg.dim
    assert tied_flops == plain_flops


def test_native_mfu_extra_inactive_experts_only_add_router_work(native_api):
    smaller = native_config(native_api, raw_config(moe=True))
    bigger_raw = ModelConfig(
        vocab_size=17,
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=12,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        num_shared_experts=1,
    )
    bigger = native_config(native_api, bigger_raw)
    small_parameters, small_flops = smaller.get_nparams_and_flops(smaller.build(), seq_len=6)
    big_parameters, big_flops = bigger.get_nparams_and_flops(bigger.build(), seq_len=6)
    extra_experts = bigger.num_experts - smaller.num_experts
    assert big_parameters > small_parameters
    assert big_flops - small_flops == 6 * smaller.n_layers * smaller.dim * extra_experts


@pytest.fixture(scope="module")
def moe_dispatch_api(native_api, npu_modules):
    """Import the real shared gate helper without unrelated model registration."""
    root = Path(__file__).resolve().parents[4]
    patch = pytest.MonkeyPatch()
    package = "torchtitan_npu.models.deepseek_v4"
    if package not in sys.modules:
        namespace = types.ModuleType(package)
        namespace.__path__ = [str(root / "torchtitan_npu/models/deepseek_v4")]
        namespace.__spec__ = importlib.machinery.ModuleSpec(package, loader=None, is_package=True)
        patch.setitem(sys.modules, package, namespace)
    module = importlib.import_module("torchtitan_npu.converters.kernels.moe_dispatch")
    yield module
    patch.undo()


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize(
    "activation_dtype,weight_dtype",
    [
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_actual_moe_dispatch_gate_matches_weight_dtype_and_gradients(
    moe_dispatch_api, device, activation_dtype, weight_dtype
):
    with torch.device(device):
        gate = torch.nn.Linear(8, 4, bias=True, dtype=weight_dtype)
        inputs = torch.randn((3, 8), dtype=activation_dtype, requires_grad=True)
    actual = moe_dispatch_api._local_gate_scores(SimpleNamespace(gate=gate), inputs)
    expected = F.linear(inputs.to(weight_dtype), gate.weight, gate.bias)
    assert actual.dtype == weight_dtype
    assert actual.shape == (3, 4)
    actual_gradients = torch.autograd.grad(actual.sum(), (inputs, gate.weight, gate.bias), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected.sum(), (inputs, gate.weight, gate.bias))
    assert actual_gradients[0].dtype == activation_dtype
    assert actual_gradients[1].dtype == actual_gradients[2].dtype == weight_dtype
    if device == "cpu":
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for gradient, reference in zip(actual_gradients, expected_gradients, strict=False):
            assert gradient.isfinite().all()
            torch.testing.assert_close(gradient, reference, atol=0, rtol=0)
    else:
        assert all(gradient.device.type == "meta" for gradient in actual_gradients)


def test_exact_recipe_selects_real_moe_dispatch_registration(native_api, npu_modules, moe_dispatch_api, monkeypatch):
    registry = importlib.import_module("torchtitan_npu.converters.registry")
    converters = sys.modules["torchtitan_npu.converters"]
    monkeypatch.setattr(converters, "get_model_converter_config", registry.get_model_converter_config, raising=False)
    recipes = importlib.import_module("torchtitan_npu.models.block_diffusion.config_registry")
    recipe = recipes.block_diffusion_exact_smoketest()
    assert recipe.model_spec.model.attention_mode == "exact"
    assert recipe.dataloader.position_policy == "window"
    selected = [
        config for config in recipe.model_converters.converters if getattr(config, "name", None) == "npu_moe_dispatch"
    ]
    assert len(selected) == 1
    assert selected[0]._owner._model_config.model_converter is moe_dispatch_api.NpuMoeDispatchConverter


@pytest.mark.parametrize(
    "owner,field,value",
    [
        ("router", "score_func", "sigmoid"),
        ("router", "route_norm", False),
        ("router", "route_scale", 1.5),
        ("router", "num_expert_groups", 2),
        ("router", "num_limited_groups", 1),
        ("moe", "score_before_experts", True),
        ("moe", "load_balance_coeff", 1e-3),
    ],
)
def test_exact_native_rejects_nested_router_policy_drift(native_api, owner, field, value):
    cfg = native_config(native_api, raw_config(moe=True))
    target = cfg.layers[0].moe.router if owner == "router" else cfg.layers[0].moe
    setattr(target, field, value)
    with pytest.raises(ValueError):
        cfg.build()


def test_fp32_gate_contract_survives_outer_cpu_autocast(moe_dispatch_api):
    gate = torch.nn.Linear(8, 4, dtype=torch.float32)
    inputs = torch.randn(3, 8, dtype=torch.bfloat16)
    expected = F.linear(inputs.float(), gate.weight, gate.bias)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = moe_dispatch_api._local_gate_scores(SimpleNamespace(gate=gate), inputs)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_proxy_preserves_explicit_unscaled_rope_override(native_api):
    cfg = native_api.registry.model_registry("dense_debug").model
    assert cfg.attention_mode == "compute_proxy"
    assert cfg.rope.scaling == "llama"
    cfg.rope.scaling = "none"
    model = cfg.build()
    model.init_states(buffer_device=torch.device("cpu"))
    assert cfg.rope.scaling == "none"
    assert model.rope.config.scaling == "none"
    assert model.rope.config.backend == "complex"
