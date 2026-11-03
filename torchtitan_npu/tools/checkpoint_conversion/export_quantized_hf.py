# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a torchtitan DCP checkpoint as a quantized HuggingFace safetensors checkpoint.

The applied recipe is also recorded in ``config.json`` as an ``FqnToConfig``
plan (``quantization_config.quant_type``): a machine-readable source of
quantization info for downstream tooling, never applied at load time (loaders
rebuild tensors from the per-instance shard-header records) and not
necessarily executable as-is. Decoding the record (torchao
``config_from_dict``) needs ``import torchao_npu.configs`` for its
``ALLOWED_AO_MODULES`` registration; plain JSON reads are unaffected. The
export fails closed unless the plan matches the shard headers
(``_validate_plan_against_shards``).
"""

import argparse
import contextlib
import copy
import importlib
import json
import logging
import os
import re
import shutil
import struct
import sys
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch import nn

_SCRIPT_DIR = Path(__file__).resolve().parent
_TORCHTITAN_NPU_DIR = _SCRIPT_DIR.parents[1]
_REPO_ROOT = _TORCHTITAN_NPU_DIR.parent


def _bootstrap_sys_path() -> None:
    """Make ``torchtitan_npu`` / ``torchao_npu`` importable from any cwd.

    ``torchao_npu`` prefers an already-importable installation (PYTHONPATH or
    pip); the repo tree under ``experiments/torchao-npu`` is only a fallback,
    so a dev checkout shadowing the path is never displaced by the repo copy.
    """
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        import torchao_npu.serialization  # noqa: F401
    except ImportError:
        cand = _REPO_ROOT / "experiments" / "torchao-npu"
        if cand.is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))


_bootstrap_sys_path()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# Logical-side weight filters (``to_hf`` output names; see
# ``DeepSeekV4StateDictAdapter.from_hf_map``).
_DENSE_WEIGHT_SUFFIXES = (
    ".attn.wq_a.weight",
    ".attn.wq_b.weight",
    ".attn.wkv.weight",
    ".attn.wo_a.weight",
    ".attn.wo_b.weight",
    ".attn.indexer.wq_b.weight",
    ".ffn.shared_experts.w1.weight",
    ".ffn.shared_experts.w2.weight",
    ".ffn.shared_experts.w3.weight",
)
_QWEN35_DENSE_WEIGHT_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".linear_attn.in_proj_qkv.weight",
    ".linear_attn.in_proj_z.weight",
    ".linear_attn.in_proj_a.weight",
    ".linear_attn.in_proj_b.weight",
    ".linear_attn.out_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)
_QWEN35_HF_PREFIXES = ("model.language_model.",)
_ROUTED_WEIGHT_PATTERN = r".*\.ffn\.experts\.\d+\.w[123]\.weight"

# The converter's own recipe list also carries all_hif8, which this tool cannot export.
_QWEN35_RECIPES = ("W4A8_MXFP", "W8A8_MXFP8")
_SUPPORTED_RECIPES = ("all_mxfp8", "mix", "all_block_fp8", *_QWEN35_RECIPES)


def _validate_recipe_model(recipe: str, model_name: str) -> None:
    """Reject recipe/model combinations that the training converter cannot build."""
    model_family = model_name.rsplit(".", 1)[-1]
    if recipe in _QWEN35_RECIPES and model_family != "qwen3_5":
        raise ValueError(f"recipe {recipe!r} is only supported for Qwen3.5, got model {model_name!r}")
    if model_family == "qwen3_5" and recipe not in _QWEN35_RECIPES:
        raise ValueError(f"Qwen3.5 export recipe must be one of {_QWEN35_RECIPES}, got {recipe!r}")


def build_recipe_plan(recipe: str, *, enable_mxfp4_qat: bool, dst_type_max: float = 0.0):
    """Recipe -> ordered ``[(ParamSwapConfig, fqn-filter)]`` pairs.

    Mirrors ``torchtitan_npu.extensions.adaptor.torchao.converter._recipe_converters`` (same configs,
    same order) with the training-time config-fqn filters replaced by their
    HF-name counterparts.
    """
    from torchao_npu.quantization.filters import any_filter, match_fqn_regex, match_fqn_suffix

    from torchtitan_npu.extensions.adaptor.torchao.converter import block_fp8_param_swap, mxfp8_param_swap

    if recipe not in _SUPPORTED_RECIPES:
        raise ValueError(f"recipe must be one of {_SUPPORTED_RECIPES}, got {recipe!r}")

    dense_filter = match_fqn_suffix(*_DENSE_WEIGHT_SUFFIXES)
    routed_filter = match_fqn_regex(_ROUTED_WEIGHT_PATTERN)

    if recipe == "all_mxfp8":
        return [(mxfp8_param_swap(), any_filter(dense_filter, routed_filter))]
    if recipe in _QWEN35_RECIPES:
        swap_config = (
            block_fp8_param_swap(enable_mxfp4_qat=True, dst_type_max=dst_type_max)
            if recipe == "W4A8_MXFP"
            else block_fp8_param_swap()
        )
        return [
            (
                swap_config,
                match_fqn_suffix(*_QWEN35_DENSE_WEIGHT_SUFFIXES),
            )
        ]

    routed_config = block_fp8_param_swap(enable_mxfp4_qat=enable_mxfp4_qat, dst_type_max=dst_type_max)
    dense_config = mxfp8_param_swap() if recipe == "mix" else block_fp8_param_swap()
    return [(dense_config, dense_filter), (routed_config, routed_filter)]


def _quantize_to_mx_tensor(weight: torch.Tensor, swap_config, device: str):
    """Quantize one plain ``weight`` per ``swap_config`` on ``device`` and return the ``MXTensor`` on CPU."""
    from torchao_npu.quantization.quant_configs import BlockMXQuantizeConfig, MXQuantizeConfig
    from torchao_npu.quantized_tensors import MXTensor
    from torchao_npu.wrapper_tensors import BlockMXTrainingWeightWrapperTensor, MXTrainingWeightWrapperTensor

    on_device = weight.to(device)
    # npu_dynamic_block_mx_quant rejects fp32 inputs.
    if on_device.dtype not in (torch.float16, torch.bfloat16):
        on_device = on_device.to(torch.bfloat16)
    weight_config = swap_config.weight_config
    if isinstance(weight_config, BlockMXQuantizeConfig):
        wrapper = BlockMXTrainingWeightWrapperTensor(
            on_device, weight_config=weight_config, activation_config=swap_config.activation_config
        )
    elif type(weight_config) is MXQuantizeConfig:
        wrapper = MXTrainingWeightWrapperTensor(
            on_device, weight_config=weight_config, activation_config=swap_config.activation_config
        )
    else:
        raise ValueError(
            f"weight_config must be `MXQuantizeConfig` or `BlockMXQuantizeConfig`, got {type(weight_config).__name__}."
        )
    mx = wrapper.to_inference_weight()
    # Storage must land on CPU: the NPU->CPU copy is bit-exact, and the
    # flatten-time clone kernel does not support FP4 qdata on the NPU.
    return MXTensor(
        mx.qdata.cpu(),
        mx.scale.cpu(),
        mx.dtype,
        mx.quant_axis,
        mx.quant_config,
        mx.act_quant_config,
        mx.pack_axis,
    )


def quantize_state_dict(hf_state_dict, plan, *, device: str):
    """Quantize the recipe-matched weights; returns ``(state_dict, fqn -> plan index)``.

    Non-matched tensors pass through unchanged (still on CPU, master dtype).
    """
    out = {}
    quantized: dict[str, int] = {}
    for fqn, value in hf_state_dict.items():
        # Non-floating-point tensors (e.g. the int64 ``tid2eid`` hash-router
        # table) are never recipe-matched and ``nn.Parameter`` rejects them;
        # pass them through untouched.
        if not value.is_floating_point():
            out[fqn] = value
            continue
        for idx, (swap_config, filter_fn) in enumerate(plan):
            if filter_fn(nn.Parameter(value), fqn):
                out[fqn] = _quantize_to_mx_tensor(value, swap_config, device)
                quantized[fqn] = idx
                break
        else:
            out[fqn] = value
    return out, quantized


# Dense renames logical ``layers.N.*`` -> HF ``model.layers.N.*``; inverse of ``_HF_TO_LOGICAL`` (``ao_hf_storage.py``).
# Routed experts merge separately (``_to_in_memory_layout``).
_LOGICAL_TO_HF = (
    (r"^(layers\.\d+)\.attn\.wq_a\.weight$", r"model.\1.self_attn.q_a_proj.weight"),
    (r"^(layers\.\d+)\.attn\.wq_b\.weight$", r"model.\1.self_attn.q_b_proj.weight"),
    (r"^(layers\.\d+)\.attn\.wkv\.weight$", r"model.\1.self_attn.kv_proj.weight"),
    (r"^(layers\.\d+)\.attn\.wo_a\.weight$", r"model.\1.self_attn.o_a_proj.weight"),
    (r"^(layers\.\d+)\.attn\.wo_b\.weight$", r"model.\1.self_attn.o_b_proj.weight"),
    (
        r"^(layers\.\d+)\.attn\.indexer\.wq_b\.weight$",
        r"model.\1.self_attn.compressor.indexer.q_b_proj.weight",
    ),
    (r"^(layers\.\d+)\.ffn\.shared_experts\.w1\.weight$", r"model.\1.mlp.shared_experts.gate_proj.weight"),
    (r"^(layers\.\d+)\.ffn\.shared_experts\.w2\.weight$", r"model.\1.mlp.shared_experts.down_proj.weight"),
    (r"^(layers\.\d+)\.ffn\.shared_experts\.w3\.weight$", r"model.\1.mlp.shared_experts.up_proj.weight"),
)
_EXPERT_FQN_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.weight$")


def _check_expert_attrs_match(reference, candidate) -> None:
    """Reject merging experts whose quantization attributes disagree.

    Configs compare by value: the block-FP8 path derives a fresh config per expert.
    """
    reference_attrs = (
        reference.dtype,
        reference.quant_axis,
        reference.pack_axis,
        reference.quant_config,
        reference.act_quant_config,
    )
    candidate_attrs = (
        candidate.dtype,
        candidate.quant_axis,
        candidate.pack_axis,
        candidate.quant_config,
        candidate.act_quant_config,
    )
    if reference_attrs != candidate_attrs:
        raise ValueError("per-expert MXTensors disagree on quantization attributes; cannot merge")


def _stack_expert_axes(per_expert, partner=None):
    """Stack per-expert qdata/scale into the merged tensors' leading expert dim.

    With ``partner``, each w1 expert is first concatenated with its w3 counterpart.
    """
    if partner is None:
        qdata = torch.stack([per_expert[e].qdata for e in sorted(per_expert)], dim=0)
        scale = torch.stack([per_expert[e].scale for e in sorted(per_expert)], dim=0)
    else:
        qdata = torch.stack(
            [torch.cat([per_expert[e].qdata, partner[e].qdata], dim=0) for e in sorted(per_expert)], dim=0
        )
        scale = torch.stack(
            [torch.cat([per_expert[e].scale, partner[e].scale], dim=0) for e in sorted(per_expert)], dim=0
        )
    return qdata, scale


def _merge_routed_experts(hf_state_dict, quantized):
    """Merge the per-expert MXTensors into the 3D routed-expert tensors.

    Returns ``(merged_tensors, merged_plan_index)``; the merged tensor inherits
    the plan index of the per-expert tensors it was stacked from.
    """
    from torchao_npu.quantized_tensors import MXTensor

    experts: dict[tuple[str, str], dict[int, MXTensor]] = {}
    group_idx: dict[tuple[str, str], int] = {}
    for fqn in sorted(quantized):
        m = _EXPERT_FQN_RE.match(fqn)
        if m:
            experts.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = hf_state_dict[fqn]
            group_idx[(m.group(1), m.group(3))] = quantized[fqn]

    merged: dict[str, MXTensor] = {}
    merged_plan: dict[str, int] = {}
    for (layer, w), per_expert in sorted(experts.items()):
        if w == "3":
            continue  # folded into gate_up_proj with w1
        if set(per_expert) != set(range(max(per_expert) + 1)):
            missing = sorted(set(range(max(per_expert) + 1)) - set(per_expert))
            raise ValueError(f"layer {layer}: incomplete per-expert quantization for w{w}, missing {missing}")
        first = per_expert[min(per_expert)]
        for tensor in per_expert.values():
            _check_expert_attrs_match(first, tensor)

        if w == "2":
            key = f"model.layers.{layer}.mlp.experts.down_proj"
            qdata, scale = _stack_expert_axes(per_expert)
        else:  # w == "1"
            partner = experts.get((layer, "3"))
            if partner is None or set(partner) != set(per_expert):
                raise ValueError(f"layer {layer}: cannot merge w1 without a matching w3 for every expert")
            for e in per_expert:
                _check_expert_attrs_match(first, partner[e])
            key = f"model.layers.{layer}.mlp.experts.gate_up_proj"
            qdata, scale = _stack_expert_axes(per_expert, partner)

        # Stacking adds a leading expert dim, so re-base quant/pack onto the
        # merged tensor's last axis (-1) instead of reusing the stale per-expert index.
        merged_pack_axis = -1 if first.pack_axis is not None else None
        merged[key] = MXTensor(
            qdata,
            scale,
            first.dtype,
            -1,
            first.quant_config,
            first.act_quant_config,
            merged_pack_axis,
        )
        merged_plan[key] = group_idx[(layer, w)]
    return merged, merged_plan


def _to_in_memory_layout(hf_state_dict, quantized_fqns):
    """Move the recipe-quantized tensors to the transformers in-memory layout.

    DeepSeek routed experts are merged into the 3D ``experts.gate_up_proj``
    (``w1`` | ``w3``) and ``experts.down_proj``; Qwen3.5's adapter has already
    emitted that HF layout, so its non-quantized experts pass through unchanged.
    Dense weights are renamed to their in-memory FQNs (the quantized loader
    rebuilds each param from the metadata under the checkpoint's original key,
    so these must be the final names).
    Returns ``(state_dict, final fqn -> plan index)``.
    """
    out = {k: v for k, v in hf_state_dict.items() if k not in quantized_fqns}
    new_quantized: dict[str, int] = {}

    for fqn, idx in sorted(quantized_fqns.items()):
        if _EXPERT_FQN_RE.match(fqn):
            continue  # merged below
        for pattern, replacement in _LOGICAL_TO_HF:
            new_key = re.sub(pattern, replacement, fqn, count=1)
            if new_key != fqn:
                out[new_key] = hf_state_dict[fqn]
                new_quantized[new_key] = idx
                break
        else:
            if fqn.startswith(_QWEN35_HF_PREFIXES) and fqn.endswith(_QWEN35_DENSE_WEIGHT_SUFFIXES):
                out[fqn] = hf_state_dict[fqn]
                new_quantized[fqn] = idx
            else:
                raise ValueError(f"quantized tensor {fqn!r} has no in-memory name mapping")

    merged, merged_plan = _merge_routed_experts(hf_state_dict, quantized_fqns)
    out.update(merged)
    new_quantized.update(merged_plan)
    return out, new_quantized


# Packaging meta files that no load path reads.
_LOAD_IRRELEVANT_ASSETS = {"config_recipes.json", "configuration.json"}


def _copy_hf_assets(assets_path: Path, output_dir: Path) -> None:
    """Copy the non-weight files (``config.json``, tokenizers, ...) from the HF assets dir."""
    if not assets_path.is_dir():
        logger.warning("--hf_assets_path %s is not a directory; skipping asset copy", assets_path)
        return
    copied = []
    for f in sorted(assets_path.iterdir()):
        if f.is_dir() or (f.is_symlink() and not f.resolve(strict=False).is_file()):
            logger.warning("skipping non-file asset %s", f)
            continue
        if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
            continue
        if f.name in _LOAD_IRRELEVANT_ASSETS:
            continue
        shutil.copy2(f, output_dir / f.name)
        copied.append(f.name)
    logger.info("Copied %d HF asset file(s): %s", len(copied), ", ".join(copied) if copied else "(none)")


# HF's per-layer spelling of ``attention.compress_ratio``; mirrors ``_LAYER_TYPE``
# in scripts/numerical_alignment/deepseek_v4/hf_model.py, keyed by the emitted HF
# spelling of "uncompressed" (0) rather than the model's (1).
_LAYER_TYPE_FOR_RATIO = {
    0: "sliding_attention",
    4: "compressed_sparse_attention",
    128: "heavily_compressed_attention",
}


def _find_rope_configs(layers):
    """Return the plain and yarn (compress) rope configs any layer carries.

    ``dataclasses.replace`` only pins the per-site ``split``; theta and the
    yarn scalars ride along, so one sample per kind is the flavor truth.
    Flavors whose every layer compresses (pro) carry no plain rope at all.
    """
    main_rope: Any = None
    compress_rope: Any = None
    for layer in layers:
        rope = getattr(getattr(layer, "attention", None), "rope", None)
        if rope is None:
            continue
        if getattr(rope, "scaling", None) == "yarn":
            compress_rope = compress_rope or rope
        elif main_rope is None:
            main_rope = rope
    return main_rope, compress_rope


def _is_hash_moe(layer) -> bool:
    return bool(getattr(getattr(layer.moe, "router", None), "hash", False))


def _model_derived_config_fields(model_config) -> dict:
    """Structural ``config.json`` fields recomputed from the model config.

    The copied assets describe the release model the dir was packaged for
    (expert count, MTP layers, ...), which need not match the flavor being
    exported. Each field is emitted only when this config declares it, so
    flavors without a sub-config keep their asset value. HF spells
    "uncompressed" layers ``compress_ratio=0`` where the model config uses 1.

    ``layer_types`` / ``mlp_layer_types`` are spelled out from the same
    per-layer walk as ``compress_ratios`` / ``num_hash_layers``; whether they
    are written back depends on the asset schema (see
    ``_apply_model_derived_config``).
    """
    fields: dict = {}

    def put(key, getter):
        try:
            value = getter()
        except (AttributeError, TypeError):
            return
        if value is not None:
            fields[key] = value

    layers = list(getattr(model_config, "layers", None) or [])
    put("vocab_size", lambda: model_config.vocab_size)
    put("hidden_size", lambda: model_config.dim)
    put("num_hidden_layers", lambda: len(layers) or None)
    put("num_nextn_predict_layers", lambda: len(model_config.mtp_layers))
    put("tie_word_embeddings", lambda: model_config.enable_weight_tying)
    put(
        "compress_ratios",
        lambda: [r if (r := layer.attention.compress_ratio) > 1 else 0 for layer in layers] or None,
    )
    # a ratio the table does not name yields no layer_types guess
    if ratios := fields.get("compress_ratios"):
        with contextlib.suppress(KeyError):
            fields["layer_types"] = [_LAYER_TYPE_FOR_RATIO[ratio] for ratio in ratios]
    if layers and all(getattr(layer, "moe", None) is not None for layer in layers):
        fields["mlp_layer_types"] = ["hash_moe" if _is_hash_moe(layer) else "moe" for layer in layers]

    attn = layers[0].attention if layers else None
    if attn is not None:
        put("num_attention_heads", lambda: attn.n_heads)
        put("head_dim", lambda: attn.head_dim)
        put("qk_rope_head_dim", lambda: attn.rope_head_dim)
        put("q_lora_rank", lambda: attn.q_lora_rank)
        put("num_key_value_heads", lambda: attn.wkv.out_features // attn.head_dim)
        put("o_lora_rank", lambda: attn.wo_a.out_features)
        put("o_groups", lambda: attn.wo_a.n_heads)
        put("sliding_window", lambda: attn.inner_attention.window_size)
        put("index_topk", lambda: attn.inner_attention.index_topk)
        put("rms_norm_eps", lambda: attn.norm_eps)

    # ``compress_rope_theta``/yarn scalars are flavor-varying like the rest; the
    # main theta comes from a non-yarn site (layer 0 may itself compress in pro).
    main_rope, compress_rope = _find_rope_configs(layers)
    put("rope_theta", lambda: main_rope.theta)
    put("compress_rope_theta", lambda: compress_rope.theta)
    put("max_position_embeddings", lambda: int(compress_rope.original_seq_len * compress_rope.rope_factor))

    indexer = next(
        (layer.attention.indexer for layer in layers if getattr(layer.attention, "indexer", None) is not None), None
    )
    if indexer is not None:
        put("index_n_heads", lambda: indexer.num_index_heads)
        put("index_head_dim", lambda: indexer.index_head_dim)

    moe = next((layer.moe for layer in layers if getattr(layer, "moe", None) is not None), None)
    if moe is not None:
        put("n_routed_experts", lambda: moe.num_experts)
        put("num_experts_per_tok", lambda: moe.router.top_k)
        put("num_hash_layers", lambda: sum(1 for layer in layers if _is_hash_moe(layer)))
        put("moe_intermediate_size", lambda: moe.routed_experts.inner_experts.hidden_dim)
        put("swiglu_limit", lambda: moe.routed_experts.inner_experts.swiglu_limit)
        put("n_shared_experts", lambda: 1 if moe.shared_experts is not None else 0)
        put("scoring_func", lambda: moe.router.score_func)
        put("routed_scaling_factor", lambda: moe.router.route_scale)
        put("norm_topk_prob", lambda: moe.router.route_norm)

    hc = next((layer.hc_attn_pre for layer in layers if getattr(layer, "hc_attn_pre", None) is not None), None)
    if hc is not None:
        put("hc_mult", lambda: hc.hc_mult)
        put("hc_sinkhorn_iters", lambda: hc.sinkhorn_iters)
        put("hc_eps", lambda: hc.eps)

    return fields


# Model ``ComplexRoPE`` attr -> HF yarn scalar name, shared by both rope
# spellings (nested ``rope_parameters`` and flat legacy ``rope_scaling``).
_YARN_SCALAR_KEYS = (
    ("rope_factor", "factor"),
    ("beta_fast", "beta_fast"),
    ("beta_slow", "beta_slow"),
    ("original_seq_len", "original_max_position_embeddings"),
)


def _sync_rope_parameters(existing: dict, layers) -> dict:
    """Refresh the main/compress rope entries an asset already spelled out.

    Only leaves the asset carries are rewritten; anything the model config
    does not model (``attention_factor`` and friends) keeps its asset value.
    """
    synced = copy.deepcopy(existing)
    main_rope, compress_rope = _find_rope_configs(layers)
    partial = None
    for layer in layers:
        attn = getattr(layer, "attention", None)
        head_dim = getattr(attn, "head_dim", None)
        rope_head_dim = getattr(attn, "rope_head_dim", None)
        if head_dim and rope_head_dim:
            partial = rope_head_dim / head_dim
            break
    for name, rope in (("main", main_rope), ("compress", compress_rope)):
        entry = synced.get(name)
        if not isinstance(entry, dict) or rope is None:
            continue
        theta = getattr(rope, "theta", None)
        if "rope_theta" in entry and theta is not None:
            entry["rope_theta"] = theta
        if "partial_rotary_factor" in entry and partial is not None:
            entry["partial_rotary_factor"] = partial
        if getattr(rope, "scaling", None) == "yarn":
            for src, dst in _YARN_SCALAR_KEYS:
                value = getattr(rope, src, None)
                if value is not None and dst in entry:
                    entry[dst] = value
    return synced


def _sync_rope_scaling(existing: dict, compress_rope) -> dict:
    """Refresh the yarn scalars in a flat legacy ``rope_scaling`` dict.

    Official release assets spell the flavor-varying compress rope here
    (the loader folds it into ``rope_parameters.compress``); same leaf rule
    as ``_sync_rope_parameters`` — keys the asset omits are not added.
    """
    synced = copy.deepcopy(existing)
    if compress_rope is None:
        return synced
    for src, dst in _YARN_SCALAR_KEYS:
        value = getattr(compress_rope, src, None)
        if value is not None and dst in synced:
            synced[dst] = value
    return synced


def _apply_model_derived_config(output_dir: Path, model_config) -> None:
    """Overwrite the copied ``config.json`` structural fields with the exported model's truth."""
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        logger.warning("no config.json in %s; skipping model-derived field rewrite", output_dir)
        return
    derived = _model_derived_config_fields(model_config)
    config = json.loads(config_path.read_text())
    # layer_types/mlp_layer_types spell the same per-layer truth as
    # compress_ratios/num_hash_layers. Official assets omit them and let the
    # loader derive, so refresh them only when the asset carries them: fix
    # staleness, do not reshape the schema.
    for key in ("layer_types", "mlp_layer_types"):
        if key not in config:
            derived.pop(key, None)
    layers = list(getattr(model_config, "layers", None) or [])
    if isinstance(config.get("rope_parameters"), dict):
        derived["rope_parameters"] = _sync_rope_parameters(config["rope_parameters"], layers)
    # legacy spelling (official release assets): yarn scalars live in a flat
    # top-level ``rope_scaling`` dict instead of ``rope_parameters.compress``.
    if isinstance(config.get("rope_scaling"), dict):
        derived["rope_scaling"] = _sync_rope_scaling(config["rope_scaling"], _find_rope_configs(layers)[1])
    changed = sorted(k for k, v in derived.items() if config.get(k) != v)
    config.update(derived)
    config_path.write_text(json.dumps(config, indent=2))
    logger.info(
        "Aligned %d/%d config.json structural field(s) with the model config%s",
        len(changed),
        len(derived),
        ": " + ", ".join(changed) if changed else "",
    )


# Static part of the config.json ``quantization_config``; without it transformers
# falls back to on-the-fly TorchAo re-quantization, which fails on NPU.
_TORCHAO_QUANTIZATION_CONFIG = {
    "include_input_output_embeddings": False,
    "modules_to_not_convert": None,
    "quant_method": "torchao",
    "untie_embedding_weights": False,
}


def _generalized_fqn_pattern(fqn: str) -> str:
    """``FqnToConfig`` ``re:`` key matching this FQN and its numeric siblings."""
    return "re:" + r"\.".join(r"\d+" if part.isdigit() else re.escape(part) for part in fqn.split("."))


def _drop_callable_fields(config):
    # torchao's ConfigJSONEncoder walks __dict__ verbatim and chokes on callables;
    # the filter still lands there as ParamSwapConfig's ``_params_filter_fn`` attr.
    names = [k for k, v in vars(config).items() if callable(v)]
    if not names:
        return config
    clone = copy.copy(config)
    for name in names:
        del clone.__dict__[name]
    return clone


def _build_fqn_to_config_record(quantized: dict[str, int], plan) -> dict | None:
    """Serialize the recipe plan as the AO-official ``FqnToConfig`` ``quant_type`` record.

    One ``re:`` key per generalized parameter FQN, valued with the applied
    ``ParamSwapConfig``. Keys name parameters (``...q_a_proj.weight``), not the
    module FQNs torchao's own matcher walks: merged 3D expert tensors put
    several quantized parameters on one module, so the parameter level is the
    precise namespace.

    The record is descriptive source material -- no load path applies it, and
    downstream consumers may need a conversion step -- but it carries one hard
    obligation: transformers parses ``quant_type`` eagerly in
    ``TorchAoConfig.from_dict``, and torchao's ``config_from_dict`` resolves
    ``_type`` names only inside the ``ALLOWED_AO_MODULES`` module whitelist, so
    decoding needs a torchao_npu whose ``configs`` package adds
    ``torchao_npu.configs`` to that whitelist. The safetensors shard-header
    path is a separate table: ``torchao_npu.serialization`` registers
    ``MXTensor`` in ``safetensors_utils.ALLOWED_CLASSES`` and does not cover
    this record.
    """
    from torchao.core.config import config_to_dict
    from torchao.quantization import FqnToConfig

    groups: dict[int, list[str]] = {}
    for fqn, idx in quantized.items():
        groups.setdefault(idx, []).append(fqn)

    fqn_to_config: OrderedDict[str, object] = OrderedDict()
    for idx, (swap_config, _) in enumerate(plan):
        config = _drop_callable_fields(swap_config)
        for fqn in sorted(groups.get(idx, ())):
            fqn_to_config.setdefault(_generalized_fqn_pattern(fqn), config)
    if not fqn_to_config:
        return None
    record = config_to_dict(FqnToConfig(fqn_to_config=fqn_to_config))
    # ``config_to_dict`` serializes the two aliased fields twice; drop the BC
    # alias — decoding an ``fqn_to_config``-only record re-aliases it.
    record["_data"].pop("module_fqn_to_config", None)
    return record


def _read_safetensors_metadata(path: Path) -> dict[str, str]:
    """``__metadata__`` of a safetensors file, read from the header alone.

    Stdlib-only so the cross-check runs in CI without safetensors installed.
    """
    with open(path, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    return header.get("__metadata__") or {}


def _shard_mx_records(shard_paths: Iterable[Path]) -> dict[str, dict]:
    """Logical FQN -> ``MXTensor`` record payload across shards: the per-instance truth."""
    records: dict[str, dict] = {}
    for path in shard_paths:
        for key, blob in _read_safetensors_metadata(path).items():
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                continue  # bookkeeping entries (``format``, ``tensor_names``)
            if isinstance(parsed, dict) and parsed.get("_type") == "MXTensor":
                records[key] = parsed["_data"]
    return records


def _deref(config_record: dict | None) -> dict | None:
    """Compare serialized configs by class name + data; the plan and header writers stamp ``_version`` differently."""
    if config_record is None:
        return None
    return {"_type": config_record["_type"], "_data": config_record["_data"]}


def _plan_weight_config(param_record: dict) -> dict:
    """The record this plan entry must produce as the tensor's ``quant_config`` in a shard.

    Mirrors the write path: an MX config verbatim; block-MX uses its nested
    mxfp4 config, else the ``MXQuantizeConfig`` that
    ``BlockMXTensor.to_mx_tensor`` rebuilds from the MX fields only.
    """
    from dataclasses import fields

    from torchao_npu.quantization.quant_configs import MXQuantizeConfig

    weight = param_record["_data"].get("weight_config")
    if not isinstance(weight, dict):
        raise ValueError(f"plan entry has no serialized weight_config record: {weight!r}")
    if weight["_type"] == "MXQuantizeConfig":
        return weight
    if weight["_type"] == "BlockMXQuantizeConfig":
        mxfp4 = weight["_data"].get("mxfp4_fake_quantize_config")
        if mxfp4 is not None:
            return mxfp4
        mx_fields = {f.name for f in fields(MXQuantizeConfig)}
        return {
            "_type": "MXQuantizeConfig",
            "_data": {k: v for k, v in weight["_data"].items() if k in mx_fields},
        }
    raise ValueError(f"unexpected plan weight_config type: {weight['_type']!r}")


def _validate_plan_against_shards(
    output_dir: Path,
    shard_names: list[str],
    fqn_to_config_record: dict,
    quantized: dict[str, int],
) -> None:
    """Fail the export unless the ``quant_type`` plan agrees with the shards just written.

    Plan and shard headers encode the same recipe for different loaders and no
    load path compares them, so drift would ship silently. Matching mirrors
    upstream ``FqnToConfig``: first ``re:`` fullmatch in insertion order wins.
    """
    payload = _shard_mx_records(output_dir / name for name in shard_names)
    patterns = fqn_to_config_record["_data"]["fqn_to_config"]

    problems: list[str] = []

    winner: dict[str, str | None] = {}
    for fqn in payload:
        winner[fqn] = next((k for k in patterns if k.startswith("re:") and re.fullmatch(k[3:], fqn)), None)
    for fqn, key in winner.items():
        if key is None:
            problems.append(f"quantized tensor {fqn!r} matches no plan pattern")
            continue
        header, pattern = payload[fqn], patterns[key]
        if _deref(_plan_weight_config(pattern)) != _deref(header.get("quant_config")):
            problems.append(
                f"{fqn!r}: pattern {key!r} declares weight_config {_deref(_plan_weight_config(pattern))}, "
                f"shard records quant_config {_deref(header.get('quant_config'))}"
            )
        if _deref(pattern["_data"].get("activation_config")) != _deref(header.get("act_quant_config")):
            problems.append(
                f"{fqn!r}: pattern {key!r} declares activation_config "
                f"{_deref(pattern['_data'].get('activation_config'))}, shard records "
                f"act_quant_config {_deref(header.get('act_quant_config'))}"
            )

    for key in patterns:
        if key in winner.values():
            continue
        matched = [f for f in payload if key.startswith("re:") and re.fullmatch(key[3:], f)]
        if matched:
            problems.append(f"pattern {key!r} never wins: every tensor it matches is claimed by an earlier pattern")
        else:
            problems.append(f"pattern {key!r} matches no quantized tensor in the shards")

    if set(quantized) != set(payload):
        problems.append(
            "plan quantized FQNs and shard MXTensor records disagree: "
            f"missing from shards {sorted(set(quantized) - set(payload))[:4]}, "
            f"absent from plan {sorted(set(payload) - set(quantized))[:4]}"
        )

    if problems:
        raise ValueError("config.json plan disagrees with the shard headers:\n  " + "\n  ".join(problems))


def _inject_torchao_quant_config(output_dir: Path, fqn_to_config_record: dict | None) -> None:
    """Write ``quantization_config`` into the copied ``config.json``.

    The block's presence gates transformers into the pre-quantized path; the
    ``quant_type`` plan inside it is descriptive (see
    ``_build_fqn_to_config_record``).
    """
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        logger.warning(
            "no config.json in %s; the export is not directly loadable via transformers.from_pretrained",
            output_dir,
        )
        return
    if fqn_to_config_record is None:
        logger.warning("no quantized tensors; leaving config.json without a torchao quantization_config")
        return
    config = json.loads(config_path.read_text())
    quantization_config: dict = dict(_TORCHAO_QUANTIZATION_CONFIG)
    quantization_config["quant_type"] = {"default": fqn_to_config_record}
    config["quantization_config"] = quantization_config
    config_path.write_text(json.dumps(config, indent=2))
    logger.info(
        "Injected torchao quantization_config into %s (%d FqnToConfig entries)",
        config_path,
        len(fqn_to_config_record["_data"]["fqn_to_config"]),
    )


def _discard_failed_export(output_dir: Path, shard_names: list[str]) -> None:
    # Targeted removal only, never rmtree: a reused output_dir may predate this run.
    names = [*shard_names, "model.safetensors.index.json"]
    removed = [name for name in dict.fromkeys(names) if (output_dir / name).is_file()]
    for name in removed:
        (output_dir / name).unlink()
    logger.error(
        "export failed after writing shards; removed %s from %s; directory is not a complete checkpoint",
        ", ".join(removed) or "nothing",
        output_dir,
    )


def _build_model_and_adapter(model_name: str, model_flavor: str, hf_assets_path: Path):
    """Build the model (for state-dict shapes only) with its HF adapter and model config."""
    from torchtitan.components.checkpoint import ModelWrapper

    # ``--model_name`` may be a dotted module path (e.g.
    # torchtitan_npu.models.deepseek_v4) or the short ``llama3`` form used
    # upstream.
    module_name = model_name if "." in model_name else f"torchtitan.models.{model_name}"
    model_module = importlib.import_module(module_name)
    model_spec = model_module.model_registry(model_flavor)
    model_config = model_spec.model
    with torch.device("cpu"):
        raw_model = model_config.build()
    sd_adapter = model_spec.state_dict_adapter(model_config, str(hf_assets_path))
    if sd_adapter is None:
        raise ValueError("trying to export to HF safetensors, but the model spec provides no state_dict_adapter.")
    return ModelWrapper(raw_model), sd_adapter, model_config


def _load_dcp_state_dict(model, input_dir: Path, master_dtype: str, read_threads: int):
    """Load the DCP checkpoint into the model's state-dict container."""
    from torchtitan.config import TORCH_DTYPE_MAP

    from torchtitan_npu.tools.checkpoint_conversion.convert_to_hf import ParallelFileSystemReader

    state_dict = model.state_dict()
    # The model is built in float32, but dcp.load copies the stored dtype into
    # the container in place: an fp32 container would silently upcast bf16
    # master weights, and the NPU block-MX quant kernel only accepts fp16/bf16
    # inputs. Rebind the container to the checkpoint's master dtype first.
    container_dtype = TORCH_DTYPE_MAP[master_dtype]
    for fqn, value in state_dict.items():
        if value.is_floating_point() and value.dtype != container_dtype and ".engram.table.weight" not in fqn:
            state_dict[fqn] = value.detach().to(container_dtype)
    reader = ParallelFileSystemReader(str(input_dir), thread_count=read_threads)
    dcp.load(state_dict, storage_reader=reader)
    return state_dict


@torch.inference_mode()
def export_quantized_hf(args: argparse.Namespace) -> None:
    """Run the export: load the DCP, quantize per recipe, save the HF safetensors layout."""
    from torchao_npu.quantized_tensors import MXTensor
    from torchao_npu.serialization.export import save_hf_safetensors
    from torchtitan.config import TORCH_DTYPE_MAP

    input_dir = args.input_dir
    output_dir = args.output_dir
    hf_assets_path = args.hf_assets_path
    model_name = args.model_name
    model_flavor = args.model_flavor
    master_dtype = args.master_dtype
    export_dtype = args.export_dtype
    recipe = args.recipe
    enable_mxfp4_qat = args.enable_mxfp4_qat
    dst_type_max = args.dst_type_max
    quant_device = args.quant_device
    max_shard_size = args.max_shard_size
    read_threads = args.read_threads

    _validate_recipe_model(recipe, model_name)
    model, sd_adapter, model_config = _build_model_and_adapter(model_name, model_flavor, hf_assets_path)
    state_dict = _load_dcp_state_dict(model, input_dir, master_dtype, read_threads)
    logger.info("Loaded DCP checkpoint %s (%d tensors)", input_dir, len(state_dict))

    hf_state_dict = sd_adapter.to_hf(state_dict)
    logger.info("to_hf: %d tensors", len(hf_state_dict))

    # to_hf emits the hash-router ``tid2eid`` lookup table as float32 (the
    # upstream DSv4 checkpoint format); transformers' in-memory buffer is
    # int64. The values are small exact integers, so the cast back is lossless.
    for fqn, value in hf_state_dict.items():
        if "tid2eid" in fqn:
            hf_state_dict[fqn] = value.to(torch.int64)

    plan = build_recipe_plan(recipe, enable_mxfp4_qat=enable_mxfp4_qat, dst_type_max=dst_type_max)
    hf_state_dict, quantized_fqns = quantize_state_dict(hf_state_dict, plan, device=quant_device)
    hf_state_dict, quantized_fqns = _to_in_memory_layout(hf_state_dict, quantized_fqns)
    logger.info(
        "recipe=%s: quantized %d/%d tensors on %s",
        recipe,
        len(quantized_fqns),
        len(hf_state_dict),
        quant_device,
    )
    if quantized_fqns:
        logger.info("  first matches: %s", ", ".join(sorted(quantized_fqns)[:8]))
    else:
        logger.warning("recipe=%s matched no weights; all tensors will be exported as plain %s", recipe, export_dtype)

    # Plain (non-quantized) floats land in export_dtype; the MXTensors carry
    # their own low-precision payloads and must pass through untouched.
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    for fqn, value in hf_state_dict.items():
        if isinstance(value, MXTensor):
            continue
        if value.is_floating_point() and value.dtype != target_dtype:
            hf_state_dict[fqn] = value.to(target_dtype)

    shard_names = save_hf_safetensors(hf_state_dict, output_dir, max_shard_size=max_shard_size)
    logger.info("Saved %d shard(s) to %s: %s", len(shard_names), output_dir, ", ".join(shard_names))

    try:
        _copy_hf_assets(hf_assets_path, output_dir)
        _apply_model_derived_config(output_dir, model_config)
        # The FqnToConfig keys are in-memory FQNs, so build the record after the relayout.
        fqn_to_config_record = _build_fqn_to_config_record(quantized_fqns, plan)
        if fqn_to_config_record is not None:
            _validate_plan_against_shards(output_dir, shard_names, fqn_to_config_record, quantized_fqns)
        _inject_torchao_quant_config(output_dir, fqn_to_config_record)
    except Exception:
        _discard_failed_export(output_dir, shard_names)
        raise


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a DCP checkpoint as a quantized HF safetensors checkpoint.")
    parser.add_argument("input_dir", type=Path, help="Input directory with DCP weights.")
    parser.add_argument("output_dir", type=Path, help="Output directory for the HF checkpoint.")
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        required=True,
        help="HF assets directory (config.json, tokenizers, ...); non-weight files are copied to output_dir.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3",
        help="Model module: short name ('llama3') or dotted path ('torchtitan_npu.models.deepseek_v4').",
    )
    parser.add_argument("--model_flavor", type=str, default="8B")
    parser.add_argument(
        "--recipe",
        choices=_SUPPORTED_RECIPES,
        default="mix",
        help="Quantization recipe (same semantics as torchtitan_npu.extensions.adaptor.torchao.converter).",
    )
    parser.add_argument(
        "--enable_mxfp4_qat",
        action="store_true",
        help=(
            "Serve routed-expert weights as MXFP4 for mix/all_block_fp8; "
            "Qwen3.5 W4A8_MXFP enables MXFP4 for dense weights intrinsically."
        ),
    )
    parser.add_argument("--dst_type_max", type=float, default=0.0)
    parser.add_argument(
        "--export_dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Dtype for the plain (non-quantized) float tensors.",
    )
    parser.add_argument(
        "--master_dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Dtype of the DCP load container; should match the checkpoint's master dtype (production bf16).",
    )
    parser.add_argument("--max_shard_size", type=str, default="5GB", help="e.g. 5368709120 or '5GB'.")
    parser.add_argument("--quant_device", type=str, default="npu:0", help="NPU used for the quantization kernels.")
    parser.add_argument(
        "--read_threads",
        type=int,
        default=min(32, os.cpu_count() or 16),
        help="Threads used to read DCP shard files in parallel.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    export_quantized_hf(args)


if __name__ == "__main__":
    main()
