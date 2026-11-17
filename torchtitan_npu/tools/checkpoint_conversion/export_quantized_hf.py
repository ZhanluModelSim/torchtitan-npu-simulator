# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a torchtitan DCP checkpoint as a quantized HuggingFace safetensors checkpoint."""

import argparse
import importlib
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import torch
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

# HF-side weight filters (post-``to_hf`` names; see
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
_ROUTED_WEIGHT_PATTERN = r".*\.ffn\.experts\.\d+\.w[123]\.weight"


def build_recipe_plan(recipe: str, *, enable_mxfp4_qat: bool, dst_type_max: float = 0.0):
    """Recipe -> ordered ``[(ParamSwapConfig, fqn-filter)]`` pairs.

    Mirrors ``torchtitan_npu.extensions.adaptor.torchao.converter._recipe_converters`` (same configs,
    same order) with the training-time config-fqn filters replaced by their
    HF-name counterparts.
    """
    from torchao_npu.quantization.filters import any_filter, match_fqn_regex, match_fqn_suffix

    from torchtitan_npu.extensions.adaptor.torchao.converter import (
        _SUPPORTED_RECIPES,
        _block_fp8_param_swap,
        _mxfp8_param_swap,
    )

    if recipe not in _SUPPORTED_RECIPES:
        raise ValueError(f"recipe must be one of {_SUPPORTED_RECIPES}, got {recipe!r}")

    dense_filter = match_fqn_suffix(*_DENSE_WEIGHT_SUFFIXES)
    routed_filter = match_fqn_regex(_ROUTED_WEIGHT_PATTERN)

    if recipe == "all_mxfp8":
        return [(_mxfp8_param_swap(), any_filter(dense_filter, routed_filter))]

    routed_config = _block_fp8_param_swap(enable_mxfp4_qat=enable_mxfp4_qat, dst_type_max=dst_type_max)
    dense_config = _mxfp8_param_swap() if recipe == "mix" else _block_fp8_param_swap()
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
        mx.orig_dtype,
        mx.quant_axis,
        mx.quant_config,
        mx.act_quant_config,
        mx.pack_axis,
    )


def quantize_state_dict(hf_state_dict, plan, *, device: str):
    """Quantize the recipe-matched weights; returns ``(state_dict, quantized_fqns)``.

    Non-matched tensors pass through unchanged (still on CPU, master dtype).
    """
    out = {}
    quantized = []
    for fqn, value in hf_state_dict.items():
        # Non-floating-point tensors (e.g. the int64 ``tid2eid`` hash-router
        # table) are never recipe-matched and ``nn.Parameter`` rejects them;
        # pass them through untouched.
        if not value.is_floating_point():
            out[fqn] = value
            continue
        for swap_config, filter_fn in plan:
            if filter_fn(nn.Parameter(value), fqn):
                out[fqn] = _quantize_to_mx_tensor(value, swap_config, device)
                quantized.append(fqn)
                break
        else:
            out[fqn] = value
    return out, quantized


# HF-name -> in-memory (transformers) name for the recipe-quantized dense weights;
# the routed experts are merged separately (``_to_in_memory_layout``).
_QUANT_HF_TO_MODEL = (
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
        reference.orig_dtype,
        reference.quant_axis,
        reference.pack_axis,
        reference.quant_config,
        reference.act_quant_config,
    )
    candidate_attrs = (
        candidate.orig_dtype,
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
    """Merge the per-expert MXTensors into the 3D routed-expert tensors."""
    from torchao_npu.quantized_tensors import MXTensor

    experts: dict[tuple[str, str], dict[int, MXTensor]] = {}
    for fqn in sorted(quantized):
        m = _EXPERT_FQN_RE.match(fqn)
        if m:
            experts.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = hf_state_dict[fqn]

    merged: dict[str, MXTensor] = {}
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
            first.orig_dtype,
            -1,
            first.quant_config,
            first.act_quant_config,
            merged_pack_axis,
        )
    return merged


def _to_in_memory_layout(hf_state_dict, quantized_fqns):
    """Move the recipe-quantized tensors to the transformers in-memory layout.

    Routed experts are merged into the 3D ``experts.gate_up_proj`` (``w1`` | ``w3``)
    and ``experts.down_proj``; dense weights are renamed to their in-memory FQNs
    (the quantized loader rebuilds each param from the metadata under the
    checkpoint's original key, so these must be the final names).
    Returns ``(state_dict, quantized_fqns)``.
    """
    quantized = set(quantized_fqns)
    out = {k: v for k, v in hf_state_dict.items() if k not in quantized}
    new_quantized = []

    for fqn in sorted(quantized):
        if _EXPERT_FQN_RE.match(fqn):
            continue  # merged below
        for pattern, replacement in _QUANT_HF_TO_MODEL:
            new_key = re.sub(pattern, replacement, fqn, count=1)
            if new_key != fqn:
                out[new_key] = hf_state_dict[fqn]
                new_quantized.append(new_key)
                break
        else:
            raise ValueError(f"quantized tensor {fqn!r} has no in-memory name mapping")

    merged = _merge_routed_experts(hf_state_dict, quantized)
    out.update(merged)
    new_quantized.extend(merged.keys())
    return out, new_quantized


def _copy_hf_assets(assets_path: Path, output_dir: Path) -> None:
    """Copy the non-weight files (``config.json``, tokenizers, ...) from the HF assets dir.

    Dirs/symlinks and safetensors shards / index are skipped: this export produces its own weights.
    """
    if not assets_path.is_dir():
        logger.warning("--hf_assets_path %s is not a directory; skipping asset copy", assets_path)
        return
    copied = []
    for f in sorted(assets_path.iterdir()):
        if f.is_dir() or f.is_symlink():
            logger.warning(
                "skipping non-regular asset %s (directory or symlink); required files must be top-level regular files",
                f,
            )
            continue
        if f.suffix == ".safetensors" or f.name == "model.safetensors.index.json":
            continue
        shutil.copy2(f, output_dir / f.name)
        copied.append(f.name)
    logger.info("Copied %d HF asset file(s): %s", len(copied), ", ".join(copied) if copied else "(none)")


# config.json must carry ``quantization_config`` or transformers falls back to
# on-the-fly TorchAo re-quantization, which fails on NPU (no CUDA). No version
# pin: the loader rebuilds each param from the shard-header metadata.
_TORCHAO_QUANTIZATION_CONFIG = {
    "include_input_output_embeddings": False,
    "modules_to_not_convert": None,
    "quant_method": "torchao",
    "quant_type": {
        "default": {
            "_data": {
                "activation_dtype": {"_data": "float8_e4m3fn", "_type": "torch.dtype"},
                "activation_value_lb": None,
                "activation_value_ub": None,
                "granularity": [
                    {"_data": {}, "_type": "PerTensor", "_version": 1},
                    {"_data": {}, "_type": "PerTensor", "_version": 1},
                ],
                "kernel_preference": {"_data": "AUTO", "_type": "KernelPreference"},
                "mm_config": {
                    "_data": {"emulate": False, "pad_inner_dim": False, "use_fast_accum": True},
                    "_type": "Float8MMConfig",
                    "_version": 1,
                },
                "packing_format": {"_data": "PLAIN", "_type": "Float8PackingFormat"},
                "set_inductor_config": True,
                "weight_dtype": {"_data": "float8_e4m3fn", "_type": "torch.dtype"},
            },
            "_type": "Float8DynamicActivationFloat8WeightConfig",
            "_version": 2,
        }
    },
    "untie_embedding_weights": False,
}


def _inject_torchao_quant_config(output_dir: Path) -> None:
    config_path = output_dir / "config.json"
    if not config_path.is_file():
        logger.warning(
            "no config.json in %s; the export is not directly loadable via transformers.from_pretrained",
            output_dir,
        )
        return
    config = json.loads(config_path.read_text())
    config["quantization_config"] = _TORCHAO_QUANTIZATION_CONFIG
    config_path.write_text(json.dumps(config, indent=2))
    logger.info("Injected torchao quantization_config into %s", config_path)


def _build_model_and_adapter(model_name: str, model_flavor: str, hf_assets_path: Path):
    """Build the model (for state-dict shapes only) and its HF state-dict adapter."""
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
    return ModelWrapper(raw_model), sd_adapter


def _load_dcp_state_dict(model, input_dir: Path, master_dtype: str, read_threads: int):
    """Load the DCP checkpoint into the model's state-dict container."""
    from torchtitan.config import TORCH_DTYPE_MAP

    from torchtitan_npu.tools.checkpoint_conversion.convert_to_hf import ParallelFileSystemReader, load_dcp_model

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
    load_dcp_model(state_dict, reader)
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

    model, sd_adapter = _build_model_and_adapter(model_name, model_flavor, hf_assets_path)
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
        logger.info("  first matches: %s", ", ".join(quantized_fqns[:8]))
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

    _copy_hf_assets(hf_assets_path, output_dir)
    _inject_torchao_quant_config(output_dir)


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
        choices=("all_mxfp8", "mix", "all_block_fp8"),
        default="mix",
        help="Quantization recipe (same semantics as torchtitan_npu.extensions.adaptor.torchao.converter).",
    )
    parser.add_argument(
        "--enable_mxfp4_qat",
        action="store_true",
        help="Serve the block_fp8 recipe's routed-expert weights as MXFP4.",
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
