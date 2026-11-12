# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, cast

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import OverrideConfig, derive
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import decoder_vocab_size

from torchtitan_npu.config import (
    CompileConfig,
    ModelCompileConfig,
    OptimConfig,
    OptimizerConfig,
    TrainingConfig,
)
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.common.alignment import AlignedTextProcessor, compression_alignment
from torchtitan_npu.models.deepseek_v4_1 import model_registry
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model, DeepSeekV41MultimodalModel
from torchtitan_npu.models.deepseek_v4_1.optimizer import _v41_optimizer_config
from torchtitan_npu.models.deepseek_v4_1.trainer import DeepSeekV41Trainer

if TYPE_CHECKING:
    from torchtitan.config.transform import ModelConfigConverter
    from torchtitan.trainer import Trainer

    from torchtitan_npu.models.deepseek_v4_1.vision.language_encoder import ReasoningEffort, ThinkingMode


def _per_doc_alignment(model: DeepSeekV41Model.Config) -> int:
    """The pooling granularity a packed document must be padded to.

    Derived from the model's compression ratios so the loader and the compressor
    cannot disagree: a document whose length is not a multiple of every ratio would
    get a group straddling its edge.
    """
    alignment = compression_alignment(model.compress_ratios)
    return int(os.environ.get("DSV41_PER_DOC_ALIGNMENT", alignment))


def _vision_dataloader_config(model: DeepSeekV41Model.Config):
    """The image-conditioned caption loader, for a multimodal model only.

    Imported here rather than at module scope: the multimodal dataset stack pulls in the
    image-preprocessing dependencies, and a text-only run must not need them.
    """
    from torchtitan_npu.models.deepseek_v4_1.vision.dataloader import caption_dataloader_config

    return caption_dataloader_config(
        dataset=os.environ.get("DATASET", "cc12m-test"),
        dataset_path=os.environ.get("DATASET_PATH") or None,
        num_packing_bins=int(os.environ.get("PACKING_BUFFER_SIZE", "0")),
        per_doc_alignment=_per_doc_alignment(model),
    )


def _text_data_files() -> str:
    path = os.environ.get("DATA_FILES", "tests/assets/c4_test/data.json")
    if not path.strip():
        raise ValueError("DATA_FILES must be a non-empty JSON/JSONL file path when set")
    return path


def _make_trainer_config(
    flavor: str,
    *,
    converters=None,
    dataloader_factory: Callable[[DeepSeekV41Model.Config], GrainDataLoader.Config] | None = None,
) -> TrainerEx.Config:
    model = model_registry(flavor, converters=converters)
    model = derive(
        model,
        type(model),
        fp32_param_patterns=(
            "layers.*.hc_attn_pre.*",
            "layers.*.hc_ffn_pre.*",
            "layers.*.attention.attn_sink",
        ),
        local_compile_regions=[],
    )
    # The upstream validator requires model, data, and RoPE context lengths to agree.
    model.max_context_length = min(4096, model.max_context_length)
    from torchtitan.models.common.rope import RoPE

    for _, rope, _, _ in model.traverse(RoPE.Config):
        rope.max_context_length = model.max_context_length
    if model.n_layers != len(model.layers):  # pyrefly: ignore [missing-attribute]
        raise ValueError("registered V4.1 model does not describe every configured layer")

    if dataloader_factory is not None:
        dataloader = dataloader_factory(model)
    elif isinstance(model, DeepSeekV41MultimodalModel.Config):
        dataloader = _vision_dataloader_config(model)
    else:
        dataloader = GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json",
                        split="train",
                        load_dataset_kwargs={"data_files": _text_data_files()},
                    ),
                    processor=AlignedTextProcessor.Config(per_doc_alignment=_per_doc_alignment(model)),
                    post_filters=(lambda sample: sample is not None,),
                ),
            ),
        )

    return DeepSeekV41Trainer.Config(
        training=TrainingConfig(
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=min(4096, model.max_context_length),
            max_context_length=min(4096, model.max_context_length),
        ),
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
            ),
        ),
        model=model,
        tokenizer=HuggingFaceTokenizer.Config(),
        dataloader=dataloader,
        optim=OptimConfig(optimizer=_v41_optimizer_config(model)),
        parallelism=ParallelismConfig(),
        activation_remat=None,
        override=OverrideConfig(imports=["torchtitan_npu.override.common.rope.workaround"]),
    )


def _vision_sft_dataloader_config(
    model: DeepSeekV41Model.Config,
    *,
    dataset_path: str,
    image_root: str,
    thinking_mode: "ThinkingMode",
    drop_thinking: bool,
    add_default_bos_token: bool,
    reasoning_effort: "ReasoningEffort | None",
) -> GrainDataLoader.Config:
    """The one-chat-context loader graph for a multimodal model.

    This is the whole loader, not a caption-loader variant: the SFT rows carry
    conversations and images, so the caption dataset, packer and collator do not
    participate at all.
    """
    if not isinstance(model, DeepSeekV41MultimodalModel.Config):
        raise ValueError(f"V4.1 vision SFT requires a multimodal flavor, got {type(model).__name__}")

    # Imported here rather than at module scope: the vision dataset stack pulls in
    # the image-preprocessing dependencies a text-only run must not need.
    from torchtitan_npu.models.deepseek_v4_1.vision.language_dataset import sft_dataloader_config

    return sft_dataloader_config(
        dataset_path=dataset_path,
        image_root=image_root,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        add_default_bos_token=add_default_bos_token,
        reasoning_effort=reasoning_effort,
        per_doc_alignment=_per_doc_alignment(model),
    )


def deepseek_v4_1_vision_sft(
    *,
    dataset_path: str | None = None,
    image_root: str = "",
    thinking_mode: "ThinkingMode" = "chat",
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: "ReasoningEffort | None" = None,
    flavor: str = "deepseek_v4_1_debugmodel",
    converters: list["ModelConfigConverter.Config"] | None = None,
) -> TrainerEx.Config:
    """A multimodal flavor trained on one chat context per Grain batch.

    The loader is installed while the training config is built, instead of being
    derived from ``_vision_dataloader_config`` afterwards: SFT and caption
    pre-training share the model and trainer policy, but no data graph.
    """
    dataset_path = dataset_path or os.environ.get("DSV41_SFT_DATASET") or os.environ.get("DATASET_PATH", "")
    if not dataset_path:
        raise ValueError("V4.1 vision SFT requires a JSON, JSONL, or Parquet dataset (DSV41_SFT_DATASET)")
    return _make_trainer_config(
        flavor,
        converters=converters,
        dataloader_factory=partial(
            _vision_sft_dataloader_config,
            dataset_path=dataset_path,
            image_root=image_root,
            thinking_mode=thinking_mode,
            drop_thinking=drop_thinking,
            add_default_bos_token=add_default_bos_token,
            reasoning_effort=reasoning_effort,
        ),
    )


def _vision_sft_env(name: str, default: str) -> str:
    """Read one SFT knob for the zero-argument registry entries."""
    return os.environ.get(f"DSV41_SFT_{name}", default)


def _vision_sft_env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(f"DSV41_SFT_{name}")
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _vision_sft_thinking_mode() -> "ThinkingMode":
    mode = _vision_sft_env("THINKING_MODE", "chat")
    if mode not in ("chat", "thinking"):
        raise ValueError(f"DSV41_SFT_THINKING_MODE must be 'chat' or 'thinking', got {mode!r}")
    return mode


def _vision_sft_recipe(flavor: str) -> TrainerEx.Config:
    """The ``--config`` entry for SFT: ``--config`` names take no arguments.

    The dataset path falls back to ``DSV41_SFT_DATASET`` / ``DATASET_PATH`` inside
    the recipe; the encoder policy is read here.
    """
    return deepseek_v4_1_vision_sft(
        flavor=flavor,
        image_root=_vision_sft_env("IMAGE_ROOT", ""),
        thinking_mode=_vision_sft_thinking_mode(),
        drop_thinking=_vision_sft_env_flag("DROP_THINKING", True),
        add_default_bos_token=_vision_sft_env_flag("ADD_DEFAULT_BOS_TOKEN", True),
        reasoning_effort=_vision_sft_env("REASONING_EFFORT", "") or None,
    )


def deepseek_v4_1_flash() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash")


def deepseek_v4_1_flash_sft() -> TrainerEx.Config:
    return _vision_sft_recipe("deepseek_v4_1_flash")


def deepseek_v4_1_flash_text() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash_text")


def deepseek_v4_1_flash_40layers_16experts_multimodal() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash_40layers_16experts_vision")


def deepseek_v4_1_flash_40layers_16experts_multimodal_sft() -> TrainerEx.Config:
    return _vision_sft_recipe("deepseek_v4_1_flash_40layers_16experts_vision")


def deepseek_v4_1_flash_40layers_16experts_text() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash_40layers_16experts_text")


def deepseek_v4_1_flash_40layers_24experts_multimodal() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash_40layers_24experts_multimodal")


def deepseek_v4_1_flash_40layers_24experts_multimodal_sft() -> TrainerEx.Config:
    return _vision_sft_recipe("deepseek_v4_1_flash_40layers_24experts_multimodal")


def deepseek_v4_1_flash_40layers_24experts_text() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash_40layers_24experts_text")


def deepseek_v4_1_debugmodel_multimodal() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_debugmodel")


def deepseek_v4_1_debugmodel_multimodal_sft() -> TrainerEx.Config:
    return _vision_sft_recipe("deepseek_v4_1_debugmodel")


def deepseek_v4_1_debugmodel_text() -> TrainerEx.Config:
    config = _make_trainer_config("deepseek_v4_1_debugmodel_text")
    config.hf_assets_path = os.environ.get("HF_ASSETS_PATH") or "tests/assets/deepseek_v3"
    return config


def deepseek_v4_1_stripped() -> TrainerEx.Config:
    """The single-card flavor: debug widths, released indexer geometry, ten layers.

    See :func:`torchtitan_npu.models.deepseek_v4_1._stripped` for what is and is not
    shrunk; the point of the flavor is to reach all three branches of the fused selector
    on one card.
    """
    return _make_trainer_config("deepseek_v4_1_stripped")


def deepseek_v4_1_flash_lora() -> TrainerEx.Config:
    return _make_lora_trainer_config("deepseek_v4_1_flash")


def deepseek_v4_1_flash_text_lora() -> TrainerEx.Config:
    return _make_lora_trainer_config("deepseek_v4_1_flash_text")


def deepseek_v4_1_flash_text_cp() -> TrainerEx.Config:
    raise NotImplementedError("Context parallelism is deferred in the v0.4 eager upgrade.")


def deepseek_v4_1_debugmodel_lora() -> TrainerEx.Config:
    return _make_lora_trainer_config("deepseek_v4_1_debugmodel_text")


def _make_lora_trainer_config(flavor: str) -> TrainerEx.Config:
    from torchtitan.config.transform import apply_transforms

    from torchtitan_npu.models.common.deepseek_lora import deepseek_v4_lora_transform

    config = _make_trainer_config(flavor)
    config.optim.optimizer = OptimizerConfig(
        _optimizer_defaults={"AdamW": [AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-6)]},
    )
    from torchtitan_npu.models.deepseek_v4_1.checkpoint import DeepSeekV41LoRACheckpointManager

    config = apply_transforms(config, [deepseek_v4_lora_transform()])
    assert isinstance(config, TrainerEx.Config)
    assert isinstance(config.model, DeepSeekV41Model.Config)
    assert isinstance(config.optim.optimizer, OptimizerConfig)
    defaults = _v41_optimizer_config(config.model)
    config.optim.optimizer._muon_profile = defaults._muon_profile
    config.optim.optimizer._optimizer_defaults["Muon"] = defaults._optimizer_defaults["Muon"]
    config.checkpointer = DeepSeekV41LoRACheckpointManager.Config()
    return config


def deepseek_v4_1_flash_cpt() -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_1_flash()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    # The registered converter promotes this upstream schema before CLI parsing.
    cast("Trainer.Config", config).checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_1_flash_40layers_16experts_multimodal_compiled() -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_1_flash_40layers_16experts_multimodal()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config


def deepseek_v4_1_flash_40layers_24experts_multimodal_compiled() -> TrainerEx.Config:
    """Launch recipe with explicit compilation and checkpoint policy."""
    config = deepseek_v4_1_flash_40layers_24experts_multimodal()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config


def deepseek_v4_1_flash_sft_compiled() -> TrainerEx.Config:
    """Launch the native SFT recipe with explicit model compilation."""
    config = deepseek_v4_1_flash_sft()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    cast("Trainer.Config", config).checkpointer = CheckpointManager.Config(load_only=True)
    return config


def deepseek_v4_1_flash_40layers_16experts_multimodal_sft_compiled() -> TrainerEx.Config:
    """Launch the native SFT recipe with explicit model compilation."""
    config = deepseek_v4_1_flash_40layers_16experts_multimodal_sft()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config


def deepseek_v4_1_flash_40layers_24experts_multimodal_sft_compiled() -> TrainerEx.Config:
    """Launch the native SFT recipe with explicit model compilation."""
    config = deepseek_v4_1_flash_40layers_24experts_multimodal_sft()
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    return config
