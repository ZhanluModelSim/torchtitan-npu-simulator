# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 vision-language dataset."""

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.tokenizer import HuggingFaceTokenizer

from .data import TEXT, ImagePatchProcessor, build_image_token_layout
from .dataloader import _MultiModalDataset
from .language_encoder import (
    DeepSeekV41VisionLanguageEncoder,
    ReasoningEffort,
    ThinkingMode,
)

_STACKED_INPUT_KEYS = (
    "input",
    "positions",
    "token_types",
)


class DeepSeekV41VisionLanguageProcessor:
    """Convert chat turns containing arbitrary interleaved text and images.

    Accepts LLaVA/ShareGPT, OpenAI-style messages, and Alpaca instruction rows.
    Text-only rows use the same DeepSeek V4.1 chat template and loss mask.
    """

    def __init__(
        self,
        *,
        dataset_path: str,
        image_root: str,
        tokenizer: HuggingFaceTokenizer,
        vision_language_encoder: DeepSeekV41VisionLanguageEncoder,
        seq_len: int,
    ):
        self.dataset_path = str(Path(dataset_path).resolve())
        self.root = Path(image_root).resolve() if image_root else Path(self.dataset_path).parent
        self.tokenizer = tokenizer
        self.vision_language_encoder = vision_language_encoder
        self.seq_len = seq_len
        self.processor = ImagePatchProcessor()
        if tokenizer.bos_id is None or tokenizer.eos_id is None:
            raise ValueError("V4.1 SFT requires BOS and EOS tokens")

    @staticmethod
    def _message_parts(row: dict) -> list[dict[str, Any]]:
        conversations = row.get("conversations")
        messages = row.get("messages")
        if conversations is not None and messages is not None:
            raise ValueError("provide one conversation format per row")
        if conversations is not None:
            source = conversations
        elif messages is not None:
            source = messages
        elif isinstance(row.get("instruction"), str) and isinstance(row.get("output"), str):
            if row.get("input") is not None and not isinstance(row["input"], str):
                raise ValueError("Alpaca input must be text")
            query = row["instruction"] + ("\n" + row["input"] if row.get("input") else "")
            source = [{"role": "user", "content": query}, {"role": "assistant", "content": row["output"]}]
        else:
            raise ValueError("provide conversations, messages, or an instruction/output pair")
        if not isinstance(source, list) or not source or not isinstance(source[0], dict):
            raise ValueError("V4.1 chat SFT requires non-empty messages")
        if row.get("images") is not None and row.get("image") is not None:
            raise ValueError("use either images or image, not both")
        raw_image_paths = row.get("images") if row.get("images") is not None else row.get("image")
        image_paths: list[object]
        if raw_image_paths is None:
            image_paths = []
        elif isinstance(raw_image_paths, list):
            image_paths = raw_image_paths
        else:
            image_paths = [raw_image_paths]
        if row.get("system") is not None and row.get("system_prompt") is not None:
            raise ValueError("provide either system or system_prompt")
        system = row.get("system") if row.get("system") is not None else row.get("system_prompt")
        if system is not None:
            if not isinstance(system, str) or source[0].get("role", source[0].get("from")) == "system":
                raise ValueError("system must be text and cannot duplicate the first system message")
            source = [{"role": "system", "content": system}, *source]

        parsed = []
        image_index = 0

        def split_text(value: str) -> list[tuple[str, object]]:
            nonlocal image_index
            parts: list[tuple[str, object]] = []
            cursor = 0
            while (start := value.find("<image>", cursor)) >= 0:
                if start > cursor:
                    parts.append(("text", value[cursor:start]))
                source_start = start + len("<image>")
                close = value.find("</image>", source_start)
                next_open = value.find("<image>", source_start)
                if close >= 0 and (next_open < 0 or close < next_open):
                    path = value[source_start:close]
                    if not path:
                        raise ValueError("tagged <image> path cannot be empty")
                    parts.append(("image", path))
                    cursor = close + len("</image>")
                else:
                    if image_index >= len(image_paths):
                        raise ValueError("each <image> marker needs an image")
                    parts.append(("image", image_paths[image_index]))
                    image_index += 1
                    cursor = source_start
            if "</image>" in value[cursor:]:
                raise ValueError("malformed <image>path</image> tag")
            if cursor < len(value):
                parts.append(("text", value[cursor:]))
            return parts

        for message in source:
            if not isinstance(message, dict):
                raise ValueError("conversation messages must be objects")
            role_name = message.get("from", message.get("role"))
            role = (
                {
                    "human": "user",
                    "gpt": "assistant",
                    "developer": "system",
                    "last_reminder": "latest_reminder",
                }.get(role_name, role_name)
                if isinstance(role_name, str)
                else role_name
            )
            if role not in {"system", "user", "assistant", "tool", "latest_reminder"}:
                raise ValueError(f"unsupported DeepSeek V4.1 message role: {role!r}")
            if conversations is not None:
                value = message.get("value", message.get("content"))
                if not isinstance(value, str):
                    raise ValueError("LLaVA message value must be text")
                parts = split_text(value)
            else:
                content = message.get("content_blocks", message.get("content"))
                parts = []
                if isinstance(content, str):
                    parts = split_text(content)
                elif isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            raise ValueError("message content parts must be objects")
                        kind = item.get("type")
                        if kind == "text":
                            if not isinstance(item.get("text"), str):
                                raise ValueError("text content must be a string")
                            parts.extend(split_text(item["text"]))
                        elif kind in ("image", "image_url"):
                            value = next(
                                (
                                    item[key]
                                    for key in ("path", "image", "image_url", "url", "source", "data")
                                    if item.get(key)
                                ),
                                None,
                            )
                            if value is None and image_index < len(image_paths):
                                value = image_paths[image_index]
                                image_index += 1
                            if value is None:
                                raise ValueError("image block needs a local image source")
                            parts.append(("image", value))
                        else:
                            parts.append(("block", item))
                elif content is None:
                    parts = []
                else:
                    raise ValueError("message content must be text or an ordered list of content parts")
            metadata = {
                key: value
                for key, value in message.items()
                if key not in {"from", "value", "role", "content", "content_blocks"}
            }
            parsed.append((role, parts, metadata))
        if image_index != len(image_paths):
            raise ValueError("each image path needs a matching <image> marker")
        messages = []
        for role, parts, metadata in parsed:
            content = []
            for kind, value in parts:
                if kind == "text":
                    content.append({"type": "text", "text": value})
                elif kind == "image":
                    content.append({"type": "image", "source": value})
                else:
                    content.append(value)
            messages.append({**metadata, "role": role, "content": content})
        if row.get("tools"):
            if messages[0]["role"] == "system":
                messages[0]["tools"] = row["tools"]
            else:
                messages.insert(0, {"role": "system", "content": "", "tools": row["tools"]})
        return messages

    def _sample(self, row: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor] | None:
        messages = self._message_parts(row)
        context = row.get("context")
        if isinstance(context, str):
            context = json.loads(context)
        if context is not None and not isinstance(context, list):
            raise ValueError("context must be a list of messages")
        if context:
            aliases = {"developer": "system", "last_reminder": "latest_reminder"}
            if any(not isinstance(message, dict) for message in context):
                raise ValueError("context messages must be objects")
            context = [
                {**message, "role": aliases.get(message.get("role"), message.get("role"))} for message in context
            ]
        _, text_ids, text_mask, image_positions, image_sources = (
            self.vision_language_encoder.encode_messages_with_assistant_mask(
                messages,
                self.tokenizer,
                context=context,
            )
        )
        ids = []
        types = []
        indices = []
        target_mask = []
        patches = []
        grids = []
        feature_base = 0

        def add_image(source: object) -> None:
            nonlocal feature_base
            if isinstance(source, dict):
                if source.get("bytes") is not None:
                    source = source["bytes"]
                elif source.get("type") == "base64":
                    source = f"data:{source.get('media_type', 'image/png')};base64,{source['data']}"
                else:
                    source = source.get("source") or source.get("path") or source.get("url") or source.get("data")
                if isinstance(source, dict):
                    if source.get("type") == "base64":
                        source = f"data:{source.get('media_type', 'image/png')};base64,{source['data']}"
                    else:
                        source = source.get("path") or source.get("url") or source.get("data")
            if isinstance(source, str) and source.startswith("data:image/"):
                header, separator, payload = source.partition(",")
                if not separator or not header.endswith(";base64"):
                    raise ValueError("image data URLs must be base64 encoded")
                source = base64.b64decode(payload, validate=True)
            if isinstance(source, bytes):
                with Image.open(BytesIO(source)) as image:
                    pixels, grid = self.processor.from_image(image)
            elif isinstance(source, Image.Image):
                pixels, grid = self.processor.from_image(source)
            elif isinstance(source, str) and "://" in source:
                raise ValueError(
                    "remote image URLs are not supported; download the image and use a local path or data URL"
                )
            elif isinstance(source, str) and source and "://" not in source:
                path = Path(source)
                image_path = (path if path.is_absolute() else self.root / path).resolve()
                if not path.is_absolute() and not image_path.is_relative_to(self.root):
                    raise ValueError("relative SFT image path must stay inside the image directory")
                pixels, grid = self.processor.from_path(image_path)
            else:
                raise ValueError("image source must be a local path, data URL, or decoded image")
            grid_hw = (int(grid[0]), int(grid[1]))
            layout_ids, layout_types, layout_indices = build_image_token_layout(
                [grid_hw],
                span_start=0,
                vocab_size=self.tokenizer.get_vocab_size(),
                downsample_ratio=self.processor.downsample_ratio,
            )
            patches.append(pixels)
            grids.append(grid_hw)
            ids.extend(layout_ids.tolist())
            types.extend(layout_types.tolist())
            indices.extend(index + feature_base if index >= 0 else -1 for index in layout_indices.tolist())
            target_mask.extend([False] * len(layout_ids))
            ratio = self.processor.downsample_ratio
            feature_base += ((grid_hw[0] + ratio - 1) // ratio) * ((grid_hw[1] + ratio - 1) // ratio)

        cursor = 0
        for image_position, image_source in zip(image_positions, image_sources, strict=True):
            if image_position < cursor:
                raise ValueError("image positions must be ordered")
            ids.extend(text_ids[cursor:image_position])
            types.extend([TEXT] * (image_position - cursor))
            indices.extend([-1] * (image_position - cursor))
            target_mask.extend(text_mask[cursor:image_position])
            add_image(image_source)
            cursor = image_position
        ids.extend(text_ids[cursor:])
        types.extend([TEXT] * (len(text_ids) - cursor))
        indices.extend([-1] * (len(text_ids) - cursor))
        target_mask.extend(text_mask[cursor:])
        if not any(target_mask):
            raise ValueError("V4.1 chat SFT requires an assistant response")
        if len(ids) > self.seq_len:
            return None
        if not patches:
            patch_dim = 3 * self.processor.patch_size**2
            patches = [torch.zeros((1, patch_dim), dtype=torch.bfloat16)]
            grids = [(1, 1)]
        pad = self.seq_len - len(ids)
        tokens = torch.tensor(ids + [self.tokenizer.eos_id] * pad, dtype=torch.long)
        token_types = torch.tensor(types + [TEXT] * pad, dtype=torch.long)
        feature_indices = torch.tensor(indices + [-1] * pad, dtype=torch.long)
        labels = torch.full((self.seq_len,), -100, dtype=torch.long)
        for position, target in enumerate(target_mask):
            if target and position:
                labels[position - 1] = tokens[position]
        inputs = {
            "input": tokens,
            "positions": torch.arange(self.seq_len),
            "pixel_values": pad_sequence(patches, batch_first=True),
            "image_grid": torch.tensor(grids),
            "token_types": token_types,
            "image_feature_indices": feature_indices,
        }
        return inputs, labels

    def __call__(
        self, row: dict | None = None, *, sample: dict | None = None, **_kwargs
    ) -> dict[str, torch.Tensor] | None:
        row = row if row is not None else sample
        if row is None:
            raise ValueError("SFT processor requires a sample")
        processed = self._sample(row)
        if processed is None:
            return None
        inputs, labels = processed
        return {**inputs, "input_ids": inputs["input"], "labels": labels}


class DeepSeekV41VisionLanguageDataset(_MultiModalDataset):
    """Adapt an SFT source while inheriting the existing V4.1 lifecycle."""

    def __init__(self, source, processor, tokenizer, batch_size, seq_len, dp_rank, dp_world_size, infinite):
        self._data = split_dataset_by_node(source, dp_rank, dp_world_size)
        self.sample_processor = processor
        self._tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.enable_packing = False
        self.infinite = infinite
        self._sample_idx = 0
        self._hf_state_restored = False

        image_processor = processor.processor
        self.patch_size = image_processor.patch_size
        self.temporal_patch_size = 1
        self.spatial_merge_size = image_processor.downsample_ratio
        self.min_pixels = image_processor.min_pixels
        self.max_pixels = 0
        self.image_mean = image_processor.mean
        self.image_std = image_processor.std
        self.resize_fn = image_processor.target_grid
        self.max_patches = image_processor.max_n_token
        self.max_patches_per_side = 0
        self.video_dir = ""
        self.video_fps = 2.0
        self.video_min_frames = 0
        self.video_max_frames = 0


class _SFTCollator:
    def __call__(self, batch: list[dict[str, Any]]):
        (sample,) = batch
        inputs = {key: torch.as_tensor(sample[key]).unsqueeze(0) for key in _STACKED_INPUT_KEYS}
        inputs["pixel_values"] = torch.as_tensor(sample["pixel_values"])
        inputs["image_grid"] = torch.as_tensor(sample["image_grid"])
        inputs["image_feature_indices"] = torch.as_tensor(sample["image_feature_indices"]).unsqueeze(0)
        return inputs, torch.as_tensor(sample["labels"]).unsqueeze(0)


class DeepSeekV41VisionLanguageDataLoader(ParallelAwareDataloader):
    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        image_root: str = ""
        infinite: bool = True
        thinking_mode: ThinkingMode = "chat"
        drop_thinking: bool = True
        add_default_bos_token: bool = True
        reasoning_effort: ReasoningEffort | None = None

    def __init__(
        self,
        config: Config,
        *,
        tokenizer: HuggingFaceTokenizer,
        dp_world_size: int,
        dp_rank: int,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = 1,
        **kwargs,
    ):
        if local_batch_size != 1:
            raise ValueError("DeepSeek V4.1 uses one sequence per data-parallel rank (local_batch_size=1)")
        if not config.dataset_path:
            raise ValueError("set --dataloader.dataset-path to a JSON, JSONL, or Parquet SFT dataset")
        dataset_path = str(Path(config.dataset_path).resolve())
        suffix = Path(dataset_path).suffix.lower()
        if suffix not in (".json", ".jsonl", ".parquet"):
            raise ValueError("vision SFT dataset must be JSON, JSONL, or Parquet")
        vision_language_encoder = DeepSeekV41VisionLanguageEncoder(
            str(Path(tokenizer.tokenizer_path) / "encoding" / "encoding.py"),
            thinking_mode=config.thinking_mode,
            drop_thinking=config.drop_thinking,
            add_default_bos_token=config.add_default_bos_token,
            reasoning_effort=config.reasoning_effort,
        )
        processor = DeepSeekV41VisionLanguageProcessor(
            dataset_path=dataset_path,
            image_root=config.image_root,
            tokenizer=tokenizer,
            vision_language_encoder=vision_language_encoder,
            seq_len=seq_len,
        )
        source = load_dataset(
            "parquet" if suffix == ".parquet" else "json",
            data_files={"train": dataset_path},
            split="train",
            streaming=True,
        )
        dataset = DeepSeekV41VisionLanguageDataset(
            source,
            processor,
            tokenizer,
            local_batch_size,
            seq_len,
            dp_rank,
            dp_world_size,
            config.infinite,
        )
        super().__init__(
            dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            num_workers=config.num_workers,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            snapshot_every_n_steps=snapshot_every_n_steps,
            batch_size=local_batch_size,
            collate_fn=_SFTCollator(),
        )
