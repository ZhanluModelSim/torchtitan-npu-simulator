# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""Composable native Grain nodes and a narrow resume compatibility guard."""

import hashlib
import importlib
import importlib.metadata
import json
import socket
import sys
from dataclasses import dataclass, fields, replace
from glob import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload

import grain.python as grain
import torch.distributed as dist
from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    IndexedJsonlSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.sources import _file_patterns_to_paths
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import derive, override

from . import chat_processor

if TYPE_CHECKING:
    from renderers import Message


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def chat_processor_digest():
    return digest([file_digest(chat_processor.__file__)])


def tokenizer_identity(tokenizer, assets_path, *, chat_template=None):
    """Bind files and effective template/tokenizer behavior without dataset re-tokenization."""
    root = Path(assets_path)
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "tokenizer.model",
        "vocab.json",
        "merges.txt",
    )
    files = {name: file_digest(root / name) for name in names if (root / name).is_file()}
    if not files:
        raise ValueError("Tokenizer assets must contain tokenizer/template files")
    probes = [
        [{"role": "user", "content": "Question?中文"}, {"role": "assistant", "content": "Answer."}],
        [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "A", "reasoning_content": "Reason"},
            {"role": "user", "content": "Next"},
            {"role": "assistant", "content": "B"},
        ],
    ]
    if chat_template is None:
        rendered = [tokenizer.apply_chat_template(messages) for messages in probes]
    else:
        # The processor renders external templates with the actual conversation and tools.
        # Its template bytes are recorded separately. Probe only the tokenizer here.
        rendered = []
        for messages in probes:
            for message in messages:
                rendered.append(message["content"])
    encoded = [tokenizer.encode(text, add_bos=True, add_eos=False) for text in rendered]
    backend = getattr(tokenizer, "tokenizer", None)
    backend_identity = digest(json.loads(backend.to_str())) if hasattr(backend, "to_str") else None
    versions = {}
    for name in ("tokenizers", "jinja2"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    encode_code = getattr(tokenizer.encode, "__code__", None)
    provider = Path(encode_code.co_filename) if encode_code is not None else None
    return {
        "files": files,
        "eos_id": tokenizer.eos_id,
        "behavior": digest(encoded),
        "backend": backend_identity,
        "versions": versions,
        "provider_sha256": file_digest(provider) if provider and provider.is_file() else None,
    }


def _tokenizer_contract(identity):
    return {key: identity[key] for key in ("files", "eos_id", "behavior")}


class _TrainableShard(grain.MapDataset):
    """Bound filtered reads while retaining Grain's map indices and cursor."""

    def __init__(self, parent, *, dp_rank, dp_world_size, context_length, shuffle):
        super().__init__(parent)
        self._shard = parent
        self._size = len(parent)
        self._search_size = self._size * (4 if shuffle else 1)
        self._description = f"Qwen SFT DP rank {dp_rank}/{dp_world_size} at context length {context_length}"
        self._first_index, self._first_sample = self._first_trainable(0)

    def _first_trainable(self, start):
        for index in range(start, start + self._search_size):
            sample = self._shard[index]
            if sample is not None:
                return index, sample
        raise ValueError(f"{self._description}: no trainable samples in {self._search_size} consecutive rows")

    def __len__(self):
        return self._size

    @overload
    def __getitem__(self, index: slice) -> grain.MapDataset:
        """Return a view of a range of rows."""

    @overload
    def __getitem__(self, index: int) -> Any:
        """Return the sampled row at an absolute index."""

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return self.slice(index)
        if index < 0:
            index %= self._size
        if index < self._first_index:
            return None
        if index == self._first_index:
            return self._first_sample
        sample = self._shard[index]
        # A shuffled repeat can have an empty shard in a later epoch too.
        if sample is None and index % self._size == 0:
            self._first_trainable(index)
        return sample


@dataclass(frozen=True, kw_only=True, slots=True)
class _QwenRawDataset(SingleDatasetConfig):
    def build(self, *, context, dataset_iteration_policy):
        if not dataset_iteration_policy.repeat:
            return SingleDatasetConfig.build(self, context=context, dataset_iteration_policy=dataset_iteration_policy)
        shard = SingleDatasetConfig.build(
            self, context=context, dataset_iteration_policy=replace(dataset_iteration_policy, repeat=False)
        )
        return _TrainableShard(
            shard,
            dp_rank=dataset_iteration_policy.dp_rank,
            dp_world_size=dataset_iteration_policy.dp_world_size,
            context_length=context.max_context_length,
            shuffle=dataset_iteration_policy.shuffle,
        ).repeat()


def _check_rank_inputs(fingerprints: dict[str, str] | None, error: Exception | None = None):
    if not dist.is_initialized() or dist.get_world_size() == 1 or dist.get_backend() == "fake":
        return
    ranks: list[tuple[str, dict[str, str], str | None]] = [("", {}, None)] * dist.get_world_size()
    failure = None if error is None else f"{type(error).__name__}: {error}"
    dist.all_gather_object(ranks, (socket.gethostname(), fingerprints or {}, failure))
    errors = [f"rank {rank} ({host}): {reason}" for rank, (host, _, reason) in enumerate(ranks) if reason is not None]
    if errors:
        raise ValueError("Qwen SFT dataloader startup failed: " + "; ".join(errors))

    reference = ranks[0][1]
    mismatches = []
    for rank, (host, inputs, _) in enumerate(ranks[1:], start=1):
        fields = sorted(key for key in inputs.keys() | reference.keys() if inputs.get(key) != reference.get(key))
        if fields:
            mismatches.append(f"rank {rank} ({host}): {', '.join(fields)}")
    if mismatches:
        raise ValueError("Qwen SFT inputs differ from rank 0: " + "; ".join(mismatches))


class SFTGrainDataLoader(GrainDataLoader):
    """Delegate all iteration/state to Grain; guard the state input contract."""

    @dataclass(kw_only=True, slots=True)
    class Config(GrainDataLoader.Config):
        input_contract: dict

    def __init__(
        self, config, *, dp_rank, dp_world_size, tokenizer, max_context_length, num_tokens_per_microbatch, **kwargs
    ):
        try:
            contract = self._build_contract(
                config, dp_rank, dp_world_size, tokenizer, max_context_length, num_tokens_per_microbatch
            )
            super().__init__(
                config,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                tokenizer=tokenizer,
                max_context_length=max_context_length,
                num_tokens_per_microbatch=num_tokens_per_microbatch,
                **kwargs,
            )
        except Exception as error:
            _check_rank_inputs(None, error)
            raise

        # Match data and tokenizer behavior across ranks. Keep runtime provenance in the saved cursor.
        shared = {key: digest(value) for key, value in contract.items() if key not in ("dp_rank", "input")}
        shared["versions"] = digest({key: contract["versions"][key] for key in ("grain", "torch", "torch-npu")})
        shared.update({key: digest(value) for key, value in contract["input"].items() if key != "transformers"})
        shared["tokenizer"] = digest(_tokenizer_contract(contract["input"]["tokenizer"]))
        try:
            _check_rank_inputs(shared)
        except Exception:
            self.close()
            raise
        self._contract_fingerprint = digest(contract)
        # Per-DP keys remain distinct when distributed DCP merges rank states.
        self._contract_key = f"qwen_sft_contract_dp_rank_{dp_rank}"

    @staticmethod
    def _build_contract(config, dp_rank, dp_world_size, tokenizer, max_context_length, num_tokens_per_microbatch):
        expected = config.input_contract
        raw = expected.get("raw_source")
        patterns = expected.get("raw_patterns")
        template = expected["chat_template"]
        if raw is None and not patterns:
            raise ValueError("Provide Qwen SFT data with --dataset-path or DATA_FILES")
        files = [raw] if raw is not None else _file_patterns_to_paths(patterns)
        hashes = [file_digest(path) for path in files]
        expected = {
            "raw_sha256": hashes[0] if len(hashes) == 1 else digest(hashes),
            "template_sha256": file_digest(template) if template is not None else None,
            "loss_mask": expected["loss_mask"],
            "tokenizer": tokenizer_identity(tokenizer, tokenizer.tokenizer_path, chat_template=template),
            "processor_sha256": chat_processor_digest(),
            "transformers": importlib.metadata.version("transformers"),
        }
        versions = {}
        for name in ("grain", "torch", "torch-npu", "tokenizers", "jinja2"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        upstream = {
            name: file_digest(importlib.import_module("torchtitan.components.data." + name).__file__)
            for name in ("loader", "dataset", "packing", "collators", "sources", "types")
        }
        return {
            "input": expected,
            "seed": config.seed,
            "shuffle": config.shuffle,
            "repeat": config.repeat,
            "context": max_context_length,
            "microbatch_tokens": num_tokens_per_microbatch,
            "dp_rank": dp_rank,
            "dp_world_size": dp_world_size,
            "max_num_documents": config.max_num_documents,
            "packing_bins": getattr(config.dataset, "num_packing_bins", None),
            "prefetch": config.num_prefetch_microbatches,
            "read_options": repr(config.read_options),
            "collator": {field.name: repr(getattr(config.collator, field.name)) for field in fields(config.collator)},
            "collator_type": type(config.collator).__module__ + "." + type(config.collator).__qualname__,
            "python": list(sys.version_info[:2]),
            "versions": versions,
            "upstream": upstream,
            "schema": "qwen.sft.loader.v2",
        }

    def state_dict(self):
        return super().state_dict() | {self._contract_key: self._contract_fingerprint}

    def load_state_dict(self, state_dict):
        if state_dict.get(self._contract_key) != self._contract_fingerprint:
            raise ValueError(
                "Incompatible Qwen Grain state: input, packing, topology, policy, runtime, or loader version"
            )
        super().load_state_dict(state_dict)


def _has_supervision(sample) -> bool:
    return sample is not None and bool((sample.labels != IGNORE_INDEX).any())


def raw_sft_dataset(
    *,
    dataset_path: str | None = None,
    patterns: tuple[str, ...] = (),
    chat_template: str | None = None,
    loss_mask: str = chat_processor.DEFAULT_LOSS_MASK,
) -> tuple[SingleDatasetConfig, dict[str, Any]]:
    """Build the same raw dataset for native recipes and command-line overrides."""
    processor = chat_processor.QwenSFTChatProcessor.Config(chat_template=chat_template, loss_mask=loss_mask)
    contract: dict[str, Any] = {"chat_template": chat_template, "loss_mask": processor.loss_mask}
    if dataset_path is None:
        contract["raw_patterns"] = patterns
    else:
        path = Path(dataset_path).resolve()
        contract["raw_source"] = str(path)
        patterns = (escape(str(path)),)
    dataset = _QwenRawDataset(
        source=IndexedJsonlSource.Config(patterns=patterns),
        processor=processor,
        post_filters=(_has_supervision,),
    )
    return dataset, contract


def _messages(sample: dict[str, Any]) -> list["Message"]:
    return cast("list[Message]", sample["messages"])


@override(target=HuggingFaceTokenizer.Config, fqns=["tokenizer"])
def qwen_sft_tokenizer(_config: HuggingFaceTokenizer.Config) -> chat_processor.QwenSFTTokenizer.Config:
    return chat_processor.QwenSFTTokenizer.Config()


@override(target=GrainDataLoader.Config, fqns=["dataloader"])
def chat_sft(
    config: GrainDataLoader.Config,
    *,
    dataset_path: str,
    chat_template: str | None = None,
    loss_mask: str | None = None,
) -> GrainDataLoader.Config:
    packing = config.dataset
    if (
        not isinstance(packing, FirstFitPackingConfig)
        or not isinstance(packing.dataset, SingleDatasetConfig)
        or not isinstance(packing.dataset.processor, chat_processor.QwenSFTChatProcessor.Config)
    ):
        raise ValueError("Select a Qwen raw SFT recipe for chat template and loss mask options")
    processor = packing.dataset.processor
    selected, input_contract = raw_sft_dataset(
        dataset_path=dataset_path,
        chat_template=chat_template if chat_template is not None else processor.chat_template,
        loss_mask=loss_mask if loss_mask is not None else processor.loss_mask,
    )
    dataset = replace(packing.dataset, source=selected.source, processor=selected.processor)
    return derive(
        config,
        SFTGrainDataLoader.Config,
        dataset=replace(packing, dataset=dataset),
        input_contract=input_contract,
    )
