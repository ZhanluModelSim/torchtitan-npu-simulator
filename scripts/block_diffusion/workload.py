# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Hardware-independent block-diffusion workload descriptions.

This is a semantic workload adapter, not a latency predictor or an eager-op trace.
Matrix FLOPs count multiply/add as two and attention uses dense QK/PV shapes;
mask-aware kernels may do less work. Vector operations are present with unknown
FLOPs rather than silently counted as free. Tensor sizes, logical KV accesses,
parameter residency, and collective payloads are separate quantities: none is a
claim about peak workspace, physical HBM traffic, or topology link traffic.

Distributed convention: batch_size is GLOBAL independent requests. EP partitions
whole batch rows contiguously (ragged/empty partitions are valid). Non-expert
parameters are replicated over EP; TP shards heads, vocabulary and every FFN's
intermediate dimension. Experts are owned contiguously over EP and their
intermediate dimension is TP-sharded. Each TP lane dispatches full hidden-state
pairs to expert owners, experts TP-reduce their packed outputs, and combine
returns results to the request owner. Routing scores remain at the source.
All-to-all activation payloads exclude local pairs; assumed int64 routing records
are listed separately. Collectives are listed once PER PARTICIPANT, not once per
group. This supported layout is called ``batch_sharded``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torchtitan_npu.simulator.raw_model.block_diff.model import ModelConfig, RoutingPlan
else:  # Direct execution/import with this directory on sys.path.
    from raw_reference import ModelConfig, RoutingPlan


def _integer(name: str, value: int, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype, device="meta").element_size()


def _json(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: _json(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, torch.dtype):
        return _dtype_name(value)
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class ParallelSpec:
    tp_size: int = 1
    ep_size: int = 1
    ep_layout: str = "batch_sharded"

    def __post_init__(self) -> None:
        _integer("tp_size", self.tp_size, 1)
        _integer("ep_size", self.ep_size, 1)
        if self.ep_layout != "batch_sharded":
            raise ValueError("only the batch_sharded EP layout is supported")


@dataclass(frozen=True)
class PhaseSpec:
    phase: str
    batch_size: int
    query_length: int
    prefix_length: int = 0
    head_positions: tuple[int, ...] | None = None
    attention_policy: str = "bidirectional"
    unresolved_tokens: int | None = None
    routing_plan: RoutingPlan | tuple[RoutingPlan, ...] | None = None
    block_index: int | None = None
    step_index: int | None = None
    accepted_tokens: int | None = None
    sampling_policy: str = "none"
    head_selection: str = "explicit"

    def __post_init__(self) -> None:
        if self.phase not in ("prefill", "denoise", "commit"):
            raise ValueError("phase must be prefill, denoise or commit")
        if self.attention_policy not in ("causal", "bidirectional"):
            raise ValueError("attention_policy must be causal or bidirectional")
        _integer("batch_size", self.batch_size, 1)
        _integer("query_length", self.query_length, 1)
        _integer("prefix_length", self.prefix_length)
        if self.head_positions is not None:
            if not isinstance(self.head_positions, tuple):
                raise ValueError("head_positions must be a tuple or None")
            for p in self.head_positions:
                if isinstance(p, bool) or not isinstance(p, int) or not 0 <= p < self.query_length:
                    raise ValueError("head position outside the current query")
        if self.unresolved_tokens is not None:
            _integer("unresolved_tokens", self.unresolved_tokens)
            if self.unresolved_tokens > self.query_length:
                raise ValueError("unresolved_tokens exceeds query_length")
        if self.head_selection not in ("explicit", "unresolved"):
            raise ValueError("head_selection must be explicit or unresolved")
        if self.head_selection == "unresolved" and (self.unresolved_tokens is None or self.head_positions is not None):
            raise ValueError("unresolved head selection requires unresolved_tokens and no explicit head_positions")
        if self.accepted_tokens is not None:
            _integer("accepted_tokens", self.accepted_tokens)
            limit = self.query_length if self.unresolved_tokens is None else self.unresolved_tokens
            if self.accepted_tokens > limit:
                raise ValueError("accepted_tokens exceeds unresolved/query tokens")
            if self.phase != "denoise":
                raise ValueError("accepted_tokens is only defined for denoise phases")
        if self.sampling_policy not in ("none", "greedy_confidence"):
            raise ValueError("sampling_policy must be none or greedy_confidence")
        if self.sampling_policy != "none":
            if self.phase != "denoise" or self.accepted_tokens is None:
                raise ValueError("sampling requires a denoise phase with explicit accepted_tokens")
            if self.head_tokens == 0:
                raise ValueError("sampling needs logits even for a zero-accepted round")
            if self.head_tokens < self.accepted_tokens:
                raise ValueError("sampling needs logits for at least accepted_tokens positions")
            if self.head_positions is not None and len(set(self.head_positions)) != len(self.head_positions):
                raise ValueError("sampling head positions must be distinct")
        for name in ("block_index", "step_index"):
            if getattr(self, name) is not None:
                _integer(name, getattr(self, name))

    @property
    def head_tokens(self) -> int:
        if self.head_selection == "unresolved":
            assert self.unresolved_tokens is not None
            return self.unresolved_tokens
        return self.query_length if self.head_positions is None else len(self.head_positions)

    @property
    def cache_length_after(self) -> int:
        return self.prefix_length + (self.query_length if self.phase != "denoise" else 0)


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str

    @property
    def numel(self) -> int:
        result = 1
        for d in self.shape:
            result *= d
        return result

    @property
    def nbytes(self) -> int:
        return self.numel * _bytes(getattr(torch, self.dtype))


@dataclass(frozen=True)
class Event:
    name: str
    kind: str
    layer: int | None
    inputs: tuple[TensorSpec, ...] = ()
    output: TensorSpec | None = None
    flops: int | None = None
    payload_bytes: int = 0
    received_payload_bytes: int = 0
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class RankWorkload:
    rank: int
    tp_rank: int
    ep_rank: int
    batch_start: int
    local_batch_size: int
    parameter_bytes: int
    kv_before_bytes: int
    kv_after_bytes: int
    routed_pairs_per_layer: tuple[int, ...]
    events: tuple[Event, ...]

    @property
    def matrix_flops(self) -> int:
        return sum(e.flops or 0 for e in self.events)


@dataclass(frozen=True)
class PhaseWorkload:
    phase: PhaseSpec
    parallel: ParallelSpec
    ranks: tuple[RankWorkload, ...]
    routing_matrices: tuple[tuple[tuple[int, ...], ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return _json(self)


@dataclass(frozen=True)
class WorkloadReport:
    phases: tuple[PhaseWorkload, ...]
    generated_tokens: int
    model_config: ModelConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scope": "inference_semantic_workload",
            "generated_tokens": self.generated_tokens,
            "model_config": _json(self.model_config),
            "flop_convention": "dense matrix multiply/add = 2; vector FLOPs unspecified",
            "traffic_convention": "logical accesses and collective payloads; not HBM or link traffic",
            "phases": [_json(p) for p in self.phases],
        }


def _batch_partition(batch_size: int, ep_size: int) -> tuple[tuple[int, int], ...]:
    q, r = divmod(batch_size, ep_size)
    start = 0
    result = []
    for ep in range(ep_size):
        size = q + int(ep < r)
        result.append((start, size))
        start += size
    return tuple(result)


def _validate_parallel(c: ModelConfig, p: ParallelSpec) -> None:
    divisible = {
        "num_attention_heads": c.num_attention_heads,
        "num_key_value_heads": c.num_key_value_heads,
        "vocab_size": c.vocab_size,
    }
    if c.num_experts == 0 or c.num_shared_experts:
        divisible["intermediate_size"] = c.intermediate_size
    if c.num_experts:
        divisible["moe_intermediate_size"] = c.moe_intermediate_size
        if c.num_experts % p.ep_size:
            raise ValueError("num_experts must be divisible by ep_size")
    elif p.ep_size != 1:
        raise ValueError("EP > 1 is only supported for routed MoE")
    for name, value in divisible.items():
        if value % p.tp_size:
            raise ValueError(f"{name} must be divisible by tp_size; replicated KV heads are not supported")


def _routing(c: ModelConfig, phase: PhaseSpec, p: ParallelSpec):
    if not c.num_experts:
        if phase.routing_plan is not None:
            raise ValueError("routing_plan requires a routed MoE model")
        return (), (), ()
    n = phase.batch_size * phase.query_length
    if phase.routing_plan is None:
        plans = (RoutingPlan.balanced(n, c.num_experts, c.num_experts_per_tok),) * c.num_hidden_layers
    elif isinstance(phase.routing_plan, RoutingPlan):
        plans = (phase.routing_plan,) * c.num_hidden_layers
    else:
        plans = tuple(phase.routing_plan)
        if len(plans) != c.num_hidden_layers or not all(isinstance(plan, RoutingPlan) for plan in plans):
            raise ValueError("routing_plan must have one RoutingPlan per layer")
    partitions = _batch_partition(phase.batch_size, p.ep_size)
    sources = [ep for ep, (_, b) in enumerate(partitions) for _ in range(b * phase.query_length)]
    matrices, counts = [], []
    for plan in plans:
        if len(plan.expert_indices) != n:
            raise ValueError("routing plan token count differs from global batch * query length")
        matrix = [[0] * p.ep_size for _ in range(p.ep_size)]
        expert_counts = [0] * c.num_experts
        for token, selected in enumerate(plan.expert_indices):
            if len(selected) != c.num_experts_per_tok or len(set(selected)) != len(selected):
                raise ValueError("routing plan must select distinct top-k experts for every token")
            for expert in selected:
                if isinstance(expert, bool) or not isinstance(expert, int) or not 0 <= expert < c.num_experts:
                    raise ValueError("expert index out of range")
                expert_counts[expert] += 1
                matrix[sources[token]][expert // (c.num_experts // p.ep_size)] += 1
        assert sum(expert_counts) == n * c.num_experts_per_tok
        matrices.append(tuple(tuple(row) for row in matrix))
        counts.append(tuple(expert_counts))
    return plans, tuple(matrices), tuple(counts)


def _parameter_bytes(c: ModelConfig, p: ParallelSpec) -> int:
    """Unique persistent parameter storage per rank for the supported layout."""
    h, layers, tp = c.hidden_size, c.num_hidden_layers, p.tp_size
    weight = _bytes(c.dtype)
    embedding = c.vocab_size // tp * h * (1 if c.tie_word_embeddings else 2)
    norms = (2 * layers + 1) * h
    attention = layers * 2 * h * c.head_dim * ((c.num_attention_heads + c.num_key_value_heads) // tp)
    if c.num_experts:
        ffn = (
            layers
            * 3
            * h
            * (
                c.num_experts // p.ep_size * (c.moe_intermediate_size // tp)
                + c.num_shared_experts * (c.intermediate_size // tp)
            )
        )
        router = layers * c.num_experts * h * 4  # replicated, fp32 router
    else:
        ffn = layers * 3 * h * (c.intermediate_size // tp)
        router = 0
    return (embedding + norms + attention + ffn) * weight + router


def analyze_phase(config: ModelConfig, phase: PhaseSpec, parallel: ParallelSpec = ParallelSpec()) -> PhaseWorkload:
    """Describe each rank. No model allocation, timing, or data-dependent tracing."""
    c, p = config, parallel
    assert c.kv_dtype is not None
    _validate_parallel(c, p)
    _, matrices, expert_counts = _routing(c, phase, p)
    partitions = _batch_partition(phase.batch_size, p.ep_size)
    tp, ep_count = p.tp_size, p.ep_size
    h, d, q, prefix = c.hidden_size, c.head_dim, phase.query_length, phase.prefix_length
    heads, kv_heads = c.num_attention_heads // tp, c.num_key_value_heads // tp
    dtype, kv_dtype = _dtype_name(c.dtype), _dtype_name(c.kv_dtype)
    itemsize, kv_itemsize = _bytes(c.dtype), _bytes(c.kv_dtype)
    ranks = []
    for ep, (batch_start, batch) in enumerate(partitions):
        n = batch * q
        for lane in range(tp):
            rank = ep * tp + lane
            events: list[Event] = []
            tp_group = tuple(ep * tp + t for t in range(tp))
            ep_group = tuple(e * tp + lane for e in range(ep_count))

            def tensor(shape, typ=dtype):
                return TensorSpec(tuple(shape), typ)

            def vector(name, layer, shape, *, _events=events, **metadata):
                _events.append(Event(name, "vector", layer, (tensor(shape),), tensor(shape), metadata=metadata))

            def mm(name, layer, rows, inner, out, typ=dtype, *, _events=events):
                if rows:
                    _events.append(
                        Event(
                            name,
                            "matmul",
                            layer,
                            (tensor((rows, inner), typ), tensor((inner, out), typ)),
                            tensor((rows, out), typ),
                            2 * rows * inner * out,
                        )
                    )

            def all_reduce(name, layer, rows, *, _events=events, _group=tp_group):
                if tp > 1 and rows:
                    data = tensor((rows, h))
                    _events.append(
                        Event(
                            name,
                            "collective",
                            layer,
                            (data,),
                            data,
                            payload_bytes=data.nbytes,
                            received_payload_bytes=data.nbytes,
                            metadata={
                                "operation": "all_reduce",
                                "group": _group,
                                "payload_scope": "logical tensor per participant",
                            },
                        )
                    )

            def ffn(name, layer, rows, inter):
                mm(name + ".gate_proj", layer, rows, h, inter)
                mm(name + ".up_proj", layer, rows, h, inter)
                if rows:
                    vector(name + ".swiglu", layer, (rows, inter))
                mm(name + ".down_proj", layer, rows, inter, h)

            if n:
                events.append(
                    Event(
                        "embedding",
                        "embedding",
                        None,
                        (tensor((batch, q), "int64"), tensor((c.vocab_size // tp, h))),
                        tensor((batch, q, h)),
                        metadata={"vocab_sharded": tp > 1},
                    )
                )
                all_reduce("embedding.all_reduce", None, n)
                events.append(
                    Event(
                        "rope.positions",
                        "position",
                        None,
                        (),
                        tensor((q, d // 2), "float32"),
                        metadata={"offset": prefix, "shared_across_layers": True, "tables": 2},
                    )
                )
            local_pairs = []
            for layer in range(c.num_hidden_layers):
                if n:
                    vector("input_layernorm", layer, (n, h), compute_dtype="float32")
                    mm("q_proj", layer, n, h, heads * d)
                    mm("k_proj", layer, n, h, kv_heads * d)
                    mm("v_proj", layer, n, h, kv_heads * d)
                    events.append(
                        Event(
                            "rope.qk",
                            "rope",
                            layer,
                            (tensor((batch, heads, q, d)), tensor((batch, kv_heads, q, d))),
                            metadata={"offset": prefix, "compute_dtype": "float32"},
                        )
                    )
                    if prefix:
                        data = tensor((2, batch, kv_heads, prefix, d), kv_dtype)
                        events.append(
                            Event(
                                "kv.read",
                                "kv_read",
                                layer,
                                (data,),
                                payload_bytes=data.nbytes,
                                metadata={"scope": "logical prefix bytes, not physical HBM traffic"},
                            )
                        )
                        if kv_dtype != dtype:
                            events.append(
                                Event(
                                    "kv.dequantize_or_cast",
                                    "cast",
                                    layer,
                                    (data,),
                                    tensor(data.shape),
                                    metadata={"quantization": False},
                                )
                            )
                    attention_inputs = (
                        tensor((batch, heads, q, d)),
                        tensor((batch, kv_heads, prefix + q, d)),
                        tensor((batch, kv_heads, prefix + q, d)),
                    )
                    visible = q * prefix + (q * (q + 1) // 2 if phase.attention_policy == "causal" else q * q)
                    events.append(
                        Event(
                            "attention",
                            "attention",
                            layer,
                            attention_inputs,
                            tensor((batch, heads, q, d)),
                            4 * batch * heads * q * (prefix + q) * d,
                            metadata={
                                "policy": phase.attention_policy,
                                "gqa": True,
                                "visible_pairs_per_head": visible,
                                "implementation": "semantic fused attention; no KV replication or score workspace assumed",
                            },
                        )
                    )
                    if phase.phase != "denoise":
                        data = tensor((2, batch, kv_heads, q, d), kv_dtype)
                        if kv_dtype != dtype:
                            events.append(Event("kv.storage_cast", "cast", layer, (tensor(data.shape),), data))
                        events.append(
                            Event(
                                "kv.write",
                                "kv_write",
                                layer,
                                (data,),
                                payload_bytes=data.nbytes,
                                metadata={"offset": prefix, "scope": "logical append, not full-prefix copy"},
                            )
                        )
                    events.append(
                        Event(
                            "attention.merge_heads",
                            "layout",
                            layer,
                            (tensor((batch, heads, q, d)),),
                            tensor((batch, q, heads * d)),
                            metadata={
                                "operation": "head/token transpose then flatten heads",
                                "physical_copy": "backend dependent; may fuse into attention output",
                            },
                        )
                    )
                    mm("o_proj", layer, n, heads * d, h)
                    all_reduce("attention.all_reduce", layer, n)
                    vector("attention.residual", layer, (n, h))
                    vector("post_attention_layernorm", layer, (n, h), compute_dtype="float32")
                if c.num_experts:
                    if n:
                        if dtype != "float32":
                            events.append(
                                Event("router.input_cast", "cast", layer, (tensor((n, h)),), tensor((n, h), "float32"))
                            )
                        mm("router", layer, n, h, c.num_experts, "float32")
                        events.append(
                            Event(
                                "router.topk",
                                "routing",
                                layer,
                                (tensor((n, c.num_experts), "float32"),),
                                tensor((n, c.num_experts_per_tok), "float32"),
                                metadata={
                                    "indices_shape": (n, c.num_experts_per_tok),
                                    "indices_dtype": "int64",
                                    "routing": "explicit replay" if phase.routing_plan else "balanced synthetic replay",
                                },
                            )
                        )
                        events.append(
                            Event(
                                "router.selected_softmax",
                                "vector",
                                layer,
                                (tensor((n, c.num_experts), "float32"), tensor((n, c.num_experts_per_tok), "int64")),
                                tensor((n, c.num_experts_per_tok), "float32"),
                                metadata={
                                    "replay_score_gather": "fused",
                                    "operation": "gather replay-selected router logits then softmax over top-k",
                                },
                            )
                        )
                        events.append(
                            Event(
                                "experts.pack",
                                "data_movement",
                                layer,
                                (tensor((n, h)), tensor((n, c.num_experts_per_tok), "int64")),
                                tensor((n * c.num_experts_per_tok, h)),
                                metadata={"layout": "expert-grouped token pairs"},
                            )
                        )
                    matrix = matrices[layer]
                    sent = sum(matrix[ep][dst] for dst in range(ep_count) if dst != ep)
                    received = sum(matrix[src][ep] for src in range(ep_count) if src != ep)
                    if ep_count > 1:
                        events.append(
                            Event(
                                "experts.dispatch",
                                "collective",
                                layer,
                                payload_bytes=sent * h * itemsize,
                                received_payload_bytes=received * h * itemsize,
                                metadata={
                                    "operation": "all_to_all",
                                    "group": ep_group,
                                    "send_counts": matrix[ep],
                                    "receive_counts": tuple(row[ep] for row in matrix),
                                    "send_metadata_bytes": sent * 16,
                                    "receive_metadata_bytes": received * 16,
                                    "metadata_record": "two int64 values: expert ID and source pair slot",
                                    "routing_weights": "retained at request owner",
                                    "local_pairs": matrix[ep][ep],
                                },
                            )
                        )
                    per_ep = c.num_experts // ep_count
                    owned = range(ep * per_ep, (ep + 1) * per_ep)
                    count = sum(expert_counts[layer][expert] for expert in owned)
                    local_pairs.append(count)
                    if count:
                        events.append(
                            Event(
                                "experts.unpack",
                                "data_movement",
                                layer,
                                (tensor((count, h)),),
                                tensor((count, h)),
                                metadata={
                                    "owned_experts": tuple(owned),
                                    "tokens_per_expert": tuple(expert_counts[layer][e] for e in owned),
                                },
                            )
                        )
                    for expert in owned:
                        ffn(f"expert.{expert}", layer, expert_counts[layer][expert], c.moe_intermediate_size // tp)
                    all_reduce("experts.all_reduce", layer, count)
                    if ep_count > 1:
                        events.append(
                            Event(
                                "experts.combine",
                                "collective",
                                layer,
                                payload_bytes=received * h * itemsize,
                                received_payload_bytes=sent * h * itemsize,
                                metadata={
                                    "operation": "all_to_all",
                                    "group": ep_group,
                                    "send_counts": tuple(row[ep] for row in matrix),
                                    "receive_counts": matrix[ep],
                                    "send_metadata_bytes": received * 8,
                                    "receive_metadata_bytes": sent * 8,
                                    "metadata_record": "one int64 source pair slot",
                                    "local_pairs": matrix[ep][ep],
                                },
                            )
                        )
                    if n:
                        events.append(
                            Event(
                                "experts.weighted_combine",
                                "vector",
                                layer,
                                (tensor((n, c.num_experts_per_tok, h)), tensor((n, c.num_experts_per_tok), "float32")),
                                tensor((n, h)),
                                metadata={"weights_cast_to": dtype},
                            )
                        )
                    for shared in range(c.num_shared_experts):
                        ffn(f"shared.{shared}", layer, n, c.intermediate_size // tp)
                        all_reduce(f"shared.{shared}.all_reduce", layer, n)
                        if n:
                            events.append(
                                Event(
                                    f"shared.{shared}.sum",
                                    "vector",
                                    layer,
                                    (tensor((n, h)), tensor((n, h))),
                                    tensor((n, h)),
                                )
                            )
                else:
                    ffn("mlp", layer, n, c.intermediate_size // tp)
                    all_reduce("mlp.all_reduce", layer, n)
                if n:
                    vector("ffn.residual", layer, (n, h))
            head_rows = batch * phase.head_tokens
            unresolved = tensor((batch, q), "bool")
            state_metadata = {
                "state": "unresolved_mask",
                "block_index": phase.block_index,
                "scope": "logical state bytes, not physical HBM traffic",
            }
            if head_rows and (phase.head_selection == "unresolved" or phase.sampling_policy == "greedy_confidence"):
                if phase.step_index == 0 and phase.unresolved_tokens == q:
                    events.append(
                        Event(
                            "sampling.unresolved_init",
                            "state_init",
                            None,
                            (),
                            unresolved,
                            payload_bytes=unresolved.nbytes,
                            metadata={**state_metadata, "initial_value": True},
                        )
                    )
                else:
                    events.append(
                        Event(
                            "sampling.unresolved_read",
                            "state_read",
                            None,
                            (unresolved,),
                            unresolved,
                            payload_bytes=unresolved.nbytes,
                            metadata=state_metadata,
                        )
                    )
            head_index_map = None
            if head_rows:
                if phase.head_selection == "unresolved":
                    head_index_map = tensor((batch, phase.head_tokens), "int64")
                    events.append(
                        Event(
                            "head.compact_unresolved",
                            "data_movement",
                            None,
                            (unresolved,),
                            head_index_map,
                            metadata={
                                "head_position_source": "unresolved_mask",
                                "count_per_request": phase.head_tokens,
                                "shape_from_replay": True,
                                "operation": "compact true mask positions independently per request",
                            },
                        )
                    )
                    events.append(
                        Event(
                            "head.select",
                            "data_movement",
                            None,
                            (tensor((batch, q, h)), head_index_map),
                            tensor((batch, phase.head_tokens, h)),
                            metadata={"head_position_source": "unresolved_mask"},
                        )
                    )
                elif phase.head_positions is not None:
                    head_index_map = tensor((phase.head_tokens,), "int64")
                    events.append(
                        Event(
                            "head.select",
                            "data_movement",
                            None,
                            (tensor((batch, q, h)), head_index_map),
                            tensor((batch, phase.head_tokens, h)),
                            metadata={"head_position_source": "explicit", "positions": phase.head_positions},
                        )
                    )
                vector("final_norm", None, (head_rows, h), compute_dtype="float32")
                mm("lm_head", None, head_rows, h, c.vocab_size // tp)
                # Output is explicitly vocab-sharded. Global sampling/argmax is
                # a separate consumer and is not silently performed here.
                events.append(
                    Event(
                        "logits.output",
                        "output",
                        None,
                        (tensor((head_rows, c.vocab_size // tp)),),
                        metadata={
                            "vocab_start": lane * (c.vocab_size // tp),
                            "vocab_sharded": tp > 1,
                            "sampling_included": phase.sampling_policy != "none",
                        },
                    )
                )
            if head_rows and phase.sampling_policy == "greedy_confidence":
                logits = tensor((batch, phase.head_tokens, c.vocab_size // tp))
                scores = tensor(logits.shape, "float32")
                if dtype != "float32":
                    events.append(Event("sampling.logits_cast", "cast", None, (logits,), scores))
                maxima = tensor((batch, phase.head_tokens), "float32")
                token_ids = tensor((batch, phase.head_tokens), "int64")
                events.append(
                    Event(
                        "sampling.local_argmax",
                        "sampling",
                        None,
                        (scores,),
                        maxima,
                        metadata={"indices": _json(token_ids), "global_vocab_offset": lane * (c.vocab_size // tp)},
                    )
                )
                if tp > 1:
                    size = maxima.nbytes + token_ids.nbytes
                    events.append(
                        Event(
                            "sampling.global_argmax",
                            "collective",
                            None,
                            (maxima, token_ids),
                            maxima,
                            payload_bytes=size,
                            received_payload_bytes=size,
                            metadata={
                                "operation": "all_reduce",
                                "reduction": "maxloc",
                                "group": tp_group,
                                "index_output": _json(token_ids),
                                "tie_break": "smallest global token ID",
                            },
                        )
                    )
                events.append(
                    Event(
                        "sampling.shifted_exp_sum",
                        "sampling",
                        None,
                        (scores, maxima),
                        maxima,
                        metadata={"operation": "sum(exp(logits - global_max)) over local vocabulary"},
                    )
                )
                if tp > 1:
                    events.append(
                        Event(
                            "sampling.global_normalizer",
                            "collective",
                            None,
                            (maxima,),
                            maxima,
                            payload_bytes=maxima.nbytes,
                            received_payload_bytes=maxima.nbytes,
                            metadata={"operation": "all_reduce", "reduction": "sum", "group": tp_group},
                        )
                    )
                events.append(
                    Event(
                        "sampling.confidence",
                        "sampling",
                        None,
                        (maxima,),
                        maxima,
                        metadata={"operation": "reciprocal of global shifted exponential sum"},
                    )
                )
                accepted = tensor((batch, phase.accepted_tokens), "int64")
                if phase.head_selection == "unresolved":
                    mapping_metadata = {
                        "head_position_source": "unresolved_mask",
                        "head_index_map": _json(head_index_map),
                    }
                else:
                    mapping = tuple(range(q)) if phase.head_positions is None else phase.head_positions
                    mapping_metadata = {"head_position_source": "explicit", "head_positions": mapping}
                selection_inputs = (maxima, unresolved)
                if head_index_map is not None:
                    selection_inputs += (head_index_map,)
                events.append(
                    Event(
                        "sampling.select_positions",
                        "sampling",
                        None,
                        selection_inputs,
                        accepted,
                        metadata={
                            "unresolved_tokens": phase.unresolved_tokens,
                            **mapping_metadata,
                            "output_indices": "block_positions",
                            "operation": "gather unresolved mask at head positions; confidence topk; map selected indices to block positions",
                        },
                    )
                )
                if phase.accepted_tokens:
                    canvas = tensor((batch, q), "int64")
                    update_inputs = (canvas, token_ids, accepted)
                    if head_index_map is not None:
                        update_inputs += (head_index_map,)
                    events.append(
                        Event(
                            "sampling.update_canvas",
                            "sampling",
                            None,
                            update_inputs,
                            canvas,
                            metadata={"accepted_tokens_per_request": phase.accepted_tokens, **mapping_metadata},
                        )
                    )
                    events.append(
                        Event(
                            "sampling.unresolved_update",
                            "state_write",
                            None,
                            (unresolved, accepted),
                            unresolved,
                            payload_bytes=batch * phase.accepted_tokens,
                            metadata={
                                **state_metadata,
                                "updated_value": False,
                                "modified_entries": batch * phase.accepted_tokens,
                            },
                        )
                    )
            per_position = 2 * batch * kv_heads * d * c.num_hidden_layers * kv_itemsize
            ranks.append(
                RankWorkload(
                    rank,
                    lane,
                    ep,
                    batch_start,
                    batch,
                    _parameter_bytes(c, p),
                    per_position * prefix,
                    per_position * phase.cache_length_after,
                    tuple(local_pairs),
                    tuple(events),
                )
            )
    return PhaseWorkload(phase, p, tuple(ranks), matrices)


def build_schedule(
    prompt_length: int,
    block_lengths: tuple[int, ...],
    commit_counts: tuple[tuple[int, ...], ...],
    *,
    batch_size: int = 1,
    commit_last: bool = False,
    commit_attention_policy: str = "bidirectional",
    prefill_attention_policy: str = "causal",
    head_policy: str = "unresolved",
) -> tuple[PhaseSpec, ...]:
    """Create a shape schedule; commit_counts[j][t] fixes accepted tokens/step.

    Unresolved-only heads compact the evolving boolean mask independently per
    request. Replay fixes the number of selected rows, not their positions; the
    resulting runtime index map also maps sampled tokens back into the block.
    Only the terminal block may omit commit because
    its KV has no later consumer in this request. No generated-token seed is
    assumed from a boundary head; every emitted block token is counted explicitly.
    """
    _integer("prompt_length", prompt_length)
    _integer("batch_size", batch_size, 1)
    if len(block_lengths) != len(commit_counts):
        raise ValueError("one commit-count schedule is required per block")
    if head_policy not in ("unresolved", "all"):
        raise ValueError("head_policy must be unresolved or all")
    if not isinstance(commit_last, bool):
        raise ValueError("commit_last must be boolean")
    phases = []
    if prompt_length:
        phases.append(
            PhaseSpec(
                "prefill", batch_size, prompt_length, head_positions=(), attention_policy=prefill_attention_policy
            )
        )
    prefix = prompt_length
    for block, (length, counts) in enumerate(zip(block_lengths, commit_counts, strict=False)):
        _integer("block length", length, 1)
        if not counts:
            raise ValueError("a block needs at least one denoising step")
        if (
            any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts)
            or sum(counts) != length
        ):
            raise ValueError("nonnegative commit counts must sum to the block length")
        unresolved = length
        for step, accepted in enumerate(counts):
            if unresolved == 0:
                raise ValueError("cannot schedule denoising after all block tokens are resolved")
            phases.append(
                PhaseSpec(
                    "denoise",
                    batch_size,
                    length,
                    prefix,
                    head_positions=None,
                    head_selection="unresolved" if head_policy == "unresolved" else "explicit",
                    unresolved_tokens=unresolved,
                    block_index=block,
                    step_index=step,
                    accepted_tokens=accepted,
                    sampling_policy="greedy_confidence",
                )
            )
            unresolved -= accepted
        if block < len(block_lengths) - 1 or commit_last:
            phases.append(
                PhaseSpec(
                    "commit",
                    batch_size,
                    length,
                    prefix,
                    head_positions=(),
                    attention_policy=commit_attention_policy,
                    unresolved_tokens=0,
                    block_index=block,
                )
            )
            prefix += length
    return tuple(phases)


def analyze_schedule(
    config: ModelConfig, phases: tuple[PhaseSpec, ...], parallel: ParallelSpec = ParallelSpec()
) -> WorkloadReport:
    """Analyze a fixed request batch, including an optionally partial terminal block.

    Initial prefix KV may be supplied externally. Subsequent blocks require a
    complete commit, but a terminal prefix of the decoding schedule is permitted;
    generated_tokens counts only explicitly accepted tokens, once per request.
    """
    previous: PhaseSpec | None = None
    blocks: dict[int, tuple[int, int]] = {}
    accepted_by_block: dict[int, int] = {}
    active_block: int | None = None
    active_committed = True
    for phase in phases:
        if previous is not None:
            if phase.batch_size != previous.batch_size:
                raise ValueError("batch changes require an external serving/cache mapping adapter")
            if phase.prefix_length != previous.cache_length_after:
                raise ValueError("phase prefix lengths do not follow preceding cache updates")
        if phase.phase == "prefill" and blocks:
            raise ValueError("prefill cannot be inserted after modeled block decoding")
        if phase.phase == "commit" and active_block is not None:
            if active_committed:
                raise ValueError("the active block has already been committed")
            if phase.block_index != active_block or (phase.batch_size, phase.query_length) != blocks[active_block]:
                raise ValueError("commit does not match the active decoded block")
            if accepted_by_block[active_block] != phase.query_length:
                raise ValueError("cannot commit KV for an unresolved block")
            active_committed = True
        if phase.phase == "denoise":
            if phase.block_index is None:
                raise ValueError("denoise phases require block_index for output-token accounting")
            spec = (phase.batch_size, phase.query_length)
            if phase.block_index != active_block:
                if not active_committed:
                    raise ValueError("a subsequent block requires the previous block's KV commit")
                if phase.block_index in blocks:
                    raise ValueError("a completed block cannot be reopened")
                active_block = phase.block_index
                active_committed = False
            elif active_committed:
                raise ValueError("cannot denoise a block after its KV commit")
            expected_unresolved = phase.query_length - accepted_by_block.get(phase.block_index, 0)
            if expected_unresolved == 0:
                raise ValueError("cannot denoise after all block tokens are resolved")
            if phase.unresolved_tokens is not None and phase.unresolved_tokens != expected_unresolved:
                raise ValueError("unresolved_tokens disagrees with preceding accepted token counts")
            if phase.block_index in blocks and blocks[phase.block_index] != spec:
                raise ValueError("block shape changed across denoising steps")
            blocks[phase.block_index] = spec
            if phase.accepted_tokens is None:
                raise ValueError("schedule denoise phases need accepted_tokens for actual output-token accounting")
            accepted_by_block[phase.block_index] = accepted_by_block.get(phase.block_index, 0) + phase.accepted_tokens
            if accepted_by_block[phase.block_index] > phase.query_length:
                raise ValueError("accepted token counts exceed the block length")
        previous = phase
    return WorkloadReport(
        tuple(analyze_phase(config, phase, parallel) for phase in phases),
        sum(blocks[block][0] * count for block, count in accepted_by_block.items()),
        config,
    )
