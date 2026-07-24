# PP L2 Capture Architecture

## 1. Goal

The simulator captures one representative process for each pipeline stage while
describing a larger logical device mesh. L2 must export the actual rank-local
execution order and enough stable identities for an external assembler or DES
to connect plans from different stages.

The capture must not depend on a specific TorchTitan schedule implementation.
1F1B, GPipe, interleaved schedules, and runtime schedules eventually execute
the same semantic interfaces:

- `PipelineStage.forward_one_chunk`
- `PipelineStage.backward_one_chunk`
- `PipelineStage.backward_weight_one_chunk`
- pipeline P2P operations
- FSDP parameter-group unshard, wait, and reshard
- gradient collectives

These interfaces are the capture boundary. TorchTitan's lowered schedule is
useful for validation, but is not the primary L2 source.

## 2. Rank Identities

A `multi_proc_meta` worker has two rank identities:

| Identity | Meaning | Example: world=64, PP=4 |
| --- | --- | --- |
| `capture_process_rank` | Real Gloo control-plane rank | `0,1,2,3` |
| `logical_global_rank` | Representative rank in the complete mesh | `0,16,32,48` |

The real Gloo group only advances PP control traffic. DeviceMesh slicing,
DTensor placement, FSDP/EP/TP groups, and rank-table metadata use the logical
rank. P2P peer ranks remain capture-process ranks.

`SimulationRankContext` is the only conversion point. Code must not silently
fall back to rank 0 when the distributed process group is not initialized.
For PP greater than one, `RANK` and `WORLD_SIZE` must both be present (or an
equivalent process group must already be initialized), and the physical world
size must equal the PP degree.

## 3. Canonical Capture Records

Each worker records a rank-local semantic action stream.

Runtime schedules also emit an intent record for every explicit action. Intent
records are an integrity oracle, not the source of tensor dependencies. A
normal `F`, `B`, `I`, or `W` intent must have a matching compute execution
record. Missing matches fail capture instead of silently assigning later work
to a stale stage.

### Compute action

Identity:

```text
(stage, comp_type, microbatch)
```

`comp_type` is `F`, `B`, `I`, or `W`. The first occurrence of each
`(stage, comp_type)` captures the full L0 graph as a reusable template. Every
occurrence records an L2 compute instance.

### Pipeline transfer

Identity:

```text
pp:{forward|backward}:s{src}->s{dst}:mb{microbatch}:t{tensor_ordinal}
```

SEND and RECV plans from different workers carry the same `transfer_id`.
Multiple tensors in one P2P batch use distinct ordinals. The external assembler
must join transfers by `transfer_id`, not by local `seq_idx`.

### FSDP residency transition

Identity:

```text
fsdp:r{capture_rank}:g{parameter_group_id}:u{transition_ordinal}
```

The transition ID connects:

1. the all-gather launched by `unshard`;
2. the full-parameter allocation observed by `wait_for_unshard`;
3. the later residency release observed by `reshard`.

FSDP may call `unshard` again while a previous asynchronous all-gather is in
flight. A no-op call must retain the in-flight transition ID. A new ID becomes
active only when that call launches a collective.

One logical transition has two observable timelines:

- schedule intent: the explicit `UNSHARD` or `RESHARD` action position;
- residency state: full-parameter allocation after async unshard wait, and
  actual release when reshard changes state.

Assembly merges both by `(transition_id, action)`. It keeps the explicit
`UNSHARD` position, but uses the actual state-loss position for `RESHARD`.
Memory replay uses state events, not intent events.

### Explicit schedule-only actions

Some runtime schedules contain `REDUCE_GRAD` even when the meta execution path
does not issue a collective. Capture records the schedule intent independently
from communication. If a matching collective is observed, both become one
action. Otherwise L2 exports an explicit no-op action with no DataSlot, so it
preserves schedule shape without blocking replay. The optimizer then depends
directly on the last real local gradient producer.

### Ordering

`action_order` is a recorder-owned monotonic semantic clock. `schedule_order`
is assigned from this stream after assembly.

`seq_idx` is retained only as a reference into captured L0 operators and memory
events. It is not a cross-rank clock and must not be used as transfer identity.

## 4. Canonical L2 Assembly

`SchedulePlan` is the authoritative L2 representation:

```text
semantic events
    -> CapturedTraceAssembler
    -> ordered ScheduleAction list
    -> typed DataSlot dependencies
    -> validation
```

The legacy `ScheduleGraph` is projected from the completed `SchedulePlan`.
It must not independently match communication events or infer another
execution order.

Current exports carry `annotations.capture_schema_version = 2` and
`annotations.capture_process_rank`. Consumers should reject an unsupported
schema instead of treating missing `schedule_order`, `external`, or stable
identity fields as empty defaults.

The downstream reconstruction contract is documented in
[`schedule-plan-dependency-reconstruction-contract.md`](./schedule-plan-dependency-reconstruction-contract.md).
It defines local DataSlot dependencies, cross-rank rendezvous, logical-rank
expansion, collective/FSDP state, and fail-fast validation without requiring
consumers to understand capture internals.

### Stage-local dependencies

- external dataloader input -> first-stage forward;
- RECV_F -> local forward;
- forward -> forward state -> corresponding backward;
- RECV_B -> local backward;
- unshard -> full parameter -> owning compute;
- owning compute -> control dependency -> reshard;
- backward -> local gradient -> reduction -> optimizer.

Size-one FSDP groups and schedule-only actions without real work create explicit
no-op actions and no blocking DataSlot. A real unshard without a compute
consumer, or a real reshard without an active unshard and preceding compute, is
an error.

### Cross-stage dependencies

The sender plan contains:

```text
producer compute -> local slot -> SEND
```

The receiver plan contains:

```text
RECV -> received slot -> consumer compute
```

The external assembler joins SEND and RECV by `transfer_id`. Local action IDs
and local sequence numbers are never compared across plans.

Adjacent virtual stages mapped to the same capture process have no P2P
collective. L2 synthesizes `is_local_transfer=True` activation and input-gradient
slots from the complete stage-to-rank mapping. The terminal loss stage is also
derived from that mapping, not from the physical PP worker count.

## 5. Compatibility Rules

- Capture semantic interfaces, not schedule class names.
- Preserve explicit runtime schedule metadata as optional validation input.
- Never infer logical mesh membership from physical Gloo rank.
- Resolve process-group ranks from c10d's registered rank map.
- Keep meta DeviceMesh device type consistent with DTensor/FSDP while using
  fake process groups for logical non-PP dimensions.
- Avoid reconstructing a distributed DTensor through `nn.Parameter`; preserve
  its local meta shard and parameter marker.
- Treat missing identity fields only as legacy-input compatibility. New capture
  paths must emit `action_order`, `transfer_id`, and FSDP `transition_id`.

## 6. Required Validation

Minimum regression matrix:

1. non-PP, single process;
2. PP=2 1F1B with multiple microbatches;
3. PP=2 GPipe;
4. PP=2, EP=2, DP-shard=2 on logical world size 4;
5. multi-tensor P2P transfer;
6. repeated/asynchronous FSDP unshard;
7. size-one E-FSDP no-op group;
8. a runtime/interleaved schedule with `I`, `W`, and explicit residency actions;
9. an explicit `REDUCE_GRAD` with and without a captured collective.

For every multi-process case verify:

- every capture worker emits a non-empty plan;
- action `rank` is the capture-process rank;
- action `stage` is the actual PP stage;
- logical communication groups differ between PP stages when expected;
- SEND/RECV transfer IDs pair exactly once;
- every non-external DataSlot has a producer;
- replay reaches every non-noop action.
