# Communication Ownership Contract

## 1. Boundary

The simulator exports immutable L1 `StepGraph` templates and an L2
`SchedulePlan`. Communication ownership is decided before either artifact is
exported. The downstream predictor or DES must not move communication between
these layers.

Ownership is semantic and schedule-independent:

| Observed behavior | Owner | Export |
| --- | --- | --- |
| communication executes inside F/B/I/W | `L1_STAGE` | node in the referenced compute template |
| PP send/receive between stages | `L2_PIPELINE` | SEND/RECV action plus a dedicated communication fragment |
| FSDP prefetch issued outside compute | `L2_PREFETCH` | UNSHARD/RESHARD actions and residency slots |
| standalone gradient collective | `L2_STANDALONE` | REDUCE_GRAD action |
| schedule intent without real work | none | omitted |

The capture implementation must not branch on `1F1B`, `DualPipeV`, or another
schedule name. It determines ownership from the real communication node,
compute span, FSDP transition identity, and explicit schedule intent.

## 2. L1 Stage Communication

All communication nodes in a compute template are part of that template's
cost and dependency graph. This includes:

- TP, CP, and EP collectives;
- FSDP all-gather triggered while entering F/B/I/W;
- FSDP or DDP gradient reduction triggered inside B/I/W.

Each such node has:

```text
annotations.communication_owner = L1_STAGE
```

When repeated microbatches have different communication shapes, capture emits
immutable variants such as:

```text
s0_F
s0_F__comm_v1
```

Every compute action references the correct variant through `template_ref`.
The downstream system predicts and replays each referenced template as-is.

Independent stage-local collectives are connected to the compute entries that
need their results. FSDP ownership uses captured parameter-group boundaries,
not the whole stage:

```text
unshard_wait(group N)
  -> compute region for group N
  -> reshard_release(group N)
```

The capture-only boundary markers are removed before export. The resulting
template records each interval in
`StepGraph.annotations["fsdp_residency_intervals"]`, including the group/module,
all-gather node, target entries, release exits, prefetch source, and placement.

The placement values are:

| value | meaning |
| --- | --- |
| `captured` | the original all-gather and its tensor edges were in this template |
| `layer_jit` | all-gather starts after the previous group and gates only its own group |
| `layer_prefetch` | all-gather starts when the source group is ready and overlaps source compute |
| `cross_action_prefetch` | all-gather is launched by one compute action for a later compute action |

Every layer or cross-action prefetch has an explicit zero-cost control operator
in the exported L1 graph:

```text
source input/gradient readiness + source parameter all-gather
                         -> FSDP_PREFETCH_LAUNCH -> source compute
                                                   -> target all-gather
```

`FSDP_PREFETCH_LAUNCH` has zero FLOPs, peak memory, parameter memory, and
communication bytes. It models the source module hook: the target all-gather
may overlap source compute, but a chain of fast all-gathers cannot recursively
prefetch later layers before each source module is reached. A prefetch without
a matching source parameter-group region is a capture error.

`cross_action_prefetch` belongs to the launch action's L1 template. It is an
exit of that template when the target use is in a later action; the later
action must not contain a duplicate all-gather. Rank-local action order then
provides target readiness.

Multiple independent FSDP all-gather groups remain parallel unless the
captured graph records a dependency between them. There is no fallback that
connects a missing all-gather to every stage entry. A sharded transition
without either captured tensor edges or a matching parameter-group boundary
is a capture error.

FSDP gradient reduction keeps a different dependency contract. Packing and
dtype-conversion scaffolding is hidden, but every parameter gradient consumed
by a parameter-group reduce-scatter remains a dependency-only input of that
RS node. Therefore an expert/eFSDP RS becomes runnable after its expert
gradient producers finish; it does not wait for unrelated attention backward
operators or for the entire transformer-block region to exit. Only
reduce-scatter nodes carrying an `fsdp_group_id` participate in this stream;
CP/TP reduce-scatter nodes keep their own communication dependencies.

Reduction stream order is explicit:

```text
RS(group N) -> RS(group N+1)       # reduce-scatter stream
RS(group N) -> AR(group N)         # corresponding HSDP data path
AR(group N) -> AR(group N+1)       # all-reduce stream
```

`FSDP_POST_BACKWARD_SYNC` represents only the preceding module's RS
backpressure and gates the first RS of the current module. It must not collect
the current module's whole-region exits. HSDP AR completion is retained for
gradient handling and final backward completion, but never gates a later RS,
all-gather, or backward compute node.

## 3. L2 Communication

### Pipeline P2P

PP P2P never remains in F/B/I/W. It is extracted into an immutable fragment
and represented by SEND/RECV plus `transfer_id`. Cross-rank reconstruction
still joins endpoints by `transfer_id`.

### External FSDP prefetch

An FSDP all-gather remains in L2 only when the schedule issued it outside the
owning compute span. It is exported as:

```text
UNSHARD -> param_full -> COMPUTE -> control -> RESHARD
```

The pair has `communication_owner=L2_PREFETCH`. `RESHARD` closes full-parameter
residency; it is not a gradient reduce-scatter.

### Standalone gradient reduction

A real reduction outside all compute templates remains:

```text
B/W -> grad_local -> REDUCE_GRAD -> grad_reduced -> OPTIMIZER
```

It has `communication_owner=L2_STANDALONE`. If the real collective is already
inside B/I/W, no L2 REDUCE_GRAD is exported. A no-op REDUCE_GRAD intent is also
omitted, and the optimizer depends on the last local gradient producer.

## 4. Consumer Rules

For each COMPUTE action:

1. resolve `plan.step_templates[action.template_ref]`;
2. replay every node in that graph, including `L1_STAGE` communication;
3. do not synthesize FSDP communication from the schedule name;
4. do not add L2 communication for a collective already in the template.

The downstream system does not need to interpret
`fsdp_residency_intervals` to reconstruct dependencies: the required edges
are already present in the L1 graph. The annotation is for validation,
visualization, and memory accounting.

For each explicit L2 communication action, replay only its own communication
fragment or `CommDetail`. L2 controls pipeline order, external prefetch, and
standalone collectives. It does not edit an already predicted L1 template.

The plan annotation `communication_ownership` reports:

```text
internal_fsdp_transitions
external_fsdp_prefetches
generated_l1_templates
internal_gradient_reductions
external_gradient_reductions
removed_noop_gradient_intents
stage_owned_collectives
```

## 5. Acceptance Checks

For 1F1B and DualPipeV with `reshard_after_forward=always`:

1. later microbatch F/B/I/W templates contain expected FSDP all-gather nodes;
2. compute actions reference the matching communication variant;
3. only explicit prefetch transitions remain as L2 UNSHARD/RESHARD;
4. reduce-scatter in B/I/W has no duplicate L2 REDUCE_GRAD;
5. PP P2P is absent from compute templates and present in SEND/RECV fragments;
6. all non-external DataSlots have a producer and replay reaches every action.
7. no exported template contains `sim.fsdp_*` marker operators;
8. a copied all-gather gates only its parameter-group entries, never every
   entry of the stage;
9. a cross-action prefetch appears in the launch template and not the target
   template.
10. every FSDP RS depends on its own parameter-group gradient producers, not
    on all exits of the containing transformer block;
11. HSDP RS and AR form independent ordered streams: AR may depend on RS, but
    AR never gates a later RS, all-gather, or backward compute node.
12. CP/TP reduce-scatter nodes never become an FSDP RS predecessor merely
    because they occur between two FSDP reductions in capture order.

For non-PP capture, stage-local communication remains in the ordinary F/B
templates and no PP fragment is created.
