# Multi-region research question register

This is the decision register for multi-region and asynchronous search. The
[optimizer note](multiregion_optimizer.md) records implementation history and
run evidence; Beads owns actionable work and dependencies. A passing test or a
currently implemented rule does not close a research question.

## How to use this register

- Keep each `MR-Q` ID stable. State the question as a choice whose answer could
  change the method or public behavior, not as an implementation task.
- Record **observed** behavior with code or run evidence, and label a proposed
  policy separately. An absent experiment is `unknown`, not evidence for a
  policy.
- Change `open` or `deferred` to `decided` only with a decision record: selected rule,
  alternatives considered, reason, evidence, scope, and a revisit trigger.
  `deferred` means the answer is intentionally pending, not silently settled.
- Beads tracks the investigation and implementation work. Link a question to
  its Beads task, but do not use task closure as a substitute for the decision
  record. Review this table when work reaches a listed gate.

The existing facts we can depend on are evaluation identity, proposal origin
as `(region slot, generation)`, and the separation of observation relevance
from progress attribution. The current host and resident limits below are
implementation facts, not permanent method requirements. In particular, the
resident portfolio has local histories; it does not have the host optimizer's
global posterior.

| ID | Open question | Status | Decision gate | Beads work |
| --- | --- | --- | --- | --- |
| MR-Q01 | How many evaluations or rounds may be in flight per region, and when may it issue another? | Decided for current capacity | More than one unsettled group per slot | `blckbaux-sjj` |
| MR-Q02 | Which completion earns region progress credit when results arrive out of order or a batch is partial? | Decided for current credit rule | Issue-time rather than settlement-time baseline | `blckbaux-sjj` |
| MR-Q03 | What happens to an old generation's late completion, cancellation, or failed evaluation? | Decided for current restart contract | Automatic late-result archival or recovery after terminal ingestion | `blckbaux-sjj` |
| MR-Q04 | How should pending points affect candidate generation and local models? | Exact exclusion decided; broader rule deferred | Distance or fantasy treatment for nearby pending points | `blckbaux-sjj` |
| MR-Q05 | When and where does an observation enter another region's model, including across restart? | Deferred | Changing the current three sharing rules or restart memory | `blckbaux-va3` |
| MR-Q06 | How are region slots allocated, started, assigned roles, and restarted? | Deferred | Replacing uniform automatic restart or adding basin-aware roles | `blckbaux-va3` |
| MR-Q07 | Which work is asynchronous, and which state stays resident across Python, CPU, and GPU paths? | Decided for current execution boundary | A public async loop or resident performance claim | `blckbaux-sjj` |
| MR-Q08 | What is the recoverable outcome when ingestion mutates some models and then fails? | Decided for current adapters | A provider offering transactional rollback or a native optimizer checkpoint | `blckbaux-i06` |

## Questions and evidence needed

### MR-Q01 — In-flight work and scheduling

The host accepts partial completion by evaluation ID and can issue another
group to a slot after that slot's prior group settles, while other slots stay
busy. It cannot issue a second group to a busy slot. The Metal resident
portfolio permits one pending ticket per slot. Decide whether a *region step*
should remain a group that settles together or become a stream with several
outstanding evaluations per slot. Specify capacity and backpressure separately
from the optimizer's numerical policy. Test unequal evaluation times and a fast
region repeatedly requesting work while another remains busy. [pySOT SOP],
[POAP], and [APOSMM] are concrete precedents, but none determines BAUX's choice.

### MR-Q02 — Attribution under partial completion

The host ingests each successful result into relevant models on arrival and
updates a slot's controller after every row issued to that slot has resolved;
other slots need not finish. The resident portfolio credits a completed ticket
immediately. The current host controller compares a group's result against
the incumbent visible at settlement. Both completion orders have been tested:
shared relevance can change success counters while leaving final incumbents
and origin trial counts equal. Decide whether future methods should preserve
this event-order rule or compare against an issue-time baseline.

### MR-Q03 — Old work after restart

Host and Metal resident restart can abandon pending work for one slot, advance
its generation, and reject completion under the old identity. A caller may
explicitly import a late measured result as external evidence; that import
has no origin controller credit and follows the current sharing and restart
memory policies. Cancellation and failed evaluations add no observation or
credit. Automatic archival of late results remains a future policy choice.
[pySOT's epoch check] keeps late results globally but excludes them from the
new local run; it is a precedent rather than BAUX's current rule.

### MR-Q04 — Pending-aware proposals

The host excludes exact pending coordinates from a new candidate pool when
issuing to an idle slot; the resident one-ticket-per-slot capacity prevents a
second proposal in that slot. Specify whether nearby pending candidates should
also be excluded by distance, represented in an acquisition function, or
handled by a sampler's joint batch rule. Test duplicate coordinates, nearby
regions, and out-of-order completion. [Trieste's
async rule] tracks pending points, while its [multi-region rule] is separate;
their combination is not established by the reviewed source.

### MR-Q05 — Observation relevance and restart memory

The three implemented labels are independent, shared, and nearest-center.
Their current host and resident routing rules differ in storage and geometry.
Decide whether relevance is evaluated at issue time, completion time, import
time, or more than once; how a moved center changes it; and what a new
generation inherits. Preserve origin and credit as separate facts. Compare
each policy on objectives where regions are both separated and overlapping,
using model-data counts and optimization quality, not only successful dispatch.
Current implementation chooses relevance at completion or explicit import,
using then-current centers and geometry. Host `Fresh` clears a restarted
slot's local model; host `RetainRelevant` replays observations admitted to
that slot. The resident engine starts a fresh local history on replacement.
The [matched resident comparison](multiregion_optimizer.md#matched-policy-comparison-in-the-resident-run-2026-11-05)
found identical reward traces across all three sharing modes, although
nearest-center delivered 16 foreign imports. It does not compare host restart
memory or establish a preferred sharing policy. A separate
[matched host run](multiregion_optimizer.md#matched-host-restart-memory-run-2026-11-05)
did compare `Fresh` and `RetainRelevant`: all 18 pairs matched before a forced
restart, their slot-0 local-model counts then diverged, and retention had
higher final best reward in six, lower in ten, and tied in two. This narrow
synthetic result does not establish a general restart-memory rule.
A second [role diagnostic](multiregion_optimizer.md#matched-host-acquisition-role-diagnostic-2026-11-05)
found the direction of the acquisition-role comparison changed when restart
memory changed, so a memory decision cannot be inferred from the uniform-role
run alone.
[pySOT's epoch check] shows global retention with a fresh local run as one
alternative.
The [three-landscape host suite](multiregion_optimizer.md#matched-host-landscape-suite-2026-11-05)
adds Rastrigin-form and Ackley-form diagnostics at the same fixed budget.
Retention's effect changes with objective, role, and center choice, so the
current evidence still does not support a universal restart-memory default.
The [automatic retirement diagnostic](multiregion_optimizer.md#automatic-host-retirement-diagnostic-2026-11-05)
also shows the first retirement event and pre-event values match in all 108
memory pairs. The replacement local-model counts differ, but final reward
effects remain mixed across objectives and roles. This supports keeping
restart memory as a policy choice.
The [automatic center comparison](multiregion_optimizer.md#automatic-restart-center-comparison-2026-11-05)
holds the actual first retirement event fixed while comparing two replacement
selectors. Its reward effects also vary by objective and memory rule, so it
does not collapse restart memory and center selection into one policy.

### MR-Q06 — Allocation, roles, and multistart

The resident balanced and reward-UCB rules are BAUX heuristics, not faithful
TuRBO-m joint Thompson sampling. Decide whether allocation compares region
acquisition values, rewards, uncertainty, explicit exploration roles, or some
combination; also decide how new starts are chosen and when a region retires.
The [original TuRBO-m] code gives one synchronous reference, and [APOSMM]
gives an asynchronous multistart reference. Neither proves one allocation
policy best for BAUX. Compare under a fixed total evaluation budget.
The host automatically draws a uniform unit-cube center when a TuRBO region
retires; `automatic_restart` verifies its generation advances and the slot
becomes active again. `automatic_retention` verifies that a retained local
model is synchronized at the same completion event; `Fresh` clears its local
model. This is a simple start rule, not basin discovery. The
current resident run forces one restart at its midpoint; it does not test
automatic start selection or an exploration/exploitation role policy. At four
regions, resident UCB had higher final reward than balanced in five of six
matched seed/layout cells and lower reward in one; the six one-region cells
tied. This is one 64-dimensional synthetic objective with three seeds. The
[matched comparison](multiregion_optimizer.md#matched-policy-comparison-in-the-resident-run-2026-11-05)
does not settle allocation. The [original TuRBO-m] selects proposals in a
synchronous global competition; [APOSMM] uses distance-based start eligibility
and active-run limits. Those are distinct methods, not validation of BAUX's
current heuristics.
Host `Balanced` now compares cumulative completed trials when dividing each
new candidate budget, so repeated one-arm asks rotate among equally evaluated
slots. The regression `balancedhistory` exercises that behavior; it does not
compare optimization quality.
The [host role diagnostic](multiregion_optimizer.md#matched-host-acquisition-role-diagnostic-2026-11-05)
compares per-region UCB β=`[2, 2, 2, 2]` with `[0, 4, 2, 2]` at the same
budget. With `Fresh` memory the split had lower final best reward in 12 of
18 matched policy cells; with `RetainRelevant` it had higher reward in 13 of
18. This is a BAUX diagnostic of heterogeneous acquisitions, not a faithful
implementation or validation of [MTRBO](https://arxiv.org/abs/2605.06618).
The cells share three seeds and one objective, so role selection remains open.
A [restart-center diagnostic](multiregion_optimizer.md#matched-restart-center-diagnostic-2026-11-05)
then compared seeded uniform and maximin center choices through the existing
explicit restart API at the same fixed budget. Maximin was farther from every
other active-center set in all 18 matched cells, but final reward favored it
in only 2/18 cells under `Fresh` memory and 6/18 under `RetainRelevant` (with
many ties). This tests center choice at a forced restart, not the policy for
when a region should retire or a learned basin selector. Three reused seeds
and one objective leave the general multistart choice open.
The [three-landscape host suite](multiregion_optimizer.md#matched-host-landscape-suite-2026-11-05)
extends the paired comparison to 216 memory pairs. Maximin still has mixed
reward effects, while the split β assignment often beats uniform β in that
factorial suite. These are correlated synthetic cells, and the earlier
midpoint-only role result changes sign under `Fresh` memory. Keep role and
start policy choices open until more seeds, actual workloads, and automatic
retirement timing are compared.
The [automatic retirement diagnostic](multiregion_optimizer.md#automatic-host-retirement-diagnostic-2026-11-05)
records actual TuRBO retirements and built-in uniform replacements at 64
evaluations, with a configured controller that produces 1–5 events per arm.
It verifies the current automatic path but leaves automatic center selection
and retirement thresholds open. The forced and automatic runs use different
controller methods, so they are not a causal timing comparison.
The subsequent [automatic center comparison](multiregion_optimizer.md#automatic-restart-center-comparison-2026-11-05)
uses that same retirement path and equal random consumption. All 108 paired
pre-event trajectories match under each memory rule, while maximin selects a
different center in 106 pairs. Final reward and replacement-count effects are
mixed across the three landscapes. Keep both selectors available; revisit the
default with more independent seeds, actual workloads, and complete
basin-discovery alternatives rather than extending this synthetic grid.

### MR-Q07 — Execution and residency

Separate asynchronous objective evaluation, nonblocking Python orchestration,
overlapped surrogate fitting, and GPU-resident optimizer state. Define which
event changes optimizer state and how Python observes pending work without
copying resident parameters. Test the same issue/complete/cancel sequence in
host and resident paths; measure proposal latency, fitting, synchronization,
and objective time separately. The current Python methods are synchronous
calls over an ID-based lifecycle; external workers may finish out of order.
The [resident phase run](multiregion_optimizer.md#resident-phase-evidence-2026-11-05)
separates host wall-clock proposal, objective, tell, and final sync time on
Apple M4, while parameter arrays stay resident. It does not measure device-only
kernel time, overlap, or a large-parameter latency target. [Xopt's future loop]
is an orchestration example, not evidence of a resident GPU implementation.

### MR-Q08 — Ingestion outcome and recovery

The host rejects validation errors before mutation; a later fit or flush
failure can leave partial ingestion and poison further operations. The resident
path can retire a poisoned ticket without credit. Define an outcome that tells
the caller whether an observation was accepted, rejected, or left in an
unrecoverable state, with no double counting on retry. This is required by the
[optimizer completion gate](multiregion_optimizer.md#completion-gate) and is
tracked by the current Beads execution-adapter task `blckbaux-i06`.

## Decision records

### MR-Q01 — 2026-11-05 — host and Metal resident

Rule: one unsettled issued group per region slot. A host group may contain
several IDs; a settled slot can issue again while another slot stays busy.
Metal has one ticket per slot. Warm-up keeps one outstanding host batch.
Capacity errors leave existing tickets unchanged.

Alternatives considered: one host-wide batch barrier and several unsettled
groups per slot. The former blocked fast slots behind slow ones; the latter
needs a separate pending-aware numerical policy.

Why this rule: it permits useful overlap without mixing results from two
unsettled groups in one controller update. Evidence: the 17-case
`optimizer-provenance` target, especially `overlapping_asks`, and the
wheel-backed 46-case `resident-python` target. Unresolved limit: neither path
issues another group to a busy slot. Revisit when a method requires that
capacity. Beads task: `blckbaux-sjj`.

### MR-Q02 — 2026-11-05 — host

Rule: each successful result enters relevant models at arrival, but a host
slot receives one controller settlement after all rows of its issued group
resolve. Cancellation adds no reward or controller credit. Success compares
against the incumbent visible at settlement, so shared foreign evidence can
change whether a later local result counts as improvement.

Alternatives considered: a host-wide barrier, per-row controller updates, and
an incumbent snapshot from issue time. Per-row updates would change the
existing group-based success/failure method; an issue-time snapshot would
ignore useful foreign evidence that arrived first.

Why this rule: origin and controller credit stay attached to issued work while
observation relevance remains independent. Evidence: `regional_settlement`
and `arrival_order` in `optimizer-provenance` (17 passed); reversing the two
shared results changed success counters but not final incumbents or trial
counts. Unresolved limit: different schedules can induce different future
regions. Revisit if reproducibility across completion order is required.
Beads task: `blckbaux-sjj`.

### MR-Q03 — 2026-11-05 — host and Metal resident

Rule: restarting one slot abandons its unresolved IDs or ticket, advances the
generation, and rejects their late completion as stale. The host returns the
abandoned IDs and preserves work in other slots. An explicit external import
may ingest a late measured result through current relevance rules, without
origin controller credit. Resident restart drops queued imports addressed to
the old generation. A post-dispatch uncertain resident failure poisons the
old engine and retires or invalidates its ticket before restart.

Alternatives considered: wait for pending work, silently admit a late result
to the new generation, or automatically archive it. Explicit restart and
explicit import keep identity and progress attribution unambiguous.

Evidence: `abandon_restart` in `optimizer-provenance` verifies partial host
completion, abandoned ID rejection, explicit late import, and another live
slot. `region-metal` passed 721 tests with 6 ignored, including terminal
cleanup fault injection. The wheel-backed Python restart test exercises the
natural-coordinate bridge. Revisit for automatic late-result archival,
cross-generation replay, or recovery from a terminal model-update failure.
Beads task: `blckbaux-sjj`.

### MR-Q04 — 2026-11-05 — host exact duplicates

Rule: a new host candidate pool excludes exact coordinates of unresolved
evaluations. If too few distinct candidates remain, ask fails before issuing
new IDs. The resident path relies on its one-ticket-per-slot capacity for
same-slot proposals.

Alternatives considered: accept duplicates, enforce a distance threshold, or
fantasize pending outcomes. Exact exclusion is deterministic and does not
invent a distance scale or surrogate requirement.

Evidence: `pending_exclusion` in `baux-unit` (662 passed, 5 ignored) and
overlapping host ask tests in `optimizer-provenance` (17 passed). Unresolved
limit: nearby pending points across regions are not excluded and no fantasy
model is used. Revisit when duplicate-adjacent proposals are observed in a
real workload. Beads task: `blckbaux-sjj`.

### MR-Q07 — 2026-11-05 — current execution boundary

Rule: issue, completion, cancellation, import, and restart are synchronous
optimizer calls with stable evaluation identities. Objective work may run
outside the optimizer and finish in any order; callers submit each result by
ID. Python exposes this contract through `AskBatch` and typed `AskId` values,
without owning an event loop. The host Rust path stores its posterior and
regional models in host memory. The Metal regional path keeps parameter and
history buffers on the device and admits one outstanding ticket per slot.
There is no CUDA resident regional search engine in this checkout.

Alternatives considered: make Python own an asyncio scheduler now, or treat
device residency as an implied asynchronous optimizer API. Neither follows
from the present completion contract or measured evidence. A scheduler can
later drive the same IDs and tickets without changing observation origin or
controller credit.

Evidence: `overlapping_asks`, `abandon_restart`, and wheel-backed Python
completion/restart tests exercise out-of-order and stale identities. The
72-case Metal run records host wall-clock phase times and zero parameter
readbacks; it has no device-only or overlap measurement. Revisit when adding
a public nonblocking loop, a CUDA resident engine, or a device latency claim.
Beads task: `blckbaux-sjj`.

### MR-Q08 — 2026-11-04 — host and Metal resident

Rule: Validation failures before mutation leave work retryable. Once a model
update or deferred flush may have mutated state, the host records a terminal
`ingestion_failure` and rejects every later submission. The caller may build a
new optimizer from independently trusted completed history, but must not replay
the uncertain observation on the old optimizer. The resident path retires the
ticket with `TellFailure::Retired`, poisons an uncertain engine, and requires
replacement by restart; an old ticket then returns `TellFailure::Stale`.

Alternatives considered: retrying on the same mutable model, silently counting
the observation, or cloning every surrogate before each update. These cannot
provide a general exactly-once guarantee for supplied Python and disk-backed
providers without a transactional provider contract.

Why this rule: no-retry makes partial mutation visible and prevents duplicate
credit. It preserves a cheap hot path for providers that cannot roll back.

Evidence: `//build/buck2/tests:optimizer-provenance` passed 13 tests;
`//rust/crates/baux:region-metal` passed 720 with 6 ignored; the wheel-backed
`//build/buck2/tests:resident-python` passed 44. See the latest
[optimizer checkpoint](multiregion_optimizer.md#shared-working-copy-integration-checkpoint-2026-11-04).

Unresolved limits: no automatic rollback or native checkpoint restores a
poisoned host optimizer. Revisit when a provider supports transactional update
or a persisted optimizer snapshot. Beads task: `blckbaux-i06`.

### Template

Add a short entry when another question is decided:

```text
MR-Qnn — date — scope (host / resident / both)
Rule:
Alternatives considered:
Why this rule:
Evidence: source links, exact tests or tune history filters, hardware if relevant
Unresolved limits:
Revisit when:
Beads task:
```

## Source pointers

These sources show mechanisms, not a single endorsed architecture:

- [pySOT SOP]: asynchronous per-center updates and pending candidate exclusion.
- [POAP]: proposal acceptance and evaluation lifecycle records.
- [pySOT's epoch check]: global retention of results from before a restart.
- [APOSMM]: routing returned simulation IDs to concurrent local runs.
- [Trieste's async rule]: pending-point acquisition state.
- [multi-region rule]: multiple local trust-region subspaces.
- [Xopt's future loop]: refill worker slots after first completion.
- [original TuRBO-m]: synchronous multi-region batch method.
- [OptunaHub TuRBOSampler]: trial-to-region membership and completion updates;
  pending-aware acquisition and late-result behavior still require tests.

[pySOT SOP]: https://github.com/dme65/pySOT/blob/c8f04fd4ed30d49bb61adb008134741319b512a4/pySOT/strategy/sop_strategy.py
[pySOT's epoch check]: https://github.com/dme65/pySOT/blob/c8f04fd4ed30d49bb61adb008134741319b512a4/pySOT/strategy/surrogate_strategy.py
[POAP]: https://github.com/dbindel/POAP/blob/2866d20e90093f6c1f98f966cfc406e98f32d4d0/poap/strategy.py
[APOSMM]: https://github.com/Libensemble/libensemble/blob/2f79ba6943083ff71875272d6f3f9b5222acd273/libensemble/gen_funcs/persistent_aposmm.py
[Trieste's async rule]: https://github.com/secondmind-labs/trieste/blob/5fb83e75b4d2677d70bb2c456ad44471fe51d653/trieste/acquisition/rule.py
[multi-region rule]: https://github.com/secondmind-labs/trieste/blob/5fb83e75b4d2677d70bb2c456ad44471fe51d653/trieste/acquisition/rule.py
[Xopt's future loop]: https://github.com/xopt-org/Xopt/blob/a7f787a8631299d118afecaa6ef7ef27bcbf2e34/xopt/asynchronous.py
[original TuRBO-m]: https://github.com/uber-research/TuRBO/blob/de0db39f481d9505bb3610b7b7aa0ebf7702e4a5/turbo/turbo_m.py
[OptunaHub TuRBOSampler]: https://github.com/optuna/optunahub-registry/blob/a23bec519fafa2565b9e9d47f2024a238f4839e5/package/samplers/turbo/sampler.py
