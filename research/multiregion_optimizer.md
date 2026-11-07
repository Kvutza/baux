# Multi-region optimizer foundations

Open method and asynchronous lifecycle questions are tracked in the
[multi-region question register](multiregion_questions.md). This note records
implementation facts and evidence; its provisional rules do not resolve those
questions.

Multi-region optimization is an evolving family of methods, not a synonym for
one trust-region algorithm.  BAUX therefore preserves proposal history as fact
and treats observation relevance, progress attribution, adaptation, allocation,
and restart as replaceable policies.

## Stable concepts

- A **region slot** is storage reserved for a local search context.
- A **region generation** identifies one lifetime of a slot.  Restarting a slot
  creates a new generation.
- A **region instance** is the pair `(slot, generation)` together with its
  geometry, role, and lifecycle state.
- An **arm** is a proposed evaluation.
- **Proposal provenance** records which region instance generated an arm.  It
  is historical fact and remains valid after that region restarts.
- An **observation** is the result returned for an arm.
- **Relevance** decides which regional or global models may learn from an
  observation.
- **Attribution** decides which region receives progress credit and adapts.

Provenance does not imply exclusive model ownership, geometric membership, or
progress attribution.  Those interpretations must remain independently
replaceable as the research develops.

## Current execution invariant

The host implementation keeps the current numerical multi-region method and
records an evaluation ID for every warm-up and regional arm. Regional arms also
retain their generating slot and generation. It permits one unsettled issued
group per region slot, with another ask to a settled slot while other slots
remain busy. Warm-up permits one outstanding batch. A capacity or validation
error does not replace existing identities.

## Research lineage

The Python loop walkthrough below is the starting point for the optimizer
interface. The region state work above is provisional until it supports that
interaction coherently.

| Work | Idea retained by this foundation |
| --- | --- |
| Eriksson et al. (2019), *Scalable Global Optimization via Local Bayesian Optimization*, NeurIPS, https://proceedings.neurips.cc/paper/8788-scalable-global-optimization-via-local-bayesian-optimization | Concurrent local-search trajectories, adaptive regions, and regional allocation. |
| Daulton et al. (2022), *Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces*, UAI, https://proceedings.mlr.press/v180/daulton22a.html | Collaborative regions can share relevant evidence without erasing local region state. |
| Namura and Takemori (2025), *Regional Expected Improvement for Efficient Trust Region Selection in High-Dimensional Bayesian Optimization*, AAAI, https://doi.org/10.1609/aaai.v39i18.34161 | A region itself can be valued and selected rather than created by an uninformed restart. |
| Saroj et al. (2026), *Learning-Based Memory-Guided Trust-Region Bayesian Optimization*, IEEE Access, https://doi.org/10.1109/ACCESS.2026.3736328 | Restart is informed by persistent search memory about promising and under-explored basins. |
| Das et al. (2026), *MTRBO: Multiple Trust-Region Based Bayesian Optimization*, https://arxiv.org/abs/2605.06618 | Simultaneous regions may have distinct exploration and exploitation roles. |
| Van Dieren, Vanzan, and Nobile (2026), *LAGO: A Local-Global Optimization Framework Combining Trust Region Methods and Bayesian Optimization*, https://arxiv.org/abs/2603.02970 | Global and local search mechanisms can cooperate or compete without being the same mechanism. |

These citations motivate replaceability; they do not establish that one paper's
observation-sharing or attribution rule is universally correct.

## Python loop walkthrough (discussion draft)

These are behavioral examples, not approved new method signatures. Only the
first snippet uses the existing interface. The later traces describe operations
without choosing a proposal container, handle type, or configuration hierarchy.

### Existing ordinary loop

Given an optimizer and a caller-supplied objective:

```python
x = optimizer.ask(num_arms=4)
y = objective(x)
optimizer.tell(x, y)
```

The caller runs the objective and keeps the result rows aligned with the input
rows. Rust owns search state. Changing region dynamics or using a compatible
surrogate-free proposal method should preserve this basic interaction.

Current path:

1. `src/baux/bo.py`: `create_optimizer` chooses a native kind through `_plan`,
   based on surrogate type. It always passes a region count of four; it does
   not select the native multi-region kind. `encode._tr` also drops the
   `RegionalSearchPolicy` configuration.
2. `Optimizer.ask` calls the native wrapper in
   `rust/crates/baux-py/src/py_optimizer.rs`, which calls `ask_natural` in Rust.
   Python receives only parameter rows, without proposal identity.
3. `Optimizer.tell` normalizes and validates shapes before applying its existing
   read-only input contract and calling native `tell_natural`.
4. For compatibility, `tell` still returns fitted means when a surrogate exists,
   otherwise observations. A prediction failure after native success now raises
   `TellPredictionError`, with `observations_accepted = True` and the original
   exception as its cause. Callers can retry `optimizer.posterior_mu(x)` without
   submitting observations again. Native ingestion failures are not wrapped in
   that exception and are not claimed to be transactional.

The experimental `RegionalSearchLoop` instead orchestrates allocation,
proposal callbacks, scoring, and selection around a standalone state. It does
not call the integrated optimizer's multi-region execution. We must account
for these existing users without duplicating the evolving regional policies
in two implementations.

### Partial or reordered completion

```text
Request four arms: A, B, C, D.
Evaluate them outside the optimizer.
Return C's result, then A's result.
B and D remain pending.
Return D's result after its originating region has restarted.
```

The host API now assigns numeric identities to each ask row, including warm-up.
It accepts partial and reordered results by ID. Identical coordinates may have
different IDs; the ordinary `tell(x, y)` path retains an ordered-coordinate
guard for whole-batch compatibility. Natural-coordinate asks record the
normalized values that a tell reconstructs. The host still allows only one
outstanding batch and cannot yet realize the late-after-restart trace above;
the resident portfolio can hold one outstanding proposal per region.

### Existing data and externally chosen evaluations

```text
Supply previously evaluated points and their observations.
Request new arms using the available knowledge.
Later supply an additional externally chosen evaluation.
```

Existing data has no optimizer-generated arm identity. It must be distinguishable
from completion of a pending proposal and must not consume that proposal's
provenance. Relevance and attribution policies decide how imported observations
affect regions. With no pending batch, the current path accepts data into the
posterior and incumbent without changing regional trial statistics or regional
adaptation. This is a conservative BAUX engineering rule, not a research-derived
sharing policy. Imports during an outstanding batch still need an explicit
operation that cannot be mistaken for completion.

### Failure and retry

An objective evaluation failure is distinct from a poor measured objective.
The caller must be able to determine whether an arm remains pending, failed,
or was cancelled. A malformed result must leave its proposal available for
correction. A failure during model updating needs a defined ingestion outcome
so a retry cannot silently count the observation twice. Repeating delivery of
one result is distinct from deliberately evaluating the same point again.

Current validation checks finite observations and shaped, finite, nonnegative
variances before mutation. Duplicate ID completion is rejected; explicit
imports remain separate. Surrogate fitting, incumbent updating, and storage
flushing can still fail after mutations. The host then blocks later operations
with a no-retry ingestion error, but has no restoration path yet.
`TellPredictionError` addresses only a later Python prediction query. Stale
generations are rejected and retained; cancellation and late-result policy
still need a shared host/resident contract.

### Earlier implementation checkpoint

At the first checkpoint, the complete-batch guard and distinguishable Python prediction error were
implemented. The next work was to carry explicit arm identity through Python, the native
adapter, and Rust, with partial completion and explicit imports that cannot
consume another proposal. Establish the native ingestion outcome separately
from surrogate-update failures. Preserve the distinction between origin,
information relevance, and progress attribution throughout. The public spelling
and transport of identity remain open for joint review. The general Python
factory's multi-region routing gap also remains open.

### Verification at the first checkpoint, 2026-11-04

| Command | Observed result |
| --- | --- |
| `./baux check` | Passed, including names and kiss. |
| `./tools/buck2w test //build/buck2/tests:optimizer-provenance --local-only --console simple` | Seven Rust integration tests passed, covering surrogate-free and ENN lifecycles. |
| `./tools/buck2w test //build/buck2/tests:optimizer-python --local-only --console simple` | Five Python boundary tests passed in the pinned CPython 3.14 environment, with a controlled native-boundary double. |
| `./baux build` | Core library and runner passed. Wheel failed on existing `py_metal.rs` incompatibilities: missing `correlate`, `enable_relative`, and `read_reference`, plus `pool_geometry` return mismatch. Shared tests also fail to resolve `//rust:rand_chacha`. |

The focused targets exercise production Rust state and the production Python
wrapper separately. They do not establish end-to-end Python-to-native parity,
general surrogate support, asynchronous completion, or performance. The build
prerequisites are recorded in Beads as `blckbaux-eb3`.

### Citation convention

At each implemented research rule, cite author, year, paper title, relevant
section or algorithm, and a stable link. For software actually adapted, cite
the repository, source path, and pinned commit or release, retaining required
license notices. Distinguish paper-prescribed behavior, adapted behavior, and
BAUX engineering decisions. The lifecycle requirements in this walkthrough are
BAUX design reasoning, not claims that the cited papers prescribe this interface.

## Resident implementation, 2026-11-04

The initial resident implementation added a backend-neutral `Portfolio<S>`
transaction ledger and a concrete `RegionalSearch` adapter over the existing
Metal `SearchState`. The unified-lifecycle checkpoint below replaces that
engine-owning container with `RegionalLedger`.
The ledger owns region lifetimes, allocation statistics and pending transaction
identities. Engines own numerical methods, parameter storage and model history.
It accepts explicit region selection, balanced allocation or a utility-UCB
heuristic. This heuristic is a BAUX engineering policy, not faithful TuRBO-m
joint Thompson sampling or a stationary-bandit regret claim.

Each resident region has a separate incumbent, history, controller and scratch
allocation. Engines reuse the shared Metal runtime and pipeline cache. Several
regions can have outstanding proposals; their results can complete in any
order. One proposal may be pending per region. Foreign, stale and duplicate
completions fail. A malformed measurement leaves the handle pending. Replacing
an idle region with an evaluated fresh start increments its generation; this is
an actual new trajectory, rather than only resetting a radius counter.

Python exposes this path experimentally as `MetalRegionOptimizer`. Its proposal
objects retain their originating owner and expose the region and generation.
`proposal.weights()` returns a leased read-only resident buffer. Release exported
views after evaluation and before any mutating operation. Mutation is currently
blocked across the whole portfolio while any exported view remains alive.

```python
first = optimizer.ask(seed=12)
second = optimizer.ask(seed=13)
weights = second.weights()
value = evaluate_on_metal(weights)  # caller owns objective evaluation
del weights
optimizer.tell(second, value)
# first remains pending and can be evaluated/completed independently.
```

The ordinary `create_optimizer` path now forwards `RegionalSearchPolicy`
region count, length bounds, tolerances and sharing choice into the Rust
multi-region strategy. ENN, no-surrogate and the existing batched Python
surrogate adapter use that same strategy. `num_regions` reports configured
region slots. The historical `region_count` property still counts observations
for compatibility. In-memory ENN now fits one local model per host region and
routes observations according to the sharing choice. Its global posterior
remains the public prediction and observation store. `ExternalSurrogate` creates
an independent global provider and a fresh provider for each regional generation;
providers implement batched `fit`, `predict`, and `draw`. The built-in Python GP
uses the same factory path in multi-region mode. A lone low-level Python
provider is rejected for modeled multi-region search. The
resident adapter now supports the same three relevance labels; its geometry
and delayed-import details are recorded below.

### Recorded experiment

`./baux tune experiments/resident-regions.toml` executes a GPU-resident BF16
double-well objective with 64 dimensions, 64 total objective evaluations,
one/four regions, balanced/UCB allocation, and seeds 17, 29 and 43. Initial
evaluations and one explicit fresh start count toward the same budget. The
objective reads the resident candidate buffer on Metal and returns one scalar;
the steady-state loop performs no parameter-array readback. This is a small
functional experiment, not a training or large-model benchmark.

First recorded run: `4c5e2a7a39e71b963e0cd1ef3f257dc27461b8781213037f9cf2be729a985464`,
source `456d65b15744f4251a16f8c55bcfc93cf51022bc`, Apple M4. Full traces are
in the run's `regional.json`, discoverable through
`./baux tune --history --task current --since today --json`.

| Regions | Allocation | Mean final best reward, three seeds |
| --- | --- | --- |
| 1 | Balanced or UCB | -0.448346 |
| 4 | Balanced | -0.526263 |
| 4 | UCB | -0.439586 |

Higher reward is better. Four-region UCB beat one region on two seeds and lost
on one. Balanced four-region allocation lost on all three. The measured loops
took 35–63 ms excluding initial construction; these timings include sequential
GPU dispatch and scalar synchronization, and are not evidence of fused GPU
scheduling or large-parameter latency. The first run's catalogue lacked plot
summaries even though raw traces existed; regional evidence rendering was then
added so subsequent runs include separate seed/allocation plots and summaries.

The completed repeat run is archived in
[`resident-regions.toml`](resident-regions.toml):
run `cb76110a5681fc3e571eb82ccbf4fd943358dceb573d3e22cf959eb1a845c218`,
source `2378adfea1cc0fa727c34da8b197f83b20c0fc8e`. All 12 objective traces
exactly matched the first run, all evaluation budgets summed to 64, and the
catalogue reported 12 complete plot summaries. Repeat timings were 45–147 ms;
these runs were not isolated latency benchmarks.

Final checks: `./baux check` passed; the focused Rust targets
`optimizer-provenance` and `resident-regions` passed seven and two tests;
`resident-python` passed 22 tests against the built wheel, including all three
legacy regional sharing choices with ENN, no surrogate, and a custom batched
Python provider. The full `baux-unit` executable built; its complete test suite
was not run. The formatter now uses one Rustfmt process: recursive module
formatting from parallel shards could otherwise write the same file at once.

### Boundaries still open

- The host complete-batch optimizer and resident proposal-handle adapter have
  distinct completion interfaces. The generic ledger is ready for other engine
  adapters; it is not yet the host optimizer's state machine.
- Host regional adaptation now requires a trial generated by that region.
  Shared or imported observations can update an eligible incumbent without
  granting success or failure credit. A test covers a foreign improvement
  arriving alongside a worse local trial. In-memory ENN host acquisition now
  uses local models with independent, shared, or nearest-center data routing.
  Warm-up observations seed every local model. A restarted slot clears its
  local model and uses an unmodeled first proposal until new local data arrives.
  Regional data counts and restart clearing have a focused unit test. Restart
  seeding and search memory remain unimplemented.
  At this checkpoint, disk ENN and a lone low-level Python provider had no
  local-model routing and failed construction instead of silently scoring from
  the global model. Disk ENN support was added in the later checkpoint below.
  A low-level `gp_factory` or high-level `ExternalSurrogate` factory supplies
  independent Python models. At that checkpoint the wheel-backed Python suite passed 26 tests,
  including separate regional provider histories and rejection of reused
  provider instances. The core Rust unit suite passed 652 tests with 5 ignored
  after the unsupported-model and empty-local checks; `./baux check` also passed.
- The concrete resident implementation uses the existing Metal ENN/TuRBO
  numerical path. Other surrogate/controller combinations and CUDA execution
  are not established by this experiment.
- Resident late completion after restart, checkpoint restoration and production
  training integration remain work. Host post-validation
  model-fit failures can still leave partial ingestion, so a host ingestion-state
  contract remains necessary.
- At this checkpoint, fresh starts were supplied explicitly. The later host
  implementation automatically restarts a retired TuRBO slot at a uniformly
  sampled unit-cube center. Automatic basin discovery or learned restart
  policies are not implemented.
- The native Metal binding now matches the available Gaussian implementation;
  unsupported correlated/reference modes fail explicitly instead of calling
  missing Rust methods. The shared test dependency alias preserves rand_chacha
  0.3.1 and survives Reindeer regeneration.

### Landing gate

The host and Metal scope now satisfies the integration gate for landing.
Generation-aware identities, partial completion, cancellation, explicit
imports, restart, and terminal post-mutation failure handling are exercised
through the host and resident adapters. Sharing changes model relevance without
changing proposal origin or granting foreign controller credit. Model-free
search, built-in ENN, and supplied surrogate providers have explicit support
boundaries. Region method, allocation, acquisition role, restart memory, and
restart-center selection remain independent policy choices rather than being
baked into the identity ledger. The Python loop and the resident experiment
exercise the applicable contracts.

After rebasing onto `main`, `./baux check` passed. The combined Buck test run
passed `baux-unit` (732 passed, 6 ignored), `optimizer-provenance` (20 passed),
`baux-runner-test` (19 passed), `resident-python`, and `region-metal`. These
checks support the host-and-Metal landing; they do not establish a preferred
research policy, a CUDA resident engine, asynchronous kernel overlap, or a
device-latency target. MR-Q05 and MR-Q06 therefore remain open research
questions with explicit revisit triggers, not landing blockers.

### Resident sharing checkpoint, 2026-11-04

`RegionalSearch::with_sharing` now applies independent, shared, or
nearest-center relevance to resident BF16 observations. Metal computes the
block-weighted distance from the evaluated row to each regional incumbent and
retained resident history row. Only distance scalars return to the CPU. A
recipient imports a stable GPU copy of the foreign row into its own history;
the recipient's incumbent, proposal origin, trial count, and trust controller
credit stay unchanged. This is model evidence, not a foreign success. Shared
admission uses the target's current length and block scales as a squared metric
radius; nearest-center uses the smallest incumbent distance with stable tie
breaking. These are BAUX resident policy choices, not claims that TuRBO or
MORBO prescribe this exact metric threshold.

An outstanding recipient proposal pins its history. Foreign rows for that
recipient wait in a bounded queue and are imported before its next ask or on
explicit `flush_sharing`. `sharing_backlog` and `delivered_imports` expose the
state. Queue capacity is 128 rows per recipient. A new generation discards
the old generation's queued rows. The source ticket is completed once; if a
post-mutation resident failure poisons the source engine, the portfolio retires
its ticket without progress credit and the caller must restart that slot.
Pre-mutation validation errors leave the ticket retryable.

The 72-case `./baux tune experiments/resident-regions.toml` run is catalogue
entry `2eaeeae5d804f0b53a0417da4a91711c6dd07e7aa726f573c18ab2580a1c9022`
on Apple M4; its raw traces are archived in
[`resident-regions-sharing.toml`](resident-regions-sharing.toml). It crosses one/four regions, two allocation rules, three sharing
rules, separated/identical starts, and three seeds at 64 dimensions and 64
evaluations. The separated-start cases delivered no foreign rows. With
identical four-region starts, nearest-center delivered eight foreign rows
across three seeds for each allocation rule; shared admitted none under its
current radius. Final rewards were identical across sharing rules in every
case. The experiment verifies routing and no readback of parameter arrays,
but gives no evidence that sharing improves this objective. Timings are short
non-isolated runs with sequential Metal dispatches, so they do not establish a
multi-region latency target.

The resident Rust tests exercise GPU distance parity, no-credit import, FIFO
retention, and ticket retirement. The wheel-backed Python suite exercises
pending-recipient delivery, nearest-center and shared routing, and the public
backlog. The full completion gate above remains open: host and resident still
have different completion interfaces, and restart memory, partial host
completion, and the host ingestion outcome remain unfinished.

### Regional disk ENN checkpoint, 2026-11-04

Built-in disk ENN now creates a separate store under
`work_dir/regions/slot-N-generation-G` for every regional model. The global
posterior keeps its configured `work_dir`. Restart clears the previous
generation's model store after constructing its replacement; dropping the
optimizer clears the current regional stores. External provider factories keep
their own lifecycle. A Rust test checks two populated stores, one restarted
slot, and cleanup after drop. The wheel-backed Python loop exercises warm-up,
regional asks, and tells with disk ENN; 29 Python tests and 653 Rust unit tests
passed (5 Rust tests ignored). This establishes the functional disk path, not
an ingestion transaction or a throughput claim.

### Host submission checkpoint, 2026-11-04

The active host multi-region path now distinguishes a generated proposal
completion from an external observation. `tell` requires the pending ordered
batch; `import_observations` supplies external data without consuming it.
Imports can arrive while a generated batch remains pending. After successful
completion, submitting the same batch through `tell` fails rather than being
interpreted as another import. The Rust provenance tests cover this case and
the wheel-backed Python loop exposes the same operation.

Multi-region ingestion now has an explicit no-retry failure state. Validation
errors before dispatch leave the pending batch available. If dispatch fails
after mutation may have begun, `ingestion_failure` records the cause and later
asks and submissions fail with a do-not-retry message. A Python test makes the
global surrogate fail after warm-up and verifies that the batch cannot be
submitted twice. This prevents duplicate accounting but does not repair a
partially updated global or regional model; recovery currently requires a new
optimizer from trusted observations. At this checkpoint warm-up asks remained
untracked; the later identity checkpoint closes that gap.

### Identified host completion checkpoint, 2026-11-04

Post-warmup regional asks now retain one process-wide `EvaluationId` per row.
`pending_ids` reports only unfinished rows. `complete_ids` accepts any nonempty
subset in any order, validates all IDs and result shapes before ingestion, and
rejects duplicate or stale IDs without consuming the pending batch. Each
successful completion updates the global and relevant local models at arrival;
the region controller receives credit for observed rows only after every row
has completed or been cancelled. The ordinary ordered `tell` path remains valid for a
whole batch, but cannot be mixed with partial completion. The Python optimizer
exposes `pending_ids` and `complete_ids`; callers continue using ordinary
`ask`/`tell` during warm-up or use IDs there too. A focused Rust test covers out-of-order results,
duplicate rejection, pending state, and deferred controller credit.

This API follows the [Google Rust Style Guide](https://google.github.io/styleguide/rustguide.html)
on distinct identity types, explicit modes in place of boolean parameters,
and checked invariants. BAUX's own `./baux check` remains the formatting and
structure gate. The completion gate above remains open: resident and host
still have separate completion interfaces, and the poisoned host ingestion
state has no restoration path. A later checkpoint adds warm-up IDs and verifies
partial warm-up completion without consuming an explicit import.

### Cancellation checkpoint, 2026-11-04

Both execution adapters now retire an unevaluated proposal without treating it
as a measured failure. Host `cancel_ids` resolves any subset of a pending batch;
only rows with actual observations enter the regional controller after the
batch is fully resolved. Cancelled warm-up rows leave the initialization budget
available for new asks. Resident `cancel` clears the validated Metal proposal
and retires its portfolio ticket; a late tell for that ticket fails, while the
same region can issue another proposal without a generation change. Python
exposes both operations. A live exported resident weight view blocks
cancellation until the caller releases it.

Post-dispatch host flush failures now enter the same terminal no-retry state as
model-update failures. Multi-region imports require one scalar objective
column before mutation. The Rust provenance target passed 12 tests, the full
Rust unit target passed 653 tests with 5 ignored, the wheel-backed Python
target passed 32 tests, and `./baux check` passed. These results verify the
new lifecycle operations, not restart-memory policy or a latency benefit.

### External resident import checkpoint, 2026-11-04

`RegionalSearch::import_region` accepts an externally evaluated BF16 row into
an explicitly chosen regional model without assigning a proposal origin or
controller credit. An idle target imports it immediately. A target with a
pending proposal receives a stable GPU copy through the existing bounded
queue, then imports it after that proposal resolves. Python exposes the same
operation. The wheel-backed Python test covers both paths, confirms zero trial
credit for the external row, and checks delivery accounting. The resident Metal
target passed 712 tests with 6 ignored; the wheel-backed Python target passed
33 tests; `./baux check` passed.

Host and resident storage still differ by design: the host can retain external
rows in its global posterior. The resident portfolio has only local models.
Its `import_observations` now applies nearest-center or shared relevance and
returns the admitted slots; independent mode requires `import_region` with an
explicit target. Admitted rows are staged once on the GPU and delivered through
the existing bounded recipient queues. The wheel-backed Python target passed
36 tests including all three policy cases. Restart memory and isolated latency
evidence remain open.

### Backend and surrogate support audit, 2026-11-04

| Execution path | Surrogate | Evidence and boundary |
| --- | --- | --- |
| Host `create_optimizer` | No surrogate | Wheel-backed loop passes under independent, shared, and nearest-center policies. There is no regional model to ingest foreign data. |
| Host `create_optimizer` | In-memory ENN | Wheel-backed loop passes under all three policies; Rust regional-model tests check data routing and restart clearing. Scalar Pareto acquisition is supported and is the default `turbo_enn` acquisition. |
| Host `create_optimizer` | Disk ENN | Wheel-backed warm-up and regional loop passes under all three policies. Per-generation local stores are verified to exist. This does not measure disk throughput. |
| Host `create_optimizer` | Built-in Python GP | A wheel-backed two-round regional loop passes with actual GP fitting and prediction. The test uses the default independent policy and does not establish numerical parity or throughput. |
| Host `create_optimizer` | External Python provider factory | Wheel-backed tests cover independent instances, policy-routed histories, and rejection of reused instances. These use test providers, not every possible third-party surrogate. |
| Metal `RegionalSearch` | Resident `SearchState` ENN | Rust and wheel-backed tests cover all three sharing modes, pending-recipient delivery, GPU row import, and no controller credit. This adapter has no arbitrary surrogate factory. |
| CUDA resident | Any | No multi-region resident CUDA adapter or runtime evidence. |

The Python `ENNSurrogate.distance_metric` config also exposed cosine distance,
but `encode` and the native optimizer do not carry that setting. Constructing
an `OptimizerConfig` with cosine now raises an explicit error instead of
silently running squared L2. Scalar Pareto is valid here: the regional Rust
selector applies Pareto acquisition to each local surrogate's one-column
prediction. Multi-objective regional completion remains unsupported because
the multi-region host strategy accepts one objective column.

Verification for this audit: `./baux check` passed; the wheel-backed
`//build/buck2/tests:resident-python` target passed 40 tests, including the
built-in GP loop, all disk ENN policy modes, and the cosine rejection. A Torch
Python 3.14 FutureWarning appeared during the GP test. These results establish
functional support on the tested Apple M4 environment, not a latency target.

### Shared-working-copy integration checkpoint, 2026-11-04

Host completion now settles each region's issued group as soon as all rows for
that region have completed or been cancelled. Other regions in the same batch
may remain pending. At this checkpoint the host permitted one outstanding ask batch; the
Python Ax/Optuna adapter reports that capacity limit without cancelling a live
trial or silently converting its eventual completion into an external import.
An external observation is imported explicitly, including for a one-region
multi-region optimizer. A deferred flush failure makes the host terminal, so
the caller cannot retry an observation whose storage outcome is uncertain.

The resident portfolio can replace a region with an outstanding ticket. It
cancels the old round, advances the generation, drops queued foreign evidence
for that generation, and rejects a late tell as stale. A resident tell exposes
retryable validation, retired post-mutation, and stale-ticket failures. A
post-dispatch failure poisons the old engine until restart. The Python API
provides typed `AskId` and `AskBatch` values over the existing native IDs,
with partial completion and cancellation; the ID is process-wide unique, while
the native pending proposal stores and validates region generation separately.

Host policy choices now separate allocation (`RewardUcb` or `Balanced`),
restart memory (`Fresh` or `RetainRelevant`), region method (`Turbo` or
`FixedLength`), and optional per-slot acquisition overrides. `FixedLength`
keeps the configured length while updating relevant incumbents and recording
origin trial counts. An empty local model falls back to a fitted global model;
model-free selection uses random proposals. Only `RetainRelevant` stores model
history, and replay uses the caller's seeded RNG. Python config and PyO3 expose
these choices with defaults matching the existing behavior. These are BAUX
policy variants motivated by the cited multi-region literature, not claims of
faithful implementations of those papers' distinct methods.

Verification: `./baux check` passed with no naming or structure violations;
`//rust/crates/baux:baux-unit` passed 661 tests (5 ignored),
`//rust/crates/baux:region-metal` passed 720 (6 ignored),
`//build/buck2/tests:optimizer-provenance` passed 13, and the wheel-backed
`//build/buck2/tests:resident-python` passed 44 (one Torch FutureWarning).
The fresh `./baux tune experiments/resident-regions.toml` run is catalogue ID
`d8d89bc82ebe6f3bbb4836251db0f34343308db8c3f3b9c377221c98423a0653`
on Apple M4. Its 72 cases match the earlier run's seeds, 64 dimensions,
64 evaluations, region counts, sharing modes, allocations, and start layouts.
All 72 reward traces, 16 delivered imports, and zero parameter readbacks match
the saved baseline exactly. Median case time excluding initialization was
0.056 seconds versus 0.076 seconds previously; these short, sequential,
non-isolated runs do not support a speedup claim or the full resident latency
target.

Open boundaries at this checkpoint: host issuance remained one outstanding batch, resident issuance
one ticket per region; neither is an unbounded asynchronous scheduler. The
Ax/Optuna adapter reports capacity when a previous regional suggestion is
still live. There is no resident CUDA multi-region adapter, no arbitrary
surrogate factory for the resident Metal engine, and no restoration of a host
optimizer after a terminal model-update failure. Those limits are not hidden
by the passing functional tests.

### Completion contract across execution paths

| Event | Host Rust and Python optimizer | Metal resident portfolio |
| --- | --- | --- |
| Issue | `ask`/`ask_batch` records a process-wide evaluation ID and slot/generation provenance for every row. A settled slot can receive another group while other slots stay busy; a busy slot cannot receive a second group. Warm-up keeps one batch. | `ask_region` returns a ticket containing owner, slot, generation, and round; one ticket may be outstanding per slot. |
| Partial result | `complete_ids` accepts any subset in any order. The global and relevant local models ingest each successful result at arrival; a slot's controller settles once all rows issued for that slot resolve. | `tell_result` completes one slot ticket; other slots can remain in flight. |
| Cancellation | `cancel_ids` retires selected rows without observation or credit. | `cancel` retires the ticket and clears the engine round without reward credit. |
| Restart | `restart_region` abandons unresolved IDs for one slot, advances its generation, returns those IDs, and preserves other slots' pending work. Its Python form takes a natural-coordinate center. | `restart_region` abandons the slot's pending ticket and advances its generation. |
| External evidence | `import_variance` updates eligible models and incumbents without consuming an ID or assigning controller credit. | `import_region` selects a target explicitly; `import_observations` routes by sharing policy. Queued GPU copies retain no proposal origin or credit. |
| Pre-mutation failure | Invalid coordinates, ID, variance, or resident score leave the relevant pending work retryable. | `TellFailure::Retryable` leaves the ticket pending. |
| Post-mutation failure | `ingestion_failure` becomes terminal; the caller must rebuild from trusted completed history, rather than replay the uncertain observation. | `TellFailure::Retired` removes the ticket and poisons the engine when state may be uncertain; restart replaces that engine. |
| Stale completion | An unknown, duplicate, cancelled, or old-generation ID is rejected before ingestion. | `TellFailure::Stale` rejects a completed, cancelled, or replaced ticket. |

These are the current bounded asynchronous capacities. The common contract is
identity, origin, relevance, credit, and retry behavior; host observation
storage and resident GPU storage remain different implementations. A public
queue that issues more than one group to the same busy slot would be a separate
scheduling policy decision under MR-Q01, not a prerequisite for correct partial
completion of the current groups.

### Bounded overlapping host asks, 2026-11-05

The host pending ledger now allows a new ask for a slot whose previous issued
group has settled while another slot remains busy. One slot has at most one
unsettled group; a group may contain several evaluation IDs. `complete_ids`
and `cancel_ids` can arrive out of order. Resolved rows remain in the ledger
until their slot receives controller credit, then are pruned before that slot
is reused. Issued IDs remain process-wide unique. Ordered `tell` is retained
for an untouched complete batch; overlapping work uses IDs. Warm-up still
permits one outstanding batch. When every slot is busy, `ask` returns a
capacity error without changing pending work.

New host candidate pools exclude exact coordinates still pending in other
slots. If exclusion leaves fewer candidates than requested arms, ask returns
an explicit error. This is an exact duplicate rule; it does not use a distance
threshold or posterior fantasizing. Natural-coordinate asks update only the
new rows in the pending ledger, preserving older outstanding coordinates.
The Python `AskBatch` returns only the IDs created by its own call, and the
Ax/Optuna adapter may issue another suggestion when a slot is free without
cancelling a live trial.

The host can explicitly restart one slot while its group is partly complete.
It returns abandoned IDs, rejects their late completion, and leaves other
slots' pending IDs valid. Already ingested observations remain evidence. A
late measured result can enter through explicit `import_observations`, without
origin controller credit. The Python restart call accepts a natural-coordinate
center and returns typed `AskId` values.

The shared-relevance controller uses the incumbent visible when a slot's group
settles. A focused two-order test shows that reversing a high and low result
can change success counters while retaining one trial count per origin and
the same final incumbent. This is an event-order rule to revisit under MR-Q02,
not numerical invariance across asynchronous schedules.

Verification: `//build/buck2/tests:optimizer-provenance` passed 17 tests,
including repeated overlapping asks, natural-coordinate preservation, and
both arrival orders; `//rust/crates/baux:baux-unit` passed 662 with 5 ignored,
including exact pending-point exclusion. The wheel-backed
`//build/buck2/tests:resident-python` passed 46 tests with one Torch warning;
`//rust/crates/baux:region-metal` passed 721 with 6 ignored after a resident
terminal-cleanup fix. These tests establish lifecycle behavior, not host
throughput or a CUDA implementation. A CUDA audit found no resident CUDA
single-region search engine to wrap; the CUDA runtime/buffer crate and Metal
resident search engine are distinct implementations.

### Resident phase evidence, 2026-11-05

`./baux tune experiments/resident-regions.toml` produced catalogue entry
`a5c7d3c1364b362a3ee07a62b7d7370c2b7cd5bb7495d5b7a3dcb64aeb71f3b7`
on Apple M4. It uses the same 72-case configuration as the saved sharing run:
64 dimensions and evaluations, three seeds, one/four regions, three sharing
modes, two allocation modes, and two start layouts. All 72 reward traces and
the aggregate 16 delivered imports match the saved run exactly.

The experiment now records host wall-clock phase times. Proposal includes a
queued sharing flush; objective includes Metal dispatch, completion, and one
scalar readback; tell includes result admission and synchronous sharing; final
sync is an explicit last sharing flush. Across the 72 cases, median case time
excluding initialization was 70.6 ms; median proposal, objective, and tell
times were 30.0, 16.2, and 20.0 ms respectively. These are medians of separate
phase distributions and need not sum to the median total. The timings are
sequential and not isolated from system activity. They do not establish native
GPU kernel time, overlap, or a device-resident latency target.

### Matched policy comparison in the resident run, 2026-11-05

The same saved 72-case artifact supports a fixed-budget comparison of the
implemented resident policies. Every case reports 64 objective evaluations,
including its initial region starts and one forced restart at evaluation 32.
For four regions, the trace begins at evaluation 4 because the four initial
starts are represented by one best-so-far point; its 61 trace entries still
end at evaluation 64. The restart is an experiment fixture, not an automatic
multistart policy.

Matching seed, region count, start layout, and allocation yields 24 groups of
three sharing modes. Within all 24 groups, independent, shared, and
nearest-center have exactly identical reward traces. Independent and shared
report zero foreign imports; nearest-center reports 16 imports across six
cases. This run therefore verifies routing activity in a few nearest-center
cases but gives no reward evidence for choosing a sharing rule.

Matching seed, start layout, and sharing at four regions, resident UCB's final
reward exceeds balanced in five of six distinct seed/layout cells and is
lower in one. Across those six cells, the mean UCB-minus-balanced final reward
is `+0.080139` on the experiment's negative double-well objective; all six
one-region cells tie. Sharing modes duplicate these traces, so counting all
three as separate evidence would overstate the sample size. Three seeds, one
synthetic objective, and two chosen layouts do not justify a general
allocation preference, role assignment, or restart-memory rule.

The host `Balanced` allocator was corrected to use completed trial counts
across asks. Previously, repeated one-arm asks after each completion always
selected slot 0 even though a single larger ask split candidates evenly.
`balancedhistory` now checks six successive one-arm allocations rotate over
three slots. The complete `baux-unit` suite passed 663 tests with 5 ignored;
`./baux check` found no naming or KISS violations. This correctness fix is
not part of the saved Metal comparison, whose resident allocator already
uses cumulative trial counts.

The host's existing TuRBO retirement path also restarts an inactive slot at
a uniformly sampled unit-cube center. `automatic_restart` forces one failure
to retire a one-slot region and verifies its generation advances, its trial
count resets, and the next ask succeeds. The local model now synchronizes at
completion rather than waiting for the next ask: after automatic restart,
`Fresh` exposes zero local observations and `RetainRelevant` exposes the two
previously admitted rows. A replacement model factory failure during an
overlapping ask now poisons the host optimizer; `restartfailure` verifies the
next ask is rejected rather than retrying uncertain model state.
`optimizer-provenance` passed 20 tests. This
confirms a uniform automatic start baseline and both restart-memory lifecycles.
The resident
experiment instead uses an explicit midpoint restart, so it cannot compare
automatic start policies.

### Matched host restart-memory run, 2026-11-05

`./baux tune experiments/host-regions.toml` produced catalogue entry
`f525c7c61506163c46560d4ebe1dae117b21e19f73f91ec1b8366c5cb4580dae`
and saved its raw observations in
`.cache/baux/runs/regional/1793856703821342000/regional.json`. The runner
uses the host ENN optimizer on an eight-dimensional negative double-well
objective. Each arm has four regions, 64 objective evaluations, balanced
allocation, a fixed-length region method, and one forced restart of slot 0
after evaluation 32. Three seeds, two start layouts, and three sharing rules
produce 18 matched `Fresh`/`RetainRelevant` pairs. Initial centers, optimizer
seed, objective, and settings match within each pair.

Every pair has exactly the same first 32 observation values and local-model
counts before restart. Every `Fresh` arm clears slot 0's local model at the
restart; every `RetainRelevant` arm keeps its admitted rows. In one independent
example the counts change from `[8, 8, 8, 8]` to `[0, 8, 8, 8]` under `Fresh`
and stay `[8, 8, 8, 8]` under `RetainRelevant`. The runner derives per-slot
evaluation counts from controller trial deltas rather than assuming a balanced
label implies equal counts after the restart.

Final best observed reward is `max(observation_values)` over all 64 calls.
The following values are `RetainRelevant` minus `Fresh` within matched pairs;
positive favors retention on this objective.

| Sharing | Matched pairs | Retention higher / lower / tied | Mean final-best difference |
| --- | ---: | ---: | ---: |
| Independent | 6 | 2 / 4 / 0 | −0.039586 |
| Shared | 6 | 2 / 4 / 0 | −0.063801 |
| Nearest-center | 6 | 2 / 2 / 2 | −0.016793 |

This establishes that restart memory changes the actual host model data and
can change later observations. It does not select a general winner: the run
uses one synthetic objective, three seeds, a forced restart and fixed-length
regions. Refit can also advance the optimizer RNG differently between arms.
Automatic multistart, role assignment, TuRBO-m allocation, and resident
restart-memory variants were not tested. `//rust/crates/runner:baux-runner-test`
passed 18 tests, and `./baux check` passed after adding the replayable host
experiment schema.

### Matched host acquisition-role diagnostic, 2026-11-05

`./baux tune experiments/host-regions-roles.toml` produced catalogue entry
`26b501ece8db5865b9ab8496bc8cfca9ad322f9c687908562ff0674fb6ce05c3`
and saved `.cache/baux/runs/regional/1793857752120635000/regional.json`.
This repeats the eight-dimensional, four-region, 64-evaluation host experiment
with two per-slot acquisition assignments. `uniform` uses UCB β=2 in all
four regions. `split_beta` uses β=`[0, 4, 2, 2]`: one mean-focused slot, one
uncertainty-weighted slot, and two baseline slots. This is a BAUX diagnostic
motivated by [MTRBO's separate exploitation and uncertainty regions](https://arxiv.org/abs/2605.06618),
not an implementation of that paper's algorithm.

The run contains 36 matched `Fresh`/`RetainRelevant` pairs: three seeds, two
start layouts, three sharing rules, and two acquisition assignments. Every
pair matches in its first 32 raw observations and local-model counts before
the forced restart. Every arm has exactly 64 observed values and 64 credited
regional evaluations. The 18 `uniform` pairs reproduce the earlier host
run's raw reward sequences exactly.

For each seed/layout/sharing/memory cell, the next table compares final best
observed reward for `split_beta` minus `uniform` (positive favors the split
on this objective):

| Restart memory | Matched cells | Split higher / lower / tied | Mean final-best difference |
| --- | ---: | ---: | ---: |
| Fresh | 18 | 5 / 12 / 1 | −0.056468 |
| RetainRelevant | 18 | 13 / 5 / 0 | +0.062487 |

The sign reversal is an observed interaction between this acquisition
assignment and restart memory. The 18 cells reuse three seeds across sharing
settings and are not 18 independent objective instances. This single
synthetic workload, fixed balanced allocation, fixed-length regions, and
forced restart do not select a general role or memory policy. They show why
the two policy axes need to remain separate and be compared together on
additional objectives and budgets. Runner tests passed 18; `./baux check`
passed with no violations.

### Matched restart-center diagnostic, 2026-11-05

`./baux tune experiments/host-regions-starts.toml` produced catalogue entry
`b4d3f0d21ad521f8c80f9a7cb1d6368d35c38cd23f1191a894f34d36a2877260`
and saved `.cache/baux/runs/regional/1793858168921332000/regional.json`.
It compares two algorithmic center choices at the same forced restart after
32 of 64 host evaluations. `uniform` uses the first seeded unit-cube draw.
`maximin` chooses the farthest of 64 seeded unit-cube draws from the nearest
other active region center in squared Euclidean distance. The latter is a
BAUX distance diagnostic related to [APOSMM's start separation](https://github.com/Libensemble/libensemble/blob/2f79ba6943083ff71875272d6f3f9b5222acd273/libensemble/gen_funcs/persistent_aposmm.py),
not APOSMM's full start-eligibility method.

The run has 18 matched seed/layout/sharing cells with two restart-memory
settings in each cell. Both center policies have identical observation values
and local-model counts before the restart. All 36 memory pairs use 64
evaluations per arm. In all 18 cells, maximin chooses a different center and
its measured nearest-active-center squared distance is larger than uniform's;
the respective means are `2.081646` and `0.872756`. Repeating the original
midpoint configuration reproduced all 18 earlier host reward sequences
exactly after the runner extension.

Final best reward for `maximin` minus `uniform` is mixed:

| Restart memory | Matched cells | Maximin higher / lower / tied | Mean final-best difference |
| --- | ---: | ---: | ---: |
| Fresh | 18 | 2 / 5 / 11 | −0.011777 |
| RetainRelevant | 18 | 6 / 6 / 6 | +0.007472 |

The center-choice mechanism is verified, but this is still one synthetic
objective, three seeds, correlated sharing arms, and a forced restart time.
It does not compare automatic retirement timing or establish a generally
better start rule. The runner's 18 tests and `./baux check` passed.

### Matched host landscape suite, 2026-11-05

`./baux tune experiments/host-regions-landscapes.toml` produced catalogue
entry `847a460af792bf0e0f9fec24c5d7b695e18b64b221d77142190ce4bff002af38`
and saved `.cache/baux/runs/regional/1793858832906000000/regional.json`.
The runner compares double-well, Rastrigin-form, and Ackley-form losses on
the same eight-dimensional domain `x = 3u − 1.5`, with higher negative loss
as reward. Each landscape has 72 `Fresh`/`RetainRelevant` pairs over three
seeds, two initial layouts, three sharing policies, two acquisition-role
assignments, and two restart-center choices. The forced restart remains at
32 of 64 evaluations; allocation is balanced and region length is fixed.
Every pair matches its 32 pre-restart observation values and local-model
counts. All 432 arms record 64 evaluations and finite rewards. The double-well
uniform-role slice reproduces all 36 pairs of the preceding start diagnostic
exactly. The catalogue report is complete with 432 separate observed incumbent
curves; an earlier run with identical `regional.json` failed only during
report plotting because that plotter expected the resident result schema.

The following comparisons use final best observed reward. Each cell reports
`higher / lower / tied` and the mean reward difference for the named policy
minus its comparator. Role comparisons pair the split β assignment with
uniform β within a fixed center choice. Start comparisons pair maximin with
uniform within a fixed role assignment. There are 36 comparisons per row and
policy axis; they reuse three seeds across layouts and sharing rules.

| Landscape | Memory | Split β vs uniform β | Maximin vs uniform start |
| --- | --- | --- | --- |
| Double-well | Fresh | 27 / 6 / 3; +0.066930 | 10 / 12 / 14; −0.001007 |
| Double-well | RetainRelevant | 20 / 13 / 3; +0.031702 | 13 / 13 / 10; −0.003332 |
| Rastrigin-form | Fresh | 17 / 14 / 5; +0.468768 | 9 / 15 / 12; −0.099151 |
| Rastrigin-form | RetainRelevant | 23 / 7 / 6; +0.770947 | 8 / 13 / 15; −0.172529 |
| Ackley-form | Fresh | 26 / 6 / 4; +0.418873 | 8 / 14 / 14; −0.080879 |
| Ackley-form | RetainRelevant | 26 / 5 / 5; +0.381506 | 17 / 12 / 7; +0.045372 |

These synthetic functions have different reward scales, so their mean
differences should not be pooled. The split assignment often wins in this
factorial suite, yet the earlier midpoint-only role experiment favored the
uniform assignment under `Fresh` memory. The center rule increases separation
but has mixed reward effects, as does restart memory across landscapes and
role/center combinations. These cells are correlated diagnostics, not
independent benchmark problems. They do not justify default changes or a
claim about automatic retirement timing, learned basin selection, or actual
asynchronous objective workloads. The report-path regression passed in
`baux-unit` (664 passed, five ignored); the runner's 18 tests passed.

### Automatic host retirement diagnostic, 2026-11-05

`./baux tune experiments/host-regions-auto.toml` produced catalogue entry
`dd4ca506f7f26bc8c94847e450e635f62f8bbc193132d7eed84e97439825194a`
and saved `.cache/baux/runs/regional/1793859636448239000/regional.json`.
This run uses the optimizer's actual TuRBO length controller and automatic
uniform replacement at batch settlement. The experiment records generation,
replacement center, evaluation count, and local-model counts at each event;
it does not invoke the explicit midpoint restart API. The controller uses
initial length 0.8, minimum 0.2, maximum 1.6, success tolerance 3, and
failure tolerance 2 to make retirement observable within 64 evaluations.
These are diagnostic settings, not a recommended TuRBO default. A preliminary
setting with minimum 0.5 and failure tolerance 1 caused 9–34 replacements
per arm; its run was superseded by this configuration.

There are 108 matched `Fresh`/`RetainRelevant` pairs over three synthetic
landscapes, three seeds, two initial layouts, three sharing rules, and two
acquisition-role assignments. All 216 arms have 64 observed values and a
complete catalogue incumbent curve. Every arm retires at least once:
`Fresh` records 359 replacement events (2–5 per arm), and
`RetainRelevant` records 349 (1–5 per arm). Within every pair, observed
values through the first retirement tell, model counts before that tell, and
the first retirement's evaluation count, slot, generation, and new center
match. The replacement model counts then differ as intended; for example,
one pair has `[6,6,6,6]` before tell and then `[7,7,7,0]` under `Fresh`
versus `[7,7,7,7]` under `RetainRelevant` after slot 3 retires at evaluation
28.

Final best observed reward for `RetainRelevant` minus `Fresh` remains mixed:

| Landscape | Acquisition roles | Retention higher / lower / tied | Mean difference |
| --- | --- | ---: | ---: |
| Double-well | Uniform β | 5 / 5 / 8 | −0.001920 |
| Double-well | Split β | 8 / 5 / 5 | +0.016794 |
| Rastrigin-form | Uniform β | 2 / 7 / 9 | −0.442817 |
| Rastrigin-form | Split β | 4 / 9 / 5 | −0.156041 |
| Ackley-form | Uniform β | 2 / 5 / 11 | +0.003994 |
| Ackley-form | Split β | 8 / 7 / 3 | −0.102127 |

Each row reuses three seeds across layouts and sharing rules, so its 18 cells
are correlated. The forced-restart and automatic runs also use different
controller methods and restart schedules, so their reward differences cannot
be attributed to timing alone. The automatic run verifies real retirement,
uniform replacement, and memory behavior at an equal budget; it does not
compare alternative automatic start selectors or justify a policy default.
The runner's 19 tests and `./baux check` passed.
Replaying the earlier forced-restart start configuration after adding the
automatic mode reproduced all 36 saved arm reports and pre-restart prefixes
exactly; its new evidence is
`.cache/baux/runs/regional/1793859762098122000/regional.json`.

### Automatic restart-center comparison, 2026-11-05

`./baux tune experiments/host-regions-autostart.toml` produced catalogue entry
`e06a49c5b66be87b76020e358467466a9a09e3b167f1ccbe65541737f3cdccc1`
and saved
`.cache/baux/runs/regional/1793860928566030000/regional.json`. It compares the
first of 64 seeded unit-cube draws with the maximin-ranked draw at the
optimizer's actual automatic TuRBO retirement event. Both policies draw all
64 candidates, so paired arms consume the same random stream. The controller,
64-evaluation budget, objectives, seeds, layouts, sharing rules, acquisition
roles, and restart-memory rules match the preceding automatic-retirement run.

There are 108 center-policy comparisons under each memory rule. Every pair
has identical observations and local-model counts through the first retirement
event, and the retirement evaluation, slot, and generation match. Maximin
selects a different replacement center in 106 of 108 comparisons under each
memory rule. All 432 arms have complete catalogue curves. Final-best reward for
maximin minus the first draw remains mixed:

| Landscape | Memory | Maximin higher / lower / tied | Mean difference |
| --- | --- | ---: | ---: |
| Double-well | Fresh | 5 / 7 / 24 | -0.001990 |
| Double-well | RetainRelevant | 4 / 10 / 22 | -0.010605 |
| Rastrigin-form | Fresh | 13 / 13 / 10 | -0.129427 |
| Rastrigin-form | RetainRelevant | 10 / 10 / 16 | +0.070216 |
| Ackley-form | Fresh | 13 / 4 / 19 | +0.062191 |
| Ackley-form | RetainRelevant | 12 / 6 / 18 | +0.067878 |

Replacement-count differences are also small and mixed. This run verifies
that the configured selector participates in the real retirement path while
preserving the paired pre-event trajectory. It does not establish maximin or
the first draw as a default: the cells reuse three seeds across correlated
synthetic settings, and neither selector implements APOSMM or pySOT's complete
multistart method.

The [pinned TuRBO-m implementation](https://github.com/uber-research/TuRBO/blob/de0db39f481d9505bb3610b7b7aa0ebf7702e4a5/turbo/turbo_m.py)
selects from regional posterior samples in a synchronous global competition;
BAUX's balanced and utility-UCB policies are different scheduling rules. The
[pinned APOSMM implementation](https://github.com/Libensemble/libensemble/blob/2f79ba6943083ff71875272d6f3f9b5222acd273/libensemble/gen_funcs/persistent_aposmm.py)
uses distance-based start eligibility and active-run limits for asynchronous
local optimization. The [pinned pySOT SOP implementation](https://github.com/dme65/pySOT/blob/c8f04fd4ed30d49bb61adb008134741319b512a4/pySOT/strategy/sop_strategy.py)
uses objective/diversity center ranking and per-center search scales. These
are distinct alternatives to test; this experiment implements none of their
complete allocation or start-selection methods.

### Unified host and resident lifecycle, 2026-11-05

Host `BayesOptimizer` and Metal `RegionalSearch` now use the same
backend-neutral `RegionalLedger`. The ledger owns region and generation
identity, one outstanding evaluation group per region, balanced or utility-UCB
allocation, trial moments, incumbents, completion, retirement, and restart.
It does not own a surrogate, trust-region controller, parameter arrays, Metal
buffers, or an execution engine.

The host adapter retains candidate generation, global and regional surrogate
models, numerical trust-region state, partial row completion, and the Python
frontend. The Metal adapter retains resident `SearchState` objects, buffer
leases, synchronization, and sharing queues. This is deliberately a lifecycle
seam rather than a generic optimizer trait: execution-specific state does not
cross it, while both adapters enforce the same ticket and generation rules.

Host proposal recording is transactional. A ticket is issued once per selected
regional segment; an append failure retires every newly issued ticket. Partial
row results update models as they arrive, but regional reward and controller
credit settle only when that ticket's whole group is observed or cancelled.
Restart invalidates an outstanding ticket by advancing its generation. Metal
tell validates and stages all fallible pre-mutation work before dispatch; an
error after dispatch retires the ticket and poisons that resident state so the
observation cannot be replayed.

These lifecycle rules are BAUX engineering policy. They preserve the host's
existing allocation formula and do not claim to implement the pinned TuRBO-m,
APOSMM, or pySOT policies linked above. The complete `//build/buck2/tests:all`
suite passes 33 targets, including host provenance, resident Metal, both Python
frontends, runner experiments, and the shared lifecycle contract. `./baux
check` reports no formatting, naming, or structural violations.
