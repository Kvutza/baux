# Website purpose

This is a library website. Help readers understand what baux does, run a first
example, and find the feature they need. The homepage contains the introduction,
one working example and links organized by task. Technical explanations belong
on linked guides; experiment details belong with the examples they explain.
The homepage does not need an argument, derivations or a bibliography.

| Section | Question | Required substance |
| --- | --- | --- |
| Overview | What goes in and comes out? | Domain, values, maximization convention, evaluation loop. |
| Regions and state | What changes after an evaluation? | Proposals, updates, sharing, pending work, replay conditions. |
| Structured data | How does my problem enter the solver? | Encoders, decoders, edge cases, vector outputs, shrinkage assumptions. |
| ENN | How do observations guide proposals? | Weights, uncertainty, acquisition, additional sampling construction. |
| Examples | What can I try? | Interactive controls and callable examples, with experiment details available on demand. |
| API | How do I call it correctly? | Signatures, shapes, state changes, errors and examples. |

Admit implementation claims from inspected code and tests, literature claims
from primary sources, and mathematical claims with their assumptions. Report
experiments with methods. A non-significant difference is not equivalence;
changing two factors cannot isolate either factor. Execution traces are not
scholarly citations.

## Findings behind this revision

- The functional API rejects stale evaluation IDs. It does not always ingest
  them while suppressing regional updates.
- Direct Stiefel construction does not orthonormalize the matrix. The example
  needs from_factors. Its adaptive contraction is not the positive-part
  James–Stein estimator.
- Snapshot hashing provides integrity, not sender authentication.
- ENN joint draws are a baux construction; the paper explicitly says it lacks
  joint sampling. Matching marginal variance does not establish calibration.
- The saved TCV report supports the final means, SEMs and paired tests. The
  comparison changes both surrogate and region count.
- Matching evidence for the former TORAX results and OxCaml throughput figures
  was not located in the inspected saved runs. Their numerical claims were
  removed from the website, including the standalone throughput figure page.
- The browser animates schematic coil schedules and shapes; these are not
  measured physical trajectories.

Editorial structure follows Gopen and Swan, [The Science of Scientific
Writing](https://cseweb.ucsd.edu/~swanson/papers/science-of-writing.pdf): establish
context before new information and make the subject perform the action.

## Inspection and verification

Implementation sources: python/baux/functional.py, python/baux/_embeddings.py,
rust/crates/baux/src/strategy/multi.rs,
rust/crates/baux/src/posterior/draw_compute.rs, and
rust/crates/baux/src/snapshot_io.rs. The TCV figures were checked against
.cache/baux/runs/incremental/tokamak-tokagym-1794598817598394000/report.json
and its configuration. These are internal inspection records, not website
citations.

The site build, baux check, citation-key check, homepage and API examples,
synthetic security cases, and static security audit passed. Examples used the
existing native wheel in .cache/baux/wheel-site. The simulation example also
passed with BAUX_OXCAML_DIR pointing at .cache/baux/ocaml-pyext. Its first run
failed because the wheel's default relative lookup could not find that optional
extension; no optimizer or extension implementation was changed.

## Documentation structure

The writing reference is [tinygrad's documentation](https://docs.tinygrad.org/)
and its [API pages](https://docs.tinygrad.org/tensor/creation/): direct explanations,
runnable examples, and useful details beside the calls they describe. Its theme
and typography are not the reference. Revision 4a879bb7 records the visual
baseline: Latin Modern guide prose, bold Inter headings, Inter API prose, IBM
Plex Mono code, a masthead with small-cap navigation links, centered 820px guides
and a 1120px API layout. The baux wordmark stays lowercase. Search and the contents
menu sit below the restored masthead; API pages retain their contents sidebar.

The first attempted restoration only changed font and color variables. It left
the replacement masthead and guide layout in place and incorrectly made every
heading serif. Its successful deployment did not establish a visual restoration.

The homepage starts with installation and a working example. Guides explain
tasks; eight API pages cover the 85 documented symbols. Each definition starts
with its typed signature, followed by behavior, contracts, examples and a source
link. Search covers every page and symbol. Old symbol URLs redirect to the
current definition. Use lowercase baux in reader-facing text.

The source audit corrected several misleading contracts: the default optimizer
has no surrogate; functional tell submits coordinates rather than validating an
AskBatch's tickets; regional sharing has no global option; DistanceMetric has
only SQUARED_L2 and COSINE; NeighborIndex has no EXACT or HNSW enum aliases;
geometry and output transforms applied to OptimizerState initialize a fresh
optimizer. The two ENN setup helpers also have different default acquisitions.

Reference checks compare documented function argument names, annotations,
return types, keyword-only arguments and literal defaults against the current
Python imports. They also check symbol anchors and execute all reference
examples. Browser checks cover cross-page search, navigation, visible examples
and responsive layout. These checks establish specific behavior;
passing them does not establish that the writing or navigation is useful.
