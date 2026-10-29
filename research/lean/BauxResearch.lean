import BauxResearch.Foundation
import BauxResearch.Extraction.Native
import BauxResearch.Extraction.NativeTest
import BauxResearch.Extraction.Conversion
import BauxResearch.Extraction.LocalCoord
import BauxResearch.Extraction.LocalApi
import BauxResearch.Extraction.Box
import BauxResearch.Extraction.Hash
import BauxResearch.Extraction.Seed
import BauxResearch.Generated.Optimizer.FunsExternal
import BauxResearch.Generated.Optimizer.Funs
import BauxResearch.Extraction.Optimizer
import BauxResearch.Representation.Blocks
import BauxResearch.Optimization.Step
import BauxResearch.Optimization.ObjectiveSearch
import BauxResearch.Optimization.Reachability
import BauxResearch.Optimization.Guide
import BauxResearch.Optimization.Interpolation
import BauxResearch.Optimization.PairRotation
import BauxResearch.Optimization.Perturbation
import BauxResearch.Optimization.Smooth
import BauxResearch.Optimization.Moments
import BauxResearch.Optimization.Field
import BauxResearch.Optimization.Rate
import BauxResearch.Optimization.Information
import BauxResearch.Optimization.Regret
import BauxResearch.Optimization.Synthesis
import BauxResearch.Optimization.Process
import BauxResearch.Optimization.Surrogate
import BauxResearch.Optimization.Neighbor
import BauxResearch.Optimization.Fitting
import BauxResearch.Optimization.Accuracy
import BauxResearch.Optimization.AdaptiveKnn
import BauxResearch.Optimization.Bayes
import BauxResearch.Optimization.Statistic
import BauxResearch.Optimization.Retention
import BauxResearch.Optimization.EnvelopeMemory
import BauxResearch.Optimization.SlopeEnvelope
import BauxResearch.Optimization.LocalMemory
import BauxResearch.Optimization.KnnLearning
import BauxResearch.Optimization.Calibration
import BauxResearch.Optimization.KnnDecision
import BauxResearch.Optimization.Quadrature
import BauxResearch.Optimization.SampledEnn
import BauxResearch.Optimization.RecentMemory
import BauxResearch.Optimization.PretrainRegret
import BauxResearch.Optimization.PretrainHorizon
import BauxResearch.Forward.LossGeometry
import BauxResearch.Optimization.Cost
import BauxResearch.Optimization.ContextualEnn
import BauxResearch.Optimization.Trajectory
import BauxResearch.Optimization.Reliability
import BauxResearch.Optimization.Operator
import BauxResearch.Optimization.LieOperator
import BauxResearch.Optimization.Metric
import BauxResearch.Optimization.PretrainWorkload
import BauxResearch.Optimization.PatchFeedback
import BauxResearch.Optimization.EvaluationSpace
import BauxResearch.Semantics.FbtProgram
import BauxResearch.Semantics.Optimizer
import BauxResearch.Semantics.Commit
import BauxResearch.Semantics.TurboApi
import BauxResearch.Semantics.Correspondence
import BauxResearch.Semantics.Dispatch
import BauxResearch.Semantics.Tile
import BauxResearch.Semantics.Partition
import BauxResearch.Semantics.Memory
import BauxResearch.Semantics.Geometry
import BauxResearch.Semantics.Enumeration
import BauxResearch.Semantics.Buffer
import BauxResearch.Semantics.Wave
import BauxResearch.Semantics.Numerical
import BauxResearch.Semantics.Float
import BauxResearch.Semantics.Gaussian
import BauxResearch.Semantics.Sampler
import BauxResearch.Semantics.Candidate
import BauxResearch.Semantics.Seed
import BauxResearch.Semantics.ChaCha
import BauxResearch.Semantics.Sampling
import BauxResearch.Semantics.Sobol
import BauxResearch.Semantics.Opt
import BauxResearch.Semantics.Morbo
import BauxResearch.Semantics.Surrogate
import BauxResearch.Semantics.Bounds
import BauxResearch.Semantics.Fitter
import BauxResearch.Semantics.Index
import BauxResearch.Semantics.Fit
import BauxResearch.Semantics.FitterAsk
import BauxResearch.Semantics.Posterior
import BauxResearch.Semantics.Neighbor
import BauxResearch.Semantics.Batch
import BauxResearch.Semantics.Calibration
import BauxResearch.Semantics.Compose
import BauxResearch.Semantics.Acq
import BauxResearch.Semantics.Incumbent
import BauxResearch.Semantics.Trust
import BauxResearch.Semantics.Lifecycle
import BauxResearch.Semantics.Append
import BauxResearch.Semantics.Backend
import BauxResearch.Semantics.Residency
import BauxResearch.Semantics.Morph
import BauxResearch.Semantics.Tree
import BauxResearch.Semantics.Bpann
import BauxResearch.Semantics.Flush
import BauxResearch.Semantics.Storage
import BauxResearch.Semantics.Knn
import BauxResearch.Semantics.Metric
import BauxResearch.Semantics.Predict
import BauxResearch.Semantics.Tell
import BauxResearch.Semantics.Hash
import BauxResearch.Semantics.Pareto
import BauxResearch.Semantics.Select
import BauxResearch.Semantics.Draw
import BauxResearch.Semantics.Pcg
import BauxResearch.Generated.Pure.Funs
import BauxResearch.Semantics.Reservoir
import BauxResearch.Semantics.Auto
import BauxResearch.Semantics.Turbo
import BauxResearch.Semantics.Acquisition
import BauxResearch.Semantics.PrivateRng
import BauxResearch.Semantics.Runtime
import BauxResearch.Semantics.Guard
import BauxResearch.Semantics.Config
import BauxResearch.Semantics.Registry
import BauxResearch.Semantics.Artifact
import BauxResearch.Forward.Encoder
import BauxResearch.Forward.Transport
import BauxResearch.Forward.Projective
import BauxResearch.Forward.Residual
import BauxResearch.Forward.SeedSensitivity
import BauxResearch.Forward.SinkhornRatio
import BauxResearch.Forward.MixingSensitivity
import BauxResearch.Forward.NormSensitivity
import BauxResearch.Optimization.EnnCommit
import BauxResearch.Optimization.Projection
import BauxResearch.Optimization.SeedSupport
import BauxResearch.Optimization.ActiveEnergy
import BauxResearch.Forward.Attention
import BauxResearch.Forward.Experts
import BauxResearch.Forward.Mixing
import BauxResearch.Forward.Sensitivity
import BauxResearch.Forward.Regularity
import BauxResearch.Forward.Selection
import BauxResearch.Forward.SelectionCell
import BauxResearch.Forward.Multipole
import BauxResearch.Forward.LocalSensitivity
import BauxResearch.Forward.Clipping
import BauxResearch.Forward.Decoder
import BauxResearch.Forward.Program
import BauxResearch.Forward.Parameters
import BauxResearch.Forward.Quality
import BauxResearch.Forward.Work
import BauxResearch.Objective.MultiObjective
import BauxResearch.Objective.MarginalWindow
import BauxResearch.Objective.NaturalWindow
import BauxResearch.Objective.CorpusMemory
import BauxResearch.Objective.Construction
import BauxResearch.Objective.Aggregation
import BauxResearch.Objective.Response
import BauxResearch.Objective.Concentration
import BauxResearch.Objective.Adaptive

/-!
# Mathematical reading convention

This comment applies to the imported BAUX formalization; it declares no Lean
notation.  Let `p` be the learned-parameter count, `Θₚ` the ideal real
parameter space, `θ ∈ Θₚ` a state, and `u ∈ Θₚ` a proposed update.  A descriptor
`d` belongs to its own space and is not an element of `Θₚ`.  Let `z` denote a
stored finite-precision state; the relation between `z` and `θ` requires the
particular precision contract, not an implicit identification.

For a population objective `f`, write `G_f(θ, θ′) = f(θ) - f(θ′)` for progress and
`R_f(θ; θ⋆) = f(θ) - f(θ⋆)` for simple regret relative to a proof-side
comparator.  A sampled loss, an ENN estimate, and `f` are different functions.
The symbol `θ⋆` is not information available to the optimizer.  Proposal laws,
commit rules, and cost measures must be identified in their own modules; a
shared glyph must not conceal different constructions or assumptions.

These symbols are a reading aid, not replacements for searchable Lean names.
Every mathematical abbreviation should expose, rather than erase, the
hypotheses on which a checked result depends.
-/
