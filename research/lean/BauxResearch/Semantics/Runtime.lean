import BauxResearch.Semantics.Optimizer
import BauxResearch.Semantics.PrivateRng

/-!
# Concrete ENN 0.4.1 optimizer operation wiring

This module gives the public optimizer one consistent operation record.  The
native transcendental functions, neighbour backend, and effect gate remain
explicit typed artifact boundaries; the Lean transition itself no longer has
an arbitrary private-RNG prefix length.
-/

namespace BauxResearch.Semantics.Runtime

/-- Native and effectful dependencies that must be related to the compiled
artifact.  Supplying this record does not assert their correctness. -/
structure Artifacts where
  draw : Draw.Ops
  exp : Bounds.ExpOp
  log : Bounds.LogOp
  pow : Fitter.PowOp
  sqrt : Fitter.SqrtOp
  powi : Fit.PowiOp
  backend : Neighbor.BackendOp
  gate : Append.Gate

/-- Total private-stream operations used by surrogate fitting. -/
def privateOps (a : Artifacts) : PrivateRng.Ops :=
  { pow := a.pow
    sqrt := a.sqrt
    log := a.log
    powi := a.powi
    backend := a.backend
    gate := a.gate }

/-- Public ask dependencies, sharing the same backend and native operations. -/
def lifecycleOps (a : Artifacts) : Lifecycle.Ops :=
  { draw := a.draw
    exp := a.exp
    log := a.log
    backend := a.backend }

/-- Public tell dependencies with the total private ChaCha12 stream. -/
def tellOps (a : Artifacts) : Tell.Ops :=
  { append := privateOps a
    exp := a.exp }

/-- One operation bundle for mixed public ask/tell histories. -/
def optimizerOps (a : Artifacts) : Optimizer.Ops :=
  { ask := lifecycleOps a
    tell := tellOps a }

theorem askProjection (a : Artifacts) :
    (optimizerOps a).ask = lifecycleOps a := by
  rfl

theorem tellProjection (a : Artifacts) :
    (optimizerOps a).tell = tellOps a := by
  rfl

theorem privateProjection (a : Artifacts) :
    (optimizerOps a).tell.append = privateOps a := by
  rfl

#print axioms askProjection
#print axioms tellProjection
#print axioms privateProjection

end BauxResearch.Semantics.Runtime
