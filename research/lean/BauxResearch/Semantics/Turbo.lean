import BauxResearch.Semantics.Draw

/-!
# ENN 0.4.1 post-resampling TuRBO ask

Candidate construction and complete acquisition selection are composed in one
state transition. The successor caller RNG and Sobol engine are retained on
every represented failure path.
-/

namespace BauxResearch.Semantics.Turbo

open Sampling

inductive TurboError where
  | candidate (cause : Failure)
  | random (cause : Failure)
  | select (cause : Select.SelectError)
  deriving Repr, DecidableEq

structure Step where
  result : Except TurboError Matrix
  engine : Option Sobol.Engine
  rng : RngState
  deriving Repr, DecidableEq

/-- `draw_turbo_candidates` followed by the complete `select_arms` dispatch.
MORBO weight resampling and Sobol reseeding occur immediately before this
transition and are represented by their checked modules. -/
def ask (ops : Draw.Ops) (expOp : Bounds.ExpOp)
    (backend : Neighbor.BackendOp) (ctx : Option Select.Context)
    (mode : Select.Mode) (config : Opt.CandidateCfg) (morbo : Bool)
    (center : Row) (length : Word) (scales : Option Row) (arms : Nat)
    (engine : Option Sobol.Engine) (rng : RngState) : Step :=
  let drawn := Opt.turboDraw config morbo center length scales arms engine rng
  match drawn.1 with
  | .error cause => ⟨.error (.candidate cause), drawn.2.1, drawn.2.2⟩
  | .ok candidates =>
      let selected := run (Draw.selectAll ops expOp backend ctx mode candidates arms)
        drawn.2.2
      let result := match selected.1 with
        | .error cause => .error (.random cause)
        | .ok (.error cause) => .error (.select cause)
        | .ok (.ok rows) => .ok rows
      ⟨result, drawn.2.1, selected.2⟩

theorem drawFailure (ops : Draw.Ops) (expOp : Bounds.ExpOp)
    (backend : Neighbor.BackendOp) (ctx : Option Select.Context)
    (mode : Select.Mode) (config : Opt.CandidateCfg) (morbo : Bool)
    (center : Row) (length : Word) (scales : Option Row) (arms : Nat)
    (engine nextEngine : Option Sobol.Engine) (rng nextRng : RngState)
    (cause : Failure)
    (h : Opt.turboDraw config morbo center length scales arms engine rng =
      (.error cause, nextEngine, nextRng)) :
    ask ops expOp backend ctx mode config morbo center length scales arms engine rng =
      ⟨.error (.candidate cause), nextEngine, nextRng⟩ := by
  simp [ask, h]

#print axioms drawFailure

end BauxResearch.Semantics.Turbo
