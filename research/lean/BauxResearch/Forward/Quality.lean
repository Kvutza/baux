import BauxResearch.Forward.Program
import BauxResearch.Objective.MultiObjective
import BauxResearch.Optimization.Trajectory
import BauxResearch.Optimization.Regret

/-!
The five-coordinate quality vector records failure, compilation failure,
held-out loss, runtime, and memory separately.  It is not a scalar reward.
The first two coordinates are functions of test evidence; neither certifies
general correctness of generated code.
-/

namespace BauxResearch.Forward.Quality
noncomputable section
open scoped BigOperators
open BauxResearch.Optimization

/-- Test-suite evidence, not a proof of general program correctness. -/
structure Evidence where
  compiled : Bool
  tests : Bool

def success (evidence : Evidence) : ℝ :=
  if evidence.compiled && evidence.tests then 1 else 0
def failure (evidence : Evidence) : ℝ := 1 - success evidence

theorem Success.bounded (evidence : Evidence) :
    0 ≤ success evidence ∧ success evidence ≤ 1 := by
  unfold success
  split <;> norm_num

theorem Failure.bounded (evidence : Evidence) :
    0 ≤ failure evidence ∧ failure evidence ≤ 1 := by
  have bound := Success.bounded evidence
  unfold failure
  constructor <;> linarith [bound.1, bound.2]

/-- Raw measurements stay separate; no scalarization is chosen by this record. -/
structure Measurements where
  evidence : Evidence
  heldoutLoss : ℝ
  runtime : ℝ
  memory : ℝ

def vector (measurements : Measurements) : Objective.MultiObjective.LossVector 5 :=
  ![failure measurements.evidence,
    if measurements.evidence.compiled then 0 else 1,
    measurements.heldoutLoss, measurements.runtime, measurements.memory]

def population {Output : Type*} [Fintype Output]
    (distribution : TrajectoryDist Output) (evaluator : Output → Evidence) : ℝ :=
  expectedReward distribution (fun output => failure (evaluator output))

theorem Population.stability {Output : Type*} [Fintype Output]
    (first second : TrajectoryDist Output) (evaluator : Output → Evidence) :
    |population first evaluator - population second evaluator| ≤ totalVariation first second :=
  tv_bound first second (fun output => failure (evaluator output))
    (fun output => (Failure.bounded (evaluator output)).1)
    (fun output => (Failure.bounded (evaluator output)).2)

/-- A proved objective approximation transfers terminal regret with two error terms. -/
theorem Regret.transfer {State : Type*} (objective approximation : State → ℝ)
    (best recommendation : State) (error bound : ℝ)
    (accurate : ∀ state, |objective state - approximation state| ≤ error)
    (approximateBound : Regret.simpleRegret approximation best recommendation ≤ bound) :
    Regret.simpleRegret objective best recommendation ≤ bound + 2 * error := by
  have first := (abs_le.mp (accurate recommendation)).2
  have second := (abs_le.mp (accurate best)).1
  unfold Regret.simpleRegret at *
  linarith

theorem Regret.distribution {State Output : Type*} [Fintype Output]
    (original approximation : State → TrajectoryDist Output) (evaluator : Output → Evidence)
    (best recommendation : State) (error bound : ℝ)
    (close : ∀ state, totalVariation (original state) (approximation state) ≤ error)
    (approximateBound : Regret.simpleRegret (fun state => population (approximation state) evaluator)
      best recommendation ≤ bound) :
    Regret.simpleRegret (fun state => population (original state) evaluator)
      best recommendation ≤ bound + 2 * error :=
  Regret.transfer _ _ best recommendation error bound
    (fun state => (Population.stability _ _ evaluator).trans (close state)) approximateBound

#print axioms Population.stability
#print axioms Regret.transfer
end
end BauxResearch.Forward.Quality
