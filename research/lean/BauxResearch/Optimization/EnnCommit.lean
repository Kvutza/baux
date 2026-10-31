import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Linarith

/-!
# Guarded scalar acceptance and commit

The deployed pretraining reward is negative NLL. The scalar decision accepts
when the candidate reward gain strictly exceeds the declared margin.
-/

namespace BauxResearch.Optimization.EnnCommit

noncomputable section

/-- The scalar acceptance law: measured reward gain must strictly clear the
declared error margin. -/
abbrev accepts (improvement margin : ℝ) : Prop :=
  margin < improvement

/-- The deployed scalar reward comparison, with its strict error margin. -/
def scalarCommit {State : Type*} (incumbent candidate : State)
    (baseReward candidateReward margin : ℝ) : State :=
  if accepts (candidateReward - baseReward) margin then candidate else incumbent

/-- Guarded exact-evaluation commit: accepts iff `f(candidate) < f(incumbent) - margin`. -/
def objectiveCommit {State : Type*} (f : State → ℝ) (incumbent candidate : State)
    (margin : ℝ) : State :=
  scalarCommit incumbent candidate (-f incumbent) (-f candidate) margin

/-- Under non-negative margin `0 ≤ margin`, `objectiveCommit` never increases `f`. -/
theorem objectiveCommit_le {State : Type*} (f : State → ℝ) (incumbent candidate : State)
    (margin : ℝ) (hMargin : 0 ≤ margin) :
    f (objectiveCommit f incumbent candidate margin) ≤ f incumbent := by
  unfold objectiveCommit scalarCommit accepts
  split_ifs with h
  · linarith
  · exact le_refl _

end
end BauxResearch.Optimization.EnnCommit
