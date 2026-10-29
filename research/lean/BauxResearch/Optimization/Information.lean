import BauxResearch.Optimization.ObjectiveSearch
import BauxResearch.Optimization.Regret

/-!
# Information supplied by objective evaluations

The first optimizer result is an impossibility boundary.  With no relationship
assumed between objective values at different parameter states, one observation
cannot determine which of two unqueried states is better.  Any positive
one-query result must therefore name additional structure rather than obtain it
from "Bayesian optimization" alone.

The research problem is to identify the weakest structure under which the
information available to an ENN proposal distinguishes a useful move.  The
counterexamples below concern unrestricted objectives and, in `Guide.noUniversal`,
a fixed proposal.  They do not rule out a history-conditioned proposal on a
restricted objective class.  Conversely, a positive theorem that merely assumes
the proposed state is useful has not explained where that information came from.
-/

namespace BauxResearch.Optimization.Information

/--
Two objectives can agree at the queried state while giving opposite strict
orderings to two unqueried states.
-/
theorem Query.unordered
    {State : Type}
    [DecidableEq State]
    (queried first second : State)
    (firstUnqueried : first ≠ queried)
    (secondUnqueried : second ≠ queried)
    (statesDistinct : first ≠ second) :
    ∃ objectiveA objectiveB : State → ℝ,
      objectiveA queried = objectiveB queried ∧
      objectiveA first < objectiveA second ∧
      objectiveB second < objectiveB first := by
  let objectiveA : State → ℝ := fun state =>
    if state = first then -1 else if state = second then 1 else 0
  let objectiveB : State → ℝ := fun state =>
    if state = first then 1 else if state = second then -1 else 0
  refine ⟨objectiveA, objectiveB, ?_, ?_, ?_⟩
  · simp [objectiveA, objectiveB, Ne.symm firstUnqueried, Ne.symm secondUnqueried]
  · simp [objectiveA, Ne.symm statesDistinct]
  · simp [objectiveB, Ne.symm statesDistinct]

/-- A fixed proposal cannot retain any positive fraction of regret uniformly
over unrestricted objectives when an unproposed state may contain the optimum.
This is the exact obstruction that forces either informative history or a
restricted objective class. -/
theorem Guide.noUniversal
    {State : Type} [DecidableEq State]
    (incumbent proposal best : State)
    (bestNotIncumbent : best ≠ incumbent)
    (bestNotProposal : best ≠ proposal)
    (share : ℝ) (sharePositive : 0 < share) :
    ∃ objective : Objective State,
      Regret.Optimal objective best ∧
      ¬ share * Regret.simpleRegret objective best incumbent ≤
        progress objective incumbent proposal := by
  let objective : Objective State := fun state => if state = best then -1 else 0
  refine ⟨objective, ?_, ?_⟩
  · intro state
    by_cases atBest : state = best
    · simp [objective, atBest]
    · simp [objective, atBest]
  · simp only [Regret.simpleRegret, progress]
    have incumbentNotBest : incumbent ≠ best := Ne.symm bestNotIncumbent
    have proposalNotBest : proposal ≠ best := Ne.symm bestNotProposal
    simpa [objective, incumbentNotBest, proposalNotBest] using sharePositive

/-- The boundary is a theorem, not a claim that structured objectives are hopeless. -/
def Query.boundary : ResearchClaim where
  statement :=
    "One observation cannot order two unqueried states for every unrestricted objective."
  proposition :=
    ∀ (State : Type) (_ : DecidableEq State)
      (queried first second : State),
      first ≠ queried → second ≠ queried → first ≠ second →
      ∃ objectiveA objectiveB : State → ℝ,
        objectiveA queried = objectiveB queried ∧
        objectiveA first < objectiveA second ∧
        objectiveB second < objectiveB first
  status := .proved (by
    intro State decidableEquality queried first second
      firstUnqueried secondUnqueried statesDistinct
    letI := decidableEquality
    exact Query.unordered
      queried first second firstUnqueried secondUnqueried statesDistinct)

#print axioms Guide.noUniversal

open scoped BigOperators
open ObjectiveSearch

/-- Screening cannot create useful candidates absent from the supplied family.
For arbitrary dependence between slots, the expected nonnegative value of any
selected slot is at most the sum of the marginal expected values. This also
covers posterior/acquisition randomness by including it in `World`.

For value = indicator of true improvement, this bounds improvement probability.
For value = max(0, incumbent loss - candidate loss), it bounds true accepted
progress of an exact accept/reject transaction. It is not a statement that all
history-conditioned generators have the same marginal value.
-/
theorem screenBound {World Slot : Type*} [Fintype World] [Fintype Slot]
    (joint : ProposalLaw World) (value : World → Slot → ℝ) (select : World → Slot)
    (nonnegative : ∀ world slot, 0 ≤ value world slot) :
    (∑ world, joint.weight world * value world (select world)) ≤
      ∑ slot, ∑ world, joint.weight world * value world slot := by
  classical
  calc
    _ ≤ ∑ world, joint.weight world * ∑ slot, value world slot := by
      apply Finset.sum_le_sum
      intro world _
      apply mul_le_mul_of_nonneg_left _ (joint.nonnegative world)
      exact Finset.single_le_sum (fun slot _ => nonnegative world slot) (Finset.mem_univ _)
    _ = _ := by simp_rw [Finset.mul_sum]; rw [Finset.sum_comm]

/-- If every supplied slot has marginal expected useful value at most `bound`,
even a perfect selector improves that bound by at most the number of slots.
There is no independence, surrogate-calibration, or fixed-dimensional premise.
The bound may depend on the retained history and machine budget.
-/
theorem screenCap {World Slot : Type*} [Fintype World] [Fintype Slot]
    (joint : ProposalLaw World) (value : World → Slot → ℝ) (select : World → Slot)
    (bound : ℝ) (nonnegative : ∀ world slot, 0 ≤ value world slot)
    (marginal : ∀ slot, (∑ world, joint.weight world * value world slot) ≤ bound) :
    (∑ world, joint.weight world * value world (select world)) ≤
      Fintype.card Slot * bound := by
  apply (screenBound joint value select nonnegative).trans
  calc
    _ ≤ ∑ _slot : Slot, bound := Finset.sum_le_sum fun slot _ => marginal slot
    _ = _ := by simp

/-- A necessary gain/work tradeoff for bounded screening. `cost` is a lower
charge per supplied proposal and `work` the total allowed search work. Neither
is identified with wall time without a backend account; cheap surrogate trials
remain distinct from the one selected objective evaluation.
-/
theorem screenBudget {World Slot : Type*} [Fintype World] [Fintype Slot]
    (joint : ProposalLaw World) (value : World → Slot → ℝ) (select : World → Slot)
    (bound gain cost work : ℝ) (nonnegative : ∀ world slot, 0 ≤ value world slot)
    (marginal : ∀ slot, (∑ world, joint.weight world * value world slot) ≤ bound)
    (achieves : gain ≤ ∑ world, joint.weight world * value world (select world))
    (boundNonneg : 0 ≤ bound) (costNonneg : 0 ≤ cost)
    (feasible : Fintype.card Slot * cost ≤ work) : gain * cost ≤ work * bound := by
  calc
    gain * cost ≤ (Fintype.card Slot * bound) * cost :=
      mul_le_mul_of_nonneg_right
        (achieves.trans (screenCap joint value select bound nonnegative marginal)) costNonneg
    _ = (Fintype.card Slot * cost) * bound := by ring
    _ ≤ work * bound := mul_le_mul_of_nonneg_right feasible boundNonneg

/-- Selecting among a singleton cannot change the represented candidate.
This concerns valid selection, not source invalid-score/failure behavior, nor
changes of the incumbent, trust radius, metric, or generator between rounds.
-/
theorem singletonLaw {World State : Type*}
    (candidate : World → Fin 1 → State) (select : World → Fin 1) (world : World) :
    candidate world (select world) = candidate world 0 := by
  congr 1
  exact Fin.eq_zero _

#print axioms screenBound
#print axioms screenCap
#print axioms screenBudget
#print axioms singletonLaw

end BauxResearch.Optimization.Information
