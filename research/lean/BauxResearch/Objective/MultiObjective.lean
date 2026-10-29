import BauxResearch.Foundation

/-!
# Multi-objective constructions

Every objective is represented as a loss, so lower is better.  A reward becomes
a loss by negation.  This avoids silently mixing minimization and maximization
inside Pareto and scalarization statements.

The definitions are independent of a surrogate or optimizer.  They can be used
for pre-training loss, coding-harness outcomes, latency, memory, safety
constraints, or any later measurement, provided the conversion to loss is
stated explicitly.
-/

namespace BauxResearch.Objective.MultiObjective

open scoped BigOperators

/-- One real loss for each of `objectiveCount` objectives. -/
abbrev LossVector (objectiveCount : ℕ) := Fin objectiveCount → ℝ

/-- `candidate` is no worse than `reference` in every objective. -/
def WeaklyDominates
    {objectiveCount : ℕ}
    (candidate reference : LossVector objectiveCount) : Prop :=
  ∀ objective, candidate objective ≤ reference objective

/-- Pareto dominance additionally requires strict improvement somewhere. -/
def StrictlyDominates
    {objectiveCount : ℕ}
    (candidate reference : LossVector objectiveCount) : Prop :=
  WeaklyDominates candidate reference ∧
    ∃ objective, candidate objective < reference objective

/-- A candidate is Pareto-optimal within an explicitly supplied feasible set. -/
def ParetoOptimal
    {objectiveCount : ℕ}
    (feasible : Set (LossVector objectiveCount))
    (candidate : LossVector objectiveCount) : Prop :=
  candidate ∈ feasible ∧
    ¬∃ alternative ∈ feasible, StrictlyDominates alternative candidate

/-- A constraint is feasible when its signed violation is nonpositive. -/
def IsFeasible
    {constraintCount : ℕ}
    (violation : Fin constraintCount → ℝ) : Prop :=
  ∀ constraint, violation constraint ≤ 0

/-- A nonnegative weighted scalarization of a vector loss. -/
def weightedLoss
    {objectiveCount : ℕ}
    (weights : Fin objectiveCount → NNReal)
    (losses : LossVector objectiveCount) : ℝ :=
  ∑ objective, (weights objective : ℝ) * losses objective

/-- Nonnegative scalarization cannot reverse a Pareto weak dominance. -/
theorem Weighted.monotone
    {objectiveCount : ℕ}
    (weights : Fin objectiveCount → NNReal)
    (candidate reference : LossVector objectiveCount)
    (dominates : WeaklyDominates candidate reference) :
    weightedLoss weights candidate ≤ weightedLoss weights reference := by
  unfold weightedLoss
  apply Finset.sum_le_sum
  intro objective _
  exact mul_le_mul_of_nonneg_left
    (dominates objective) (weights objective).property

/--
A lower weighted loss does not imply Pareto dominance.  The two-objective
counterexample trades the first loss against the second.
-/
theorem Weighted.notPareto :
    ∃ (weights : Fin 2 → NNReal) (candidate reference : LossVector 2),
      weightedLoss weights candidate < weightedLoss weights reference ∧
      ¬WeaklyDominates candidate reference := by
  let weights : Fin 2 → NNReal := fun _ => 1
  let candidate : LossVector 2 := fun objective =>
    if objective = 0 then 0 else 2
  let reference : LossVector 2 := fun objective =>
    if objective = 0 then 3 else 0
  refine ⟨weights, candidate, reference, ?_, ?_⟩
  · norm_num [weightedLoss, weights, candidate, reference, Fin.sum_univ_two]
  · intro dominates
    have secondObjective := dominates (1 : Fin 2)
    norm_num [candidate, reference] at secondObjective

/-- Pareto dominance is transitive. -/
theorem WeaklyDominates.trans
    {objectiveCount : ℕ}
    {first second third : LossVector objectiveCount}
    (firstDominatesSecond : WeaklyDominates first second)
    (secondDominatesThird : WeaklyDominates second third) :
    WeaklyDominates first third := by
  intro objective
  exact le_trans
    (firstDominatesSecond objective)
    (secondDominatesThird objective)

/-- A candidate minimizing a strictly positive weighted loss is Pareto optimal.
This establishes that strictly positive scalarization (such as UCB with beta > 0)
always selects an element from the non-dominated Pareto frontier. -/
theorem Weighted.paretoOptimal
    {objectiveCount : ℕ}
    (weights : Fin objectiveCount → NNReal)
    (hPos : ∀ i, 0 < weights i)
    (feasible : Set (LossVector objectiveCount))
    (candidate : LossVector objectiveCount)
    (hMem : candidate ∈ feasible)
    (hMin : ∀ alt ∈ feasible, weightedLoss weights candidate ≤ weightedLoss weights alt) :
    ParetoOptimal feasible candidate := by
  refine ⟨hMem, ?_⟩
  rintro ⟨alt, hAltMem, hDom⟩
  rcases hDom with ⟨hWeak, j, hStrict⟩
  have hDiff : 0 < weightedLoss weights candidate - weightedLoss weights alt := by
    unfold weightedLoss
    rw [← Finset.sum_sub_distrib]
    have hNonneg : ∀ i ∈ Finset.univ, 0 ≤ (weights i : ℝ) * candidate i - (weights i : ℝ) * alt i := by
      intro i _
      rw [← mul_sub]
      have hW : 0 ≤ (weights i : ℝ) := (weights i).property
      have hSub : 0 ≤ candidate i - alt i := sub_nonneg.mpr (hWeak i)
      exact mul_nonneg hW hSub
    have hStrictTerm : 0 < (weights j : ℝ) * candidate j - (weights j : ℝ) * alt j := by
      rw [← mul_sub]
      have hW : 0 < (weights j : ℝ) := hPos j
      have hSub : 0 < candidate j - alt j := sub_pos.mpr hStrict
      exact mul_pos hW hSub
    have hLe := Finset.single_le_sum hNonneg (Finset.mem_univ j)
    exact lt_of_lt_of_le hStrictTerm hLe
  have hAltMin := hMin alt hAltMem
  linarith

end BauxResearch.Objective.MultiObjective
