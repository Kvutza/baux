import BauxResearch.Optimization.Step
import Mathlib.Data.Real.Basic
import Mathlib.Topology.MetricSpace.Basic

/-!
# Operator perturbation theory and reached-state reduction

This module formalizes the reached-state reduction: only acted-on input
directions are visible locally.

In high-dimensional transformer optimization, parameters act through operators
`W : Input → Output`.  At any round, the model visits a finite set of reached
inputs `h_1, ..., h_N`.

A perturbation `ΔW` decomposes into:
1. Acted-on components that alter outputs `(ΔW) h_j`.
2. Null components in the kernel of reached states with `(ΔW) h_j = 0`.

We prove:
1. Any perturbation in the kernel of reached states leaves all reached outputs unchanged.
2. Downstream objective functions depending on reached outputs are invariant under null perturbations.
3. The opportunity and progress of null perturbations are identically zero, proving that
   Euclidean parameter distance is misleading unless quotiented by reached states.
-/

namespace BauxResearch.Optimization

variable {Input Output : Type*}

/-- An objective depending on an operator only through its outputs on reached states. -/
def reachedObjective
    {ι : Type*}
    (inputs : ι → Input)
    (evaluator : (ι → Output) → ℝ)
    (operator : Input → Output) : ℝ :=
  evaluator (fun i ↦ operator (inputs i))

/--
Null perturbation invariance: If a perturbation leaves every reached input unchanged,
the downstream objective value is identically preserved.
-/
theorem Operator.nullInvariant
    {ι : Type*}
    (inputs : ι → Input)
    (evaluator : (ι → Output) → ℝ)
    (base perturbed : Input → Output)
    (hNull : ∀ i, perturbed (inputs i) = base (inputs i)) :
    reachedObjective inputs evaluator perturbed =
      reachedObjective inputs evaluator base := by
  unfold reachedObjective
  have hEq : (fun i ↦ perturbed (inputs i)) = (fun i ↦ base (inputs i)) := by
    funext i
    exact hNull i
  rw [hEq]

/--
Zero opportunity theorem: A perturbation in the null space of reached states
provides identically zero opportunity, regardless of its Euclidean parameter norm.
-/
theorem Operator.zeroOpportunity
    {ι : Type*}
    (inputs : ι → Input)
    (evaluator : (ι → Output) → ℝ)
    (base perturbed : Input → Output)
    (hNull : ∀ i, perturbed (inputs i) = base (inputs i)) :
    opportunity (reachedObjective inputs evaluator) base perturbed = 0 := by
  unfold opportunity
  have hEq := Operator.nullInvariant inputs evaluator base perturbed hNull
  linarith

/--
Zero progress theorem: A candidate perturbation in the null space of reached states
yields identically zero progress on the reached task evaluation.
-/
theorem Operator.zeroProgress
    {ι : Type*}
    (inputs : ι → Input)
    (evaluator : (ι → Output) → ℝ)
    (base perturbed : Input → Output)
    (hNull : ∀ i, perturbed (inputs i) = base (inputs i)) :
    progress (reachedObjective inputs evaluator) base perturbed = 0 := by
  unfold progress
  have hEq := Operator.nullInvariant inputs evaluator base perturbed hNull
  linarith

/--
Unmeasured operator bound: If an operator perturbation has bounded effect
`dist (perturbed inputs) (base inputs) ≤ opBound * inputBound`, and downstream evaluation is Lipschitz,
the objective shift is bounded analytically without in-flight activation measurement.
-/
theorem Operator.effectBound
    {Input Output : Type*}
    [PseudoMetricSpace Output]
    (inputs : Input)
    (evaluator : Output → ℝ)
    (base perturbed : Input → Output)
    (K opBound inputBound : ℝ)
    (hK : 0 ≤ K)
    (hLip : ∀ y1 y2, |evaluator y1 - evaluator y2| ≤ K * dist y1 y2)
    (hDist : dist (perturbed inputs) (base inputs) ≤ opBound * inputBound) :
    |evaluator (perturbed inputs) - evaluator (base inputs)| ≤
      K * opBound * inputBound := by
  have hStep := hLip (perturbed inputs) (base inputs)
  have hMul : K * dist (perturbed inputs) (base inputs) ≤ K * (opBound * inputBound) :=
    mul_le_mul_of_nonneg_left hDist hK
  linarith

end BauxResearch.Optimization
