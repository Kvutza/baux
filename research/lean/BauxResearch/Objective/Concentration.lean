import BauxResearch.Objective.Aggregation
import Mathlib.Probability.Moments.SubGaussian

/-!
# Concentration of finite objective batches

Unbiasedness and concentration are different statements.  This module obtains
a one-sided empirical-mean tail bound only after adding independent,
sub-Gaussian centered-loss assumptions.
-/

namespace BauxResearch.Objective.Concentration

noncomputable section

open scoped BigOperators
open MeasureTheory ProbabilityTheory
open BauxResearch.Objective.MultiObjective
open BauxResearch.Objective.Aggregation

theorem Mean.centered
    {sampleCount objectiveCount : ℕ}
    (sampleLoss : Fin sampleCount → LossVector objectiveCount)
    (target : LossVector objectiveCount)
    (sampleCountPositive : 0 < sampleCount)
    (objective : Fin objectiveCount) :
    empiricalMean sampleLoss objective - target objective =
      (sampleCount : ℝ)⁻¹ *
        ∑ sample, (sampleLoss sample objective - target objective) := by
  unfold empiricalMean
  rw [Finset.sum_sub_distrib]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have sampleCountNonzero : (sampleCount : ℝ) ≠ 0 := by
    exact_mod_cast Nat.ne_of_gt sampleCountPositive
  field_simp

/--
One-sided concentration for one objective coordinate of a finite harness batch.
The bound requires independence and a sub-Gaussian MGF for every centered sample.
-/
theorem Mean.upperTail
    {World : Type}
    [MeasurableSpace World]
    {sampleCount objectiveCount : ℕ}
    (measure : Measure World)
    (sampleLoss : Fin sampleCount → World → LossVector objectiveCount)
    (target : LossVector objectiveCount)
    (sampleCountPositive : 0 < sampleCount)
    (objective : Fin objectiveCount)
    (scale : Fin sampleCount → NNReal)
    (independent : iIndepFun
      (fun sample world => sampleLoss sample world objective - target objective)
      measure)
    (subGaussian : ∀ sample,
      HasSubgaussianMGF
        (fun world => sampleLoss sample world objective - target objective)
        (scale sample) measure)
    {ε : ℝ}
    (epsilonNonnegative : 0 ≤ ε) :
    measure.real {world |
      ε ≤ empiricalMean (fun sample => sampleLoss sample world) objective -
        target objective} ≤
      Real.exp (-((sampleCount : ℝ) * ε) ^ 2 /
        (2 * ∑ sample, scale sample)) := by
  have sampleCountRealPositive : 0 < (sampleCount : ℝ) := by
    exact_mod_cast sampleCountPositive
  have eventEquality :
      {world |
        ε ≤ empiricalMean (fun sample => sampleLoss sample world) objective -
          target objective} =
      {world |
        (sampleCount : ℝ) * ε ≤
          ∑ sample,
            (sampleLoss sample world objective - target objective)} := by
    ext world
    simp only [Set.mem_setOf_eq]
    rw [Mean.centered
      (fun sample => sampleLoss sample world) target sampleCountPositive objective]
    exact le_inv_mul_iff₀ sampleCountRealPositive
  rw [eventEquality]
  simpa using
    (HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun
      (s := Finset.univ)
      independent
      (fun sample _ => subGaussian sample)
      (mul_nonneg sampleCountRealPositive.le epsilonNonnegative))

/-- Lower-tail companion needed by loss minimization: it bounds the event in
which an empirical loss makes a candidate look better than its population
loss.  Negation preserves both independence and the sub-Gaussian parameter. -/
theorem Mean.lowerTail
    {World : Type}
    [MeasurableSpace World]
    {sampleCount objectiveCount : ℕ}
    (measure : Measure World)
    (sampleLoss : Fin sampleCount → World → LossVector objectiveCount)
    (target : LossVector objectiveCount)
    (sampleCountPositive : 0 < sampleCount)
    (objective : Fin objectiveCount)
    (scale : Fin sampleCount → NNReal)
    (independent : iIndepFun
      (fun sample world => sampleLoss sample world objective - target objective)
      measure)
    (subGaussian : ∀ sample,
      HasSubgaussianMGF
        (fun world => sampleLoss sample world objective - target objective)
        (scale sample) measure)
    {ε : ℝ}
    (epsilonNonnegative : 0 ≤ ε) :
    measure.real {world |
      ε ≤ target objective -
        empiricalMean (fun sample => sampleLoss sample world) objective} ≤
      Real.exp (-((sampleCount : ℝ) * ε) ^ 2 /
        (2 * ∑ sample, scale sample)) := by
  let centered := fun sample world =>
    sampleLoss sample world objective - target objective
  have independentNeg : iIndepFun (fun sample world => -centered sample world) measure := by
    have composed := independent.comp
      (fun _ => fun value : ℝ => -value)
      (fun _ => (measurable_neg : Measurable fun value : ℝ => -value))
    simpa [centered, Function.comp_def] using composed
  have subGaussianNeg : ∀ sample,
      HasSubgaussianMGF (fun world => -centered sample world) (scale sample) measure :=
    fun sample => (subGaussian sample).neg
  have sampleCountRealPositive : 0 < (sampleCount : ℝ) := by
    exact_mod_cast sampleCountPositive
  have eventEquality :
      {world |
        ε ≤ target objective -
          empiricalMean (fun sample => sampleLoss sample world) objective} =
      {world |
        (sampleCount : ℝ) * ε ≤
          ∑ sample, -centered sample world} := by
    ext world
    simp only [Set.mem_setOf_eq]
    rw [show target objective -
        empiricalMean (fun sample => sampleLoss sample world) objective =
      -(empiricalMean (fun sample => sampleLoss sample world) objective -
        target objective) by ring]
    rw [Mean.centered
      (fun sample => sampleLoss sample world) target sampleCountPositive objective]
    change ε ≤ -((sampleCount : ℝ)⁻¹ * ∑ sample, centered sample world) ↔ _
    rw [← mul_neg, ← Finset.sum_neg_distrib]
    exact le_inv_mul_iff₀ sampleCountRealPositive
  rw [eventEquality]
  simpa using
    (HasSubgaussianMGF.measure_sum_ge_le_of_iIndepFun
      (s := Finset.univ)
      independentNeg
      (fun sample _ => subGaussianNeg sample)
      (mul_nonneg sampleCountRealPositive.le epsilonNonnegative))

/-- The two dangerous score errors for acceptance—overestimating and
underestimating population loss—have total probability at most twice the
one-sided bound.  This is a union bound; it does not assert that contiguous
corpus tokens satisfy the independence premise. -/
theorem Mean.errorTail
    {World : Type}
    [MeasurableSpace World]
    {sampleCount objectiveCount : ℕ}
    (measure : Measure World)
    (sampleLoss : Fin sampleCount → World → LossVector objectiveCount)
    (target : LossVector objectiveCount)
    (sampleCountPositive : 0 < sampleCount)
    (objective : Fin objectiveCount)
    (scale : Fin sampleCount → NNReal)
    (independent : iIndepFun
      (fun sample world => sampleLoss sample world objective - target objective)
      measure)
    (subGaussian : ∀ sample,
      HasSubgaussianMGF
        (fun world => sampleLoss sample world objective - target objective)
        (scale sample) measure)
    {ε : ℝ}
    (epsilonNonnegative : 0 ≤ ε) :
    measure.real
        ({world |
          ε ≤ empiricalMean (fun sample => sampleLoss sample world) objective -
            target objective} ∪
        {world |
          ε ≤ target objective -
            empiricalMean (fun sample => sampleLoss sample world) objective}) ≤
      2 * Real.exp (-((sampleCount : ℝ) * ε) ^ 2 /
        (2 * ∑ sample, scale sample)) := by
  have upper := Mean.upperTail measure sampleLoss target sampleCountPositive
    objective scale independent subGaussian epsilonNonnegative
  have lower := Mean.lowerTail measure sampleLoss target sampleCountPositive
    objective scale independent subGaussian epsilonNonnegative
  exact (measureReal_union_le _ _).trans (by linarith)

end

end BauxResearch.Objective.Concentration
