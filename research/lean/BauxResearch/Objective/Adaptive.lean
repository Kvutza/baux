import BauxResearch.Objective.Concentration

/-!
# Adaptive objective batches

An optimizer may choose the next task, seed, or scoring slice from earlier
observations.  Such a batch is not independent.  Adaptivity alone also does not
preserve a population target: the next centered loss must remain conditionally
sub-Gaussian given the history used to choose it.

`AdaptiveBatch` records that contract.  Its center may depend on the past, but
the center for the next loss must already be measurable in the current history.
The checked tail bound is therefore around the recorded centers.  Identifying
those centers with one fixed population objective is a separate obligation.
-/

namespace BauxResearch.Objective.Adaptive

noncomputable section

open scoped BigOperators
open MeasureTheory ProbabilityTheory

/-- A history-aware sequence of losses and the centers they estimate. -/
structure AdaptiveBatch
    (World : Type*)
    [MeasurableSpace World]
    [StandardBorelSpace World]
    (measure : Measure World)
    [IsZeroOrProbabilityMeasure measure] where
  loss : ℕ → World → ℝ
  center : ℕ → World → ℝ
  scale : ℕ → NNReal
  history : Filtration ℕ ‹MeasurableSpace World›
  lossAdapted : StronglyAdapted history loss
  centerAdapted : StronglyAdapted history center
  centerKnown : ∀ index,
    StronglyMeasurable[history index] (center (index + 1))
  initialSubg : HasSubgaussianMGF
    (fun world => loss 0 world - center 0 world) (scale 0) measure
  nextSubg : ∀ index,
    HasCondSubgaussianMGF
      (history index)
      (history.le index)
      (fun world => loss (index + 1) world - center (index + 1) world)
      (scale (index + 1))
      measure

/-- The loss deviation from the center declared for one adaptive step. -/
def AdaptiveBatch.deviation
    {World : Type*}
    [MeasurableSpace World]
    [StandardBorelSpace World]
    {measure : Measure World}
    [IsZeroOrProbabilityMeasure measure]
    (batch : AdaptiveBatch World measure)
    (index : ℕ)
    (world : World) : ℝ :=
  batch.loss index world - batch.center index world

/-- Mean deviation of the first `sampleCount` adaptive losses. -/
def meanDeviation
    {World : Type*}
    [MeasurableSpace World]
    [StandardBorelSpace World]
    {measure : Measure World}
    [IsZeroOrProbabilityMeasure measure]
    (batch : AdaptiveBatch World measure)
    (sampleCount : ℕ)
    (world : World) : ℝ :=
  (sampleCount : ℝ)⁻¹ *
    ∑ index ∈ Finset.range sampleCount, batch.deviation index world

/-- Mean of the losses observed in the first `sampleCount` adaptive steps. -/
def meanLoss
    {World : Type*}
    [MeasurableSpace World]
    [StandardBorelSpace World]
    {measure : Measure World}
    [IsZeroOrProbabilityMeasure measure]
    (batch : AdaptiveBatch World measure)
    (sampleCount : ℕ)
    (world : World) : ℝ :=
  (sampleCount : ℝ)⁻¹ *
    ∑ index ∈ Finset.range sampleCount, batch.loss index world

/-- Mean of the centers used in the first `sampleCount` adaptive steps. -/
def meanCenter
    {World : Type*}
    [MeasurableSpace World]
    [StandardBorelSpace World]
    {measure : Measure World}
    [IsZeroOrProbabilityMeasure measure]
    (batch : AdaptiveBatch World measure)
    (sampleCount : ℕ)
    (world : World) : ℝ :=
  (sampleCount : ℝ)⁻¹ *
    ∑ index ∈ Finset.range sampleCount, batch.center index world

/-- Exact split between centered noise and adaptive-center drift. -/
theorem Mean.decompose
    {World : Type*}
    [MeasurableSpace World]
    [StandardBorelSpace World]
    {measure : Measure World}
    [IsZeroOrProbabilityMeasure measure]
    (batch : AdaptiveBatch World measure)
    (sampleCount : ℕ)
    (target : ℝ)
    (world : World) :
    meanLoss batch sampleCount world - target =
      meanDeviation batch sampleCount world +
        (meanCenter batch sampleCount world - target) := by
  unfold meanLoss meanDeviation meanCenter AdaptiveBatch.deviation
  rw [Finset.sum_sub_distrib]
  ring

/--
Adaptive upper-tail concentration.  Independence is replaced by a filtration,
strong adaptation, and a conditional sub-Gaussian obligation at every step.
-/
theorem upperTail
    {World : Type*}
    [MeasurableSpace World]
    [StandardBorelSpace World]
    (measure : Measure World)
    [IsZeroOrProbabilityMeasure measure]
    (batch : AdaptiveBatch World measure)
    (sampleCount : ℕ)
    (countPositive : 0 < sampleCount)
    {epsilon : ℝ}
    (epsilonNonnegative : 0 ≤ epsilon) :
    measure.real {world | epsilon ≤ meanDeviation batch sampleCount world} ≤
      Real.exp (-((sampleCount : ℝ) * epsilon) ^ 2 /
        (2 * ∑ index ∈ Finset.range sampleCount, batch.scale index)) := by
  have countRealPositive : 0 < (sampleCount : ℝ) := by
    exact_mod_cast countPositive
  have adapted : StronglyAdapted batch.history batch.deviation := by
    intro index
    exact (batch.lossAdapted index).sub (batch.centerAdapted index)
  have eventEquality :
      {world | epsilon ≤ meanDeviation batch sampleCount world} =
      {world |
        (sampleCount : ℝ) * epsilon ≤
          ∑ index ∈ Finset.range sampleCount, batch.deviation index world} := by
    ext world
    simp only [Set.mem_setOf_eq]
    unfold meanDeviation
    exact le_inv_mul_iff₀ countRealPositive
  rw [eventEquality]
  exact measure_sum_ge_le_of_hasCondSubgaussianMGF
    adapted
    batch.initialSubg
    sampleCount
    (fun index _ => batch.nextSubg index)
    (mul_nonneg countRealPositive.le epsilonNonnegative)

/-- Adaptive lower-tail concentration.  This is the dangerous direction when
an observed candidate loss can make that candidate look better than its
conditional center. -/
theorem lowerTail
    {World : Type*}
    [MeasurableSpace World]
    [StandardBorelSpace World]
    (measure : Measure World)
    [IsZeroOrProbabilityMeasure measure]
    (batch : AdaptiveBatch World measure)
    (sampleCount : ℕ)
    (countPositive : 0 < sampleCount)
    {epsilon : ℝ}
    (epsilonNonnegative : 0 ≤ epsilon) :
    measure.real {world | epsilon ≤ -meanDeviation batch sampleCount world} ≤
      Real.exp (-((sampleCount : ℝ) * epsilon) ^ 2 /
        (2 * ∑ index ∈ Finset.range sampleCount, batch.scale index)) := by
  have countRealPositive : 0 < (sampleCount : ℝ) := by
    exact_mod_cast countPositive
  let negDeviation := fun index world => -batch.deviation index world
  have adapted : StronglyAdapted batch.history negDeviation := by
    intro index
    exact (batch.lossAdapted index).sub (batch.centerAdapted index) |>.neg
  have initial : HasSubgaussianMGF
      (negDeviation 0) (batch.scale 0) measure := by
    convert batch.initialSubg.neg using 1
    funext world
    change -(batch.loss 0 world - batch.center 0 world) =
      -(batch.loss 0 world - batch.center 0 world)
    rfl
  have next : ∀ index,
      HasCondSubgaussianMGF
        (batch.history index)
        (batch.history.le index)
        (negDeviation (index + 1))
        (batch.scale (index + 1))
        measure := by
    intro index
    change Kernel.HasSubgaussianMGF
      (negDeviation (index + 1))
      (batch.scale (index + 1))
      (condExpKernel measure (batch.history index))
      (measure.trim (batch.history.le index))
    convert (batch.nextSubg index).neg using 1
    funext world
    change -(batch.loss (index + 1) world - batch.center (index + 1) world) =
      -(batch.loss (index + 1) world - batch.center (index + 1) world)
    rfl
  have eventEquality :
      {world | epsilon ≤ -meanDeviation batch sampleCount world} =
      {world |
        (sampleCount : ℝ) * epsilon ≤
          ∑ index ∈ Finset.range sampleCount, negDeviation index world} := by
    ext world
    simp only [Set.mem_setOf_eq]
    unfold meanDeviation
    rw [← mul_neg, ← Finset.sum_neg_distrib]
    exact le_inv_mul_iff₀ countRealPositive
  rw [eventEquality]
  exact measure_sum_ge_le_of_hasCondSubgaussianMGF
    adapted
    initial
    sampleCount
    (fun index _ => next index)
    (mul_nonneg countRealPositive.le epsilonNonnegative)

/-- Both directions of adaptive mean error are controlled by a union bound.
The statement remains centered on the history-dependent conditional centers;
it does not identify them with one population objective. -/
theorem errorTail
    {World : Type*}
    [MeasurableSpace World]
    [StandardBorelSpace World]
    (measure : Measure World)
    [IsZeroOrProbabilityMeasure measure]
    (batch : AdaptiveBatch World measure)
    (sampleCount : ℕ)
    (countPositive : 0 < sampleCount)
    {epsilon : ℝ}
    (epsilonNonnegative : 0 ≤ epsilon) :
    measure.real
        ({world | epsilon ≤ meanDeviation batch sampleCount world} ∪
          {world | epsilon ≤ -meanDeviation batch sampleCount world}) ≤
      2 * Real.exp (-((sampleCount : ℝ) * epsilon) ^ 2 /
        (2 * ∑ index ∈ Finset.range sampleCount, batch.scale index)) := by
  have upper := upperTail measure batch sampleCount countPositive epsilonNonnegative
  have lower := lowerTail measure batch sampleCount countPositive epsilonNonnegative
  exact (measureReal_union_le _ _).trans (by linarith)

/--
A tail bound around one fixed target requires a separate deterministic bound on
adaptive-center drift.  The drift penalty appears explicitly in the threshold.
-/
theorem fixedTail
    {World : Type*}
    [MeasurableSpace World]
    [StandardBorelSpace World]
    (measure : Measure World)
    [IsZeroOrProbabilityMeasure measure]
    (batch : AdaptiveBatch World measure)
    (sampleCount : ℕ)
    (countPositive : 0 < sampleCount)
    (target bias : ℝ)
    (centerBound : ∀ world,
      meanCenter batch sampleCount world - target ≤ bias)
    {epsilon : ℝ}
    (epsilonNonnegative : 0 ≤ epsilon) :
    measure.real {world |
      epsilon + bias ≤ meanLoss batch sampleCount world - target} ≤
      Real.exp (-((sampleCount : ℝ) * epsilon) ^ 2 /
        (2 * ∑ index ∈ Finset.range sampleCount, batch.scale index)) := by
  calc
    measure.real {world |
        epsilon + bias ≤ meanLoss batch sampleCount world - target} ≤
      measure.real {world | epsilon ≤ meanDeviation batch sampleCount world} := by
        refine measureReal_mono ?_ (by finiteness)
        intro world lossHigh
        simp only [Set.mem_setOf_eq] at lossHigh ⊢
        rw [Mean.decompose batch sampleCount target world] at lossHigh
        linarith [centerBound world]
    _ ≤ Real.exp (-((sampleCount : ℝ) * epsilon) ^ 2 /
        (2 * ∑ index ∈ Finset.range sampleCount, batch.scale index)) :=
      upperTail measure batch sampleCount countPositive epsilonNonnegative

#print axioms upperTail
#print axioms lowerTail
#print axioms errorTail
#print axioms fixedTail

end

end BauxResearch.Objective.Adaptive
