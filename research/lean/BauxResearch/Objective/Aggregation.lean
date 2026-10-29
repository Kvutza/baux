import BauxResearch.Objective.Construction
import Mathlib.MeasureTheory.Integral.Bochner.Basic

/-!
# Finite-batch and population aggregation

This module distinguishes an outer optimizer query from the work performed
inside that query.  A finite harness batch may contain many task/randomness
pairs.  Its empirical mean is different from the corresponding population
expectation.
-/

/-!
For a finite batch of size `n`, the empirical component is
`L̂ⱼ = (1/n) ∑ᵢ ℓᵢⱼ`; for a specified measure `μ`, the population component
is `Lⱼ = ∫ ℓⱼ dμ`.  Equality of their expectations is not equality of the
random empirical value and the population value.  Neither notation makes the
internal task evaluations free in an outer optimizer query.
-/

namespace BauxResearch.Objective.Aggregation

noncomputable section

open scoped BigOperators
open MeasureTheory
open BauxResearch.Objective.MultiObjective
open BauxResearch.Objective.Construction

/-- The componentwise empirical mean of a finite vector-valued batch. -/
def empiricalMean
    {sampleCount objectiveCount : ℕ}
    (values : Fin sampleCount → LossVector objectiveCount) :
    LossVector objectiveCount :=
  fun objective =>
    (sampleCount : ℝ)⁻¹ * ∑ sample, values sample objective

/-- The concrete task/randomness pairs evaluated inside one outer query. -/
structure EvaluationBatch
    (Task Randomness : Type)
    (sampleCount : ℕ) where
  task : Fin sampleCount → Task
  randomness : Fin sampleCount → Randomness

/-- Finite empirical-mean aggregation for one explicit evaluation batch. -/
def Empirical.aggregation
    {Task Randomness : Type}
    {sampleCount objectiveCount : ℕ}
    (batch : EvaluationBatch Task Randomness sampleCount) :
    AggregationRule Task Randomness objectiveCount where
  name := "finite empirical mean"
  aggregate := fun losses =>
    empiricalMean fun sample =>
      losses (batch.task sample) (batch.randomness sample)

/-- Componentwise population expectation over task/randomness pairs. -/
def populationMean
    {World : Type}
    [MeasurableSpace World]
    {objectiveCount : ℕ}
    (measure : Measure World)
    (losses : World → LossVector objectiveCount) : LossVector objectiveCount :=
  fun objective => ∫ world, losses world objective ∂measure

/-- Population-mean aggregation for a joint task/randomness measure. -/
def Population.aggregation
    {Task Randomness : Type}
    [MeasurableSpace Task]
    [MeasurableSpace Randomness]
    {objectiveCount : ℕ}
    (measure : Measure (Task × Randomness)) :
    AggregationRule Task Randomness objectiveCount where
  name := "population expectation"
  aggregate := fun losses =>
    populationMean measure fun world => losses world.1 world.2

/--
An empirical mean is unbiased when every sample coordinate is integrable and
has the target expectation.  Independence is not needed for unbiasedness; it
will be needed by later concentration results.
-/
theorem Empirical.unbiased
    {World : Type}
    [MeasurableSpace World]
    {sampleCount objectiveCount : ℕ}
    (measure : Measure World)
    (sampleLoss : Fin sampleCount → World → LossVector objectiveCount)
    (target : LossVector objectiveCount)
    (sampleCountPositive : 0 < sampleCount)
    (integrable : ∀ sample objective,
      Integrable (fun world => sampleLoss sample world objective) measure)
    (sameExpectation : ∀ sample objective,
      (∫ world, sampleLoss sample world objective ∂measure) = target objective) :
    ∀ objective,
      (∫ world,
        empiricalMean (fun sample => sampleLoss sample world) objective
          ∂measure) = target objective := by
  intro objective
  unfold empiricalMean
  rw [integral_const_mul]
  have sumIntegral := integral_finsetSum Finset.univ
    (fun sample _ => integrable sample objective)
  rw [sumIntegral]
  simp_rw [sameExpectation]
  simp [Nat.ne_of_gt sampleCountPositive]

/--
Unbiasedness is coordinatewise and therefore applies to any finite number of
objectives without scalarizing them first.
-/
def Empirical.claim
    {World : Type}
    [MeasurableSpace World]
    {sampleCount objectiveCount : ℕ}
    (measure : Measure World)
    (sampleLoss : Fin sampleCount → World → LossVector objectiveCount)
    (target : LossVector objectiveCount)
    (sampleCountPositive : 0 < sampleCount)
    (integrable : ∀ sample objective,
      Integrable (fun world => sampleLoss sample world objective) measure)
    (sameExpectation : ∀ sample objective,
      (∫ world, sampleLoss sample world objective ∂measure) = target objective) :
    ResearchClaim where
  statement :=
    "The finite-batch empirical loss vector is an unbiased population-loss estimator."
  proposition := ∀ objective,
    (∫ world,
      empiricalMean (fun sample => sampleLoss sample world) objective
        ∂measure) = target objective
  status := .proved (Empirical.unbiased measure sampleLoss target
    sampleCountPositive integrable sameExpectation)

end

end BauxResearch.Objective.Aggregation
