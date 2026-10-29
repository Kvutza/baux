import BauxResearch.Objective.Aggregation
import Mathlib.Probability.Distributions.Gaussian.Real

/-!
# Paired responses and dependent evaluation patches

Real-arithmetic views of the patch mean/scatter in `backends/cuda/src/model/run.rs`
and baseline-plus-improvement observations in both runners. These are not
floating-point refinement proofs. Mathlib's variance is population variance;
`patchVariance` is the statistic computed from one realized batch.
-/

namespace BauxResearch.Objective.Response

noncomputable section

open scoped BigOperators
open MeasureTheory ProbabilityTheory
open BauxResearch.Objective.Aggregation

/-- Scalar instance of the existing componentwise evaluation mean. -/
def patchMean {count : ℕ} (values : Fin count → ℝ) : ℝ :=
  empiricalMean (fun index => fun _ : Fin 1 => values index) 0

/-- Sample scatter divided by batch size, not a population variance. -/
def patchVariance {count : ℕ} (values : Fin count → ℝ) : ℝ :=
  (∑ index, (values index - patchMean values) ^ 2) /
    (((count - 1) * count : ℕ) : ℝ)

theorem Mean.repeated {count : ℕ} (positive : 0 < count) (value : ℝ) :
    patchMean (fun _ : Fin count => value) = value := by
  simp [patchMean, empiricalMean, Nat.ne_of_gt positive]

theorem Scatter.repeated {count : ℕ} (positive : 0 < count) (value : ℝ) :
    patchVariance (fun _ : Fin count => value) = 0 := by
  simp [patchVariance, Mean.repeated positive value]

/-- The full covariance matrix, not just its diagonal, controls batch variance. -/
theorem Weighted.variance
    {World Index : Type*} [MeasurableSpace World] [Fintype Index]
    (measure : Measure World) [IsFiniteMeasure measure]
    (values : Index → World → ℝ) (weights : Index → ℝ)
    (finiteMoment : ∀ index, MemLp (values index) 2 measure) :
    Var[fun world => ∑ index, weights index * values index world; measure] =
      ∑ index, ∑ other,
        weights index * weights other * cov[values index, values other; measure] := by
  rw [variance_fun_sum (fun index => (finiteMoment index).const_mul (weights index))]
  simp_rw [covariance_const_mul_left, covariance_const_mul_right]
  congr 1
  funext index
  congr 1
  funext other
  ring

/-- Squared-weight noise is justified by zero off-diagonal covariances. -/
theorem Weighted.diagonal
    {World Index : Type*} [MeasurableSpace World] [Fintype Index] [DecidableEq Index]
    (measure : Measure World) [IsFiniteMeasure measure]
    (values : Index → World → ℝ) (weights : Index → ℝ)
    (finiteMoment : ∀ index, MemLp (values index) 2 measure)
    (uncorrelated : ∀ index other, index ≠ other →
      cov[values index, values other; measure] = 0) :
    Var[fun world => ∑ index, weights index * values index world; measure] =
      ∑ index, weights index ^ 2 * Var[values index; measure] := by
  rw [Weighted.variance measure values weights finiteMoment]
  apply Finset.sum_congr rfl
  intro index _
  rw [Finset.sum_eq_single index]
  · rw [covariance_self (finiteMoment index).aemeasurable]
    ring
  · intro other _ different
    rw [uncorrelated index other (Ne.symm different)]
    ring
  · simp

/-- Paired losses need a covariance term, even before observations accumulate. -/
theorem Pair.variance
    {World : Type*} [MeasurableSpace World]
    (measure : Measure World) [IsFiniteMeasure measure]
    (incumbent candidate : World → ℝ)
    (incumbentMoment : MemLp incumbent 2 measure)
    (candidateMoment : MemLp candidate 2 measure) :
    Var[fun world => incumbent world - candidate world; measure] =
      Var[incumbent; measure] - 2 * cov[incumbent, candidate; measure] +
        Var[candidate; measure] :=
  variance_fun_sub incumbentMoment candidateMoment

/-- Paired evaluation strictly reduces variance below candidate alone iff twice covariance exceeds incumbent variance (Glasserman & Yao 1992). -/
theorem Pair.reduced
    {World : Type*} [MeasurableSpace World]
    (measure : Measure World) [IsFiniteMeasure measure]
    (incumbent candidate : World → ℝ)
    (incumbentMoment : MemLp incumbent 2 measure)
    (candidateMoment : MemLp candidate 2 measure) :
    Var[fun world => incumbent world - candidate world; measure] < Var[candidate; measure] ↔
      Var[incumbent; measure] < 2 * cov[incumbent, candidate; measure] := by
  rw [Pair.variance measure incumbent candidate incumbentMoment candidateMoment]
  constructor
  · intro h
    linarith
  · intro h
    linarith

/-- Uncoupled evaluation variance equals the sum of individual variances. -/
theorem Pair.uncoupled
    {World : Type*} [MeasurableSpace World]
    (measure : Measure World) [IsFiniteMeasure measure]
    (incumbent candidate : World → ℝ)
    (incumbentMoment : MemLp incumbent 2 measure)
    (candidateMoment : MemLp candidate 2 measure)
    (uncorrelated : cov[incumbent, candidate; measure] = 0) :
    Var[fun world => incumbent world - candidate world; measure] =
      Var[incumbent; measure] + Var[candidate; measure] := by
  rw [Pair.variance measure incumbent candidate incumbentMoment candidateMoment, uncorrelated]
  ring

/-- Positive covariance strictly reduces variance below uncoupled evaluations. -/
theorem Pair.beneficial
    {World : Type*} [MeasurableSpace World]
    (measure : Measure World) [IsFiniteMeasure measure]
    (incumbent candidate : World → ℝ)
    (incumbentMoment : MemLp incumbent 2 measure)
    (candidateMoment : MemLp candidate 2 measure)
    (positiveCov : 0 < cov[incumbent, candidate; measure]) :
    Var[fun world => incumbent world - candidate world; measure] <
      Var[incumbent; measure] + Var[candidate; measure] := by
  rw [Pair.variance measure incumbent candidate incumbentMoment candidateMoment]
  linarith


/-- Adding a stored baseline and an improvement does not remove covariance. -/
theorem Baseline.variance
    {World : Type*} [MeasurableSpace World]
    (measure : Measure World) [IsFiniteMeasure measure]
    (baseline improvement : World → ℝ)
    (baselineMoment : MemLp baseline 2 measure)
    (improvementMoment : MemLp improvement 2 measure) :
    Var[fun world => baseline world + improvement world; measure] =
      Var[baseline; measure] + 2 * cov[baseline, improvement; measure] +
        Var[improvement; measure] :=
  variance_fun_add baselineMoment improvementMoment

/-- Paired evaluation cancels a shared additive component, patch by patch. -/
theorem Pair.invariant {count : ℕ}
    (incumbent candidate noise : Fin count → ℝ) :
    patchMean (fun index => (incumbent index + noise index) -
      (candidate index + noise index)) =
        patchMean (fun index => incumbent index - candidate index) ∧
    patchVariance (fun index => (incumbent index + noise index) -
      (candidate index + noise index)) =
        patchVariance (fun index => incumbent index - candidate index) := by
  have equality : (fun index => (incumbent index + noise index) -
      (candidate index + noise index)) =
      (fun index => incumbent index - candidate index) := by
    funext index
    ring
  rw [equality]
  exact ⟨rfl, rfl⟩

/-- A fixed-world paired chain cancels its intermediate loss exactly. -/
theorem Anchor.cancel (anchor incumbent candidate : ℝ) :
    (anchor - incumbent) + (incumbent - candidate) = anchor - candidate := by
  ring

/-- Changing the incumbent's evaluation world leaves an explicit drift term. -/
theorem Anchor.drift (anchor oldLoss newLoss candidate : ℝ) :
    (anchor - oldLoss) + (newLoss - candidate) =
      (anchor - candidate) + (newLoss - oldLoss) := by
  ring

/-- The cancellation also holds for the actual population variance. -/
theorem Anchor.variance
    {World : Type*} [MeasurableSpace World]
    (measure : Measure World) (anchor incumbent candidate : World → ℝ) :
    Var[fun world => (anchor world - incumbent world) +
      (incumbent world - candidate world); measure] =
      Var[fun world => anchor world - candidate world; measure] := by
  have equality : (fun world => (anchor world - incumbent world) +
      (incumbent world - candidate world)) =
      (fun world => anchor world - candidate world) := by
    funext world
    exact Anchor.cancel _ _ _
  rw [equality]

/-- More copies of the same random response do not average its uncertainty away. -/
theorem Shared.variance
    {World : Type*} [MeasurableSpace World]
    {count : ℕ} (positive : 0 < count)
    (measure : Measure World) (response : World → ℝ) :
    Var[fun world => patchMean (fun _ : Fin count => response world); measure] =
      Var[response; measure] := by
  simp_rw [Mean.repeated positive]

/-- A concrete probability model: zero sample scatter but unit mean variance. -/
theorem Shared.counterexample {count : ℕ} (positive : 0 < count) :
    (∀ world : ℝ, patchVariance (fun _ : Fin count => world) = 0) ∧
      Var[fun world : ℝ => patchMean (fun _ : Fin count => world);
        gaussianReal 0 1] = 1 := by
  constructor
  · intro world
    exact Scatter.repeated positive world
  · rw [Shared.variance positive]
    simp

#print axioms Weighted.variance
#print axioms Weighted.diagonal
#print axioms Pair.variance
#print axioms Pair.reduced
#print axioms Pair.uncoupled
#print axioms Pair.beneficial
#print axioms Baseline.variance
#print axioms Pair.invariant
#print axioms Anchor.drift
#print axioms Anchor.variance
#print axioms Shared.counterexample

end

end BauxResearch.Objective.Response
