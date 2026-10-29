import BauxResearch.Foundation
import Mathlib.Probability.Moments.Covariance
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.LinearAlgebra.Matrix.Trace

/-!
# Full-parameter proposal moments

Cross-parameter dependence is retained. This is a mathematical view of a
proposal law, not a claim that an indexed GPU generator samples that law.
The trace formulas are symbolic: no dense billion-by-billion matrix is proposed
as an implementation. A covariance alone does not certify exploration or progress.
-/
namespace BauxResearch.Optimization.Moments
noncomputable section
open scoped BigOperators
open MeasureTheory ProbabilityTheory Matrix

variable {Index World : Type*} [Fintype Index] [MeasurableSpace World]

def mean (measure : Measure World) (delta : World → Index → ℝ) : Index → ℝ :=
  fun index => ∫ world, delta world index ∂measure

def covariance (measure : Measure World) (delta : World → Index → ℝ) :
    Matrix Index Index ℝ :=
  fun index other => cov[fun world => delta world index, fun world => delta world other; measure]

def quadratic (matrix : Matrix Index Index ℝ) (delta : Index → ℝ) : ℝ :=
  ∑ index, ∑ other, matrix index other * (delta index * delta other)

theorem integrable (measure : Measure World) (delta : World → Index → ℝ)
    (matrix : Matrix Index Index ℝ)
    (moments : ∀ index, MemLp (fun world => delta world index) 2 measure) :
    Integrable (fun world => quadratic matrix (delta world)) measure := by
  exact integrable_finsetSum _ (fun index _ => integrable_finsetSum _
    (fun other _ => ((moments index).integrable_mul (moments other)).const_mul _))

/-- The expected quadratic form uses covariance plus the outer product of the mean. -/
theorem expected (measure : Measure World) [IsProbabilityMeasure measure]
    (delta : World → Index → ℝ) (matrix : Matrix Index Index ℝ)
    (moments : ∀ index, MemLp (fun world => delta world index) 2 measure) :
    (∫ world, quadratic matrix (delta world) ∂measure) =
      ∑ index, ∑ other, matrix index other *
        (covariance measure delta index other + mean measure delta index * mean measure delta other) := by
  unfold quadratic
  rw [integral_finsetSum]
  · apply Finset.sum_congr rfl
    intro index _
    rw [integral_finsetSum]
    · apply Finset.sum_congr rfl
      intro other _
      rw [integral_const_mul]
      have identity := covariance_eq_sub (moments index) (moments other)
      change _ = matrix index other *
        (covariance measure delta index other + mean measure delta index * mean measure delta other)
      congr 1
      simpa [covariance, mean, sub_add_cancel] using
        (eq_add_of_sub_eq (show (∫ world, delta world index * delta world other ∂measure) -
          mean measure delta index * mean measure delta other =
          covariance measure delta index other from identity.symm))
    · exact fun other _ => ((moments index).integrable_mul (moments other)).const_mul _
  · exact fun index _ => integrable_finsetSum _
      (fun other _ => ((moments index).integrable_mul (moments other)).const_mul _)

theorem traceForm (measure : Measure World) [IsProbabilityMeasure measure]
    (delta : World → Index → ℝ) (matrix : Matrix Index Index ℝ)
    (moments : ∀ index, MemLp (fun world => delta world index) 2 measure) :
    (∫ world, quadratic matrix (delta world) ∂measure) =
      Matrix.trace (matrix * covariance measure delta) + quadratic matrix (mean measure delta) := by
  rw [expected measure delta matrix moments]
  simp only [Matrix.trace, Matrix.diag, Matrix.mul_apply, quadratic, mul_add,
    Finset.sum_add_distrib]
  congr 1
  apply Finset.sum_congr rfl
  intro index _
  apply Finset.sum_congr rfl
  intro other _
  rw [covariance, covariance, covariance_comm]

/-- Isotropy charges the trace, but this is not an objective-improvement theorem. -/
theorem isotropic [DecidableEq Index]
    (measure : Measure World) [IsProbabilityMeasure measure]
    (delta : World → Index → ℝ) (matrix : Matrix Index Index ℝ) (variance : ℝ)
    (moments : ∀ index, MemLp (fun world => delta world index) 2 measure)
    (centered : mean measure delta = 0)
    (isotropic : covariance measure delta = Matrix.diagonal (fun _ => variance)) :
    (∫ world, quadratic matrix (delta world) ∂measure) = variance * Matrix.trace matrix := by
  rw [traceForm measure delta matrix moments, centered, isotropic]
  simp [quadratic, Matrix.trace, Matrix.mul_diagonal, ← Finset.mul_sum, mul_comm]

/-- The multivariate Gaussian's mean is inherited through the coordinate projection. -/
theorem Gaussian.mean [DecidableEq Index]
    (location : EuclideanSpace ℝ Index) (cov : Matrix Index Index ℝ) :
    mean (multivariateGaussian location cov) (fun delta index => delta index) = location := by
  funext index
  have identity := (EuclideanSpace.proj index).integral_comp_comm
    (IsGaussian.integrable_fun_id (μ := multivariateGaussian location cov))
  change (∫ delta : EuclideanSpace ℝ Index, delta index ∂multivariateGaussian location cov) = location index
  calc
    _ = (EuclideanSpace.proj index) (∫ delta, delta ∂multivariateGaussian location cov) := identity
    _ = _ := by simp

theorem Gaussian.covariance [DecidableEq Index]
    (location : EuclideanSpace ℝ Index) (cov : Matrix Index Index ℝ)
    (positive : cov.PosSemidef) :
    covariance (multivariateGaussian location cov) (fun delta index => delta index) = cov := by
  ext index other
  exact covariance_eval_multivariateGaussian positive index other

theorem Gaussian.quadratic [DecidableEq Index]
    (location : EuclideanSpace ℝ Index) (cov matrix : Matrix Index Index ℝ)
    (positive : cov.PosSemidef) :
    (∫ delta, Moments.quadratic matrix (fun index => delta index)
      ∂multivariateGaussian location cov) =
      Matrix.trace (matrix * cov) + Moments.quadratic matrix location := by
  have moments : ∀ index, MemLp (fun delta : EuclideanSpace ℝ Index => delta index)
      2 (multivariateGaussian location cov) := by
    intro index
    have vectorMoment : MemLp (id : EuclideanSpace ℝ Index → EuclideanSpace ℝ Index)
        2 (multivariateGaussian location cov) := IsGaussian.memLp_two_id
    simpa only [EuclideanSpace.coe_proj, id_eq] using
      vectorMoment.continuousLinearMap_comp (EuclideanSpace.proj index : EuclideanSpace ℝ Index →L[ℝ] ℝ)
  rw [traceForm _ _ _ moments, Gaussian.mean, Gaussian.covariance _ _ positive]

#print axioms expected
#print axioms traceForm
#print axioms isotropic
#print axioms Gaussian.mean
#print axioms Gaussian.covariance
#print axioms Gaussian.quadratic
end
end BauxResearch.Optimization.Moments
