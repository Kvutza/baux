import Mathlib.Analysis.Calculus.TaylorIntegral
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import BauxResearch.Optimization.Perturbation

/-!
# Objective perturbations from Mathlib calculus

The parameter space may be any real normed space, including the full parameter
space in Foundation. Smoothness is required only along the stated segment.
Derivatives are proof-side objects, not an optimizer oracle. The library's
multidimensional Taylor formula derives the remainder rather than assuming a
linear/quadratic approximation. Routing boundaries and floating-point execution
require separate bounds; no smoothness of the deployed forward program is asserted.
-/
namespace BauxResearch.Optimization.Smooth
noncomputable section
open Set

variable {Space : Type*} [NormedAddCommGroup Space] [NormedSpace ℝ Space]

def remainder (objective : Space → ℝ) (state delta : Space) : ℝ :=
  ∫ time in (0 : ℝ)..1,
    (1 - time) * iteratedFDeriv ℝ 2 objective (state + time • delta) (fun _ => delta)

/-- Exact first-order expansion with a second-order integral remainder. -/
theorem expansion (objective : Space → ℝ) (state delta : Space)
    (smooth : ∀ time ∈ Icc (0 : ℝ) 1,
      ContDiffAt ℝ 2 objective (state + time • delta)) :
    objective (state + delta) = objective state + fderiv ℝ objective state delta +
      remainder objective state delta := by
  have formula := map_add_eq_sum_add_integral_iteratedFDeriv
    (n := 1) (x := state) (y := delta) smooth
  simpa [remainder, Finset.sum_range_succ, iteratedFDeriv_zero_apply,
    iteratedFDeriv_one_apply, smul_eq_mul] using formula

/-- A directional curvature cap integrates to half that cap. It need not be
nonnegative: a negative upper cap describes concavity along this segment. -/
theorem remainderBound (objective : Space → ℝ) (state delta : Space) (cap : ℝ)
    (smooth : ∀ time ∈ Icc (0 : ℝ) 1,
      ContDiffAt ℝ 2 objective (state + time • delta))
    (curvature : ∀ time ∈ Icc (0 : ℝ) 1,
      iteratedFDeriv ℝ 2 objective (state + time • delta) (fun _ => delta) ≤ cap) :
    remainder objective state delta ≤ cap / 2 := by
  have continuous : ContinuousOn
      (fun time : ℝ => iteratedFDeriv ℝ 2 objective
        (state + time • delta) (fun _ => delta)) (Icc 0 1) := by
    intro time member
    have derivative := ((smooth time member).iteratedFDeriv_right
      (i := 2) (m := 0) (by simp)).continuousAt
    exact (derivative.comp (x := time) (by fun_prop)).continuousWithinAt.eval_const _
  have first : IntervalIntegrable
      (fun time : ℝ => (1 - time) * iteratedFDeriv ℝ 2 objective
        (state + time • delta) (fun _ => delta)) MeasureTheory.volume 0 1 := by
    apply ContinuousOn.intervalIntegrable
    rw [uIcc_of_le zero_le_one]
    exact (by fun_prop : ContinuousOn (fun time : ℝ => 1 - time) (Icc 0 1)).mul continuous
  have second : IntervalIntegrable (fun time : ℝ => (1 - time) * cap)
      MeasureTheory.volume 0 1 := (by fun_prop : Continuous _).intervalIntegrable _ _
  have bound := intervalIntegral.integral_mono_on zero_le_one first second
    (fun time member => mul_le_mul_of_nonneg_left (curvature time member)
      (sub_nonneg.mpr member.2))
  have constant : (∫ time in (0 : ℝ)..1, (1 - time) * cap) = cap / 2 := by
    have unit : (∫ time in (0 : ℝ)..1, 1 - time) = (1 : ℝ) / 2 := by
      calc
        _ = (∫ _ in (0 : ℝ)..1, (1 : ℝ)) - (∫ time in (0 : ℝ)..1, time) :=
          intervalIntegral.integral_sub (continuous_const.intervalIntegrable 0 1)
            (continuous_id.intervalIntegrable 0 1)
        _ = _ := by norm_num
    rw [intervalIntegral.integral_mul_const, unit]
    ring
  simpa only [remainder, constant] using bound

theorem upperBound (objective : Space → ℝ) (state delta : Space) (cap : ℝ)
    (smooth : ∀ time ∈ Icc (0 : ℝ) 1,
      ContDiffAt ℝ 2 objective (state + time • delta))
    (curvature : ∀ time ∈ Icc (0 : ℝ) 1,
      iteratedFDeriv ℝ 2 objective (state + time • delta) (fun _ => delta) ≤ cap) :
    objective (state + delta) - objective state ≤
      fderiv ℝ objective state delta + cap / 2 := by
  rw [expansion objective state delta smooth]
  have bound := remainderBound objective state delta cap smooth curvature
  linarith

/-- The better mirrored value retains, rather than cancels, the curvature cost. -/
theorem mirrored (objective : Space → ℝ) (state delta : Space) (cap : ℝ)
    (positive : ∀ time ∈ Icc (0 : ℝ) 1,
      ContDiffAt ℝ 2 objective (state + time • delta))
    (negative : ∀ time ∈ Icc (0 : ℝ) 1,
      ContDiffAt ℝ 2 objective (state + time • (-delta)))
    (positiveCap : ∀ time ∈ Icc (0 : ℝ) 1,
      iteratedFDeriv ℝ 2 objective (state + time • delta) (fun _ => delta) ≤ cap)
    (negativeCap : ∀ time ∈ Icc (0 : ℝ) 1,
      iteratedFDeriv ℝ 2 objective (state + time • (-delta)) (fun _ => -delta) ≤ cap) :
    min (objective (state + delta)) (objective (state + (-delta))) - objective state ≤
      -|fderiv ℝ objective state delta| + cap / 2 := by
  have first := upperBound objective state delta cap positive positiveCap
  have second := upperBound objective state (-delta) cap negative negativeCap
  simp only [map_neg] at second
  rcases le_total (fderiv ℝ objective state delta) 0 with nonpos | nonneg
  · rw [abs_of_nonpos nonpos]
    have chosen := min_le_left (objective (state + delta)) (objective (state + (-delta)))
    linarith
  · rw [abs_of_nonneg nonneg]
    have chosen := min_le_right (objective (state + delta)) (objective (state + (-delta)))
    linarith

#print axioms expansion
#print axioms remainderBound
#print axioms mirrored
end
end BauxResearch.Optimization.Smooth
