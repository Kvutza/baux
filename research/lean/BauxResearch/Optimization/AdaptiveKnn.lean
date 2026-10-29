import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import BauxResearch.Optimization.Step

open scoped BigOperators

/-!
# Adaptive k-Nearest Neighbors on Metric Spaces

This module supplies deterministic metric error bounds and conditional oracle
inequalities for neighborhood selection, using Mathlib's `PseudoMetricSpace`
and `LipschitzWith`.  It does not prove the probabilistic rates in the cited
adaptive-neighbor literature or identify the source risk-prefix rule with Lepski's
procedure.

The observation premise below bounds the average absolute error.  That quantity
need not shrink with neighbor count, even for independent centered noise.
Statistical cancellation instead concerns a signed weighted noise sum, exposed
for the source estimator in `Accuracy.errorSplit`.  Dimension adaptation, noise
concentration, and validity after fitting and adaptive queries remain obligations.
-/

namespace BauxResearch.Optimization.AdaptiveKnn
noncomputable section

variable {State : Type*} [PseudoMetricSpace State]

/-- The k-nearest neighbor empirical mean on a metric space. -/
def knnMean {k : ℕ} (values : Fin k → ℝ) : ℝ :=
  (∑ i : Fin k, values i) / (k : ℝ)

/-- Geometric contract for an approximate neighbor result. `radius` is the
exact neighborhood radius, `stretch` is multiplicative distortion, and
`offset` is additive distortion. No search implementation is identified with
this contract without a separate refinement proof. -/
def ApproxCover {k : ℕ} (points : Fin k → State) (query : State)
    (radius stretch offset : ℝ) : Prop :=
  ∀ index, dist (points index) query ≤ stretch * radius + offset

/-- Metric bias-variance error envelope for k-NN regression. -/
theorem errorBound {k : ℕ} (hk : 0 < k)
    (objective : State → ℝ) (points : Fin k → State) (values : Fin k → ℝ)
    (query : State) (L : ℝ) (R sigma : ℝ)
    (hL : 0 ≤ L)
    (hLip : ∀ x y, |objective x - objective y| ≤ L * dist x y)
    (hDist : ∀ i : Fin k, dist (points i) query ≤ R)
    (hNoise : (∑ i : Fin k, |values i - objective (points i)|) / (k : ℝ) ≤ sigma) :
    |knnMean values - objective query| ≤ L * R + sigma := by
  have hkPos : 0 < (k : ℝ) := Nat.cast_pos.mpr hk
  have hSumDiff : knnMean values - objective query =
      (∑ i : Fin k, (values i - objective query)) / (k : ℝ) := by
    unfold knnMean
    have hEq : ∑ i : Fin k, (values i - objective query) =
        (∑ i : Fin k, values i) - (k : ℝ) * objective query := by
      rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [hEq, sub_div, mul_div_cancel_left₀ (objective query) (ne_of_gt hkPos)]
  rw [hSumDiff, abs_div, abs_of_pos hkPos]
  have hTerm : ∀ i : Fin k, |values i - objective query| ≤
      |values i - objective (points i)| + L * dist (points i) query := by
    intro i
    have hSplit : values i - objective query =
        (values i - objective (points i)) + (objective (points i) - objective query) := by ring
    rw [hSplit]
    have hTri := abs_add_le (values i - objective (points i)) (objective (points i) - objective query)
    have hCont := hLip (points i) query
    linarith
  have hSumTerms : ∑ i : Fin k, |values i - objective query| ≤
      ∑ i : Fin k, (|values i - objective (points i)| + L * dist (points i) query) :=
    Finset.sum_le_sum (fun i _ => hTerm i)
  rw [Finset.sum_add_distrib] at hSumTerms
  have hDistSum : ∑ i : Fin k, L * dist (points i) query ≤ ∑ i : Fin k, L * R := by
    apply Finset.sum_le_sum
    intro i _
    exact mul_le_mul_of_nonneg_left (hDist i) hL
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hDistSum
  have hTotalSum : ∑ i : Fin k, |values i - objective query| ≤
      (∑ i : Fin k, |values i - objective (points i)|) + (k : ℝ) * (L * R) := by
    linarith
  have hAbsSum : |∑ i : Fin k, (values i - objective query)| ≤
      ∑ i : Fin k, |values i - objective query| :=
    Finset.abs_sum_le_sum_abs (fun i => values i - objective query) Finset.univ
  have hUpper : |∑ i : Fin k, (values i - objective query)| ≤
      (∑ i : Fin k, |values i - objective (points i)|) + (k : ℝ) * (L * R) :=
    hAbsSum.trans hTotalSum
  have hDivBound : |∑ i : Fin k, (values i - objective query)| / (k : ℝ) ≤
      ((∑ i : Fin k, |values i - objective (points i)|) + (k : ℝ) * (L * R)) / (k : ℝ) :=
    div_le_div_of_nonneg_right hUpper (le_of_lt hkPos)
  rw [add_div, mul_div_cancel_left₀ (L * R) (ne_of_gt hkPos)] at hDivBound
  linarith

/-- Approximate retrieval changes the geometric bias through its certified
multiplicative and additive radius distortion. -/
theorem approxBound {k : ℕ} (hk : 0 < k)
    (objective : State → ℝ) (points : Fin k → State) (values : Fin k → ℝ)
    (query : State) (L radius stretch offset noise : ℝ)
    (hL : 0 ≤ L)
    (hLip : ∀ x y, |objective x - objective y| ≤ L * dist x y)
    (cover : ApproxCover points query radius stretch offset)
    (observed : (∑ index : Fin k, |values index - objective (points index)|) / (k : ℝ) ≤ noise) :
    |knnMean values - objective query| ≤
      L * (stretch * radius + offset) + noise :=
  errorBound hk objective points values query L
    (stretch * radius + offset) noise hL hLip cover observed

/-- Choosing an adaptive neighborhood by an approximately minimal certified
envelope gives an oracle inequality against every comparison neighborhood. -/
theorem oracleBound {Scale : Type*}
    (truth : ℝ) (estimate envelope : Scale → ℝ)
    (selected comparison : Scale) (slack : ℝ)
    (accurate : ∀ scale, |estimate scale - truth| ≤ envelope scale)
    (chosen : ApproxMin envelope selected slack) :
    |estimate selected - truth| ≤ envelope comparison + slack :=
  (accurate selected).trans (chosen comparison)

/-- The adaptive approximate-KNN composition. Every neighborhood may have its
own count, exact radius, observations, and noise envelope. The selected scale
competes with any comparison scale up to envelope-selection slack. -/
theorem adaptiveBound {Scale : Type*}
    (count : Scale → ℕ) (positive : ∀ scale, 0 < count scale)
    (objective : State → ℝ)
    (points : (scale : Scale) → Fin (count scale) → State)
    (values : (scale : Scale) → Fin (count scale) → ℝ)
    (query : State) (L stretch offset : ℝ)
    (radius noise : Scale → ℝ)
    (hL : 0 ≤ L)
    (hLip : ∀ x y, |objective x - objective y| ≤ L * dist x y)
    (cover : ∀ scale, ApproxCover (points scale) query (radius scale) stretch offset)
    (observed : ∀ scale,
      (∑ index : Fin (count scale),
        |values scale index - objective (points scale index)|) / (count scale : ℝ) ≤ noise scale)
    (selected comparison : Scale) (slack : ℝ)
    (chosen : ApproxMin
      (fun scale => L * (stretch * radius scale + offset) + noise scale)
      selected slack) :
    |knnMean (values selected) - objective query| ≤
      L * (stretch * radius comparison + offset) + noise comparison + slack := by
  apply oracleBound (objective query)
    (fun scale => knnMean (values scale))
    (fun scale => L * (stretch * radius scale + offset) + noise scale)
    selected comparison slack
  · intro scale
    exact approxBound (positive scale) objective (points scale) (values scale)
      query L (radius scale) stretch offset (noise scale) hL hLip
      (cover scale) (observed scale)
  · exact chosen

/-- Denoised 1-NN error bound (Xue & Kpotufe, arXiv:1712.02369).
    Given an anchor point whose value has been pre-denoised within `sigmaAnchor`
    of the true objective, the 1-NN prediction at `query` has error bounded by
    the local 1-NN metric distance `L * R1` plus the pre-smoothed variance `sigmaAnchor`. -/
theorem denoisedNn (objective : State → ℝ) (anchor : State) (yAnchor : ℝ)
    (query : State) (L : ℝ) (R1 sigmaAnchor : ℝ)
    (hL : 0 ≤ L)
    (hLip : ∀ x y, |objective x - objective y| ≤ L * dist x y)
    (hDist : dist anchor query ≤ R1)
    (hDenoised : |yAnchor - objective anchor| ≤ sigmaAnchor) :
    |yAnchor - objective query| ≤ L * R1 + sigmaAnchor := by
  have hSplit : yAnchor - objective query =
      (yAnchor - objective anchor) + (objective anchor - objective query) := by ring
  rw [hSplit]
  have hTri := abs_add_le (yAnchor - objective anchor) (objective anchor - objective query)
  have hCont := hLip anchor query
  have hDistBound : L * dist anchor query ≤ L * R1 := mul_le_mul_of_nonneg_left hDist hL
  linarith

#print axioms errorBound
#print axioms approxBound
#print axioms oracleBound
#print axioms adaptiveBound
#print axioms denoisedNn

end
end BauxResearch.Optimization.AdaptiveKnn
