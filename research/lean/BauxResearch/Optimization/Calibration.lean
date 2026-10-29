import BauxResearch.Optimization.Accuracy
import Mathlib.Analysis.Convex.Mul

/-!
# Calibrating source inverse precision

Mathlib Jensen for squared loss connects the source weighted mean to its actual
total precision. A residual/denominator bound is then derived from Lipschitz
regularity, squared metric-distance bounds with explicit estimation remainders,
and simultaneous observation-error bounds. Fitting and query selection may be
adaptive. The source weights remain those of the original dissimilarities.

The resulting confidence multiplier depends on chosen neighbor count and fitted
scales. Neither the source beta nor its shadow dissimilarities are assumed to
satisfy this calibration. No posterior interpretation or independent-weight
noise calculation is used.
-/

namespace BauxResearch.Optimization.Calibration

open BauxResearch.Forward
open scoped BigOperators NNReal

noncomputable section

/-- Two-term Cauchy-Schwarz couples geometric and observation error without
an unnecessary factor of two. -/
private theorem pairBound (x y radius epistemic aleatoric : ℝ)
    (epistemicPos : 0 < epistemic) (aleatoricPos : 0 < aleatoric) :
    (x * radius + y)^2 ≤ (x^2 / epistemic + y^2 / aleatoric) *
      (epistemic * radius^2 + aleatoric) := by
  have es : Real.sqrt epistemic ≠ 0 := (Real.sqrt_pos.mpr epistemicPos).ne'
  have al : Real.sqrt aleatoric ≠ 0 := (Real.sqrt_pos.mpr aleatoricPos).ne'
  have bound := Finset.sum_mul_sq_le_sq_mul_sq (Finset.univ : Finset (Fin 2))
    (![x / Real.sqrt epistemic, y / Real.sqrt aleatoric])
    (![Real.sqrt epistemic * radius, Real.sqrt aleatoric])
  simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val_fin_one] at bound
  have cancel : x / Real.sqrt epistemic * (Real.sqrt epistemic * radius) = x * radius := by
    field_simp
  simpa only [cancel, div_mul_cancel₀ _ al, div_pow, mul_pow,
    Real.sq_sqrt epistemicPos.le, Real.sq_sqrt aleatoricPos.le] using bound

/-- Jensen preserves an additional per-observation residual without changing
the source mean or any of its sampled-distance weights. -/
theorem residualBound {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (scale : ℝ) (distances : Vector capacity)
    (nearest : List (Fin capacity)) (truth coefficient : ℝ)
    (extra : Fin nearest.length → ℝ)
    (positive : ∀ position : Fin nearest.length,
      0 < Neighbor.weight data params scale distances nearest[position.val])
    (inactive : (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances nearest)
    (residual : ∀ position : Fin nearest.length,
      (data.values nearest[position.val] - truth)^2 ≤
        coefficient * Neighbor.denominator data params scale distances nearest[position.val] + extra position) :
    (Neighbor.mean data params scale distances nearest - truth)^2 ≤
      (coefficient * nearest.length + ∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] * extra position) /
          Accuracy.weightTotal data params scale distances nearest := by
  have convex : ConvexOn ℝ Set.univ (fun value : ℝ => (value - truth)^2) := by
    simpa [Function.comp_def, sub_eq_add_neg, add_comm] using
      (Even.convexOn_pow (by decide : Even 2) : ConvexOn ℝ Set.univ (fun value : ℝ => value^2)).translate_right (-truth)
  have sumWeights : (∑ position : Fin nearest.length,
      Neighbor.weight data params scale distances nearest[position.val]) =
      Accuracy.weightTotal data params scale distances nearest := by
    simp only [Accuracy.weightTotal, ← List.sum_ofFn, List.ofFn_getElem_eq_map]
  have totalPositive : 0 < Accuracy.weightTotal data params scale distances nearest :=
    lt_of_lt_of_le (by norm_num) inactive
  have jensen := Neighbor.Mean.jensen data params scale distances nearest _ convex
    (fun position => (positive position).le) (by rwa [sumWeights])
  rw [← Neighbor.Mean.library data params scale distances nearest inactive] at jensen
  have numerator : (∑ position : Fin nearest.length,
      Neighbor.weight data params scale distances nearest[position.val] *
        (data.values nearest[position.val] - truth)^2) ≤
      coefficient * nearest.length + ∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] * extra position := by
    calc
      _ ≤ ∑ position : Fin nearest.length,
          Neighbor.weight data params scale distances nearest[position.val] *
            (coefficient * Neighbor.denominator data params scale distances nearest[position.val] + extra position) :=
        Finset.sum_le_sum fun position _ => mul_le_mul_of_nonneg_left (residual position) (positive position).le
      _ = _ := by
        have cancel (position : Fin nearest.length) :
            Neighbor.weight data params scale distances nearest[position.val] *
              (coefficient * Neighbor.denominator data params scale distances nearest[position.val]) = coefficient := by
          have denominator : Neighbor.denominator data params scale distances nearest[position.val] ≠ 0 :=
            ne_of_gt (one_div_pos.mp (positive position))
          unfold Neighbor.weight
          field_simp
        simp only [mul_add, Finset.sum_add_distrib, cancel, Finset.sum_const,
          Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
        ring
  apply jensen.trans
  simp only [Finset.centerMass, smul_eq_mul, sumWeights]
  change (Accuracy.weightTotal data params scale distances nearest)⁻¹ * _ ≤ _
  rw [← div_eq_inv_mul]
  exact div_le_div_of_nonneg_right numerator totalPositive.le

/-- The original inverse-precision bound is the zero-extra specialization. -/
theorem precisionBound {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (scale : ℝ) (distances : Vector capacity)
    (nearest : List (Fin capacity)) (truth coefficient : ℝ)
    (positive : ∀ position : Fin nearest.length,
      0 < Neighbor.weight data params scale distances nearest[position.val])
    (inactive : (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances nearest)
    (residual : ∀ position : Fin nearest.length,
      (data.values nearest[position.val] - truth)^2 ≤
        coefficient * Neighbor.denominator data params scale distances nearest[position.val]) :
    (Neighbor.mean data params scale distances nearest - truth)^2 ≤
      coefficient * nearest.length / Accuracy.weightTotal data params scale distances nearest := by
  simpa using residualBound data params scale distances nearest truth coefficient (fun _ => 0)
    positive inactive (fun position => by simpa using residual position)

theorem confidenceBound {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (scale : ℝ) (distances : Vector capacity)
    (nearest : List (Fin capacity)) (truth coefficient : ℝ)
    (nonnegative : 0 ≤ coefficient)
    (positive : ∀ position : Fin nearest.length,
      0 < Neighbor.weight data params scale distances nearest[position.val])
    (inactive : (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances nearest)
    (residual : ∀ position : Fin nearest.length,
      (data.values nearest[position.val] - truth)^2 ≤
        coefficient * Neighbor.denominator data params scale distances nearest[position.val]) :
    |Neighbor.mean data params scale distances nearest - truth| ≤
      Real.sqrt (coefficient * nearest.length) *
        Real.sqrt (1 / Accuracy.weightTotal data params scale distances nearest) := by
  have bound := precisionBound data params scale distances nearest truth coefficient positive inactive residual
  have root := Real.le_sqrt_of_sq_le (by simpa only [sq_abs] using bound :
    |Neighbor.mean data params scale distances nearest - truth|^2 ≤ _)
  simpa only [div_eq_mul_inv, ← Real.sqrt_mul (mul_nonneg nonnegative (Nat.cast_nonneg nearest.length)),
    one_mul] using root

/-- A deterministic distance-estimation remainder propagates through the
actual source mean and its original weights. Fitting or increasing the neighbor
count does not automatically make this precision-weighted geometry term small. -/
theorem sourceRemainder {State : Type*} [PseudoMetricSpace State] {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances remainder : Vector capacity) (nearest : List (Fin capacity))
    (points : Fin capacity → State) (objective : State → ℝ) (query : State)
    (constant : ℝ≥0) (error : ℝ) (errorNonneg : 0 ≤ error)
    (regular : LipschitzWith constant objective)
    (epistemic : 0 < params.epistemic) (aleatoric : 0 < params.aleatoric)
    (variances : ∀ index, 0 ≤ data.variances index)
    (dissimilarities : ∀ index, 0 ≤ distances index)
    (geometry : ∀ index, dist (points index) query ^ 2 ≤ distances index + remainder index)
    (inactive : (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances nearest)
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] - objective (points nearest[position.val])| ≤ error) :
    |Neighbor.mean data params scale distances nearest - objective query| ≤
      Real.sqrt (((constant : ℝ)^2 / params.epistemic + error^2 / params.aleatoric) *
        (nearest.length + params.epistemic * ∑ position : Fin nearest.length,
          Neighbor.weight data params scale distances nearest[position.val] * remainder nearest[position.val]) /
            Accuracy.weightTotal data params scale distances nearest) := by
  let coefficient := (constant : ℝ)^2 / params.epistemic + error^2 / params.aleatoric
  have coefficientNonneg : 0 ≤ coefficient := by dsimp [coefficient]; positivity
  have residuals : ∀ position : Fin nearest.length,
      (data.values nearest[position.val] - objective query)^2 ≤
        coefficient * Neighbor.denominator data params scale distances nearest[position.val] +
          coefficient * params.epistemic * remainder nearest[position.val] := by
    intro position
    let index := nearest[position.val]
    let denominator := Neighbor.denominator data params scale distances index
    have normalized : 0 ≤ data.variances index / max (1 / 10^12 : ℝ) (scale^2) :=
      div_nonneg (variances _) (by positivity)
    have combined : params.epistemic * dist (points index) query ^ 2 + params.aleatoric ≤
        denominator + params.epistemic * remainder index := by
      have dominated := mul_le_mul_of_nonneg_left (geometry index) epistemic.le
      dsimp [denominator, Neighbor.denominator, Neighbor.noise]
      linarith
    have regularity := regular.dist_le_mul (points index) query
    rw [Real.dist_eq] at regularity
    have residual : |data.values index - objective query| ≤
        (constant : ℝ) * dist (points index) query + error := by
      have triangle := (abs_sub_le (data.values index) (objective (points index)) (objective query)).trans
        (add_le_add (observed position) regularity)
      linarith
    have sumNonneg : 0 ≤ (constant : ℝ) * dist (points index) query + error :=
      add_nonneg (mul_nonneg constant.coe_nonneg dist_nonneg) errorNonneg
    have squared : (data.values index - objective query)^2 ≤
        ((constant : ℝ) * dist (points index) query + error)^2 :=
      sq_le_sq.mpr (by simpa only [abs_of_nonneg sumNonneg] using residual)
    have paired := pairBound (constant : ℝ) error (dist (points index) query)
      params.epistemic params.aleatoric epistemic aleatoric
    have bound := squared.trans (paired.trans (mul_le_mul_of_nonneg_left combined coefficientNonneg))
    change (data.values index - objective query)^2 ≤ _
    dsimp only [coefficient] at bound ⊢
    nlinarith
  have bound := residualBound data params scale distances nearest (objective query) coefficient
    (fun position => coefficient * params.epistemic * remainder nearest[position.val])
    (fun position => Neighbor.Weight.positive data params scale distances _
      epistemic.le aleatoric.le (variances _) (dissimilarities _)) inactive residuals
  have sums : (∑ position : Fin nearest.length,
      Neighbor.weight data params scale distances nearest[position.val] *
        (coefficient * params.epistemic * remainder nearest[position.val])) =
      coefficient * params.epistemic * ∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] * remainder nearest[position.val] := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro position _
    ring
  rw [sums] at bound
  have factor : coefficient * nearest.length + coefficient * params.epistemic *
      (∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] * remainder nearest[position.val]) =
      coefficient * (nearest.length + params.epistemic *
        ∑ position : Fin nearest.length,
          Neighbor.weight data params scale distances nearest[position.val] * remainder nearest[position.val]) := by ring
  rw [factor] at bound
  exact Real.le_sqrt_of_sq_le (by simpa only [sq_abs] using bound)

/-- Exact distance domination recovers the earlier confidence bound. -/
theorem sourceConfidence {State : Type*} [PseudoMetricSpace State] {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : Vector capacity) (nearest : List (Fin capacity))
    (points : Fin capacity → State) (objective : State → ℝ) (query : State)
    (constant : ℝ≥0) (error : ℝ) (errorNonneg : 0 ≤ error)
    (regular : LipschitzWith constant objective)
    (epistemic : 0 < params.epistemic) (aleatoric : 0 < params.aleatoric)
    (variances : ∀ index, 0 ≤ data.variances index)
    (geometry : ∀ index, dist (points index) query ^ 2 ≤ distances index)
    (inactive : (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale distances nearest)
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] - objective (points nearest[position.val])| ≤ error) :
    |Neighbor.mean data params scale distances nearest - objective query| ≤
      Real.sqrt (((constant : ℝ)^2 / params.epistemic + error^2 / params.aleatoric) * nearest.length) *
        Real.sqrt (1 / Accuracy.weightTotal data params scale distances nearest) := by
  have bound := sourceRemainder data params scale distances (fun _ => 0) nearest points objective query
    constant error errorNonneg regular epistemic aleatoric variances
    (fun index => (sq_nonneg _).trans (geometry index))
    (fun index => by simpa using geometry index) inactive observed
  simp only [mul_zero, Finset.sum_const_zero, add_zero] at bound
  have nonnegative : 0 ≤ (constant : ℝ)^2 / params.epistemic + error^2 / params.aleatoric := by positivity
  rw [div_eq_mul_inv, Real.sqrt_mul (mul_nonneg nonnegative (Nat.cast_nonneg nearest.length))] at bound
  simpa only [one_div] using bound

#print axioms residualBound
#print axioms precisionBound
#print axioms confidenceBound
#print axioms sourceRemainder
#print axioms sourceConfidence

end
end BauxResearch.Optimization.Calibration
