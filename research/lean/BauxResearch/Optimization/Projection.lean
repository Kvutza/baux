import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Analysis.InnerProductSpace.Projection.Reflection
import Mathlib.Analysis.Convex.Combination

/-!
# Projected-ray surrogate geometry from pairwise squared distances

In a real inner product space `E`, a 0th-order radial KNN surrogate (`Finset.centerMass`
with positive weights) is bounded above by the incumbent reward whenever the
incumbent is the best point in the archive (`convexTrap`).

Using the three pairwise squared distances already maintained by the surrogate
between the incumbent `x_b`, a historical neighbor `x_i = x_b + v`, and a
candidate `x_c = x_b + w` (`‖w‖^2`, `‖v‖^2`, and `‖w - v‖^2`), polarization
recovers the inner product `⟪v, w⟫_ℝ` (`innerDist`), Mathlib's
`Submodule.starProjection_singleton` gives the exact 1D orthogonal projection
onto the historical step ray `ℝ ∙ v` (`rayProjection`), and
`Submodule.norm_sq_eq_add_norm_sq_starProjection` gives the orthogonal residual
squared norm (`orthogonalResidual`). Consequently, at the incumbent (`w = 0`)
every ray predicts `y_inc` identically (`rayBaseline`), while a rejected neighbor
(`y_i < y_inc`) in the opposite half-space (`‖w - v‖^2 > ‖w‖^2 + ‖v‖^2`)
yields a strictly positive predicted improvement (`rayGain`).
-/

namespace BauxResearch.Optimization.Projection

noncomputable section
open scoped BigOperators InnerProductSpace RealInnerProductSpace

/-- Any positive-weight KNN convex combination (`Finset.centerMass`) of archive
outcomes `y i ≤ y_inc` is trapped at or below `y_inc`. -/
theorem convexTrap {ι : Type*} (t : Finset ι) (w : ι → ℝ) (y : ι → ℝ)
    (y_inc : ℝ) (hw : ∀ i ∈ t, 0 ≤ w i) (hpos : 0 < ∑ i ∈ t, w i)
    (hy : ∀ i ∈ t, y i ≤ y_inc) :
    t.centerMass w y ≤ y_inc := by
  simp only [Finset.centerMass, smul_eq_mul]
  rw [inv_mul_le_iff₀ hpos]
  calc ∑ i ∈ t, w i * y i
    _ ≤ ∑ i ∈ t, w i * y_inc := by
      apply Finset.sum_le_sum
      intro i hi
      exact mul_le_mul_of_nonneg_left (hy i hi) (hw i hi)
    _ = (∑ i ∈ t, w i) * y_inc := by rw [← Finset.sum_mul]

/-- Polarization recovers the inner product `⟪v, w⟫` of two steps from the
incumbent purely from the three squared distances `‖w‖^2`, `‖v‖^2`, and `‖w - v‖^2`. -/
theorem innerDist {E : Type*} [SeminormedAddCommGroup E] [InnerProductSpace ℝ E]
    (v w : E) :
    ⟪v, w⟫ = (‖w‖ ^ 2 + ‖v‖ ^ 2 - ‖w - v‖ ^ 2) / 2 := by
  have h := norm_sub_sq_real w v
  rw [real_inner_comm] at h
  linarith

/-- Mathlib's `Submodule.starProjection_singleton` expressed purely in terms of
the three squared distances `‖w‖^2`, `‖v‖^2`, and `‖w - v‖^2`. -/
theorem rayProjection {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (v w : E) :
    (ℝ ∙ v).starProjection w =
      (((‖w‖ ^ 2 + ‖v‖ ^ 2 - ‖w - v‖ ^ 2) / (2 * ‖v‖ ^ 2))) • v := by
  rw [Submodule.starProjection_singleton, RCLike.ofReal_real_eq_id, id_eq, innerDist]
  ring_nf

/-- By Mathlib's `Submodule.norm_sq_eq_add_norm_sq_starProjection`, the squared
distance from candidate step `w` to the 1D subspace `ℝ ∙ v` is the Pythagorean
residual `‖w‖^2 - ‖(ℝ ∙ v).starProjection w‖^2`. -/
theorem orthogonalResidual {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (v w : E) :
    ‖(ℝ ∙ v)ᗮ.starProjection w‖ ^ 2 =
      ‖w‖ ^ 2 - ‖(ℝ ∙ v).starProjection w‖ ^ 2 := by
  have h := Submodule.norm_sq_eq_add_norm_sq_starProjection w (ℝ ∙ v)
  linarith

/-- By the parallelogram law (`norm_add_sq_real` and `norm_sub_sq_real`), the
squared distance `‖w + v‖^2` from candidate step `w` to the antipodal reflection
`-v` of a historical step `v` is determined purely by `‖w‖^2`, `‖v‖^2`, and `‖w - v‖^2`. -/
theorem antipodalDist {E : Type*} [SeminormedAddCommGroup E] [InnerProductSpace ℝ E]
    (v w : E) :
    ‖w + v‖ ^ 2 = 2 * ‖w‖ ^ 2 + 2 * ‖v‖ ^ 2 - ‖w - v‖ ^ 2 := by
  have h1 := norm_add_sq_real w v
  have h2 := norm_sub_sq_real w v
  linarith

/-- At the incumbent (`w = 0`), the projected-ray coordinate is identically zero
and the ray prediction equals `y_inc`, preventing rejected neighbors from
biasing the incumbent baseline downward. -/
theorem rayBaseline {E : Type*} [SeminormedAddCommGroup E]
    (v : E) (y_inc y_i : ℝ) :
    let α := (‖(0 : E)‖ ^ 2 + ‖v‖ ^ 2 - ‖(0 : E) - v‖ ^ 2) / (2 * ‖v‖ ^ 2)
    y_inc + α * (y_i - y_inc) = y_inc := by
  intro α
  have hα : α = 0 := by
    dsimp [α]
    simp
  rw [hα, zero_mul, add_zero]

/-- For a rejected neighbor (`y_i < y_inc`) with `v ≠ 0`, any candidate step `w`
in the opposite half-space (`‖w - v‖^2 > ‖w‖^2 + ‖v‖^2`) has negative projection
coordinate `α < 0` onto `ℝ ∙ v` and strictly positive predicted gain over `y_inc`. -/
theorem rayGain {E : Type*} [NormedAddCommGroup E]
    (v w : E) (hv : v ≠ 0) (y_inc y_i : ℝ)
    (h_rej : y_i < y_inc)
    (h_opp : ‖w‖ ^ 2 + ‖v‖ ^ 2 < ‖w - v‖ ^ 2) :
    let α := (‖w‖ ^ 2 + ‖v‖ ^ 2 - ‖w - v‖ ^ 2) / (2 * ‖v‖ ^ 2)
    y_inc < y_inc + α * (y_i - y_inc) := by
  intro α
  have hv2 : 0 < 2 * ‖v‖ ^ 2 := by positivity
  have hnum : ‖w‖ ^ 2 + ‖v‖ ^ 2 - ‖w - v‖ ^ 2 < 0 := by linarith
  have hα : α < 0 := div_neg_of_neg_of_pos hnum hv2
  have hdiff : y_i - y_inc < 0 := by linarith
  have hprod : 0 < α * (y_i - y_inc) := mul_pos_of_neg_of_neg hα hdiff
  linarith

#print axioms convexTrap
#print axioms innerDist
#print axioms rayProjection
#print axioms orthogonalResidual
#print axioms rayBaseline
#print axioms rayGain

end
end BauxResearch.Optimization.Projection
