import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
For patch responses `r : Fin m → ℝ`, `patchMean r` is `(∑ᵢ rᵢ)/m`.
The results here depend on this mean and the second raw moment, both invariant
under patch permutation.  A patch index is therefore not treated as a stable
statistical identity across evaluations.
-/

open scoped BigOperators

namespace BauxResearch.Optimization.PatchFeedback
noncomputable section

/-- Mean causal patch response over M patches. -/
def patchMean {m : ℕ} (responses : Fin m → ℝ) : ℝ :=
  (∑ i : Fin m, responses i) / (m : ℝ)

/-- The retained patch mean depends on the response multiset, not patch-index
identity. -/
theorem meanPerm {m : ℕ} (responses : Fin m → ℝ) (order : Fin m ≃ Fin m) :
    patchMean (responses ∘ order) = patchMean responses := by
  unfold patchMean
  congr 1
  simpa [Function.comp_apply] using order.sum_comp responses

/-- The second raw moment is likewise invariant under patch reindexing. -/
theorem energyPerm {m : ℕ} (responses : Fin m → ℝ) (order : Fin m ≃ Fin m) :
    (∑ i : Fin m, (responses (order i)) ^ 2) =
      ∑ i : Fin m, (responses i) ^ 2 := by
  simpa using order.sum_comp (fun i ↦ (responses i) ^ 2)

/-- Mean response error is bounded by the uniform patch discrepancy bound. -/
theorem meanBound {m : ℕ} (hm : 0 < m) (responses : Fin m → ℝ) (target delta : ℝ)
    (hBound : ∀ i : Fin m, |responses i - target| ≤ delta) :
    |patchMean responses - target| ≤ delta := by
  have hmPos : 0 < (m : ℝ) := Nat.cast_pos.mpr hm
  have hSumDiff : patchMean responses - target =
      (∑ i : Fin m, (responses i - target)) / (m : ℝ) := by
    unfold patchMean
    have hEq : ∑ i : Fin m, (responses i - target) =
        (∑ i : Fin m, responses i) - (m : ℝ) * target := by
      rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
    rw [hEq, sub_div, mul_div_cancel_left₀ target (ne_of_gt hmPos)]
  rw [hSumDiff, abs_div, abs_of_pos hmPos]
  have hSumBound : |∑ i : Fin m, (responses i - target)| ≤ (m : ℝ) * delta := by
    have hLe : |∑ i : Fin m, (responses i - target)| ≤ ∑ i : Fin m, |responses i - target| :=
      Finset.abs_sum_le_sum_abs _ _
    have hDelta : ∑ i : Fin m, |responses i - target| ≤ ∑ i : Fin m, delta :=
      Finset.sum_le_sum (fun i _ => hBound i)
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hDelta
    exact hLe.trans hDelta
  have hDiv : |∑ i : Fin m, (responses i - target)| / (m : ℝ) ≤ ((m : ℝ) * delta) / (m : ℝ) :=
    div_le_div_of_nonneg_right hSumBound (le_of_lt hmPos)
  rw [mul_div_cancel_left₀ delta (ne_of_gt hmPos)] at hDiv
  exact hDiv

/-- Energy decomposition: sum of squares equals mean correction plus centered sum of squares. -/
theorem energySplit {m : ℕ} (responses : Fin m → ℝ) (c : ℝ) :
    ∑ i : Fin m, (responses i)^2 =
      ∑ i : Fin m, (responses i - c)^2 + 2 * c * (∑ i : Fin m, responses i) - (m : ℝ) * c^2 := by
  have hTerm : ∀ i : Fin m, (responses i)^2 = (responses i - c)^2 + 2 * c * responses i - c^2 := by
    intro i
    ring
  have hSum : ∑ i : Fin m, (responses i)^2 =
      ∑ i : Fin m, ((responses i - c)^2 + 2 * c * responses i - c^2) := by
    apply Finset.sum_congr rfl
    intro i _
    exact hTerm i
  rw [hSum]
  rw [Finset.sum_sub_distrib]
  rw [Finset.sum_add_distrib]
  rw [← Finset.mul_sum]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- Linear scaling of patch responses by trust-region radius `r` scales `patchMean` by `r`. -/
theorem scaleMean {m : ℕ} (r : ℝ) (responses : Fin m → ℝ) :
    patchMean (fun i ↦ r * responses i) = r * patchMean responses := by
  unfold patchMean
  rw [← Finset.mul_sum, mul_div_assoc]

/-- Linear scaling of patch responses by trust-region radius `r` scales the centered
sum of squares by `r^2`, proving that the paired signal-to-dispersion ratio is
scale-invariant in `r` and depends only on the sample count `m`. -/
theorem scaleCentered {m : ℕ} (r : ℝ) (responses : Fin m → ℝ) (c : ℝ) :
    (∑ i : Fin m, (r * responses i - r * c) ^ 2) =
      r ^ 2 * ∑ i : Fin m, (responses i - c) ^ 2 := by
  rw [ Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  ring

#print axioms meanBound
#print axioms energySplit
#print axioms meanPerm
#print axioms energyPerm
#print axioms scaleMean
#print axioms scaleCentered

end
end BauxResearch.Optimization.PatchFeedback
