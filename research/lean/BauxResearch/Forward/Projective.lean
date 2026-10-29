import BauxResearch.Forward.Transport

/-!
# Projective transport geometry

Finite Sinkhorn normalization changes a positive matrix only by alternating row
and column scalings. The scaling factors matter to the executed transport, but
the matrix's projective cross-ratios are exact invariants. This is the correct
quotient geometry in which to seek a non-exploding sensitivity bound.
-/

namespace BauxResearch.Forward.Projective

noncomputable section

open Transport

/-- Two matrices differ only by a left and right diagonal scaling. -/
def ScaleEq {streams : ℕ} (first second : Square streams) : Prop :=
  ∃ left right : Fin streams → ℝ,
    ∀ row column, second row column = left row * first row column * right column

theorem ScaleEq.refl {streams : ℕ} (matrix : Square streams) :
    ScaleEq matrix matrix := by
  exact ⟨fun _ => 1, fun _ => 1, by simp⟩

theorem ScaleEq.trans {streams : ℕ} {first second third : Square streams}
    (firstSecond : ScaleEq first second) (secondThird : ScaleEq second third) :
    ScaleEq first third := by
  obtain ⟨firstLeft, firstRight, firstScale⟩ := firstSecond
  obtain ⟨secondLeft, secondRight, secondScale⟩ := secondThird
  refine ⟨fun row => secondLeft row * firstLeft row,
    fun column => firstRight column * secondRight column, ?_⟩
  intro row column
  rw [secondScale, firstScale]
  ring

theorem rowScale {streams : ℕ} (floor : ℝ) (matrix : Square streams) :
    ScaleEq matrix (rowNormalize floor matrix) := by
  refine ⟨fun row => (max floor (rowSum matrix row))⁻¹, fun _ => 1, ?_⟩
  intro row column
  simp only [rowNormalize, div_eq_mul_inv, mul_one]
  ring

theorem columnScale {streams : ℕ} (floor : ℝ) (matrix : Square streams) :
    ScaleEq matrix (columnNormalize floor matrix) := by
  refine ⟨fun _ => 1, fun column => (max floor (columnSum matrix column))⁻¹, ?_⟩
  intro row column
  simp only [columnNormalize, div_eq_mul_inv, one_mul]

theorem stepScale {streams : ℕ} (floor : ℝ) (matrix : Square streams) :
    ScaleEq matrix (step floor matrix) := by
  exact (rowScale floor matrix).trans (columnScale floor (rowNormalize floor matrix))

theorem sinkhornScale {streams : ℕ} (iterations : ℕ) (floor : ℝ)
    (matrix : Square streams) : ScaleEq matrix (sinkhorn iterations floor matrix) := by
  induction iterations with
  | zero => exact ScaleEq.refl matrix
  | succ iterations hypothesis =>
      rw [sinkhorn, Function.iterate_succ_apply']
      exact hypothesis.trans (stepScale floor (sinkhorn iterations floor matrix))

/-- Diagonal equivalence preserves every division-free two-by-two cross-ratio
identity. No normalization denominator enters this equality. -/
theorem ScaleEq.cross {streams : ℕ} {first second : Square streams}
    (equivalent : ScaleEq first second) (row other column another : Fin streams) :
    second row column * second other another * first row another * first other column =
      first row column * first other another * second row another * second other column := by
  obtain ⟨left, right, scale⟩ := equivalent
  rw [scale row column, scale other another, scale row another, scale other column]
  ring

theorem sinkhornCross {streams : ℕ} (iterations : ℕ) (floor : ℝ)
    (matrix : Square streams) (row other column another : Fin streams) :
    sinkhorn iterations floor matrix row column *
          sinkhorn iterations floor matrix other another *
          matrix row another * matrix other column =
      matrix row column * matrix other another *
          sinkhorn iterations floor matrix row another *
          sinkhorn iterations floor matrix other column :=
  (sinkhornScale iterations floor matrix).cross row other column another

#print axioms ScaleEq.refl
#print axioms ScaleEq.trans
#print axioms rowScale
#print axioms columnScale
#print axioms stepScale
#print axioms sinkhornScale
#print axioms ScaleEq.cross
#print axioms sinkhornCross

end

end BauxResearch.Forward.Projective
