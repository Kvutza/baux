import BauxResearch.Forward.Transport

/-!
# Finite transport residual

The finite source iteration need not already be exactly doubly stochastic. Its
reported row/column residual nevertheless certifies mass bounds and an absolute
Euclidean transport-energy bound.
-/

namespace BauxResearch.Forward.Transport.Residual

noncomputable section
open scoped BigOperators
open Matrix

/-- Every individual row discrepancy is bounded by the total residual. -/
theorem rowError {streams : ℕ} (matrix : Square streams) (row : Fin streams) :
    |rowSum matrix row - 1| ≤ residual matrix := by
  have rowPart : |rowSum matrix row - 1| ≤
      ∑ other, |rowSum matrix other - 1| :=
    Finset.single_le_sum (f := fun other => |rowSum matrix other - 1|)
      (fun other _ => abs_nonneg _) (Finset.mem_univ row)
  unfold residual
  exact rowPart.trans (le_add_of_nonneg_right <| Finset.sum_nonneg fun _ _ => abs_nonneg _)

/-- Every individual column discrepancy is bounded by the total residual. -/
theorem columnError {streams : ℕ} (matrix : Square streams) (column : Fin streams) :
    |columnSum matrix column - 1| ≤ residual matrix := by
  have columnPart : |columnSum matrix column - 1| ≤
      ∑ other, |columnSum matrix other - 1| :=
    Finset.single_le_sum (f := fun other => |columnSum matrix other - 1|)
      (fun other _ => abs_nonneg _) (Finset.mem_univ column)
  unfold residual
  exact columnPart.trans (le_add_of_nonneg_left <| Finset.sum_nonneg fun _ _ => abs_nonneg _)

theorem rowBounds {streams : ℕ} (matrix : Square streams) (epsilon : ℝ)
    (bounded : residual matrix ≤ epsilon) (row : Fin streams) :
    1 - epsilon ≤ rowSum matrix row ∧ rowSum matrix row ≤ 1 + epsilon := by
  have error := (rowError matrix row).trans bounded
  exact ⟨by linarith [neg_le_of_abs_le error], by linarith [le_of_abs_le error]⟩

theorem columnBounds {streams : ℕ} (matrix : Square streams) (epsilon : ℝ)
    (bounded : residual matrix ≤ epsilon) (column : Fin streams) :
    1 - epsilon ≤ columnSum matrix column ∧ columnSum matrix column ≤ 1 + epsilon := by
  have error := (columnError matrix column).trans bounded
  exact ⟨by linarith [neg_le_of_abs_le error], by linarith [le_of_abs_le error]⟩

theorem nonnegative {streams : ℕ} (matrix : Square streams) :
    0 ≤ residual matrix := by
  unfold residual
  exact add_nonneg (Finset.sum_nonneg fun _ _ => abs_nonneg _)
    (Finset.sum_nonneg fun _ _ => abs_nonneg _)

/-- Residual at most epsilon makes the finite nonnegative transport amplify
squared Euclidean energy by at most `(1 + epsilon)^2`. -/
theorem energyBound {streams : ℕ} (matrix : Square streams)
    (nonnegative : ∀ row column, 0 ≤ matrix row column)
    (epsilon : ℝ) (epsilonNonnegative : 0 ≤ epsilon)
    (bounded : residual matrix ≤ epsilon) (value : Vector streams) :
    ∑ row, (matrix *ᵥ value) row ^ 2 ≤
      (1 + epsilon) ^ 2 * ∑ column, value column ^ 2 := by
  have result := Approximate.energy matrix nonnegative
    (1 + epsilon) (1 + epsilon) (by linarith)
    (fun row => (rowBounds matrix epsilon bounded row).2)
    (fun column => (columnBounds matrix epsilon bounded column).2) value
  simpa [pow_two] using result

/-- The matrix's own computable residual is an immediate energy certificate. -/
theorem energy {streams : ℕ} (matrix : Square streams)
    (entries : ∀ row column, 0 ≤ matrix row column) (value : Vector streams) :
    ∑ row, (matrix *ᵥ value) row ^ 2 ≤
      (1 + residual matrix) ^ 2 * ∑ column, value column ^ 2 :=
  energyBound matrix entries (residual matrix) (nonnegative matrix) le_rfl value

/-- Every nonnegative floored Sinkhorn result carries an explicit energy bound
whose only finite-iteration penalty is its final row/column residual. -/
theorem sinkhornEnergy {streams : ℕ} (iterations : ℕ) (floor : ℝ)
    (matrix : Square streams) (floorNonnegative : 0 ≤ floor)
    (entries : ∀ row column, 0 ≤ matrix row column) (value : Vector streams) :
    ∑ row, (sinkhorn iterations floor matrix *ᵥ value) row ^ 2 ≤
      (1 + residual (sinkhorn iterations floor matrix)) ^ 2 *
        ∑ column, value column ^ 2 :=
  energy (sinkhorn iterations floor matrix)
    (sinkhornNonnegative iterations floor matrix floorNonnegative entries) value

theorem blendRow {streams : ℕ} (lambda : ℝ) (matrix : Square streams)
    (row : Fin streams) :
    rowSum (blend lambda matrix) row =
      (1 - lambda) + lambda * rowSum matrix row := by
  simp only [rowSum, blend, Matrix.add_apply, Matrix.smul_apply,
    Matrix.one_apply, smul_eq_mul]
  rw [Finset.sum_add_distrib]
  simp [← Finset.mul_sum]

theorem blendColumn {streams : ℕ} (lambda : ℝ) (matrix : Square streams)
    (column : Fin streams) :
    columnSum (blend lambda matrix) column =
      (1 - lambda) + lambda * columnSum matrix column := by
  simp only [columnSum, blend, Matrix.add_apply, Matrix.smul_apply,
    Matrix.one_apply, smul_eq_mul]
  rw [Finset.sum_add_distrib]
  simp [← Finset.mul_sum]

theorem blendResidual {streams : ℕ} (lambda : ℝ) (matrix : Square streams)
    (lambdaNonnegative : 0 ≤ lambda) :
    residual (blend lambda matrix) = lambda * residual matrix := by
  unfold residual
  simp_rw [blendRow, blendColumn]
  have rowAbs : ∀ row : Fin streams,
      |(1 - lambda) + lambda * rowSum matrix row - 1| =
        lambda * |rowSum matrix row - 1| := by
    intro row
    rw [show (1 - lambda) + lambda * rowSum matrix row - 1 =
      lambda * (rowSum matrix row - 1) by ring, abs_mul, abs_of_nonneg lambdaNonnegative]
  have columnAbs : ∀ column : Fin streams,
      |(1 - lambda) + lambda * columnSum matrix column - 1| =
        lambda * |columnSum matrix column - 1| := by
    intro column
    rw [show (1 - lambda) + lambda * columnSum matrix column - 1 =
      lambda * (columnSum matrix column - 1) by ring,
      abs_mul, abs_of_nonneg lambdaNonnegative]
  simp_rw [rowAbs, columnAbs, ← Finset.mul_sum]
  ring

theorem blendNonnegative {streams : ℕ} (lambda : ℝ) (matrix : Square streams)
    (lambdaNonnegative : 0 ≤ lambda) (lambdaUpper : lambda ≤ 1)
    (entries : ∀ row column, 0 ≤ matrix row column) :
    ∀ row column, 0 ≤ blend lambda matrix row column := by
  intro row column
  by_cases same : row = column
  · simp only [blend, Matrix.add_apply, Matrix.smul_apply,
      Matrix.one_apply, same, if_true, smul_eq_mul, mul_one]
    exact add_nonneg (sub_nonneg.mpr lambdaUpper)
      (mul_nonneg lambdaNonnegative (entries _ _))
  · simp only [blend, Matrix.add_apply, Matrix.smul_apply,
      Matrix.one_apply, same, if_false, smul_eq_mul, mul_zero, zero_add]
    exact mul_nonneg lambdaNonnegative (entries _ _)

/-- The actual identity-blended residual transport has squared-energy factor
`(1 + lambda * residual)^2`; clipping lambda into `[0,1]` cannot worsen the raw
Sinkhorn residual penalty. -/
theorem blendEnergy {streams : ℕ} (lambda : ℝ) (matrix : Square streams)
    (lambdaNonnegative : 0 ≤ lambda) (lambdaUpper : lambda ≤ 1)
    (entries : ∀ row column, 0 ≤ matrix row column) (value : Vector streams) :
    ∑ row, (blend lambda matrix *ᵥ value) row ^ 2 ≤
      (1 + lambda * residual matrix) ^ 2 * ∑ column, value column ^ 2 := by
  have result := energy (blend lambda matrix)
    (blendNonnegative lambda matrix lambdaNonnegative lambdaUpper entries) value
  rwa [blendResidual lambda matrix lambdaNonnegative] at result

/-- Source-shaped certificate for the clipped identity blend of a finite
four-stream Sinkhorn transport. -/
theorem sourceEnergy (iterations : ℕ) (maximum : ℝ) (raw : Square 4)
    (lambda : ℝ) (value : Vector 4) :
    ∑ row, (blend (min 1 (max 0 lambda))
          (sinkhorn iterations (1 / (10 : ℝ) ^ 20) (seed maximum raw)) *ᵥ value) row ^ 2 ≤
      (1 + min 1 (max 0 lambda) *
          residual (sinkhorn iterations (1 / (10 : ℝ) ^ 20) (seed maximum raw))) ^ 2 *
        ∑ column, value column ^ 2 := by
  have lambdaNonnegative : 0 ≤ min 1 (max 0 lambda) :=
    le_min (by norm_num) (le_max_left _ _)
  have lambdaUpper : min 1 (max 0 lambda) ≤ 1 := min_le_left _ _
  have seedNonnegative : ∀ row column, 0 ≤ seed maximum raw row column :=
    fun row column => (Real.exp_pos _).le
  have transportNonnegative : ∀ row column,
      0 ≤ sinkhorn iterations (1 / (10 : ℝ) ^ 20) (seed maximum raw) row column :=
    sinkhornNonnegative iterations (1 / (10 : ℝ) ^ 20) (seed maximum raw)
      (by norm_num) seedNonnegative
  exact blendEnergy (min 1 (max 0 lambda))
    (sinkhorn iterations (1 / (10 : ℝ) ^ 20) (seed maximum raw))
    lambdaNonnegative lambdaUpper transportNonnegative value

#print axioms rowError
#print axioms columnError
#print axioms rowBounds
#print axioms columnBounds
#print axioms nonnegative
#print axioms energyBound
#print axioms energy
#print axioms sinkhornEnergy
#print axioms blendRow
#print axioms blendColumn
#print axioms blendResidual
#print axioms blendNonnegative
#print axioms blendEnergy
#print axioms sourceEnergy

end

end BauxResearch.Forward.Transport.Residual
