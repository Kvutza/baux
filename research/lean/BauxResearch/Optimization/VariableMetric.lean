import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Analysis.Real.Sqrt
import Mathlib.Basic.Real.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Tactic

/-!
# Variable-Metric (1+1) Operator and Seed-Reconstructed Covariance Shaping

Formalizes the single-offspring variable-metric shaping operator of Glasmachers
(arXiv:2412.15647, §3.1) and the unit-variance identity for seed-reconstructed
historical trajectory combinations in `resident_search.metal`:

* Given `k` remembered vectors in `M : Matrix (Fin k) (Fin d) ℝ`, the shaping
  operator `S(M) = I_d + Mᵀ * M` maps `z` to `z + Mᵀ (M z)` in `O(k d)` work.
* Its quadratic form satisfies `u ⬝ᵥ S(M) u = ∑ j, u j ^ 2 + ∑ i, (M u) i ^ 2`,
  bounding `S(M) ⪰ I_d` from below so every direction in `ℝ^d` remains active
  while strictly amplifying variance along any direction with `M u ≠ 0`.
* For seed-reconstructed directions with weights `∑ i, α i ^ 2 ≤ 1`, combining
  `∑ i, α i * z_i + √(1 - ∑ i, α i ^ 2) * z_fresh` preserves exact unit second
  moment while maintaining a strictly positive full-rank innovation when
  `∑ i, α i ^ 2 < 1`.
-/

namespace BauxResearch.Optimization.VariableMetric

noncomputable section
open scoped BigOperators Matrix

variable {k d : ℕ}

/-- The `d × d` variable-metric shaping matrix `I_d + Mᵀ M`. -/
def shapingMatrix (M : Matrix (Fin k) (Fin d) ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  1 + Mᵀ * M

/-- Two-pass `O(k d)` evaluation of `z + Mᵀ (M z)`. -/
def applyMetric (M : Matrix (Fin k) (Fin d) ℝ) (z : Fin d → ℝ) : Fin d → ℝ :=
  z + Mᵀ.mulVec (M.mulVec z)

/-- Induced proposal covariance matrix `σ² (I_d + Mᵀ M)²`. -/
def covarianceMatrix (M : Matrix (Fin k) (Fin d) ℝ) (sigma : ℝ) : Matrix (Fin d) (Fin d) ℝ :=
  (sigma ^ 2) • (shapingMatrix M * shapingMatrix M)

/-- The `O(k d)` two-pass vector product `z + Mᵀ (M z)` equals `(I_d + Mᵀ M) z`. -/
theorem apply_mulvec (M : Matrix (Fin k) (Fin d) ℝ) (z : Fin d → ℝ) :
    applyMetric M z = (shapingMatrix M).mulVec z := by
  unfold applyMetric shapingMatrix
  rw [Matrix.add_mulVec, Matrix.one_mulVec, Matrix.mulVec_mulVec]

/-- The shaping matrix `I_d + Mᵀ M` is symmetric. -/
theorem shaping_transpose (M : Matrix (Fin k) (Fin d) ℝ) :
    (shapingMatrix M)ᵀ = shapingMatrix M := by
  unfold shapingMatrix
  rw [Matrix.transpose_add, Matrix.transpose_one, Matrix.transpose_mul, Matrix.transpose_transpose]

/-- Euclidean squared norm of a vector in `ℝ^n`. -/
def normSq {n : ℕ} (u : Fin n → ℝ) : ℝ :=
  ∑ i : Fin n, (u i) ^ 2

theorem norm_dot {n : ℕ} (u : Fin n → ℝ) :
    normSq u = dotProduct u u := by
  unfold normSq dotProduct
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem norm_nonneg {n : ℕ} (u : Fin n → ℝ) : 0 ≤ normSq u := by
  unfold normSq
  exact Finset.sum_nonneg (fun i _ ↦ sq_nonneg (u i))

/-- The quadratic form of `Mᵀ M` equals the squared norm of `M u` in `ℝ^k`. -/
theorem gram_dot (M : Matrix (Fin k) (Fin d) ℝ) (u : Fin d → ℝ) :
    dotProduct u ((Mᵀ * M).mulVec u) = normSq (M.mulVec u) := by
  rw [← Matrix.mulVec_mulVec, Matrix.dotProduct_mulVec, Matrix.vecMul_transpose,
    norm_dot]

/-- Exact quadratic-form decomposition of the shaping operator `I_d + Mᵀ M`. -/
theorem shaping_dot (M : Matrix (Fin k) (Fin d) ℝ) (u : Fin d → ℝ) :
    dotProduct u ((shapingMatrix M).mulVec u) = normSq u + normSq (M.mulVec u) := by
  unfold shapingMatrix
  rw [Matrix.add_mulVec, Matrix.one_mulVec, dotProduct_add, ← norm_dot,
    gram_dot]

/-- Loewner lower bound `S(M) ⪰ I_d`: the shaping operator never contracts the
quadratic form below the isotropic norm `‖u‖²`. -/
theorem norm_le (M : Matrix (Fin k) (Fin d) ℝ) (u : Fin d → ℝ) :
    normSq u ≤ dotProduct u ((shapingMatrix M).mulVec u) := by
  rw [shaping_dot]
  linarith [norm_nonneg (M.mulVec u)]

/-- Full-dimensional reachability: for any coordinate `j` with `u j ≠ 0`, the
shaped quadratic form is strictly positive regardless of rank `k`. -/
theorem shaping_pos (M : Matrix (Fin k) (Fin d) ℝ) (u : Fin d → ℝ)
    (j : Fin d) (hj : u j ≠ 0) :
    0 < dotProduct u ((shapingMatrix M).mulVec u) := by
  have hterm : 0 < (u j) ^ 2 := sq_pos_of_ne_zero hj
  have hle : (u j) ^ 2 ≤ normSq u := by
    unfold normSq
    exact Finset.single_le_sum (fun i _ ↦ sq_nonneg (u i)) (Finset.mem_univ j)
  exact lt_of_lt_of_le (lt_of_lt_of_le hterm hle) (norm_le M u)

/-- Strict directional amplification: along any direction `u` having nonzero
projection `(M u) i ≠ 0` onto the remembered steps, `u ⬝ᵥ S(M) u > ‖u‖²`. -/
theorem norm_lt (M : Matrix (Fin k) (Fin d) ℝ) (u : Fin d → ℝ)
    (i : Fin k) (hi : M.mulVec u i ≠ 0) :
    normSq u < dotProduct u ((shapingMatrix M).mulVec u) := by
  rw [shaping_dot]
  have hterm : 0 < (M.mulVec u i) ^ 2 := sq_pos_of_ne_zero hi
  have hle : (M.mulVec u i) ^ 2 ≤ normSq (M.mulVec u) := by
    unfold normSq
    exact Finset.single_le_sum (fun r _ ↦ sq_nonneg (M.mulVec u r)) (Finset.mem_univ i)
  linarith

/-- Innovation weight `√(1 - ∑ j, α j ^ 2)` for seed-reconstructed drift + fresh noise. -/
def innovationWeight (alpha : Fin k → ℝ) : ℝ :=
  Real.sqrt (1 - normSq alpha)

/-- Combining `k` orthogonal unit-variance historical seed draws with weights `α`
(`‖α‖² ≤ 1`) and fresh noise weighted by `innovationWeight α` preserves exact
unit variance. -/
theorem combo_variance (alpha : Fin k → ℝ) (hBound : normSq alpha ≤ 1) :
    normSq alpha + (innovationWeight alpha) ^ 2 = 1 := by
  unfold innovationWeight
  have hNonneg : 0 ≤ 1 - normSq alpha := sub_nonneg.mpr hBound
  rw [Real.sq_sqrt hNonneg]
  ring

/-- Whenever `‖α‖² < 1`, the fresh isotropic innovation weight is strictly positive,
guaranteeing full-dimensional support across all `d` coordinates. -/
theorem innovation_pos (alpha : Fin k → ℝ) (hStrict : normSq alpha < 1) :
    0 < innovationWeight alpha := by
  unfold innovationWeight
  exact Real.sqrt_pos.mpr (sub_pos.mpr hStrict)

#print axioms apply_mulvec
#print axioms shaping_dot
#print axioms norm_le
#print axioms shaping_pos
#print axioms norm_lt
#print axioms combo_variance
#print axioms innovation_pos

end
end BauxResearch.Optimization.VariableMetric
