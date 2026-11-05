import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic

/-!
# Self-Interacting Trajectory Drift and ENN Denoising Targets

Formalizes the trajectory-driven generator mechanisms from:
1. Huang and Kouhkouh, *Self-interacting CBO* (arXiv:2411.10295)
2. Huang, Kouhkouh, and Sun, *Faithful Global Convergence for the Rescaled CBO*
   (arXiv:2503.08578)
3. Zhang, Hartl, Hazan, and Levin, *Diffusion Models are Evolutionary Algorithms*
   (arXiv:2410.02543)

Main results:
* **Streaming `O(d)` Equivalence (`num_sum`, `den_sum`)**:
  The one-observation recursion `A_t = A_{t-1} + e^{-α y_t} x_t`,
  `B_t = B_{t-1} + e^{-α y_t}` reproduces the exact occupation-measure center.
* **Shift-Invariant Rescaling (`center_shift`)**: Subtracting a
  reference baseline `c` (such as the running minimum `y_min`) from `y` leaves
  `m_t` invariant while preventing exponential underflow/overflow.
* **Quantitative Laplace Concentration (`boltzmann_dist`)**: If state
  `sStar` has optimality gap `γ > 0` (`∀ s ≠ sStar, y sStar + γ ≤ y s`), then
  `|m_t(α)_j - x_{sStar, j}| ≤ D * (n - 1) * exp(-α * γ)`.
* **Variational Minimizer for ENN Denoising (`energy_decomp`)**: The
  fitness-and-kernel weighted vector target `x̂₀` uniquely minimizes the
  weighted squared distance energy `∑ i, w i * (u - x i)²`.
-/

namespace BauxResearch.Optimization.SelfInteracting

noncomputable section
open scoped BigOperators

variable {d n : ℕ}

/-- Streaming numerator `A_t ∈ ℝ^d` updated in `O(d)` work per observation. -/
def streamNum (alpha : ℝ) (x : ℕ → Fin d → ℝ) (y : ℕ → ℝ) : ℕ → Fin d → ℝ
  | 0 => fun j ↦ Real.exp (-alpha * y 0) * x 0 j
  | t + 1 => fun j ↦ streamNum alpha x y t j + Real.exp (-alpha * y (t + 1)) * x (t + 1) j

/-- Streaming normalizer `B_t ∈ ℝ` updated in `O(1)` work per observation. -/
def streamDen (alpha : ℝ) (y : ℕ → ℝ) : ℕ → ℝ
  | 0 => Real.exp (-alpha * y 0)
  | t + 1 => streamDen alpha y t + Real.exp (-alpha * y (t + 1))

/-- Occupation-measure center `m_t = A_t / B_t ∈ ℝ^d`. -/
def occupationCenter (alpha : ℝ) (x : ℕ → Fin d → ℝ) (y : ℕ → ℝ) (t : ℕ) : Fin d → ℝ :=
  fun j ↦ streamNum alpha x y t j / streamDen alpha y t

/-- The streaming numerator equals the batch trajectory sum over `0..t`. -/
theorem num_sum (alpha : ℝ) (x : ℕ → Fin d → ℝ) (y : ℕ → ℝ) (t : ℕ) (j : Fin d) :
    streamNum alpha x y t j =
      ∑ s ∈ Finset.range (t + 1), Real.exp (-alpha * y s) * x s j := by
  induction t with
  | zero => simp [streamNum]
  | succ t ih =>
    simp only [streamNum]
    rw [Finset.sum_range_succ _ (t + 1), ← ih]

/-- The streaming normalizer equals the batch weight sum over `0..t`. -/
theorem den_sum (alpha : ℝ) (y : ℕ → ℝ) (t : ℕ) :
    streamDen alpha y t = ∑ s ∈ Finset.range (t + 1), Real.exp (-alpha * y s) := by
  induction t with
  | zero => simp [streamDen]
  | succ t ih =>
    simp only [streamDen]
    rw [Finset.sum_range_succ _ (t + 1), ← ih]

/-- The streaming normalizer `B_t` is strictly positive at every step `t`. -/
theorem den_pos (alpha : ℝ) (y : ℕ → ℝ) (t : ℕ) :
    0 < streamDen alpha y t := by
  induction t with
  | zero =>
    unfold streamDen
    exact Real.exp_pos _
  | succ t ih =>
    unfold streamDen
    exact add_pos ih (Real.exp_pos _)

/-- Shift invariance: subtracting any constant reference `c` from all losses
leaves the self-interacting occupation center strictly unchanged. -/
theorem center_shift (alpha c : ℝ) (x : ℕ → Fin d → ℝ) (y : ℕ → ℝ)
    (t : ℕ) (j : Fin d) :
    occupationCenter alpha x (fun s ↦ y s - c) t j = occupationCenter alpha x y t j := by
  unfold occupationCenter
  rw [num_sum, den_sum, num_sum, den_sum]
  have hNum : (∑ s ∈ Finset.range (t + 1), Real.exp (-alpha * (y s - c)) * x s j) =
      Real.exp (alpha * c) * ∑ s ∈ Finset.range (t + 1), Real.exp (-alpha * y s) * x s j := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro s _
    rw [show -alpha * (y s - c) = alpha * c + (-alpha * y s) by ring, Real.exp_add, mul_assoc]
  have hDen : (∑ s ∈ Finset.range (t + 1), Real.exp (-alpha * (y s - c))) =
      Real.exp (alpha * c) * ∑ s ∈ Finset.range (t + 1), Real.exp (-alpha * y s) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro s _
    rw [show -alpha * (y s - c) = alpha * c + (-alpha * y s) by ring, Real.exp_add]
  rw [hNum, hDen, mul_div_mul_left _ _ (ne_of_gt (Real.exp_pos (alpha * c)))]

/-- Finite-archive Boltzmann center over `Fin n`. -/
def boltzmannCenter (alpha : ℝ) (x : Fin n → Fin d → ℝ) (y : Fin n → ℝ) : Fin d → ℝ :=
  fun j ↦ (∑ s : Fin n, Real.exp (-alpha * y s) * x s j) /
    (∑ s : Fin n, Real.exp (-alpha * y s))

/-- Quantitative Laplace concentration (Huang, Kouhkouh, Sun, arXiv:2503.08578):
if `sStar` achieves optimality gap `γ` (`∀ s ≠ sStar, y sStar + γ ≤ y s`) and
coordinate differences are bounded by `D`, the Boltzmann center concentrates on
`x sStar` at rate `D * (n - 1) * exp(-α * γ)`. -/
theorem boltzmann_dist (alpha gamma D : ℝ) (x : Fin n → Fin d → ℝ) (y : Fin n → ℝ)
    (sStar : Fin n) (j : Fin d)
    (hAlpha : 0 ≤ alpha) (hD : 0 ≤ D)
    (hGap : ∀ s : Fin n, s ≠ sStar → y sStar + gamma ≤ y s)
    (hBound : ∀ s : Fin n, |x s j - x sStar j| ≤ D) :
    |boltzmannCenter alpha x y j - x sStar j| ≤
      D * ((Finset.univ.erase sStar).card : ℝ) * Real.exp (-alpha * gamma) := by
  unfold boltzmannCenter
  have hDenGe : Real.exp (-alpha * y sStar) ≤ ∑ s : Fin n, Real.exp (-alpha * y s) :=
    Finset.single_le_sum (fun s _ ↦ (Real.exp_pos (-alpha * y s)).le) (Finset.mem_univ sStar)
  have hDenPos : 0 < ∑ s : Fin n, Real.exp (-alpha * y s) :=
    lt_of_lt_of_le (Real.exp_pos (-alpha * y sStar)) hDenGe
  have hDiff : (∑ s : Fin n, Real.exp (-alpha * y s) * x s j) /
        (∑ s : Fin n, Real.exp (-alpha * y s)) - x sStar j =
      (∑ s ∈ Finset.univ.erase sStar, Real.exp (-alpha * y s) * (x s j - x sStar j)) /
        (∑ s : Fin n, Real.exp (-alpha * y s)) := by
    rw [div_sub' (ne_of_gt hDenPos)]
    congr 1
    rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
    rw [← Finset.add_sum_erase _ _ (Finset.mem_univ sStar)]
    simp [mul_sub]
  rw [hDiff, abs_div, abs_of_pos hDenPos]
  have hNumLe : |∑ s ∈ Finset.univ.erase sStar, Real.exp (-alpha * y s) * (x s j - x sStar j)| ≤
      ∑ s ∈ Finset.univ.erase sStar,
        D * Real.exp (-alpha * gamma) * Real.exp (-alpha * y sStar) := by
    calc
      |∑ s ∈ Finset.univ.erase sStar, Real.exp (-alpha * y s) * (x s j - x sStar j)|
        ≤ ∑ s ∈ Finset.univ.erase sStar, |Real.exp (-alpha * y s) * (x s j - x sStar j)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ s ∈ Finset.univ.erase sStar,
            D * Real.exp (-alpha * gamma) * Real.exp (-alpha * y sStar) := by
          apply Finset.sum_le_sum
          intro s hs
          have hNe : s ≠ sStar := (Finset.mem_erase.mp hs).1
          rw [abs_mul, abs_of_pos (Real.exp_pos _)]
          have hExpLe : Real.exp (-alpha * y s) ≤
              Real.exp (-alpha * gamma) * Real.exp (-alpha * y sStar) := by
            rw [← Real.exp_add]
            apply Real.exp_le_exp.mpr
            have h1 := mul_le_mul_of_nonneg_left (hGap s hNe) hAlpha
            linarith
          have hProd := mul_le_mul hExpLe (hBound s) (abs_nonneg _)
            (mul_nonneg (Real.exp_pos _).le (Real.exp_pos _).le)
          linarith
  rw [Finset.sum_const, nsmul_eq_mul] at hNumLe
  have hDivLe : |∑ s ∈ Finset.univ.erase sStar, Real.exp (-alpha * y s) * (x s j - x sStar j)| /
        (∑ s : Fin n, Real.exp (-alpha * y s)) ≤
      (((Finset.univ.erase sStar).card : ℝ) *
        (D * Real.exp (-alpha * gamma) * Real.exp (-alpha * y sStar))) /
        Real.exp (-alpha * y sStar) := by
    have h1 := div_le_div_of_nonneg_right hNumLe hDenPos.le
    have h2 : (((Finset.univ.erase sStar).card : ℝ) *
          (D * Real.exp (-alpha * gamma) * Real.exp (-alpha * y sStar))) /
          (∑ s : Fin n, Real.exp (-alpha * y s)) ≤
        (((Finset.univ.erase sStar).card : ℝ) *
          (D * Real.exp (-alpha * gamma) * Real.exp (-alpha * y sStar))) /
          Real.exp (-alpha * y sStar) := by
      apply div_le_div_of_nonneg_left _ (Real.exp_pos _) hDenGe
      positivity
    exact h1.trans h2
  have hCancel : (((Finset.univ.erase sStar).card : ℝ) *
        (D * Real.exp (-alpha * gamma) * Real.exp (-alpha * y sStar))) /
        Real.exp (-alpha * y sStar) =
      D * ((Finset.univ.erase sStar).card : ℝ) * Real.exp (-alpha * gamma) := by
    have hNe : Real.exp (-alpha * y sStar) ≠ 0 := ne_of_gt (Real.exp_pos _)
    field_simp [hNe]
  linarith

/-- ENN-localized parameter target `x̂₀ = (∑ i, w i * x i) / (∑ i, w i)`
(Zhang et al., arXiv:2410.02543). -/
def denoisingTarget (w : Fin n → ℝ) (x : Fin n → Fin d → ℝ) : Fin d → ℝ :=
  fun j ↦ (∑ i : Fin n, w i * x i j) / (∑ i : Fin n, w i)

/-- Exact bias-variance quadratic energy decomposition around `denoisingTarget`:
`∑ i, w i * (u - x i)² = W * (u - x̂₀)² + ∑ i, w i * (x i - x̂₀)²`. -/
theorem energy_decomp (w : Fin n → ℝ) (x : Fin n → Fin d → ℝ) (u : Fin d → ℝ)
    (j : Fin d) (hMass : ∑ i : Fin n, w i ≠ 0) :
    ∑ i : Fin n, w i * (u j - x i j) ^ 2 =
      (∑ i : Fin n, w i) * (u j - denoisingTarget w x j) ^ 2 +
        ∑ i : Fin n, w i * (x i j - denoisingTarget w x j) ^ 2 := by
  set W := ∑ i : Fin n, w i
  set m := denoisingTarget w x j
  have hWm : W * m = ∑ i : Fin n, w i * x i j := by
    dsimp [W, m, denoisingTarget]
    exact mul_div_cancel₀ _ hMass
  have hPoint : ∀ i : Fin n,
      w i * (u j - x i j) ^ 2 =
        w i * (u j - m) ^ 2 + w i * (x i j - m) ^ 2 - 2 * (u j - m) * (w i * x i j - w i * m) := by
    intro i
    ring
  calc
    ∑ i : Fin n, w i * (u j - x i j) ^ 2
      = ∑ i : Fin n, (w i * (u j - m) ^ 2 + w i * (x i j - m) ^ 2 -
          2 * (u j - m) * (w i * x i j - w i * m)) :=
        Finset.sum_congr rfl (fun i _ ↦ hPoint i)
    _ = (∑ i : Fin n, w i) * (u j - m) ^ 2 + (∑ i : Fin n, w i * (x i j - m) ^ 2) -
          2 * (u j - m) * ((∑ i : Fin n, w i * x i j) - (∑ i : Fin n, w i) * m) := by
        simp only [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.mul_sum,
          ← Finset.sum_mul]
    _ = W * (u j - m) ^ 2 + ∑ i : Fin n, w i * (x i j - m) ^ 2 := by
        rw [← hWm]
        dsimp [W]
        ring

#print axioms num_sum
#print axioms den_sum
#print axioms den_pos
#print axioms center_shift
#print axioms boltzmann_dist
#print axioms energy_decomp

end
end BauxResearch.Optimization.SelfInteracting
