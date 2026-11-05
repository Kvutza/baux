import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Basic.Real.Basic
import Mathlib.Tactic

/-!
# Online EM Conditional-Count Updates and Nonnegative Tensor Chains

Formalizes the incremental conditional-count probability generator from:
1. Cappé and Moulines, *Online EM Algorithm for Latent Data Models*
   (arXiv:0712.4273, JRSS B 2009)
2. Ghalamkari, Hinrich, and Mørup, *E²M: Double Bounded α-Divergence
   Optimization for Tensor-based Discrete Density Estimation* (arXiv:2405.18220)

Main results:
* **Simplex and Positivity Invariance (`step_nonneg`, `step_pos`,
  `step_simplex`)**: The one-observation update
  `S_{t+1} = (1 - γ_t) S_t + γ_t s_{t+1}` preserves nonnegativity, strict
  positivity (for `γ_t < 1`), and probability normalization without gradient
  optimization.
* **Closed-Form M-Step Normalization (`normalize_sum`)**: Rescaling any
  positive-mass sufficient-statistic table yields an exact probability law.
* **Forward-Only Tensor-Train / Markov Chain Normalization
  (`push_sum`, `chain_sum`)**: Left-to-right
  contraction across a sequence of row-stochastic cores preserves exact unit
  total mass at every block `b = 1..B`.
-/

namespace BauxResearch.Optimization.TensorEm

noncomputable section
open scoped BigOperators

variable {r : ℕ}

/-- Cappé–Moulines Online EM running sufficient-statistic update
`S_{t+1} = (1 - γ) S_t + γ s`. -/
def onlineStep (gamma : ℝ) (S s : Fin r → ℝ) : Fin r → ℝ :=
  fun i ↦ (1 - gamma) * S i + gamma * s i

/-- Closed-form M-step normalization `M(S)_i = S_i / ∑ j, S_j`. -/
def normalizeCounts (S : Fin r → ℝ) : Fin r → ℝ :=
  fun i ↦ S i / (∑ j : Fin r, S j)

/-- Online EM preserves nonnegativity of sufficient statistics for `γ ∈ [0, 1]`. -/
theorem step_nonneg (gamma : ℝ) (S s : Fin r → ℝ)
    (hGamma0 : 0 ≤ gamma) (hGamma1 : gamma ≤ 1)
    (hS : ∀ i, 0 ≤ S i) (hs : ∀ i, 0 ≤ s i) (i : Fin r) :
    0 ≤ onlineStep gamma S s i := by
  unfold onlineStep
  have h1 : 0 ≤ 1 - gamma := sub_nonneg.mpr hGamma1
  exact add_nonneg (mul_nonneg h1 (hS i)) (mul_nonneg hGamma0 (hs i))

/-- For `γ ∈ [0, 1)`, strict positivity of running sufficient statistics `S` is
preserved by any nonnegative observation statistic `s`, preventing zero
denominators in the closed-form M-step. -/
theorem step_pos (gamma : ℝ) (S s : Fin r → ℝ)
    (hGamma0 : 0 ≤ gamma) (hGamma1 : gamma < 1)
    (hS : ∀ i, 0 < S i) (hs : ∀ i, 0 ≤ s i) (i : Fin r) :
    0 < onlineStep gamma S s i := by
  unfold onlineStep
  have h1 : 0 < 1 - gamma := sub_pos.mpr hGamma1
  have hLeft : 0 < (1 - gamma) * S i := mul_pos h1 (hS i)
  have hRight : 0 ≤ gamma * s i := mul_nonneg hGamma0 (hs i)
  linarith

/-- Total mass of the Online EM sufficient-statistic update. -/
theorem step_sum (gamma : ℝ) (S s : Fin r → ℝ) :
    ∑ i : Fin r, onlineStep gamma S s i =
      (1 - gamma) * (∑ i : Fin r, S i) + gamma * (∑ i : Fin r, s i) := by
  unfold onlineStep
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]

/-- Simplex invariance: if `S` and `s` sum to `1`, so does `onlineStep gamma S s`. -/
theorem step_simplex (gamma : ℝ) (S s : Fin r → ℝ)
    (hS : ∑ i : Fin r, S i = 1) (hs : ∑ i : Fin r, s i = 1) :
    ∑ i : Fin r, onlineStep gamma S s i = 1 := by
  rw [step_sum, hS, hs]
  ring

/-- Closed-form M-step normalization yields unit total probability mass whenever
the total sufficient-statistic count is nonzero. -/
theorem normalize_sum (S : Fin r → ℝ) (hMass : ∑ j : Fin r, S j ≠ 0) :
    ∑ i : Fin r, normalizeCounts S i = 1 := by
  unfold normalizeCounts
  rw [← Finset.sum_div, div_self hMass]

/-- Left-to-right forward state propagation across one rank-`r` transition core. -/
def pushForward (pi : Fin r → ℝ) (T : Fin r → Fin r → ℝ) : Fin r → ℝ :=
  fun j ↦ ∑ i : Fin r, pi i * T i j

/-- Forward propagation preserves nonnegativity. -/
theorem push_nonneg (pi : Fin r → ℝ) (T : Fin r → Fin r → ℝ)
    (hPi : ∀ i, 0 ≤ pi i) (hT : ∀ i j, 0 ≤ T i j) (j : Fin r) :
    0 ≤ pushForward pi T j := by
  unfold pushForward
  exact Finset.sum_nonneg (fun i _ ↦ mul_nonneg (hPi i) (hT i j))

/-- Forward propagation through a row-stochastic core preserves unit probability mass. -/
theorem push_sum (pi : Fin r → ℝ) (T : Fin r → Fin r → ℝ)
    (hPi : ∑ i : Fin r, pi i = 1) (hT : ∀ i : Fin r, ∑ j : Fin r, T i j = 1) :
    ∑ j : Fin r, pushForward pi T j = 1 := by
  unfold pushForward
  rw [Finset.sum_comm]
  calc
    ∑ i : Fin r, ∑ j : Fin r, pi i * T i j
      = ∑ i : Fin r, pi i * (∑ j : Fin r, T i j) := by
          apply Finset.sum_congr rfl
          intro i _
          rw [← Finset.mul_sum]
    _ = ∑ i : Fin r, pi i * 1 := by
          apply Finset.sum_congr rfl
          intro i _
          rw [hT i]
    _ = 1 := by simpa using hPi

/-- Multi-block left-to-right contraction across a chain of transition cores. -/
def chainForward (pi : Fin r → ℝ) (cores : List (Fin r → Fin r → ℝ)) : Fin r → ℝ :=
  cores.foldl pushForward pi

/-- Left-to-right contraction across any number of row-stochastic tensor-train
cores preserves unit total probability mass without a backward pass. -/
theorem chain_sum (pi : Fin r → ℝ) (cores : List (Fin r → Fin r → ℝ))
    (hPi : ∑ i : Fin r, pi i = 1)
    (hCores : ∀ T ∈ cores, ∀ i : Fin r, ∑ j : Fin r, T i j = 1) :
    ∑ j : Fin r, chainForward pi cores j = 1 := by
  induction cores generalizing pi with
  | nil => simpa [chainForward] using hPi
  | cons head tail ih =>
    simp only [chainForward, List.foldl_cons]
    apply ih
    · exact push_sum pi head hPi (hCores head List.mem_cons_self)
    · intro T hMem
      exact hCores T (List.mem_cons_of_mem head hMem)

#print axioms step_nonneg
#print axioms step_pos
#print axioms step_simplex
#print axioms normalize_sum
#print axioms push_sum
#print axioms chain_sum

end
end BauxResearch.Optimization.TensorEm
