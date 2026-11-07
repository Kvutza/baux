module

public import Mathlib.Algebra.Order.BigOperators.Ring.Finset
public import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
public import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.NormNum

/-!
# Two evaluated movements with residual feedback

A control problem, not a model of FBT: a noiseless translated unit quadratic,
uniform signed coordinate perturbations, and a known curvature normalization.
The update sees only two consecutive function values and the sampled coordinate.
The unknown minimizer appears only in the objective and proof statements.

The first evaluated point is retained unconditionally. The second evaluated point
adds a fresh perturbation after a correction from the first observed residual.
The intermediate correction is arithmetic, not an extra objective evaluation.
Finite averages enumerate possible executions, not candidates evaluated in a round.

Motivation: Zhang et al., *A New One-Point Residual-Feedback Oracle for Black-Box
Learning and Control*, Lemma 5 and Eq. (5), https://arxiv.org/abs/2006.10820.
Our transition differs from that paper: it starts the next movement at the
previously queried point. The results below are derived for this transition.
-/

public section

namespace BauxResearch.Optimization.Residual
open scoped BigOperators
noncomputable section

variable {d : ℕ}

/-- Translated quadratic; the center is unavailable to the update. -/
def loss (center x : Fin d → ℝ) : ℝ :=
  ∑ i, (x i - center i) ^ 2

/-- Change one coordinate by the signed perturbation. -/
def probe (x : Fin d → ℝ) (i : Fin d) (h : ℝ) : Fin d → ℝ :=
  Function.update x i (x i + h)

/-- Secant feedback from the old value and one new observation. -/
def slope (f : (Fin d → ℝ) → ℝ) (x : Fin d → ℝ) (i : Fin d) (h : ℝ) : ℝ :=
  (f (probe x i h) - f x) / h

/-- Correct from the evaluated point; a fresh perturbation follows this correction. -/
def correct (f : (Fin d → ℝ) → ℝ) (x : Fin d → ℝ) (i : Fin d) (h : ℝ) : Fin d → ℝ :=
  Function.update x i (x i + h - slope f x i h / 2)

/-- Average loss over a fresh uniform coordinate and an independent uniform sign. -/
def meanLoss (f : (Fin d → ℝ) → ℝ) (x : Fin d → ℝ) (r : ℝ) : ℝ :=
  (∑ i, (f (probe x i r) + f (probe x i (-r))) / 2) / d

/-- Two-round terminal loss, with only one actual candidate evaluated per round. -/
def terminal (f : (Fin d → ℝ) → ℝ) (x : Fin d → ℝ) (r : ℝ) : ℝ :=
  (∑ i, (meanLoss f (correct f x i r) r +
    meanLoss f (correct f x i (-r)) r) / 2) / d

/-- Execution needs only current weights, the coordinate, and the cached scalar slope. -/
theorem correct_state (f : (Fin d → ℝ) → ℝ) (x : Fin d → ℝ) (i : Fin d) (h : ℝ) :
    correct f x i h = Function.update (probe x i h) i
      (probe x i h i - slope f x i h / 2) := by
  funext j
  by_cases hij : j = i <;> simp [correct, probe, hij]

theorem loss_update (center x : Fin d → ℝ) (i : Fin d) (y : ℝ) :
    loss center (Function.update x i y) =
      loss center x - (x i - center i) ^ 2 + (y - center i) ^ 2 := by
  classical
  have update_eq : (fun j ↦ (Function.update x i y j - center j) ^ 2) =
      Function.update (fun j ↦ (x j - center j) ^ 2) i ((y - center i) ^ 2) := by
    funext j
    by_cases h : j = i <;> simp [h]
  unfold loss
  rw [update_eq, Finset.sum_update_of_mem (Finset.mem_univ i)]
  rw [Finset.sdiff_singleton_eq_erase]
  have total := Finset.sum_erase_add Finset.univ
    (fun j ↦ (x j - center j) ^ 2) (Finset.mem_univ i)
  linarith

theorem probe_loss (center x : Fin d → ℝ) (i : Fin d) (h : ℝ) :
    loss center (probe x i h) =
      loss center x + 2 * (x i - center i) * h + h ^ 2 := by
  rw [probe, loss_update]
  ring

/-- The residual estimates the old coordinate slope plus a displacement error. -/
theorem slope_exact (center x : Fin d → ℝ) (i : Fin d) (h : ℝ) (hh : h ≠ 0) :
    slope (loss center) x i h = 2 * (x i - center i) + h := by
  unfold slope
  rw [probe_loss]
  field_simp
  ring

/-- Exact loss after the arithmetic correction, before the next fresh perturbation. -/
theorem correct_loss (center x : Fin d → ℝ) (i : Fin d) (h : ℝ) (hh : h ≠ 0) :
    loss center (correct (loss center) x i h) =
      loss center x - (x i - center i) ^ 2 + h ^ 2 / 4 := by
  rw [correct, loss_update, slope_exact center x i h hh]
  ring

theorem mean_exact (center x : Fin d → ℝ) (r : ℝ) (hd : 0 < d) :
    meanLoss (loss center) x r = loss center x + r ^ 2 := by
  have hn : (d : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hd)
  unfold meanLoss
  simp_rw [probe_loss]
  have pair : ∀ i : Fin d,
      (loss center x + 2 * (x i - center i) * r + r ^ 2 +
        (loss center x + 2 * (x i - center i) * (-r) + (-r) ^ 2)) / 2 =
      loss center x + r ^ 2 := by intro i; ring
  simp_rw [pair]
  simp [hn]

/-- Exact dimension dependence after two actual evaluations, given the initial value. -/
theorem terminal_exact (center x : Fin d → ℝ) (r : ℝ) (hd : 0 < d) (hr : r ≠ 0) :
    terminal (loss center) x r =
      (1 - 1 / (d : ℝ)) * loss center x + 5 * r ^ 2 / 4 := by
  have hn : (d : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hd)
  unfold terminal
  simp_rw [mean_exact center _ r hd, correct_loss center x _ r hr,
    correct_loss center x _ (-r) (neg_ne_zero.mpr hr)]
  have pair : ∀ i : Fin d,
      (loss center x - (x i - center i) ^ 2 + r ^ 2 / 4 + r ^ 2 +
        (loss center x - (x i - center i) ^ 2 + (-r) ^ 2 / 4 + r ^ 2)) / 2 =
      loss center x + 5 * r ^ 2 / 4 - (x i - center i) ^ 2 := by intro i; ring
  simp_rw [pair]
  rw [Finset.sum_sub_distrib]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  change ((d : ℝ) * (loss center x + 5 * r ^ 2 / 4) - loss center x) / d = _
  field_simp
  ring

/-- Comparator: two independent perturbations, retaining both evaluated points. -/
def walk (f : (Fin d → ℝ) → ℝ) (x : Fin d → ℝ) (r : ℝ) : ℝ :=
  meanLoss (fun y ↦ meanLoss f y r) x r

theorem walk_exact (center x : Fin d → ℝ) (r : ℝ) (hd : 0 < d) :
    walk (loss center) x r = loss center x + 2 * r ^ 2 := by
  have hn : (d : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hd)
  unfold walk
  simp_rw [mean_exact center _ r hd]
  unfold meanLoss
  simp_rw [probe_loss]
  have pair : ∀ i : Fin d,
      (loss center x + 2 * (x i - center i) * r + r ^ 2 + r ^ 2 +
        (loss center x + 2 * (x i - center i) * (-r) + (-r) ^ 2 + r ^ 2)) / 2 =
      loss center x + 2 * r ^ 2 := by intro i; ring
  simp_rw [pair]
  simp [hn]

/-- Residual feedback strictly beats the memory-free walk on this control problem. -/
theorem terminal_better (center x : Fin d → ℝ) (r : ℝ) (hd : 0 < d) (hr : r ≠ 0) :
    terminal (loss center) x r < walk (loss center) x r := by
  rw [terminal_exact center x r hd hr, walk_exact center x r hd]
  have hq : 0 ≤ loss center x := Finset.sum_nonneg (fun i _ ↦ sq_nonneg _)
  have hn : 0 < (d : ℝ) := by exact_mod_cast hd
  have hdiv : 0 ≤ loss center x / d := div_nonneg hq hn.le
  have hr2 : 0 < r ^ 2 := sq_pos_of_ne_zero hr
  have rearrange : (1 - 1 / (d : ℝ)) * loss center x =
      loss center x - loss center x / d := by ring
  rw [rearrange]
  linarith

/-- Improvement over the comparator does not mean descent from every starting point. -/
theorem optimum_rise (center : Fin d → ℝ) (r : ℝ) (hd : 0 < d) (hr : r ≠ 0) :
    loss center center < terminal (loss center) center r := by
  rw [terminal_exact center center r hd hr]
  have zero : loss center center = 0 := by simp [loss]
  rw [zero]
  have hr2 : 0 < r ^ 2 := sq_pos_of_ne_zero hr
  linarith

/-- Changing curvature alone can reverse the comparison: terminal losses are 44 versus 12. -/
theorem curvature_fail :
    terminal (fun x : Fin 1 → ℝ ↦ 4 * x 0 ^ 2) (fun _ ↦ 1) 1 = 44 ∧
      walk (fun x : Fin 1 → ℝ ↦ 4 * x 0 ^ 2) (fun _ ↦ 1) 1 = 12 := by
  norm_num [terminal, walk, meanLoss, correct, slope, probe]

#print axioms slope_exact
#print axioms correct_state
#print axioms terminal_exact
#print axioms terminal_better
#print axioms optimum_rise
#print axioms curvature_fail

end
end BauxResearch.Optimization.Residual
