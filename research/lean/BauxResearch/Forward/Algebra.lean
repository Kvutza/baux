import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Convex.Birkhoff
import Mathlib.Tactic

/-! Real-arithmetic forward operations. `store` is an explicit numeric boundary,
not a proved BF16/FP16 implementation. F32 accumulation order is a separate
refinement obligation. Shapes remain symbolic rather than allocating tensors. -/
namespace BauxResearch.Forward
noncomputable section
open scoped BigOperators

abbrev Vector (width : ℕ) := Fin width → ℝ
abbrev Rows (rows width : ℕ) := Fin rows → Vector width

structure Arithmetic where
  store : ℝ → ℝ

def Arithmetic.exact : Arithmetic := ⟨id⟩

def dot {width : ℕ} (left right : Vector width) : ℝ :=
  ∑ index, left index * right index

def project {input output : ℕ} (math : Arithmetic)
    (weights : Fin input → Fin output → ℝ) (value : Vector input) : Vector output :=
  fun column => math.store (∑ index, value index * weights index column)

def sigmoid (value : ℝ) : ℝ := 1 / (1 + Real.exp (-value))
def silu (value : ℝ) : ℝ := value * sigmoid value

def rms {width : ℕ} (math : Arithmetic) (epsilon : ℝ)
    (weights value : Vector width) : Vector width :=
  fun index => math.store (value index * weights index /
    Real.sqrt ((∑ column, value column ^ 2) / width + epsilon))

def average {width : ℕ} (values : Vector width) : ℝ :=
  (∑ index, values index) / width

/-- Canonical finite-score ordering: decreasing score, then increasing index. -/
def rank {width : ℕ} (scores : Vector width) : List (Fin width) :=
  (List.ofFn id).mergeSort fun left right =>
    decide (scores right < scores left ∨
      (scores right = scores left ∧ left.val ≤ right.val))

def winner {width : ℕ} (positive : 0 < width) (scores : Vector width) : Fin width :=
  (rank scores).headD ⟨0, positive⟩

def probability {width : ℕ} (scores : Vector width) (index : Fin width) : ℝ :=
  Real.exp (scores index) / ∑ other, Real.exp (scores other)

def logLoss {width : ℕ} (scores : Vector width) (target : Fin width) : ℝ :=
  Real.log (∑ index, Real.exp (scores index)) - scores target

theorem Probability.positive {width : ℕ} (scores : Vector width) (index : Fin width) :
    0 < ∑ other, Real.exp (scores other) := by
  exact lt_of_lt_of_le (Real.exp_pos (scores index))
    (Finset.single_le_sum (fun other _ => (Real.exp_pos (scores other)).le)
      (Finset.mem_univ index))

theorem Probability.normalized {width : ℕ} (positive : 0 < width)
    (scores : Vector width) : ∑ index, probability scores index = 1 := by
  have denominator := Probability.positive scores (⟨0, positive⟩ : Fin width)
  simp only [probability]
  rw [← Finset.sum_div, div_self denominator.ne']

theorem Probability.nonnegative {width : ℕ} (scores : Vector width)
    (index : Fin width) : 0 ≤ probability scores index :=
  (div_pos (Real.exp_pos _) (Probability.positive scores index)).le

theorem Loss.nonnegative {width : ℕ} (scores : Vector width) (target : Fin width) :
    0 ≤ logLoss scores target := by
  have lower : Real.exp (scores target) ≤ ∑ index, Real.exp (scores index) :=
    Finset.single_le_sum (fun index _ => (Real.exp_pos _).le) (Finset.mem_univ _)
  have bound := Real.log_le_log (Real.exp_pos (scores target)) lower
  rw [Real.log_exp] at bound
  exact sub_nonneg.mpr bound

end
end BauxResearch.Forward
