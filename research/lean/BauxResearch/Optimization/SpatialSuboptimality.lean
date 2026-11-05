import BauxResearch.Optimization.EnnCommit
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Mathlib.MeasureTheory.Measure.MeasureSpace
import Mathlib.Tactic

/-!
# Spatial Suboptimality and Log-Volume Drift (Akimoto et al., arXiv:2009.08647)

Formalizes the measure-theoretic progress certificate for value-based `(1+1)`
evolution strategies from Akimoto, Auger, Glasmachers, and Morinaga (2022):

* `sublevelSet f x = {y | f y ≤ f x}` and
  `spatialSuboptimality μ d f x = (μ {y | f y ≤ f x}) ^ (1 / d)`.
* Unlike Taylor or gradient expansions, `spatialSuboptimality` requires no
  differentiability or continuity of `f`, remaining well-defined across
  piecewise-routed selection cells and quantized parameter lattices.
* Monotonicity of `f` implies set inclusion of sublevel sets and monotonicity of
  `spatialSuboptimality`, so `EnnCommit.objectiveCommit` contracts
  `spatialSuboptimality` pathwise on every round.
* A per-step log-volume drift of `-c / d` yields the `O(d log(V₀ / ε))`
  hitting-time bound to enter the `ε`-suboptimal sublevel set.
-/

namespace BauxResearch.Optimization.SpatialSuboptimality

noncomputable section
open MeasureTheory
open scoped ENNReal

variable {State : Type*} [MeasurableSpace State]

/-- The sublevel set `{y | f y ≤ f x}` of `f` at state `x`. -/
def sublevelSet (f : State → ℝ) (x : State) : Set State :=
  {y | f y ≤ f x}

/-- Spatial suboptimality `V_f(x) = μ({y | f y ≤ f x}) ^ (1 / d)`. -/
def spatialSuboptimality (μ : Measure State) (d : ℕ) (f : State → ℝ) (x : State) : ℝ≥0∞ :=
  (μ (sublevelSet f x)) ^ (1 / (d : ℝ))

/-- Every state belongs to its own sublevel set. -/
theorem mem_sublevel (f : State → ℝ) (x : State) :
    x ∈ sublevelSet f x :=
  le_refl (f x)

/-- Decreasing objective values induce nested sublevel sets without smoothness assumptions. -/
theorem sublevel_mono (f : State → ℝ) {x' x : State} (h : f x' ≤ f x) :
    sublevelSet f x' ⊆ sublevelSet f x :=
  fun _ hy ↦ le_trans hy h

/-- Spatial suboptimality is monotone in the objective value `f x`. -/
theorem suboptimality_mono (μ : Measure State) (d : ℕ) (f : State → ℝ)
    {x' x : State} (h : f x' ≤ f x) :
    spatialSuboptimality μ d f x' ≤ spatialSuboptimality μ d f x := by
  unfold spatialSuboptimality
  have hMeas : μ (sublevelSet f x') ≤ μ (sublevelSet f x) :=
    measure_mono (sublevel_mono f h)
  have hExp : (0 : ℝ) ≤ 1 / (d : ℝ) := one_div_nonneg.mpr (Nat.cast_nonneg d)
  exact ENNReal.rpow_le_rpow hMeas hExp

/-- Guarded `(1+1)` commit (`EnnCommit.objectiveCommit`) never increases spatial suboptimality. -/
theorem suboptimality_commit (μ : Measure State) (d : ℕ) (f : State → ℝ)
    (incumbent candidate : State) (margin : ℝ) (hMargin : 0 ≤ margin) :
    spatialSuboptimality μ d f (EnnCommit.objectiveCommit f incumbent candidate margin) ≤
      spatialSuboptimality μ d f incumbent :=
  suboptimality_mono μ d f (EnnCommit.commit_le f incumbent candidate margin hMargin)

/-- Telescoping negative drift of a real log-volume potential `Φ(t) = log V_f(x_t)`. -/
theorem potential_drift (Phi : ℕ → ℝ) (delta : ℝ) (T : ℕ)
    (hStep : ∀ t < T, Phi (t + 1) - Phi t ≤ -delta) :
    Phi T ≤ Phi 0 - (T : ℝ) * delta := by
  induction T with
  | zero => simp
  | succ t ih =>
    have hPrev := ih (fun s hs ↦ hStep s (Nat.lt_succ_of_lt hs))
    have hLast := hStep t (Nat.lt_succ_self t)
    push_cast
    linarith

/-- Dimension-linear hitting-horizon bound (Akimoto et al., arXiv:2009.08647):
under per-step log-volume drift `-c / d` (`0 < c`, `0 < d`), any step `T` before
reaching spatial suboptimality `ε` (`log ε < Phi T`, `Phi 0 = log V₀`) satisfies
`T < (d / c) * log (V₀ / ε)`. -/
theorem hitting_bound (Phi : ℕ → ℝ) (V0 eps c : ℝ) (d T : ℕ)
    (hV0 : 0 < V0) (hEps : 0 < eps) (hc : 0 < c) (hd : 0 < d)
    (hInit : Phi 0 = Real.log V0)
    (hStep : ∀ t < T, Phi (t + 1) - Phi t ≤ -(c / (d : ℝ)))
    (hNotHit : Real.log eps < Phi T) :
    (T : ℝ) < ((d : ℝ) / c) * Real.log (V0 / eps) := by
  have hdPos : 0 < (d : ℝ) := Nat.cast_pos.mpr hd
  have hDrift := potential_drift Phi (c / (d : ℝ)) T hStep
  rw [hInit] at hDrift
  have hDiff : (T : ℝ) * (c / (d : ℝ)) < Real.log V0 - Real.log eps := by linarith
  rw [← Real.log_div (ne_of_gt hV0) (ne_of_gt hEps)] at hDiff
  have hcdPos : 0 < c / (d : ℝ) := div_pos hc hdPos
  have hDiv := (lt_div_iff₀ hcdPos).mpr hDiff
  have hcNe : c ≠ 0 := ne_of_gt hc
  have hdNe : (d : ℝ) ≠ 0 := ne_of_gt hdPos
  calc
    (T : ℝ) < Real.log (V0 / eps) / (c / (d : ℝ)) := hDiv
    _ = ((d : ℝ) / c) * Real.log (V0 / eps) := by
      field_simp [hcNe, hdNe]

#print axioms sublevel_mono
#print axioms suboptimality_mono
#print axioms suboptimality_commit
#print axioms potential_drift
#print axioms hitting_bound

end
end BauxResearch.Optimization.SpatialSuboptimality
