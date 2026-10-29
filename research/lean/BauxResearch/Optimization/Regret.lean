import BauxResearch.Optimization.Step
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.Real.Sqrt

/-!
# Terminal simple regret

The recommendation and comparator are executable states. The optimum is a
proof-side comparator, never an input to an optimizer policy. A comparator is
globally optimal only when `Optimal` is established on the declared state type.
For a population objective, the function here must denote that population
objective, rather than a noisy observed score.
-/

/-!
Write `R_f(θ; θ⋆) = f(θ) - f(θ⋆)`, where `θ⋆` is a proof-side comparator.
`Optimal f θ⋆` is required before interpreting `R_f` as nonnegative regret to
an optimum.  For two states sharing the same comparator,
`R_f(θ₁; θ⋆) - R_f(θ₂; θ⋆) = f(θ₁) - f(θ₂)`; the comparator cancels.
-/

namespace BauxResearch.Optimization.Regret

variable {State : Type*}

def Optimal (objective : State → ℝ) (best : State) : Prop :=
  ∀ state, objective best ≤ objective state

def simpleRegret (objective : State → ℝ) (best recommendation : State) : ℝ :=
  objective recommendation - objective best

theorem nonnegative (objective : State → ℝ) (best recommendation : State)
    (optimal : Optimal objective best) :
    0 ≤ simpleRegret objective best recommendation :=
  sub_nonneg.mpr (optimal recommendation)

/-- Coverage and selection contribute separately to terminal simple regret. -/
theorem decompose (objective : State → ℝ) (best comparison recommendation : State) :
    simpleRegret objective best recommendation =
      (objective recommendation - objective comparison) +
      simpleRegret objective best comparison := by
  unfold simpleRegret
  ring

/-- A coverage certificate composes with the existing uniform selector bound. -/
theorem poolBound {Candidate : Type*}
    (objective : State → ℝ) (best : State) (candidate : Candidate → State)
    (estimate : Candidate → ℝ) (selected comparison : Candidate)
    (coverage error slack : ℝ)
    (covered : simpleRegret objective best (candidate comparison) ≤ coverage)
    (accurate : UniformError (fun index => objective (candidate index)) estimate error)
    (selectedWithin : ApproxMin estimate selected slack) :
    simpleRegret objective best (candidate selected) ≤ coverage + 2 * error + slack := by
  have selection := Selection.regretBound
    (fun index => objective (candidate index)) estimate selected comparison
    error slack accurate selectedWithin
  rw [decompose objective best (candidate comparison) (candidate selected)]
  linarith

/-- A common comparator cancels when comparing two algorithms' terminal regret. -/
theorem compare (objective : State → ℝ) (best first second : State) :
    simpleRegret objective best first - simpleRegret objective best second =
      objective first - objective second := by
  unfold simpleRegret
  ring

open scoped BigOperators

/-- Cauchy-Schwarz bound on the sum of surrogate standard deviations. -/
theorem cauchyBound {ι : Type*} (s : Finset ι) (sigma : ι → ℝ) :
    (∑ i ∈ s, sigma i) ^ 2 ≤ (s.card : ℝ) * ∑ i ∈ s, (sigma i) ^ 2 := by
  have cs := Finset.sum_mul_sq_le_sq_mul_sq s (fun _ => (1 : ℝ)) sigma
  simp only [one_mul, one_pow, Finset.sum_const, nsmul_eq_mul, mul_one] at cs
  exact cs

/-- Information-gain variance summation bound for sublinear cumulative regret. -/
theorem infoGain {ι : Type*} (s : Finset ι) (sigma : ι → ℝ) (gamma : ℝ)
    (hSum : ∑ i ∈ s, (sigma i) ^ 2 ≤ 2 * gamma) :
    (∑ i ∈ s, sigma i) ^ 2 ≤ 2 * (s.card : ℝ) * gamma := by
  have cs := cauchyBound s sigma
  calc
    (∑ i ∈ s, sigma i) ^ 2 ≤ (s.card : ℝ) * ∑ i ∈ s, (sigma i) ^ 2 := cs
    _ ≤ (s.card : ℝ) * (2 * gamma) := mul_le_mul_of_nonneg_left hSum (Nat.cast_nonneg s.card)
    _ = 2 * (s.card : ℝ) * gamma := by ring

/-- Bound on sum of standard deviations under bounded information gain. -/
theorem sumSigma {ι : Type*} (s : Finset ι) (sigma : ι → ℝ) (gamma : ℝ)
    (hSum : ∑ i ∈ s, (sigma i) ^ 2 ≤ 2 * gamma) :
    ∑ i ∈ s, sigma i ≤ Real.sqrt (2 * (s.card : ℝ) * gamma) := by
  have hSq := infoGain s sigma gamma hSum
  exact Real.le_sqrt_of_sq_le hSq

/-- Sublinear cumulative regret under surrogate confidence exploration. -/
theorem sublinear {ι : Type*} (s : Finset ι) (regret sigma : ι → ℝ) (beta gamma : ℝ)
    (hBeta : 0 ≤ beta)
    (hRound : ∀ i ∈ s, regret i ≤ 2 * beta * sigma i)
    (hSum : ∑ i ∈ s, (sigma i) ^ 2 ≤ 2 * gamma) :
    ∑ i ∈ s, regret i ≤ 2 * beta * Real.sqrt (2 * (s.card : ℝ) * gamma) := by
  have hSumR : ∑ i ∈ s, regret i ≤ ∑ i ∈ s, 2 * beta * sigma i :=
    Finset.sum_le_sum hRound
  rw [← Finset.mul_sum] at hSumR
  have hSig := sumSigma s sigma gamma hSum
  have h2b : 0 ≤ 2 * beta := by linarith [hBeta]
  have hMul := mul_le_mul_of_nonneg_left hSig h2b
  exact hSumR.trans hMul

/-- Sublinear simple regret bounded by C / √T. -/
theorem simpleBound (cumul C : ℝ) (T : ℕ) (hT : 0 < T)
    (hCumul : cumul ≤ C * Real.sqrt (T : ℝ)) :
    cumul / (T : ℝ) ≤ C / Real.sqrt (T : ℝ) := by
  have hTPos : 0 < (T : ℝ) := Nat.cast_pos.mpr hT
  have hSqrtPos : 0 < Real.sqrt (T : ℝ) := Real.sqrt_pos.mpr hTPos
  have hSqrtSq : Real.sqrt (T : ℝ) * Real.sqrt (T : ℝ) = (T : ℝ) :=
    Real.mul_self_sqrt (le_of_lt hTPos)
  calc
    cumul / (T : ℝ) ≤ (C * Real.sqrt (T : ℝ)) / (T : ℝ) :=
      div_le_div_of_nonneg_right hCumul (le_of_lt hTPos)
    _ = (C * Real.sqrt (T : ℝ)) / (Real.sqrt (T : ℝ) * Real.sqrt (T : ℝ)) := by
      rw [hSqrtSq]
    _ = C / Real.sqrt (T : ℝ) := by
      have hNe : Real.sqrt (T : ℝ) ≠ 0 := ne_of_gt hSqrtPos
      field_simp

end BauxResearch.Optimization.Regret
