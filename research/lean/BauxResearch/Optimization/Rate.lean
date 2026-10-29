import BauxResearch.Optimization.Synthesis
import Mathlib.Analysis.SpecificLimits.Basic

/-!
# From a local progress condition to terminal regret

This result derives a finite-horizon rate for arbitrary states and retained
memory, not a two-state example. Conditional finite feedback weights may depend
on time, memory, and query. The local contraction must be proved for that policy
and objective class; it is not a consequence of calling an estimator Bayesian.
Search supplies one-step analytical ingredients. Transferring its continuous
proposal laws to a complete finite-precision policy remains an obligation.
-/
namespace BauxResearch.Optimization.Rate
noncomputable section
open scoped BigOperators Topology
open Filter Synthesis Regret

variable {State Feedback Memory Noise : Type*} [Fintype Noise]

/-- A geometric/estimation certificate implies the local regret contraction.
The policy never receives the proof-side objective, derivative, or optimum. -/
theorem localBound (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise) (objective : State → ℝ) (best : State)
    (time : ℕ) (memory : Memory) (alignment curvature errors : Noise → ℝ)
    (factor penalty : ℝ)
    (step : ∀ noise,
      objective (policy.recommend (advance policy oracle time memory noise)) -
        objective (policy.recommend memory) ≤ -alignment noise + curvature noise + errors noise)
    (dominance : (1 - factor) * simpleRegret objective best (policy.recommend memory) - penalty ≤
      ∑ noise, oracle.weight time memory (policy.query time memory) noise *
        (alignment noise - curvature noise - errors noise)) :
    (∑ noise, oracle.weight time memory (policy.query time memory) noise *
      simpleRegret objective best (policy.recommend (advance policy oracle time memory noise))) ≤
      factor * simpleRegret objective best (policy.recommend memory) + penalty := by
  have bound : (∑ noise, oracle.weight time memory (policy.query time memory) noise *
      (objective (policy.recommend (advance policy oracle time memory noise)) -
        objective (policy.recommend memory))) ≤
      ∑ noise, oracle.weight time memory (policy.query time memory) noise *
        (-alignment noise + curvature noise + errors noise) := by
    apply Finset.sum_le_sum
    intro noise _
    exact mul_le_mul_of_nonneg_left (step noise)
      (oracle.nonnegative time memory (policy.query time memory) noise)
  have leftEq : (∑ noise, oracle.weight time memory (policy.query time memory) noise *
      (objective (policy.recommend (advance policy oracle time memory noise)) -
        objective (policy.recommend memory))) =
      (∑ noise, oracle.weight time memory (policy.query time memory) noise *
        simpleRegret objective best (policy.recommend (advance policy oracle time memory noise))) -
      simpleRegret objective best (policy.recommend memory) := by
    simp only [simpleRegret, mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, oracle.mass, one_mul]
    ring
  have rightEq : (∑ noise, oracle.weight time memory (policy.query time memory) noise *
      (-alignment noise + curvature noise + errors noise)) =
      -(∑ noise, oracle.weight time memory (policy.query time memory) noise *
        (alignment noise - curvature noise - errors noise)) := by
    rw [← Finset.sum_neg_distrib]
    apply Finset.sum_congr rfl
    intro noise _
    ring
  rw [leftEq, rightEq] at bound
  linarith

/-- Uniform local contraction and additive error yield the geometric accumulated
error bound, including adaptive laws and arbitrary retained-memory types. -/
theorem expected (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise) (objective : State → ℝ) (best : State)
    (factor penalty : ℝ) (nonnegative : 0 ≤ factor)
    (localBound : ∀ time memory,
      (∑ noise, oracle.weight time memory (policy.query time memory) noise *
        simpleRegret objective best (policy.recommend (advance policy oracle time memory noise))) ≤
      factor * simpleRegret objective best (policy.recommend memory) + penalty)
    (time steps : ℕ) (memory : Memory) :
    Synthesis.expected policy oracle objective best time steps memory ≤
      factor ^ steps * simpleRegret objective best (policy.recommend memory) +
        penalty * ∑ index ∈ Finset.range steps, factor ^ index := by
  induction steps generalizing time memory with
  | zero => simp [Synthesis.expected]
  | succ steps induction =>
    have distributed :
        (∑ noise, oracle.weight time memory (policy.query time memory) noise *
          (factor ^ steps * simpleRegret objective best
            (policy.recommend (advance policy oracle time memory noise)) +
            penalty * ∑ index ∈ Finset.range steps, factor ^ index)) =
        factor ^ steps * (∑ noise, oracle.weight time memory (policy.query time memory) noise *
          simpleRegret objective best (policy.recommend (advance policy oracle time memory noise))) +
          penalty * ∑ index ∈ Finset.range steps, factor ^ index := by
      calc
        _ = ∑ noise, (factor ^ steps * (oracle.weight time memory (policy.query time memory) noise *
              simpleRegret objective best (policy.recommend (advance policy oracle time memory noise))) +
            (penalty * ∑ index ∈ Finset.range steps, factor ^ index) *
              oracle.weight time memory (policy.query time memory) noise) := by
          apply Finset.sum_congr rfl
          intro noise _
          ring
        _ = _ := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, oracle.mass, mul_one]
    calc
      Synthesis.expected policy oracle objective best time (steps + 1) memory ≤
          ∑ noise, oracle.weight time memory (policy.query time memory) noise *
            (factor ^ steps * simpleRegret objective best
              (policy.recommend (advance policy oracle time memory noise)) +
              penalty * ∑ index ∈ Finset.range steps, factor ^ index) := by
        unfold Synthesis.expected
        apply Finset.sum_le_sum
        intro noise _
        exact mul_le_mul_of_nonneg_left (induction (time + 1) _)
          (oracle.nonnegative time memory (policy.query time memory) noise)
      _ = _ := distributed
      _ ≤ factor ^ steps *
          (factor * simpleRegret objective best (policy.recommend memory) + penalty) +
          penalty * ∑ index ∈ Finset.range steps, factor ^ index :=
        add_le_add (mul_le_mul_of_nonneg_left (localBound time memory)
          (pow_nonneg nonnegative steps)) (le_refl _)
      _ = _ := by rw [Finset.sum_range_succ, pow_succ]; ring

def envelope (initial factor penalty : ℝ) (steps : ℕ) : ℝ :=
  factor ^ steps * (initial - penalty / (1 - factor)) + penalty / (1 - factor)

/-- Nonzero local errors produce a floor; contraction alone does not erase them. -/
theorem floorBound (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise) (objective : State → ℝ) (best : State)
    (factor penalty : ℝ) (nonnegative : 0 ≤ factor) (contractive : factor < 1)
    (localBound : ∀ time memory,
      (∑ noise, oracle.weight time memory (policy.query time memory) noise *
        simpleRegret objective best (policy.recommend (advance policy oracle time memory noise))) ≤
      factor * simpleRegret objective best (policy.recommend memory) + penalty)
    (time steps : ℕ) (memory : Memory) :
    Synthesis.expected policy oracle objective best time steps memory ≤
      envelope (simpleRegret objective best (policy.recommend memory)) factor penalty steps := by
  have bound := expected policy oracle objective best factor penalty nonnegative localBound time steps memory
  have identity := geom_sum_mul_neg factor steps
  have denominator : 1 - factor ≠ 0 := ne_of_gt (sub_pos.mpr contractive)
  have equality : factor ^ steps * simpleRegret objective best (policy.recommend memory) +
      penalty * ∑ index ∈ Finset.range steps, factor ^ index =
      envelope (simpleRegret objective best (policy.recommend memory)) factor penalty steps := by
    rw [show (∑ index ∈ Finset.range steps, factor ^ index) =
      (1 - factor ^ steps) / (1 - factor) from (eq_div_iff denominator).2 identity]
    unfold envelope
    ring
  rwa [equality] at bound

/-- Mathlib supplies the limit of the derived envelope, not an assumed BAUX rate. -/
theorem asymptote (initial factor penalty : ℝ)
    (nonnegative : 0 ≤ factor) (contractive : factor < 1) :
    Tendsto (envelope initial factor penalty) atTop (𝓝 (penalty / (1 - factor))) := by
  change Tendsto (fun steps => factor ^ steps * (initial - penalty / (1 - factor)) +
    penalty / (1 - factor)) atTop (𝓝 (penalty / (1 - factor)))
  have powers := tendsto_pow_atTop_nhds_zero_of_lt_one nonnegative contractive
  simpa only [zero_mul, zero_add] using
    (powers.mul_const (initial - penalty / (1 - factor))).add_const (penalty / (1 - factor))

/-- Across a token-indexed family of synthesis tasks, a contracting geometric
envelope vanishes when its certified additive error also vanishes.  Unlike
`asymptote`, the error bound here may improve with the available information
budget.  This does not claim that one fixed run knows its terminal horizon. -/
theorem movingFloor (initial factor : ℝ) (penalty : ℕ → ℝ)
    (nonnegative : 0 ≤ factor) (contractive : factor < 1)
    (penaltyZero : Tendsto penalty atTop (nhds 0)) :
    Tendsto (fun steps ↦ envelope initial factor (penalty steps) steps)
      atTop (nhds 0) := by
  have powers : Tendsto (fun steps : ℕ ↦ factor ^ steps) atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_lt_one nonnegative contractive
  have floorZero : Tendsto (fun steps ↦ penalty steps / (1 - factor))
      atTop (nhds 0) := by
    simpa only [zero_div] using penaltyZero.div_const (1 - factor)
  have shifted : Tendsto
      (fun steps ↦ initial - penalty steps / (1 - factor))
      atTop (nhds initial) := by
    simpa only [sub_zero] using tendsto_const_nhds.sub floorZero
  change Tendsto
    (fun steps ↦ factor ^ steps *
      (initial - penalty steps / (1 - factor)) +
        penalty steps / (1 - factor)) atTop (nhds 0)
  simpa only [zero_mul, zero_add] using powers.mul shifted |>.add floorZero

theorem nonnegative (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise) (objective : State → ℝ) (best : State)
    (optimal : Optimal objective best) (time steps : ℕ) (memory : Memory) :
    0 ≤ Synthesis.expected policy oracle objective best time steps memory := by
  induction steps generalizing time memory with
  | zero => exact Regret.nonnegative objective best (policy.recommend memory) optimal
  | succ steps induction =>
    apply Finset.sum_nonneg
    intro noise _
    exact mul_nonneg (oracle.nonnegative time memory (policy.query time memory) noise)
      (induction (time + 1) _)

/-- With zero additive error and a proved global comparator, the policy's actual
expected regret tends to zero. Neither premise is established for BAUX here. -/
theorem vanishes (policy : Policy State Feedback Memory)
    (oracle : Oracle State Feedback Memory Noise) (objective : State → ℝ) (best : State)
    (optimal : Optimal objective best) (factor : ℝ)
    (factorPositive : 0 ≤ factor) (contractive : factor < 1)
    (localBound : ∀ time memory,
      (∑ noise, oracle.weight time memory (policy.query time memory) noise *
        simpleRegret objective best (policy.recommend (advance policy oracle time memory noise))) ≤
      factor * simpleRegret objective best (policy.recommend memory)) :
    Tendsto (fun steps => Synthesis.expected policy oracle objective best 0 steps policy.initial)
      atTop (𝓝 0) := by
  apply squeeze_zero (fun steps => nonnegative policy oracle objective best optimal 0 steps policy.initial)
    (fun steps => show Synthesis.expected policy oracle objective best 0 steps policy.initial ≤
      factor ^ steps * simpleRegret objective best (policy.recommend policy.initial) from by
        simpa using expected policy oracle objective best factor 0 factorPositive
          (by intro time memory; simpa using localBound time memory) 0 steps policy.initial)
  simpa using (tendsto_pow_atTop_nhds_zero_of_lt_one factorPositive contractive).mul_const
    (simpleRegret objective best (policy.recommend policy.initial))

#print axioms localBound
#print axioms expected
#print axioms floorBound
#print axioms asymptote
#print axioms movingFloor
#print axioms vanishes
end
end BauxResearch.Optimization.Rate
