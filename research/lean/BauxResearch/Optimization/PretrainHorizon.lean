import BauxResearch.Optimization.PretrainRegret
import BauxResearch.Optimization.PretrainWorkload
import Mathlib.Algebra.Order.Field.GeomSum

/-!
# Population regret over a pretraining horizon

`PretrainRegret.regretTail` gives one natural-window transaction.  This module
composes such transactions without identifying the pretraining horizon with an
arbitrary round count.  Transactions partition evaluated tokens; the resulting
failure budget is a finite union bound and the regret envelope is pathwise.
-/

namespace BauxResearch.Optimization.PretrainHorizon

/-!
For a world `ω`, set `Rₜ(ω) = regret t ω`.  The pathwise input is
`Rₜ₊₁(ω) ≤ factor * Rₜ(ω) + penalty` outside a declared failure event.
Iteration produces a geometric envelope; the probability of leaving it is
at most the sum of the stepwise failure bounds.  The recurrence is the
premise to obtain from the adaptive source policy, not a consequence of
writing a long horizon or a token clock.
-/

noncomputable section

open scoped BigOperators
open MeasureTheory

variable {World : Type*}

/-- The local contraction fails at this transaction. -/
def badAt
    (regret : ℕ → World → ℝ)
    (factor penalty : ℝ)
    (step : ℕ) : Set World :=
  {world | factor * regret step world + penalty < regret (step + 1) world}

/-- At least one local contraction fails before the declared horizon. -/
def badSet
    (regret : ℕ → World → ℝ)
    (factor penalty : ℝ)
    (steps : ℕ) : Set World :=
  ⋃ step : Fin steps, badAt regret factor penalty step

/-- Outside the finite failure union, every transaction before the horizon
satisfies its local regret recurrence. -/
theorem localBound
    (regret : ℕ → World → ℝ)
    (factor penalty : ℝ)
    (steps : ℕ)
    (world : World)
    (clean : world ∉ badSet regret factor penalty steps) :
    ∀ step < steps,
      regret (step + 1) world ≤ factor * regret step world + penalty := by
  intro step before
  have notBad : world ∉ badAt regret factor penalty step := by
    intro bad
    apply clean
    exact Set.mem_iUnion_of_mem ⟨step, before⟩ bad
  exact le_of_not_gt notBad

/-- A finite union bound accumulates the declared transaction failure
probabilities.  No independence between transactions is required. -/
theorem unionBound
    [MeasurableSpace World]
    (measure : Measure World)
    [IsFiniteMeasure measure]
    (regret : ℕ → World → ℝ)
    (factor penalty : ℝ)
    (steps : ℕ)
    (failure : Fin steps → ℝ)
    (stepBound : ∀ step : Fin steps,
      measure.real (badAt regret factor penalty step) ≤ failure step) :
    measure.real (badSet regret factor penalty steps) ≤ ∑ step, failure step := by
  unfold badSet
  exact (measureReal_iUnion_fintype_le
      (μ := measure)
      (fun step : Fin steps => badAt regret factor penalty step)).trans
    (Finset.sum_le_sum fun step _ => stepBound step)

/-- A uniform local contraction gives the pathwise geometric regret envelope
through an arbitrary number of transactions. -/
theorem pathBound
    (regret : ℕ → ℝ)
    (factor penalty : ℝ)
    (factorNonnegative : 0 ≤ factor)
    (steps : ℕ)
    (stepwise : ∀ step < steps,
      regret (step + 1) ≤ factor * regret step + penalty) :
    regret steps ≤
      factor ^ steps * regret 0 +
        penalty * ∑ index ∈ Finset.range steps, factor ^ index := by
  induction steps with
  | zero => simp
  | succ steps induction =>
    have previous := induction
      (fun step before => stepwise step (Nat.lt_succ_of_lt before))
    have final := stepwise steps (Nat.lt_succ_self steps)
    have geomShift :
        factor * (∑ index ∈ Finset.range steps, factor ^ index) + 1 =
          ∑ index ∈ Finset.range (steps + 1), factor ^ index := by
      rw [Finset.sum_range_succ']
      simp only [pow_zero]
      rw [Finset.mul_sum]
      congr 1
      apply Finset.sum_congr rfl
      intro index _
      rw [pow_succ]
      ring
    calc
      regret (steps + 1) ≤ factor * regret steps + penalty := final
      _ ≤ factor *
          (factor ^ steps * regret 0 +
            penalty * ∑ index ∈ Finset.range steps, factor ^ index) + penalty :=
        add_le_add (mul_le_mul_of_nonneg_left previous factorNonnegative) (le_refl penalty)
      _ = factor ^ (steps + 1) * regret 0 +
          penalty * ∑ index ∈ Finset.range (steps + 1), factor ^ index := by
        rw [pow_succ]
        rw [← geomShift]
        ring

/-- When local transactions achieve strict contraction (`penalty = 0`), regret decays
geometrically to zero at rate `factor ^ steps`. -/
theorem strictBound
    (regret : ℕ → ℝ)
    (factor : ℝ)
    (factorNonnegative : 0 ≤ factor)
    (steps : ℕ)
    (stepwise : ∀ step < steps, regret (step + 1) ≤ factor * regret step) :
    regret steps ≤ factor ^ steps * regret 0 := by
  have general := pathBound regret factor 0 factorNonnegative steps
    (fun step before => by simpa using stepwise step before)
  simpa using general

/-- For strictly contracting factors (`factor < 1`), the geometric series sum is bounded
by `1 / (1 - factor)`, giving an asymptotic regret tolerance `penalty / (1 - factor)`. -/
theorem geomBound
    (regret : ℕ → ℝ)
    (factor penalty : ℝ)
    (factorNonneg : 0 ≤ factor)
    (factorLt : factor < 1)
    (penaltyNonneg : 0 ≤ penalty)
    (steps : ℕ)
    (stepwise : ∀ step < steps,
      regret (step + 1) ≤ factor * regret step + penalty) :
    regret steps ≤ factor ^ steps * regret 0 + penalty / (1 - factor) := by
  have general := pathBound regret factor penalty factorNonneg steps stepwise
  have hpos : 0 < 1 - factor := sub_pos.mpr factorLt
  have hmul : (∑ index ∈ Finset.range steps, factor ^ index) * (1 - factor) =
      1 - factor ^ steps := geom_sum_mul_neg factor steps
  have hpow : 0 ≤ factor ^ steps := pow_nonneg factorNonneg steps
  have hle : (∑ index ∈ Finset.range steps, factor ^ index) * (1 - factor) ≤ 1 := by
    linarith [hmul, hpow]
  have geomLe : ∑ index ∈ Finset.range steps, factor ^ index ≤ 1 / (1 - factor) := by
    exact (le_div_iff₀ hpos).mpr (by linarith [hle])
  have mulLe : penalty * ∑ index ∈ Finset.range steps, factor ^ index ≤
      penalty / (1 - factor) := by
    have step1 := mul_le_mul_of_nonneg_left geomLe penaltyNonneg
    have step2 : penalty * (1 / (1 - factor)) = penalty / (1 - factor) := by ring
    linarith [step1, step2]
  linarith [general, mulLe]

/-- Per-transaction failure bounds yield a high-probability terminal population
regret envelope.  Dependence across adaptive transactions is allowed. -/
theorem regretBound
    [MeasurableSpace World]
    (measure : Measure World)
    [IsFiniteMeasure measure]
    (regret : ℕ → World → ℝ)
    (factor penalty : ℝ)
    (factorNonnegative : 0 ≤ factor)
    (steps : ℕ)
    (failure : Fin steps → ℝ)
    (stepBound : ∀ step : Fin steps,
      measure.real (badAt regret factor penalty step) ≤ failure step) :
    measure.real {world |
        factor ^ steps * regret 0 world +
            penalty * ∑ index ∈ Finset.range steps, factor ^ index <
          regret steps world} ≤
      ∑ step, failure step := by
  calc
    measure.real {world |
        factor ^ steps * regret 0 world +
            penalty * ∑ index ∈ Finset.range steps, factor ^ index <
          regret steps world} ≤
      measure.real (badSet regret factor penalty steps) := by
        refine measureReal_mono ?_ (by finiteness)
        intro world failed
        by_contra clean
        have stepwise := localBound regret factor penalty steps world clean
        have bound := pathBound (fun step => regret step world)
          factor penalty factorNonnegative steps stepwise
        exact (not_lt_of_ge bound) failed
    _ ≤ ∑ step, failure step :=
      unionBound measure regret factor penalty steps failure stepBound

/-- A common per-transaction failure probability accumulates linearly with the
number of natural-window evaluations.  Thus a fixed terminal failure budget
requires the local probability to shrink with the pretraining horizon. -/
theorem flatBound
    [MeasurableSpace World]
    (measure : Measure World)
    [IsFiniteMeasure measure]
    (regret : ℕ → World → ℝ)
    (factor penalty : ℝ)
    (factorNonnegative : 0 ≤ factor)
    (steps : ℕ)
    (failure : ℝ)
    (stepBound : ∀ step : Fin steps,
      measure.real (badAt regret factor penalty step) ≤ failure) :
    measure.real {world |
        factor ^ steps * regret 0 world +
            penalty * ∑ index ∈ Finset.range steps, factor ^ index <
          regret steps world} ≤
      (steps : ℝ) * failure := by
  have bound := regretBound measure regret factor penalty factorNonnegative
    steps (fun _ => failure) stepBound
  simpa using bound

/-- Connecting the multi-transaction regret horizon to the active token clock
(`PretrainWorkload.tokenClock`) and a base-model loss threshold
(`PretrainWorkload.LossGoal`): when the geometric regret envelope clears the
target loss gap after `steps * windowTokens` evaluated tokens, the probability
of failing to reach the loss goal is bounded by the transaction failure sum.

This is a composition theorem, not a pretraining rate for the source optimizer.
In particular, `stepBound` must be proved for the adaptive sequence of commits;
`lossEq` must connect terminal model loss to the stated population regret; and
`envelopeReached` must be established from a feasible token budget and derived
per-step constants.  Supplying these as assumptions cannot by itself certify
that the deployed policy reaches the threshold. -/
theorem goalBound
    [MeasurableSpace World]
    (measure : Measure World)
    [IsFiniteMeasure measure]
    (goal : World → PretrainWorkload.LossGoal)
    (regret : ℕ → World → ℝ)
    (optimalLoss factor penalty : ℝ)
    (factorNonnegative : 0 ≤ factor)
    (steps : ℕ)
    (failure : Fin steps → ℝ)
    (lossEq : ∀ world,
      (goal world).lossAfter (steps * PretrainWorkload.windowTokens) =
        optimalLoss + regret steps world)
    (envelopeReached : ∀ world,
      optimalLoss +
        (factor ^ steps * regret 0 world +
          penalty * ∑ index ∈ Finset.range steps, factor ^ index) ≤
        (goal world).threshold)
    (stepBound : ∀ step : Fin steps,
      measure.real (badAt regret factor penalty step) ≤ failure step) :
    measure.real {world |
        ¬(goal world).reached
          (PretrainWorkload.tokenClock
            (fun _ => PretrainWorkload.windowTokens) steps)} ≤
      ∑ step, failure step := by
  calc
    measure.real {world |
        ¬(goal world).reached
          (PretrainWorkload.tokenClock
            (fun _ => PretrainWorkload.windowTokens) steps)} ≤
      measure.real {world |
        factor ^ steps * regret 0 world +
            penalty * ∑ index ∈ Finset.range steps, factor ^ index <
          regret steps world} := by
        refine measureReal_mono ?_ (by finiteness)
        intro world missed
        simp only [Set.mem_ofPred_eq, PretrainWorkload.LossGoal.reached,
          PretrainWorkload.clockFixed, lossEq, not_le] at missed ⊢
        have target := envelopeReached world
        linarith
    _ ≤ ∑ step, failure step :=
      regretBound measure regret factor penalty factorNonnegative
        steps failure stepBound

#print axioms localBound
#print axioms unionBound
#print axioms pathBound
#print axioms strictBound
#print axioms geomBound
#print axioms regretBound
#print axioms flatBound
#print axioms goalBound

end

end BauxResearch.Optimization.PretrainHorizon
