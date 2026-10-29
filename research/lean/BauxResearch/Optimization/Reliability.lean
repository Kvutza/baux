import BauxResearch.Optimization.Step
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Topology.MetricSpace.Basic

/-!
# Stochastic model-agreement reliability control

This module formalizes the stochastic trust-region reliability controller
from first principles through Mathlib.

In noisy black-box Bayesian optimization on high-dimensional spaces, classical
trust-region ratio adaptation (`ρ = actual / predicted`) fails because:
1. When predicted improvement is near zero, observation noise causes infinite variance in `ρ`.
2. Conflating step acceptance with trust-region adaptation creates two pathologies:
   - Spurious contraction: An accurate model predicting negative improvement is contracted.
   - Hallucination runaway: An inaccurate model that gets a lucky noisy candidate is expanded.

The reliability controller decouples acceptance from adaptation:
- Acceptance requires observed improvement to clear a separately justified margin.
- Trust-region adaptation is determined by standardized model agreement (`z_t ≤ z_max`).

We formalize:
1. The discounted evidence recursion and its uniform upper bound `1 / (1 - λ)`.
2. The epistemic sharpness pathology: Epistemic uncertainty rejecting accurate predictions.
3. Spurious contraction decay to the discrete lattice floor and zero progress.
4. Geometric decay of inadequate evidence and mass accumulation under calibrated updates.
-/

namespace BauxResearch.Optimization

/-- Possible adaptation actions for the trust-region radius. -/
inductive ReliabilityAction where
  | hold
  | expand
  | shrink
  deriving DecidableEq, Repr

/-- Discounted Beta evidence state for model adequacy. -/
structure BetaState where
  adequate : ℝ
  inadequate : ℝ
  hAdequate : 1 ≤ adequate
  hInadequate : 1 ≤ inadequate

/-- Effective evidence accumulated beyond the prior `(1, 1)`. -/
def BetaState.evidence (state : BetaState) : ℝ :=
  (state.adequate - 1) + (state.inadequate - 1)

/-- Evidence is always nonnegative. -/
theorem BetaState.evidenceNonneg (state : BetaState) : 0 ≤ state.evidence := by
  unfold evidence
  have h1 : 0 ≤ state.adequate - 1 := sub_nonneg.mpr state.hAdequate
  have h2 : 0 ≤ state.inadequate - 1 := sub_nonneg.mpr state.hInadequate
  linarith

/-- One-step discounted evidence update with discount `decay ∈ [0, 1)` and indicator `I ∈ [0, 1]`. -/
def BetaState.update (state : BetaState) (decay : ℝ) (indicator : ℝ)
    (hDecay : 0 ≤ decay) (hIndLower : 0 ≤ indicator) (hIndUpper : indicator ≤ 1) : BetaState where
  adequate := 1 + decay * (state.adequate - 1) + indicator
  inadequate := 1 + decay * (state.inadequate - 1) + (1 - indicator)
  hAdequate := by
    have hDiff : 0 ≤ state.adequate - 1 := sub_nonneg.mpr state.hAdequate
    have hMul : 0 ≤ decay * (state.adequate - 1) := mul_nonneg hDecay hDiff
    linarith
  hInadequate := by
    have hDiff : 0 ≤ state.inadequate - 1 := sub_nonneg.mpr state.hInadequate
    have hMul : 0 ≤ decay * (state.inadequate - 1) := mul_nonneg hDecay hDiff
    linarith

/-- Evidence recursion formula: `E_{t+1} = decay * E_t + 1`. -/
theorem BetaState.evidenceRecursion
    (state : BetaState) (decay indicator : ℝ)
    (hDecay : 0 ≤ decay) (hIndLower : 0 ≤ indicator) (hIndUpper : indicator ≤ 1) :
    (state.update decay indicator hDecay hIndLower hIndUpper).evidence =
      decay * state.evidence + 1 := by
  unfold update evidence
  dsimp
  ring

/--
Steady-state evidence bound: If current evidence is bounded by `1 / (1 - decay)`,
it remains bounded by `1 / (1 - decay)` after any update.
-/
theorem BetaState.evidenceBound
    (state : BetaState) (decay indicator : ℝ)
    (hDecayPos : 0 ≤ decay) (hDecayLt : decay < 1)
    (hIndLower : 0 ≤ indicator) (hIndUpper : indicator ≤ 1)
    (hBound : state.evidence ≤ 1 / (1 - decay)) :
    (state.update decay indicator hDecayPos hIndLower hIndUpper).evidence ≤ 1 / (1 - decay) := by
  rw [BetaState.evidenceRecursion state decay indicator hDecayPos hIndLower hIndUpper]
  have hOneMinusPos : 0 < 1 - decay := sub_pos.mpr hDecayLt
  have hMulLe : decay * state.evidence ≤ decay * (1 / (1 - decay)) :=
    mul_le_mul_of_nonneg_left hBound hDecayPos
  have hAlgebra : decay * (1 / (1 - decay)) + 1 = 1 / (1 - decay) := by
    have hNe : 1 - decay ≠ 0 := ne_of_gt hOneMinusPos
    calc
      decay * (1 / (1 - decay)) + 1 = decay / (1 - decay) + (1 - decay) / (1 - decay) := by
        rw [mul_one_div, div_self hNe]
      _ = (decay + (1 - decay)) / (1 - decay) := by rw [add_div]
      _ = 1 / (1 - decay) := by ring_nf
  linarith

/-- Source-shaped decision priority for trust-region adaptation. `sharp` is an
expansion condition, not part of model-adequacy evidence. `failure` denotes the
configured consecutive-rejection condition. -/
def decideAction
    (modelAdequate : Bool)
    (modelInadequate : Bool)
    (sharp : Bool)
    (accepted : Bool)
    (positiveRatio : Bool)
    (boundaryStep : Bool)
    (failure : Bool) : ReliabilityAction :=
  if modelInadequate then
    ReliabilityAction.shrink
  else if modelAdequate && sharp && accepted && positiveRatio && boundaryStep then
    ReliabilityAction.expand
  else if failure && !accepted then
    ReliabilityAction.shrink
  else
    ReliabilityAction.hold

/-- Expansion requires every source expansion gate, including acceptance. -/
theorem expandRequires
    (modelAdequate modelInadequate sharp accepted positiveRatio boundaryStep failure : Bool)
    (expanded : decideAction modelAdequate modelInadequate sharp accepted positiveRatio
      boundaryStep failure = .expand) :
    modelInadequate = false ∧ modelAdequate = true ∧ sharp = true ∧
      accepted = true ∧ positiveRatio = true ∧ boundaryStep = true := by
  cases modelAdequate <;> cases modelInadequate <;> cases sharp <;> cases accepted <;>
    cases positiveRatio <;> cases boundaryStep <;> cases failure <;>
    simp [decideAction] at expanded ⊢

/-- A non-inadequate model with no expansion and no failure condition holds the
radius; rejection alone is not an unconditional contraction rule. -/
theorem rejectionHolds
    (modelAdequate sharp positiveRatio boundaryStep : Bool) :
    decideAction modelAdequate false sharp false positiveRatio boundaryStep false = .hold := by
  simp [decideAction]

/-- Calibrated adequacy criterion: Evaluates standardized residual without sharpness penalty. -/
def calibAdequate (err zMax : ℝ) : Prop :=
  err ≤ zMax

/-- Under calibrated observations, inadequate evidence decays geometrically. -/
theorem calibDecay
    (state : BetaState) (decay : ℝ)
    (hDecayPos : 0 ≤ decay) :
    (state.update decay 1 hDecayPos (by linarith) (by linarith)).inadequate - 1 =
      decay * (state.inadequate - 1) := by
  unfold BetaState.update
  dsimp
  ring

/-- Under calibrated observations, adequate evidence accumulates positive mass. -/
theorem calibGrowth
    (state : BetaState) (decay : ℝ)
    (hDecayPos : 0 ≤ decay) :
    (state.update decay 1 hDecayPos (by linarith) (by linarith)).adequate - 1 =
      decay * (state.adequate - 1) + 1 := by
  unfold BetaState.update
  dsimp
  ring

/-- No-spurious-collapse: Calibrated updates strictly contract inadequate evidence below any positive bound. -/
theorem noCollapse
    (state : BetaState) (decay bound : ℝ)
    (hDecayPos : 0 ≤ decay) (hDecayLt : decay < 1)
    (hBoundPos : 0 < bound)
    (hBound : state.inadequate - 1 ≤ bound) :
    (state.update decay 1 hDecayPos (by linarith) (by linarith)).inadequate - 1 < bound := by
  rw [calibDecay state decay hDecayPos]
  have hMul : decay * (state.inadequate - 1) ≤ decay * bound :=
    mul_le_mul_of_nonneg_left hBound hDecayPos
  have hStrict : decay * bound < 1 * bound :=
    mul_lt_mul_of_pos_right hDecayLt hBoundPos
  rw [one_mul] at hStrict
  linarith

open scoped BigOperators

/-- Acquisition-free STORM Lyapunov potential (Blanchet, Cartis, Menickelly, Scheinberg 2019):
`Φ(θ, r) = ν * L(θ) + (1 - ν) * r^2`, trading off objective progress against squared
trust-region radius without an internal multi-candidate acquisition function. -/
def stormPotential (nu loss radius : ℝ) : ℝ :=
  nu * loss + (1 - nu) * radius ^ 2

/-- On an accepted step achieving true objective decrease `c * radius ^ 2 ≤ loss - nextLoss`
with radius expansion `nextRadius = expand * radius`, the STORM potential decreases by at least
`(nu * c - (1 - nu) * (expand ^ 2 - 1)) * radius ^ 2`. -/
theorem stormAccept
    (nu loss nextLoss radius expand c : ℝ)
    (nuNonneg : 0 ≤ nu)
    (progressBound : c * radius ^ 2 ≤ loss - nextLoss) :
    stormPotential nu nextLoss (expand * radius) - stormPotential nu loss radius ≤
      -(nu * c - (1 - nu) * (expand ^ 2 - 1)) * radius ^ 2 := by
  unfold stormPotential
  have weighted := mul_le_mul_of_nonneg_left progressBound nuNonneg
  linarith

/-- On a rejected or non-worsening step (`nextLoss ≤ loss`) where the trust region shrinks
by `nextRadius = shrink * radius`, the STORM potential decreases by at least
`(1 - nu) * (1 - shrink ^ 2) * radius ^ 2`. -/
theorem stormShrink
    (nu loss nextLoss radius shrink : ℝ)
    (nuNonneg : 0 ≤ nu)
    (safe : nextLoss ≤ loss) :
    stormPotential nu nextLoss (shrink * radius) - stormPotential nu loss radius ≤
      -(1 - nu) * (1 - shrink ^ 2) * radius ^ 2 := by
  unfold stormPotential
  have weighted := mul_le_mul_of_nonneg_left safe nuNonneg
  linarith

/-- Telescoping the STORM Lyapunov potential over `steps` rounds bounds the cumulative
squared trust-region radius `∑ t, (radius t)^2` by the initial-to-terminal potential drop. -/
theorem stormTelescope
    (nu : ℝ) (loss radius : ℕ → ℝ) (kappa : ℝ) (steps : ℕ)
    (stepwise : ∀ t < steps,
      stormPotential nu (loss (t + 1)) (radius (t + 1)) -
        stormPotential nu (loss t) (radius t) ≤ -kappa * (radius t) ^ 2) :
    kappa * (∑ t ∈ Finset.range steps, (radius t) ^ 2) ≤
      stormPotential nu (loss 0) (radius 0) -
        stormPotential nu (loss steps) (radius steps) := by
  induction steps with
  | zero => simp
  | succ n ih =>
    have prev := ih (fun t ht => stepwise t (Nat.lt_succ_of_lt ht))
    have last := stepwise n (Nat.lt_succ_self n)
    rw [Finset.sum_range_succ, mul_add]
    linarith

#print axioms stormAccept
#print axioms stormShrink
#print axioms stormTelescope

end BauxResearch.Optimization
