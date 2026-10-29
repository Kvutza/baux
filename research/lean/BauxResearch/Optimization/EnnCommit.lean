import BauxResearch.Optimization.Accuracy
import BauxResearch.Optimization.Regret
import BauxResearch.Forward.LossGeometry
import BauxResearch.Forward.Sensitivity

/-!
# A retained ENN mean in the one-candidate commit

The incumbent reward is the source weighted neighbor mean, including its
denominator floor. The deployed pretraining reward is negative NLL. The scalar
decision accepts when the candidate reward strictly exceeds the predicted
incumbent reward by its threshold. Vector control can override this decision;
the theorem below concerns the scalar path only.
-/

namespace BauxResearch.Optimization.EnnCommit

noncomputable section
open scoped BigOperators
open BauxResearch.Forward

/-- Weighted local observation and geometry error, with the zero-anchor bias
induced by the source denominator floor. -/
def errorRadius {capacity : ℕ} (data : Neighbor.Data capacity)
    (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (nearest : List (Fin capacity))
    (truth : ℝ) (noise geometry : Vector capacity) : ℝ :=
  ((∑ position : Fin nearest.length,
      Neighbor.weight data params scale distances nearest[position.val] *
        (noise nearest[position.val] + geometry nearest[position.val])) +
    (Accuracy.mass data params scale distances nearest -
      Accuracy.weightTotal data params scale distances nearest) * |truth|) /
    Accuracy.mass data params scale distances nearest

/-- The actual source reward mean is accurate when each retained measurement
and its local transfer to the incumbent have separately certified errors. -/
theorem baseBound {State : Type*} {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (nearest : List (Fin capacity))
    (objective : Objective State) (points : Fin capacity → State) (incumbent : State)
    (noise geometry : Vector capacity)
    (weights : ∀ position : Fin nearest.length,
      0 ≤ Neighbor.weight data params scale distances nearest[position.val])
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] -
        objective (points nearest[position.val])| ≤ noise nearest[position.val])
    (near : ∀ position : Fin nearest.length,
      |objective (points nearest[position.val]) - objective incumbent| ≤
        geometry nearest[position.val]) :
    |Neighbor.mean data params scale distances nearest - objective incumbent| ≤
      errorRadius data params scale distances nearest (objective incumbent)
        noise geometry := by
  have pointBound : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] - objective incumbent| ≤
        noise nearest[position.val] + geometry nearest[position.val] := by
    intro position
    exact (abs_sub_le _ _ _).trans (add_le_add (observed position) (near position))
  have weighted :
      (∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] *
          |data.values nearest[position.val] - objective incumbent|) ≤
      ∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] *
          (noise nearest[position.val] + geometry nearest[position.val]) := by
    apply Finset.sum_le_sum
    intro position _
    exact mul_le_mul_of_nonneg_left (pointBound position) (weights position)
  have positive : 0 < Accuracy.mass data params scale distances nearest :=
    lt_of_lt_of_le (by norm_num : (0 : ℝ) < 1 / 10^12) (le_max_left _ _)
  have source := Accuracy.clamped data params scale distances nearest
    (objective incumbent) weights
  exact source.trans (by
    unfold errorRadius
    apply div_le_div_of_nonneg_right _ positive.le
    exact add_le_add_left weighted _)

/-- A per-token logit displacement supplies the local reward geometry needed
by the source ENN mean for a fixed aligned evaluation sequence. The factor two
is independent of vocabulary size and the number of scored tokens. -/
theorem logitBase {State : Type*} {capacity tokens width : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (nearest : List (Fin capacity))
    (points : Fin capacity → State) (incumbent : State)
    (logits : State → Rows tokens width) (targets : Fin tokens → Fin width)
    (noise : Vector capacity) (radius : Fin capacity → Vector tokens)
    (weights : ∀ position : Fin nearest.length,
      0 ≤ Neighbor.weight data params scale distances nearest[position.val])
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] +
        average (fun token => logLoss
          (logits (points nearest[position.val]) token) (targets token))| ≤
        noise nearest[position.val])
    (bounded : ∀ (position : Fin nearest.length) token index,
      |logits (points nearest[position.val]) token index -
        logits incumbent token index| ≤ radius nearest[position.val] token) :
    |Neighbor.mean data params scale distances nearest +
      average (fun token => logLoss (logits incumbent token) (targets token))| ≤
      errorRadius data params scale distances nearest
        (-average (fun token => logLoss (logits incumbent token) (targets token)))
        noise (fun row => 2 * average (radius row)) := by
  let objective : State → ℝ := fun state =>
    -average (fun token => logLoss (logits state token) (targets token))
  have observedReward : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] -
        objective (points nearest[position.val])| ≤ noise nearest[position.val] := by
    intro position
    simpa only [objective, sub_neg_eq_add] using observed position
  have near : ∀ position : Fin nearest.length,
      |objective (points nearest[position.val]) - objective incumbent| ≤
        (fun row => 2 * average (radius row)) nearest[position.val] := by
    intro position
    change |(-average (fun token => logLoss
        (logits (points nearest[position.val]) token) (targets token))) -
      (-average (fun token => logLoss (logits incumbent token) (targets token)))| ≤ _
    calc
      _ = |average (fun token => logLoss (logits incumbent token) (targets token)) -
          average (fun token => logLoss
            (logits (points nearest[position.val]) token) (targets token))| := by
            congr 1
            ring
      _ = |average (fun token => logLoss
            (logits (points nearest[position.val]) token) (targets token)) -
          average (fun token => logLoss (logits incumbent token) (targets token))| :=
            abs_sub_comm _ _
      _ ≤ 2 * average (radius nearest[position.val]) :=
        LossGeometry.averageBound _ _ targets _ (bounded position)
  simpa only [objective, sub_neg_eq_add] using
    (baseBound data params scale distances nearest objective points incumbent
      noise (fun row => 2 * average (radius row)) weights observedReward near)

/-- The scalar acceptance law: measured reward gain must strictly clear the
declared error margin.  The margin is an obligation of the statistical model;
this definition does not manufacture one from a variance estimate. -/
abbrev accepts (improvement margin : ℝ) : Prop :=
  margin < improvement

/-- The deployed scalar reward comparison, with its strict error margin. -/
def scalarCommit {State : Type*} (incumbent candidate : State)
    (baseReward candidateReward margin : ℝ) : State :=
  if accepts (candidateReward - baseReward) margin then candidate else incumbent

/-- When the acceptance margin covers the ENN incumbent prediction error plus
candidate observation error, `scalarCommit` is non-worsening (`0 ≤ progress`). -/
theorem commitSafe {State : Type*} {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (nearest : List (Fin capacity))
    (objective : Objective State) (points : Fin capacity → State)
    (incumbent candidate : State)
    (noise geometry : Vector capacity) (candidateReward : ℝ)
    (candidateError margin : ℝ)
    (weights : ∀ position : Fin nearest.length,
      0 ≤ Neighbor.weight data params scale distances nearest[position.val])
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] -
        -objective (points nearest[position.val])| ≤ noise nearest[position.val])
    (near : ∀ position : Fin nearest.length,
      |-objective (points nearest[position.val]) - -objective incumbent| ≤
        geometry nearest[position.val])
    (candidateAccurate :
      |candidateReward + objective candidate| ≤ candidateError)
    (safe : errorRadius data params scale distances nearest
        (-objective incumbent) noise geometry + candidateError ≤ margin) :
    0 ≤ progress objective incumbent
      (scalarCommit incumbent candidate
        (Neighbor.mean data params scale distances nearest) candidateReward margin) := by
  have baseAccurate := baseBound data params scale distances nearest
    (fun state => -objective state) points incumbent noise geometry weights
    observed near
  by_cases accepted : accepts
      (candidateReward - Neighbor.mean data params scale distances nearest) margin
  · simp only [scalarCommit, accepted, ite_true]
    have baseLower := (abs_le.mp baseAccurate).1
    have candidateUpper := (abs_le.mp candidateAccurate).2
    unfold accepts at accepted
    unfold progress
    linarith
  · simp only [scalarCommit, accepted, ite_false]
    unfold progress
    linarith

/-- One useful trust-region candidate is committed when its NLL gain strictly
clears the source ENN reward error, candidate error, and threshold. -/
theorem commitBound {State : Type*} {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (nearest : List (Fin capacity))
    (objective : Objective State) (points : Fin capacity → State)
    (incumbent candidate : State)
    (noise geometry : Vector capacity) (candidateReward : ℝ)
    (gain candidateError margin : ℝ)
    (weights : ∀ position : Fin nearest.length,
      0 ≤ Neighbor.weight data params scale distances nearest[position.val])
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] -
        -objective (points nearest[position.val])| ≤ noise nearest[position.val])
    (near : ∀ position : Fin nearest.length,
      |-objective (points nearest[position.val]) - -objective incumbent| ≤
        geometry nearest[position.val])
    (useful : gain ≤ progress objective incumbent candidate)
    (candidateAccurate :
      |candidateReward + objective candidate| ≤ candidateError)
    (separated : errorRadius data params scale distances nearest
        (-objective incumbent) noise geometry + candidateError + margin < gain) :
    gain ≤ progress objective incumbent
      (scalarCommit incumbent candidate
        (Neighbor.mean data params scale distances nearest) candidateReward margin) := by
  have baseAccurate := baseBound data params scale distances nearest
    (fun state => -objective state) points incumbent noise geometry weights
    observed near
  have baseUpper := (abs_le.mp baseAccurate).2
  have candidateLower := (abs_le.mp candidateAccurate).1
  have accepted : accepts
      (candidateReward - Neighbor.mean data params scale distances nearest) margin := by
    unfold accepts
    unfold progress at useful
    linarith
  simp only [scalarCommit, accepted, ite_true]
  exact useful

/-- The same source ENN decision contracts simple regret when the candidate's
gain covers a stated fraction of current regret. -/
theorem regretBound {State : Type*} {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (nearest : List (Fin capacity))
    (objective : Objective State) (points : Fin capacity → State)
    (incumbent candidate best : State)
    (noise geometry : Vector capacity) (candidateReward : ℝ)
    (gain candidateError margin gainShare loss : ℝ)
    (weights : ∀ position : Fin nearest.length,
      0 ≤ Neighbor.weight data params scale distances nearest[position.val])
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] -
        -objective (points nearest[position.val])| ≤ noise nearest[position.val])
    (near : ∀ position : Fin nearest.length,
      |-objective (points nearest[position.val]) - -objective incumbent| ≤
        geometry nearest[position.val])
    (useful : gain ≤ progress objective incumbent candidate)
    (candidateAccurate :
      |candidateReward + objective candidate| ≤ candidateError)
    (separated : errorRadius data params scale distances nearest
        (-objective incumbent) noise geometry + candidateError + margin < gain)
    (retained : gainShare * Regret.simpleRegret objective best incumbent - loss ≤ gain) :
    Regret.simpleRegret objective best
      (scalarCommit incumbent candidate
        (Neighbor.mean data params scale distances nearest) candidateReward margin) ≤
      (1 - gainShare) * Regret.simpleRegret objective best incumbent + loss := by
  have committed := commitBound data params scale distances nearest objective
    points incumbent candidate noise geometry candidateReward gain
    candidateError margin weights observed near useful candidateAccurate separated
  simp only [Regret.simpleRegret, progress] at retained committed ⊢
  linarith

/-- Multi-layer forward schedule sensitivity (`Sensitivity.scheduleBound`)
composed with a Lipschitz decoder head supplies the per-token logit radius
consumed by `logitBase`, bounding the source ENN mean reward error directly
from the layer schedule recurrence `Sensitivity.foldBound`. -/
theorem scheduleBase {State Hidden Layer : Type*} [PseudoMetricSpace Hidden]
    {capacity tokens width : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (nearest : List (Fin capacity))
    (points : Fin capacity → State) (incumbent : State)
    (schedule : List Layer)
    (embed : State → Fin tokens → Hidden)
    (step : State → Hidden → Layer → Hidden)
    (decode : State → Hidden → Vector width)
    (targets : Fin tokens → Fin width)
    (noise : Vector capacity)
    (layerScale layerShift : Fin capacity → Fin tokens → Layer → ℝ)
    (initial decodeShift : Fin capacity → Vector tokens)
    (decodeLip : ℝ) (decodeNonneg : 0 ≤ decodeLip)
    (weights : ∀ position : Fin nearest.length,
      0 ≤ Neighbor.weight data params scale distances nearest[position.val])
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] +
        average (fun token => logLoss
          (decode (points nearest[position.val])
            (schedule.foldl (step (points nearest[position.val]))
              (embed (points nearest[position.val]) token)))
          (targets token))| ≤
        noise nearest[position.val])
    (scaleNonneg : ∀ (position : Fin nearest.length) token,
      ∀ layer ∈ schedule, 0 ≤ layerScale nearest[position.val] token layer)
    (embedBound : ∀ (position : Fin nearest.length) token,
      dist (embed (points nearest[position.val]) token)
        (embed incumbent token) ≤ initial nearest[position.val] token)
    (stepBound : ∀ (position : Fin nearest.length) token,
      ∀ layer ∈ schedule, ∀ before after : Hidden,
        dist (step (points nearest[position.val]) after layer)
          (step incumbent before layer) ≤
          layerScale nearest[position.val] token layer * dist after before +
            layerShift nearest[position.val] token layer)
    (decodeBound : ∀ (position : Fin nearest.length) token (before after : Hidden) index,
      |decode (points nearest[position.val]) after index -
        decode incumbent before index| ≤
        decodeLip * dist after before + decodeShift nearest[position.val] token) :
    |Neighbor.mean data params scale distances nearest +
      average (fun token => logLoss
        (decode incumbent
          (schedule.foldl (step incumbent) (embed incumbent token)))
        (targets token))| ≤
      errorRadius data params scale distances nearest
        (-average (fun token => logLoss
          (decode incumbent
            (schedule.foldl (step incumbent) (embed incumbent token)))
          (targets token)))
        noise (fun row => 2 * average (fun token =>
          decodeLip * Sensitivity.foldBound
            (layerScale row token) (layerShift row token) schedule (initial row token) +
            decodeShift row token)) := by
  apply logitBase data params scale distances nearest points incumbent
    (fun state token => decode state (schedule.foldl (step state) (embed state token)))
    targets noise
    (fun row token =>
      decodeLip * Sensitivity.foldBound
        (layerScale row token) (layerShift row token) schedule (initial row token) +
        decodeShift row token)
    weights observed
  intro position token index
  have hiddenBound := Sensitivity.scheduleBound
    (step incumbent) (step (points nearest[position.val]))
    (layerScale nearest[position.val] token)
    (layerShift nearest[position.val] token)
    schedule (scaleNonneg position token) (stepBound position token)
    (embed incumbent token) (embed (points nearest[position.val]) token)
    (initial nearest[position.val] token) (embedBound position token)
  have headBound := decodeBound position token
    (schedule.foldl (step incumbent) (embed incumbent token))
    (schedule.foldl (step (points nearest[position.val]))
      (embed (points nearest[position.val]) token))
    index
  linarith [mul_le_mul_of_nonneg_left hiddenBound decodeNonneg]

/-- Window-to-window difficulty shift between a historical corpus window and
the current evaluation window. -/
def windowShift {State : Type*} (window current : Objective State)
    (state : State) : ℝ :=
  window state - current state

/-- Exact decomposition of a historical window score into the current window
score plus the window-to-window difficulty shift. -/
theorem shiftSplit {State : Type*} (window current : Objective State)
    (state : State) :
    window state = current state + windowShift window current state := by
  unfold windowShift
  ring

/-- When historical observations on window `s` are centered against a window
baseline `base s` whose shift `base s - currentBase` tracks the incumbent
window shift `windowShift (window s) current incumbent` within `delta s`, the
common window difficulty shift cancels out of the ENN incumbent prediction
error. Setting `base = 0` and `currentBase = 0` recovers the uncentered window
shift bound where raw window drift enters `errorRadius` directly. -/
theorem windowError {State : Type*} {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (nearest : List (Fin capacity))
    (window : Fin capacity → Objective State) (current : Objective State)
    (points : Fin capacity → State) (incumbent : State)
    (base : Vector capacity) (currentBase : ℝ)
    (noise delta geometry : Vector capacity)
    (weights : ∀ position : Fin nearest.length,
      0 ≤ Neighbor.weight data params scale distances nearest[position.val])
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] -
        (window nearest[position.val] (points nearest[position.val]) -
          base nearest[position.val])| ≤
        noise nearest[position.val])
    (near : ∀ position : Fin nearest.length,
      |window nearest[position.val] (points nearest[position.val]) -
        window nearest[position.val] incumbent| ≤
        geometry nearest[position.val])
    (tracked : ∀ position : Fin nearest.length,
      |windowShift (window nearest[position.val]) current incumbent -
        (base nearest[position.val] - currentBase)| ≤
        delta nearest[position.val]) :
    |Neighbor.mean data params scale distances nearest -
      (current incumbent - currentBase)| ≤
      errorRadius data params scale distances nearest
        (current incumbent - currentBase)
        (fun row => noise row + delta row) geometry := by
  have pointBound : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] - (current incumbent - currentBase)| ≤
        (noise nearest[position.val] + delta nearest[position.val]) +
          geometry nearest[position.val] := by
    intro position
    let row := nearest[position.val]
    have split :
        data.values row - (current incumbent - currentBase) =
          (data.values row - (window row (points row) - base row)) +
            (windowShift (window row) current incumbent -
              (base row - currentBase)) +
            (window row (points row) - window row incumbent) := by
      unfold windowShift
      ring
    rw [split]
    have first := (abs_add_le _ _).trans
      (add_le_add (observed position) (tracked position))
    exact (abs_add_le _ _).trans (add_le_add first (near position))
  have weighted :
      (∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] *
          |data.values nearest[position.val] -
            (current incumbent - currentBase)|) ≤
      ∑ position : Fin nearest.length,
        Neighbor.weight data params scale distances nearest[position.val] *
          ((noise nearest[position.val] + delta nearest[position.val]) +
            geometry nearest[position.val]) := by
    apply Finset.sum_le_sum
    intro position _
    exact mul_le_mul_of_nonneg_left (pointBound position) (weights position)
  have positive : 0 < Accuracy.mass data params scale distances nearest :=
    lt_of_lt_of_le (by norm_num : (0 : ℝ) < 1 / 10^12) (le_max_left _ _)
  have source := Accuracy.clamped data params scale distances nearest
    (current incumbent - currentBase) weights
  exact source.trans (by
    unfold errorRadius
    apply div_le_div_of_nonneg_right _ positive.le
    exact add_le_add_left weighted _)

/-- Window-baseline-centered ENN commit guarantee: common window difficulty
shifts cancel out of both the incumbent prediction and the current-window
candidate comparison, so the acceptance condition pays only the residual shift
tracking error `delta` rather than the raw window-to-window NLL shift. -/
theorem windowCommit {State : Type*} {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters) (scale : ℝ)
    (distances : Vector capacity) (nearest : List (Fin capacity))
    (window : Fin capacity → Objective State) (objective : Objective State)
    (points : Fin capacity → State)
    (incumbent candidate : State)
    (base : Vector capacity) (currentBase : ℝ)
    (noise delta geometry : Vector capacity) (candidateReward : ℝ)
    (gain candidateError margin : ℝ)
    (weights : ∀ position : Fin nearest.length,
      0 ≤ Neighbor.weight data params scale distances nearest[position.val])
    (observed : ∀ position : Fin nearest.length,
      |data.values nearest[position.val] -
        (-window nearest[position.val] (points nearest[position.val]) -
          base nearest[position.val])| ≤
        noise nearest[position.val])
    (near : ∀ position : Fin nearest.length,
      |-window nearest[position.val] (points nearest[position.val]) -
        -window nearest[position.val] incumbent| ≤
        geometry nearest[position.val])
    (tracked : ∀ position : Fin nearest.length,
      |windowShift (fun state => -window nearest[position.val] state)
        (fun state => -objective state) incumbent -
        (base nearest[position.val] - currentBase)| ≤
        delta nearest[position.val])
    (useful : gain ≤ progress objective incumbent candidate)
    (candidateAccurate :
      |candidateReward - (-objective candidate - currentBase)| ≤
        candidateError)
    (separated : errorRadius data params scale distances nearest
        (-objective incumbent - currentBase)
        (fun row => noise row + delta row) geometry +
        candidateError + margin < gain) :
    gain ≤ progress objective incumbent
      (scalarCommit incumbent candidate
        (Neighbor.mean data params scale distances nearest)
        candidateReward margin) := by
  have baseAccurate := windowError data params scale distances nearest
    (fun row state => -window row state) (fun state => -objective state)
    points incumbent base currentBase noise delta geometry
    weights observed near tracked
  have baseUpper := (abs_le.mp baseAccurate).2
  have candidateLower := (abs_le.mp candidateAccurate).1
  have accepted : accepts
      (candidateReward - Neighbor.mean data params scale distances nearest) margin := by
    unfold accepts
    unfold progress at useful
    linarith
  simp only [scalarCommit, accepted, ite_true]
  exact useful

/-- Observation world provenance (Section 6.3 of
`TYPED_TENSOR_INFERENCE_AND_OPTIMIZER_ARCHITECTURE.txt`, Obligations P0 & P2). -/
inductive WorldId
  | fixedCorpus (seed tokens : ℕ)
  | rotatingCondition (step windowId tokens : ℕ)
  deriving DecidableEq

/-- Paired improvement evaluated on a shared window objective `window`. -/
def pairedImprovement {State : Type*} (window : Objective State)
    (incumbent challenger : State) : ℝ :=
  window challenger - window incumbent

/-- Obligation P2(a): Evaluating challenger and incumbent on the same `WorldId`
cancels the baseline window difficulty shift `windowShift window current incumbent`,
leaving only the differential shift between challenger and incumbent. -/
theorem pairedShift {State : Type*} (window current : Objective State)
    (incumbent challenger : State) :
    pairedImprovement window incumbent challenger =
      (current challenger - current incumbent) +
        (windowShift window current challenger -
          windowShift window current incumbent) := by
  unfold pairedImprovement windowShift
  ring

/-- Obligation P2(b): When a window difficulty shift is additive (`window x =
current x + shift` on the local pair), paired evaluation on the same `WorldId`
recovers the exact reference improvement with zero window-shift bias. -/
theorem constantShift {State : Type*} (window current : Objective State)
    (incumbent challenger : State) (shift : ℝ)
    (incumbentShift : windowShift window current incumbent = shift)
    (challengerShift : windowShift window current challenger = shift) :
    pairedImprovement window incumbent challenger =
      current challenger - current incumbent := by
  rw [pairedShift window current incumbent challenger,
    incumbentShift, challengerShift]
  ring

/-- Exact telescoping sum of anchored single-world step differences along a
trajectory `trajectory : ℕ → State` (`ExactAnchored`, Obligation P3(a)). -/
theorem chainTelescope {State : Type*} (world : Objective State)
    (trajectory : ℕ → State) (steps : ℕ) :
    (∑ t ∈ Finset.range steps,
      pairedImprovement world (trajectory t) (trajectory (t + 1))) =
      world (trajectory steps) - world (trajectory 0) := by
  induction steps with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ, ih]
    unfold pairedImprovement
    ring

/-- Accumulated bias of a `DriftedChain` where each step delta `stepDelta t`
deviates from the reference improvement `current (trajectory (t + 1)) -
current (trajectory t)` by `drift t` (Obligation P3(b)). -/
theorem chainBias {State : Type*} (current : Objective State)
    (trajectory : ℕ → State) (stepDelta drift : ℕ → ℝ) (steps : ℕ)
    (stepBound : ∀ t ∈ Finset.range steps,
      |stepDelta t - pairedImprovement current (trajectory t) (trajectory (t + 1))| ≤
        drift t) :
    |(∑ t ∈ Finset.range steps, stepDelta t) -
      (current (trajectory steps) - current (trajectory 0))| ≤
      ∑ t ∈ Finset.range steps, drift t := by
  rw [← chainTelescope current trajectory steps, ← Finset.sum_sub_distrib]
  exact (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum stepBound)

#print axioms baseBound
#print axioms logitBase
#print axioms commitSafe
#print axioms commitBound
#print axioms regretBound
#print axioms scheduleBase
#print axioms shiftSplit
#print axioms windowError
#print axioms windowCommit
#print axioms pairedShift
#print axioms constantShift
#print axioms chainTelescope
#print axioms chainBias

end
end BauxResearch.Optimization.EnnCommit
