import BauxResearch.Optimization.Reachability

/-!
# Metric structure for history-conditioned proposals

An unrestricted objective admits no universal useful proposal.  This module
states one derivative-free objective class on which a proposal can retain a
fixed share of simple regret: objective regret is comparable to distance from
an optimum, and the proposed state contracts that distance.

The metric need not be the ambient Euclidean metric.  It may encode a quotient,
a learned representation, a product geometry, or another task-specific state
space, provided the two-sided growth and proposal contraction are proved.
-/

namespace BauxResearch.Optimization.Guide

open Reachability

variable {State Move : Type*}

/-- Objective regret is bounded above and below by metric distance from the
declared optimum.  Both inequalities are explicit objective-class premises. -/
def MetricGrowth [PseudoMetricSpace State]
    (objective : Objective State) (best : State) (lower upper : ℝ) : Prop :=
  ∀ state,
    lower * dist state best ≤ Regret.simpleRegret objective best state ∧
    Regret.simpleRegret objective best state ≤ upper * dist state best

/-- A proposed state contracts distance to the declared optimum relative to
the incumbent.  The optimizer does not receive `best`; this is a proof-side
property to be established from history and the objective class. -/
def DistanceContract [PseudoMetricSpace State]
    (incumbent proposed best : State) (ratio : ℝ) : Prop :=
  dist proposed best ≤ ratio * dist incumbent best

/-- Two-sided metric growth transports distance contraction into simple-regret
contraction without a derivative, convexity, or ambient dimension premise. -/
theorem metricRegret [PseudoMetricSpace State]
    (objective : Objective State) (best incumbent proposed : State)
    (lower upper ratio factor : ℝ)
    (growth : MetricGrowth objective best lower upper)
    (contract : DistanceContract incumbent proposed best ratio)
    (upperNonneg : 0 ≤ upper) (factorNonneg : 0 ≤ factor)
    (rates : upper * ratio ≤ factor * lower) :
    Regret.simpleRegret objective best proposed ≤
      factor * Regret.simpleRegret objective best incumbent := by
  calc
    Regret.simpleRegret objective best proposed ≤ upper * dist proposed best :=
      (growth proposed).2
    _ ≤ upper * (ratio * dist incumbent best) :=
      mul_le_mul_of_nonneg_left contract upperNonneg
    _ = (upper * ratio) * dist incumbent best := by ring
    _ ≤ (factor * lower) * dist incumbent best :=
      mul_le_mul_of_nonneg_right rates dist_nonneg
    _ = factor * (lower * dist incumbent best) := by ring
    _ ≤ factor * Regret.simpleRegret objective best incumbent :=
      mul_le_mul_of_nonneg_left (growth incumbent).1 factorNonneg

/-- The same metric premises expose the retained fraction of incumbent regret
as true objective progress. -/
theorem metricGain [PseudoMetricSpace State]
    (objective : Objective State) (best incumbent proposed : State)
    (lower upper ratio factor : ℝ)
    (growth : MetricGrowth objective best lower upper)
    (contract : DistanceContract incumbent proposed best ratio)
    (upperNonneg : 0 ≤ upper) (factorNonneg : 0 ≤ factor)
    (rates : upper * ratio ≤ factor * lower) :
    (1 - factor) * Regret.simpleRegret objective best incumbent ≤
      progress objective incumbent proposed := by
  have bound := metricRegret objective best incumbent proposed lower upper
    ratio factor growth contract upperNonneg factorNonneg rates
  calc
    (1 - factor) * Regret.simpleRegret objective best incumbent =
        Regret.simpleRegret objective best incumbent -
          factor * Regret.simpleRegret objective best incumbent := by ring
    _ ≤ Regret.simpleRegret objective best incumbent -
        Regret.simpleRegret objective best proposed :=
      sub_le_sub_left bound _
    _ = progress objective incumbent proposed := by
      unfold Regret.simpleRegret progress
      ring

/-- A history-derived target produces a perturbation, not an updated state.
The distinction is explicit so execution backends can refine the perturbation
and commit operations separately. -/
def anchorMove {Space : Type*} [NormedAddCommGroup Space]
    [NormedSpace ℝ Space] (step : ℝ) (incumbent anchor : Space) : Space :=
  step • (anchor - incumbent)

/-- Apply the target-directed perturbation to the incumbent. -/
def anchorState {Space : Type*} [NormedAddCommGroup Space]
    [NormedSpace ℝ Space] (step : ℝ) (incumbent anchor : Space) : Space :=
  incumbent + anchorMove step incumbent anchor

/-- For squared distance to a proof-side optimum, one dense perturbation has
an exact derivative-free progress identity.  Its coordinates may have either
sign; the useful quantity is alignment with the optimum displacement. -/
theorem squareProgress {Space : Type*} [NormedAddCommGroup Space]
    [InnerProductSpace ℝ Space]
    (best incumbent direction : Space) (step : ℝ) :
    progress (fun state => dist state best ^ 2) incumbent
        (incumbent + step • direction) =
      2 * step * inner ℝ (best - incumbent) direction -
        step ^ 2 * ‖direction‖ ^ 2 := by
  unfold progress
  change dist incumbent best ^ 2 -
      dist (incumbent + step • direction) best ^ 2 = _
  rw [dist_eq_norm, dist_eq_norm]
  have stateEq : incumbent + step • direction - best =
      (incumbent - best) + step • direction := by
    abel
  rw [stateEq, norm_add_sq_real, real_inner_smul_right, norm_smul]
  have innerEq : inner ℝ (incumbent - best) direction =
      -inner ℝ (best - incumbent) direction := by
    simp only [inner_sub_left]
    ring
  rw [innerEq, Real.norm_eq_abs, mul_pow, sq_abs]
  ring

/-- Positive alignment exceeding the step-energy penalty makes a single
target-directed perturbation strictly improving for squared distance. -/
theorem alignedStep {Space : Type*} [NormedAddCommGroup Space]
    [InnerProductSpace ℝ Space]
    (best incumbent direction : Space) (step : ℝ)
    (stepPositive : 0 < step)
    (aligned : step * ‖direction‖ ^ 2 <
      2 * inner ℝ (best - incumbent) direction) :
    0 < progress (fun state => dist state best ^ 2) incumbent
      (incumbent + step • direction) := by
  rw [squareProgress]
  nlinarith

/-- Under the Roberts-Royer and Jin-Cen-Tan directional alignment condition
`inner ℝ (best - incumbent) direction ≥ kappa * ‖best - incumbent‖ * ‖direction‖`,
taking step `step = kappa * ‖best - incumbent‖ / ‖direction‖` yields
`kappa ^ 2 * ‖best - incumbent‖ ^ 2` progress, achieving linear distance contraction
independent of the ambient dimension. -/
theorem steerProgress {Space : Type*} [NormedAddCommGroup Space]
    [InnerProductSpace ℝ Space]
    (best incumbent direction : Space) (kappa : ℝ)
    (positiveDir : 0 < ‖direction‖) (nonnegKappa : 0 ≤ kappa)
    (aligned : kappa * ‖best - incumbent‖ * ‖direction‖ ≤
      inner ℝ (best - incumbent) direction) :
    kappa ^ 2 * ‖best - incumbent‖ ^ 2 ≤
      progress (fun state => dist state best ^ 2) incumbent
        (incumbent + (kappa * ‖best - incumbent‖ / ‖direction‖) • direction) := by
  rw [squareProgress]
  have posDirSq : 0 < ‖direction‖ ^ 2 := sq_pos_of_pos positiveDir
  have dirNe : ‖direction‖ ≠ 0 := ne_of_gt positiveDir
  have stepEq : (kappa * ‖best - incumbent‖ / ‖direction‖) ^ 2 * ‖direction‖ ^ 2 =
      kappa ^ 2 * ‖best - incumbent‖ ^ 2 := by
    rw [div_pow, div_mul_cancel₀ _ (ne_of_gt posDirSq)]
    ring
  rw [stepEq]
  have crossTerm :
      2 * kappa ^ 2 * ‖best - incumbent‖ ^ 2 ≤
        2 * (kappa * ‖best - incumbent‖ / ‖direction‖) *
          inner ℝ (best - incumbent) direction := by
    have mulNonneg : 0 ≤ 2 * (kappa * ‖best - incumbent‖ / ‖direction‖) := by
      refine mul_nonneg (by norm_num) (div_nonneg (mul_nonneg nonnegKappa (norm_nonneg _)) (le_of_lt positiveDir))
    have innerBound : 2 * (kappa * ‖best - incumbent‖ / ‖direction‖) *
        (kappa * ‖best - incumbent‖ * ‖direction‖) ≤
        2 * (kappa * ‖best - incumbent‖ / ‖direction‖) *
          inner ℝ (best - incumbent) direction :=
      mul_le_mul_of_nonneg_left aligned mulNonneg
    have cancelTerm : 2 * (kappa * ‖best - incumbent‖ / ‖direction‖) *
        (kappa * ‖best - incumbent‖ * ‖direction‖) =
        2 * kappa ^ 2 * ‖best - incumbent‖ ^ 2 := by
      calc
        2 * (kappa * ‖best - incumbent‖ / ‖direction‖) *
            (kappa * ‖best - incumbent‖ * ‖direction‖) =
          2 * (kappa * ‖best - incumbent‖) * (kappa * ‖best - incumbent‖) *
            (‖direction‖ / ‖direction‖) := by ring
        _ = 2 * (kappa * ‖best - incumbent‖) * (kappa * ‖best - incumbent‖) * 1 := by
          rw [div_self dirNe]
        _ = 2 * kappa ^ 2 * ‖best - incumbent‖ ^ 2 := by ring
    linarith [innerBound, cancelTerm]
  linarith [crossTerm]

/-- The dimension-free geometric distance contraction under directional steering. -/
theorem steerContract {Space : Type*} [NormedAddCommGroup Space]
    [InnerProductSpace ℝ Space]
    (best incumbent direction : Space) (kappa : ℝ)
    (positiveDir : 0 < ‖direction‖) (nonnegKappa : 0 ≤ kappa)
    (aligned : kappa * ‖best - incumbent‖ * ‖direction‖ ≤
      inner ℝ (best - incumbent) direction) :
    dist (incumbent + (kappa * ‖best - incumbent‖ / ‖direction‖) • direction) best ^ 2 ≤
      (1 - kappa ^ 2) * dist incumbent best ^ 2 := by
  have prog := steerProgress best incumbent direction kappa positiveDir nonnegKappa aligned
  unfold progress at prog
  dsimp only at prog
  have distEq : dist incumbent best = ‖best - incumbent‖ := by
    rw [dist_comm, dist_eq_norm]
  rw [distEq] at prog ⊢
  linarith

/-- Exact gain-transport identity for a second step `scale • direction` taken
after committing `incumbent + direction`.  The second step's progress is
completely determined by the first step's realized progress and the known
perturbation energy `‖direction‖ ^ 2`. -/
theorem repeatProgress {Space : Type*} [NormedAddCommGroup Space]
    [InnerProductSpace ℝ Space]
    (best incumbent direction : Space) (scale : ℝ) :
    progress (fun state => dist state best ^ 2) (incumbent + direction)
        ((incumbent + direction) + scale • direction) =
      scale *
          progress (fun state => dist state best ^ 2) incumbent
            (incumbent + direction) -
        scale * (scale + 1) * ‖direction‖ ^ 2 := by
  have firstStep :
      progress (fun state => dist state best ^ 2) incumbent
          (incumbent + direction) =
        2 * inner ℝ (best - incumbent) direction - ‖direction‖ ^ 2 := by
    have h := squareProgress best incumbent direction 1
    simpa [one_smul] using h
  have secondStep :=
    squareProgress best (incumbent + direction) direction scale
  have innerShift :
      inner ℝ (best - (incumbent + direction)) direction =
        inner ℝ (best - incumbent) direction - ‖direction‖ ^ 2 := by
    have subEq : best - (incumbent + direction) = (best - incumbent) - direction := by
      abel
    rw [subEq, inner_sub_left, real_inner_self_eq_norm_sq]
  rw [secondStep, innerShift, firstStep]
  ring

/-- Cumulative two-step progress from `incumbent` to
`(incumbent + direction) + scale • direction` expressed purely in terms of the
committed first-step progress and the step energy `‖direction‖ ^ 2`. -/
theorem pairProgress {Space : Type*} [NormedAddCommGroup Space]
    [InnerProductSpace ℝ Space]
    (best incumbent direction : Space) (scale : ℝ) :
    progress (fun state => dist state best ^ 2) incumbent
        ((incumbent + direction) + scale • direction) =
      (1 + scale) *
          progress (fun state => dist state best ^ 2) incumbent
            (incumbent + direction) -
        scale * (1 + scale) * ‖direction‖ ^ 2 := by
  have split :
      progress (fun state => dist state best ^ 2) incumbent
          ((incumbent + direction) + scale • direction) =
        progress (fun state => dist state best ^ 2) incumbent
            (incumbent + direction) +
          progress (fun state => dist state best ^ 2) (incumbent + direction)
            ((incumbent + direction) + scale • direction) := by
    unfold progress
    ring
  rw [split, repeatProgress]
  ring

/-- Sign-reversal recovery after an uphill committed probe: when `direction` has
negative alignment `inner ℝ (best - incumbent) direction < 0`, reversing the
committed state `incumbent + direction` by `scale = -(1 + step)` lands at
`incumbent - step • direction` and achieves strictly positive cumulative
two-step progress over `incumbent`. -/
theorem reverseProgress {Space : Type*} [NormedAddCommGroup Space]
    [InnerProductSpace ℝ Space]
    (best incumbent direction : Space) (step : ℝ)
    (stepPositive : 0 < step)
    (uphillAligned : step * ‖direction‖ ^ 2 <
      -2 * inner ℝ (best - incumbent) direction) :
    0 < progress (fun state => dist state best ^ 2) incumbent
      ((incumbent + direction) + (-(1 + step)) • direction) := by
  have combine :
      (incumbent + direction) + (-(1 + step)) • direction =
        incumbent + (-step) • direction := by
    module
  rw [combine, squareProgress]
  nlinarith

/-- Sign-conditioned directional steering: conditioning the perturbation sign on
`inner ℝ (best - incumbent) direction` converts both positive and negative
aligned probes (`kappa * ‖best - incumbent‖ * ‖direction‖ ≤ |inner|`) into
certified `kappa ^ 2 * ‖best - incumbent‖ ^ 2` progress. -/
theorem signedProgress {Space : Type*} [NormedAddCommGroup Space]
    [InnerProductSpace ℝ Space]
    (best incumbent direction : Space) (kappa : ℝ)
    (positiveDir : 0 < ‖direction‖) (nonnegKappa : 0 ≤ kappa)
    (absAligned : kappa * ‖best - incumbent‖ * ‖direction‖ ≤
      |inner ℝ (best - incumbent) direction|) :
    let signedDir :=
      if 0 ≤ inner ℝ (best - incumbent) direction then direction else -direction
    kappa ^ 2 * ‖best - incumbent‖ ^ 2 ≤
      progress (fun state => dist state best ^ 2) incumbent
        (incumbent + (kappa * ‖best - incumbent‖ / ‖signedDir‖) • signedDir) := by
  dsimp only
  split_ifs with hSign
  · rw [abs_of_nonneg hSign] at absAligned
    exact steerProgress best incumbent direction kappa positiveDir nonnegKappa
      absAligned
  · have negInner : inner ℝ (best - incumbent) direction < 0 :=
      not_le.mp hSign
    rw [abs_of_neg negInner] at absAligned
    have posNeg : 0 < ‖-direction‖ := by simpa [norm_neg] using positiveDir
    have alignedNeg :
        kappa * ‖best - incumbent‖ * ‖-direction‖ ≤
          inner ℝ (best - incumbent) (-direction) := by
      simpa [norm_neg, inner_neg_right] using absAligned
    exact steerProgress best incumbent (-direction) kappa posNeg nonnegKappa
      alignedNeg

/-- Product-space sign-conditioned steering: when each coordinate factor
satisfies the unsigned alignment premise, orienting the perturbation sign
independently on each factor yields additive squared-distance progress across
the product. -/
theorem blockProgress {m : ℕ} {Space : Type*} [NormedAddCommGroup Space]
    [InnerProductSpace ℝ Space]
    (best incumbent direction : Fin m → Space) (kappa : Fin m → ℝ)
    (positiveDir : ∀ i, 0 < ‖direction i‖)
    (nonnegKappa : ∀ i, 0 ≤ kappa i)
    (absAligned : ∀ i,
      kappa i * ‖best i - incumbent i‖ * ‖direction i‖ ≤
        |inner ℝ (best i - incumbent i) (direction i)|) :
    let signedDir := fun i =>
      if 0 ≤ inner ℝ (best i - incumbent i) (direction i) then direction i else -direction i
    (∑ i : Fin m, kappa i ^ 2 * ‖best i - incumbent i‖ ^ 2) ≤
      ∑ i : Fin m,
        progress (fun state => dist state (best i) ^ 2) (incumbent i)
          (incumbent i +
            (kappa i * ‖best i - incumbent i‖ / ‖signedDir i‖) • signedDir i) := by
  dsimp only
  apply Finset.sum_le_sum
  intro i _
  exact signedProgress (best i) (incumbent i) (direction i) (kappa i)
    (positiveDir i) (nonnegKappa i) (absAligned i)

/-- The perturbation update is the affine interpolation between incumbent and
the history-derived target. -/
theorem stateEq {Space : Type*} [NormedAddCommGroup Space]
    [NormedSpace ℝ Space] (step : ℝ) (incumbent anchor : Space) :
    anchorState step incumbent anchor =
      (1 - step) • incumbent + step • anchor := by
  unfold anchorState anchorMove
  module

/-- If the history-derived target is within a relative distance `estimate` of
the optimum, a step in `[0,1]` contracts distance by the explicit ratio
`1-step + step*estimate`. -/
theorem anchorContract {Space : Type*} [NormedAddCommGroup Space]
    [NormedSpace ℝ Space]
    (incumbent anchor best : Space) (step estimate : ℝ)
    (stepNonneg : 0 ≤ step) (stepAtMost : step ≤ 1)
    (anchorBound : dist anchor best ≤ estimate * dist incumbent best) :
    DistanceContract incumbent (anchorState step incumbent anchor) best
      ((1 - step) + step * estimate) := by
  unfold DistanceContract
  rw [stateEq]
  rw [dist_eq_norm, dist_eq_norm] at anchorBound ⊢
  have identity :
      (1 - step) • incumbent + step • anchor - best =
        (1 - step) • (incumbent - best) + step • (anchor - best) := by
    module
  rw [identity]
  calc
    ‖(1 - step) • (incumbent - best) + step • (anchor - best)‖ ≤
        ‖(1 - step) • (incumbent - best)‖ +
          ‖step • (anchor - best)‖ := norm_add_le _ _
    _ = (1 - step) * ‖incumbent - best‖ +
        step * ‖anchor - best‖ := by
      simp [norm_smul, abs_of_nonneg stepNonneg,
        abs_of_nonneg (sub_nonneg.mpr stepAtMost)]
    _ ≤ (1 - step) * ‖incumbent - best‖ +
        step * (estimate * ‖incumbent - best‖) :=
      add_le_add_right (mul_le_mul_of_nonneg_left anchorBound stepNonneg) _
    _ = ((1 - step) + step * estimate) * ‖incumbent - best‖ := by ring

/-- A target strictly closer to the optimum and a nonzero interpolation step
give a strict distance-contraction ratio. -/
theorem ratioStrict (step estimate : ℝ)
    (stepPositive : 0 < step) (stepAtMost : step ≤ 1)
    (estimateNonneg : 0 ≤ estimate) (estimateLtOne : estimate < 1) :
    0 ≤ (1 - step) + step * estimate ∧
      (1 - step) + step * estimate < 1 := by
  constructor
  · exact add_nonneg (sub_nonneg.mpr stepAtMost)
      (mul_nonneg (le_of_lt stepPositive) estimateNonneg)
  · have productPositive : 0 < step * (1 - estimate) :=
      mul_pos stepPositive (sub_pos.mpr estimateLtOne)
    nlinarith

/-- The explicit target-directed perturbation inherits a regret factor from
the objective's metric condition numbers. -/
theorem anchorRegret {Space : Type*} [NormedAddCommGroup Space]
    [NormedSpace ℝ Space]
    (objective : Objective Space) (best incumbent anchor : Space)
    (lower upper step estimate : ℝ)
    (growth : MetricGrowth objective best lower upper)
    (lowerPositive : 0 < lower) (upperNonneg : 0 ≤ upper)
    (stepNonneg : 0 ≤ step) (stepAtMost : step ≤ 1)
    (anchorBound : dist anchor best ≤ estimate * dist incumbent best)
    (ratioNonneg : 0 ≤ (1 - step) + step * estimate) :
    Regret.simpleRegret objective best (anchorState step incumbent anchor) ≤
      (upper * ((1 - step) + step * estimate) / lower) *
        Regret.simpleRegret objective best incumbent := by
  let ratio := (1 - step) + step * estimate
  let factor := upper * ratio / lower
  apply metricRegret objective best incumbent (anchorState step incumbent anchor)
    lower upper ratio factor growth
  · exact anchorContract incumbent anchor best step estimate stepNonneg
      stepAtMost anchorBound
  · exact upperNonneg
  · exact div_nonneg (mul_nonneg upperNonneg ratioNonneg) (le_of_lt lowerPositive)
  · dsimp only [factor]
    rw [div_mul_cancel₀ _ (ne_of_gt lowerPositive)]

/-- The target-directed regret factor is strictly contractive exactly when its
distance ratio clears the objective-class condition-number requirement. -/
theorem factorStrict (lower upper ratio : ℝ)
    (lowerPositive : 0 < lower) (clears : upper * ratio < lower) :
    upper * ratio / lower < 1 := by
  exact (div_lt_one lowerPositive).2 clears

/-- Scalar score selection produces a state-space guarantee when one candidate
has certified metric coverage.  The conclusion separates coverage radius,
selected-candidate error, comparison-candidate error, and selection slack. -/
theorem selectedDist [PseudoMetricSpace State]
    {Candidate : Type*}
    (objective : Objective State) (best : State)
    (candidate : Candidate → State) (estimate error : Candidate → ℝ)
    (selected comparison : Candidate) (lower upper radius slack : ℝ)
    (growth : MetricGrowth objective best lower upper)
    (lowerPositive : 0 < lower) (upperNonneg : 0 ≤ upper)
    (accurate : PointError
      (fun index => objective (candidate index)) estimate error)
    (chosen : ApproxMin estimate selected slack)
    (covered : dist (candidate comparison) best ≤ radius) :
    dist (candidate selected) best ≤
      (upper * radius + error selected + error comparison + slack) / lower := by
  have scoreBound := Selection.pointBound
    (fun index => objective (candidate index)) estimate error
    selected comparison slack accurate chosen
  have compareBound :
      Regret.simpleRegret objective best (candidate comparison) ≤
        upper * radius := by
    calc
      Regret.simpleRegret objective best (candidate comparison) ≤
          upper * dist (candidate comparison) best :=
        (growth (candidate comparison)).2
      _ ≤ upper * radius :=
        mul_le_mul_of_nonneg_left covered upperNonneg
  have selectedBound :
      Regret.simpleRegret objective best (candidate selected) ≤
        upper * radius + error selected + error comparison + slack := by
    unfold Regret.simpleRegret at scoreBound compareBound ⊢
    linarith
  apply (le_div_iff₀ lowerPositive).2
  have lowerBound := (growth (candidate selected)).1
  nlinarith

/-- If the complete coverage-and-selection penalty fits a relative metric
budget, the score-selected parameter state is a valid perturbation anchor. -/
theorem selectedAnchor [PseudoMetricSpace State]
    {Candidate : Type*}
    (objective : Objective State) (best incumbent : State)
    (candidate : Candidate → State) (estimate error : Candidate → ℝ)
    (selected comparison : Candidate)
    (lower upper radius slack ratio : ℝ)
    (growth : MetricGrowth objective best lower upper)
    (lowerPositive : 0 < lower) (upperNonneg : 0 ≤ upper)
    (accurate : PointError
      (fun index => objective (candidate index)) estimate error)
    (chosen : ApproxMin estimate selected slack)
    (covered : dist (candidate comparison) best ≤ radius)
    (budget : upper * radius + error selected + error comparison + slack ≤
      lower * ratio * dist incumbent best) :
    DistanceContract incumbent (candidate selected) best ratio := by
  have chosenDist := selectedDist objective best candidate estimate error
    selected comparison lower upper radius slack growth lowerPositive
    upperNonneg accurate chosen covered
  unfold DistanceContract
  have scaled :
      (upper * radius + error selected + error comparison + slack) / lower ≤
        ratio * dist incumbent best := by
    rw [div_le_iff₀ lowerPositive]
    nlinarith
  exact chosenDist.trans scaled

/-- Metric contraction, score accuracy, and an acceptance margin compose into
the one-atom expected-regret recurrence.  The law weight is the only penalty
for making the geometrically contracting proposal stochastic. -/
theorem metricCommit [Fintype Move] [PseudoMetricSpace State]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (scores : ObjectiveSearch.Scores Move) (guide : Move)
    (objective : Objective State) (best incumbent : State)
    (lower upper ratio factor baseError margin : ℝ)
    (candidateError : Move → ℝ)
    (growth : MetricGrowth objective best lower upper)
    (contract : DistanceContract incumbent (candidate guide) best ratio)
    (upperNonneg : 0 ≤ upper) (factorNonneg : 0 ≤ factor)
    (rates : upper * ratio ≤ factor * lower)
    (accurate : ScoreEnvelope objective incumbent candidate scores
      baseError candidateError)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin)
    (separated : baseError + candidateError guide + margin ≤
      (1 - factor) * Regret.simpleRegret objective best incumbent) :
    (∑ move, law.weight move * Regret.simpleRegret objective best
      (ObjectiveSearch.commit incumbent candidate scores margin move)) ≤
        (1 - law.weight guide * (1 - factor)) *
          Regret.simpleRegret objective best incumbent := by
  let gain := (1 - factor) * Regret.simpleRegret objective best incumbent
  have trueGain : gain ≤ progress objective incumbent (candidate guide) :=
    metricGain objective best incumbent (candidate guide) lower upper ratio
      factor growth contract upperNonneg factorNonneg rates
  have accepted : ObjectiveSearch.Accept scores margin guide := by
    have incumbentLower := (abs_le.mp accurate.1).1
    have candidateUpper := (abs_le.mp (accurate.2 guide)).2
    unfold ObjectiveSearch.Accept
    unfold progress at trueGain
    dsimp only [gain] at trueGain
    linarith
  have bound := atomRegret law objective incumbent best
    (ObjectiveSearch.commit incumbent candidate scores margin) guide
    gain (1 - factor) 0
    (pointSafe objective incumbent candidate scores baseError margin
      candidateError accurate marginSafe)
    (by simpa [ObjectiveSearch.commit, accepted] using trueGain)
    (by dsimp only [gain]; linarith)
  simpa using bound

#print axioms metricRegret
#print axioms metricGain
#print axioms squareProgress
#print axioms alignedStep
#print axioms steerProgress
#print axioms steerContract
#print axioms repeatProgress
#print axioms pairProgress
#print axioms reverseProgress
#print axioms signedProgress
#print axioms blockProgress
#print axioms stateEq
#print axioms anchorContract
#print axioms ratioStrict
#print axioms anchorRegret
#print axioms factorStrict
#print axioms selectedDist
#print axioms selectedAnchor
#print axioms metricCommit

end BauxResearch.Optimization.Guide
