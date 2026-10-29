import BauxResearch.Optimization.ObjectiveSearch
import BauxResearch.Optimization.Regret
import BauxResearch.Optimization.Accuracy
import BauxResearch.Optimization.Rate
import Mathlib.MeasureTheory.Measure.Doubling
import Mathlib.Probability.Kernel.Basic
import Mathlib.Topology.MetricSpace.CoveringNumbers

/-!
# Reachability of improving parameter states

This module states the geometric and probabilistic part of objective-query
optimization without choosing a perturbation family, a surrogate, a controller,
or a model architecture.

A proposal is a Mathlib Markov kernel on outcome identities, conditioned on
retained history, the incumbent state, and a declared trust radius.  Explicit
realization and commit maps preserve the distinction between a sampled move,
its parameter state, and the accepted update.  Objective sublevel sets carry
Mathlib covering and packing numbers.  The central implication says that local
objective regularity and proposal mass near a better state produce objective
improvement mass.  Establishing that mass for an executable proposal remains a
separate mathematical obligation.
-/

namespace BauxResearch.Optimization.Reachability

open MeasureTheory ProbabilityTheory Set
open scoped ENNReal NNReal

noncomputable section

variable {History State Target Move Sample : Type*}

/-- States whose true objective improves on the incumbent by at least `gain`. -/
def betterSet (objective : Objective State) (incumbent : State) (gain : ℝ) : Set State :=
  {candidate | gain ≤ progress objective incumbent candidate}

/-- A finite-width objective sublevel set around a supplied comparator. -/
def nearSet (objective : Objective State) (best : State) (width : ℝ) : Set State :=
  {state | objective state ≤ objective best + width}

/-- Internal covering complexity of a near-optimal set at a declared scale. -/
def coverProfile [PseudoMetricSpace State]
    (objective : Objective State) (best : State) (width : ℝ) (scale : ℝ≥0) : ℕ∞ :=
  Metric.coveringNumber scale (nearSet objective best width)

/-- Packing complexity of a near-optimal set at a declared scale. -/
def packProfile [PseudoMetricSpace State]
    (objective : Objective State) (best : State) (width : ℝ) (scale : ℝ≥0) : ℕ∞ :=
  Metric.packingNumber scale (nearSet objective best width)

/-- A history-conditioned kernel on proposal outcomes.  An explicit realization
map supplies the complete parameter state represented by each outcome. -/
abbrev Proposal (History State Sample : Type*)
    [MeasurableSpace History] [MeasurableSpace State] [MeasurableSpace Sample] :=
  Kernel (History × State × ℝ≥0) Sample

/-- The probability measure denoted by one finite executable proposal law.
Move identity is retained even when two moves realize the same state. -/
noncomputable def finiteMeasure [Fintype Move] [MeasurableSpace Move]
    (law : ObjectiveSearch.ProposalLaw Move) : Measure Move :=
  Measure.sum fun move =>
    ENNReal.ofReal (law.weight move) • Measure.dirac move

instance finiteMeasure.instProbability [Fintype Move] [MeasurableSpace Move]
    (law : ObjectiveSearch.ProposalLaw Move) :
    IsProbabilityMeasure (finiteMeasure law) := by
  unfold finiteMeasure
  have total : HasSum law.weight 1 := by
    simpa [law.total] using hasSum_fintype law.weight
  exact total.isProbabilityMeasure_sum_dirac law.nonnegative

/-- A fixed finite executable proposal law viewed as a Mathlib Markov kernel.
History and trust conditioning are constant for this single-round bridge. -/
noncomputable def finiteProposal [Fintype Move]
    [MeasurableSpace History] [MeasurableSpace State] [MeasurableSpace Move]
    (law : ObjectiveSearch.ProposalLaw Move) : Proposal History State Move :=
  Kernel.const _ (finiteMeasure law)

instance finiteProposal.instMarkov [Fintype Move]
    [MeasurableSpace History] [MeasurableSpace State] [MeasurableSpace Move]
    (law : ObjectiveSearch.ProposalLaw Move) :
    IsMarkovKernel (finiteProposal (History := History) (State := State) law) := by
  unfold finiteProposal
  infer_instance

/-- Conditional proposal mass assigned to a true improvement set. -/
def capture [MeasurableSpace History] [MeasurableSpace State] [MeasurableSpace Sample]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent : State)
    (trust : ℝ≥0) (realize : Sample → State)
    (objective : Objective State) (gain : ℝ) : ℝ≥0∞ :=
  proposal (history, incumbent, trust)
    (realize ⁻¹' betterSet objective incumbent gain)

/-- Conditional proposal mass assigned to a metric neighborhood. -/
def ballMass [MeasurableSpace History] [MeasurableSpace State] [MeasurableSpace Sample]
    [PseudoMetricSpace State]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent : State)
    (trust : ℝ≥0) (realize : Sample → State)
    (center : State) (radius : ℝ) : ℝ≥0∞ :=
  proposal (history, incumbent, trust)
    (realize ⁻¹' Metric.closedBall center radius)

/-- A quantitative accessibility statement at one history, incumbent, center,
and scale.  It deliberately does not prescribe how the bound is obtained. -/
def Accessible [MeasurableSpace History] [MeasurableSpace State] [MeasurableSpace Sample]
    [PseudoMetricSpace State]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent : State)
    (trust : ℝ≥0) (realize : Sample → State)
    (center : State) (radius : ℝ) (mass : ℝ≥0∞) : Prop :=
  mass ≤ ballMass proposal history incumbent trust realize center radius

theorem measurableBetter [MeasurableSpace State]
    (objective : Objective State) (incumbent : State) (gain : ℝ)
    (measurable : Measurable objective) :
    MeasurableSet (betterSet objective incumbent gain) := by
  unfold betterSet progress
  exact measurableSet_le measurable_const (measurable_const.sub measurable)

theorem measurableNear [MeasurableSpace State]
    (objective : Objective State) (best : State) (width : ℝ)
    (measurable : Measurable objective) :
    MeasurableSet (nearSet objective best width) := by
  unfold nearSet
  exact measurableSet_le measurable measurable_const

/-- At a fixed transaction, the Mathlib-kernel capture of the finite executable
law is exactly the older finite-law `goodMass`, embedded in `ℝ≥0∞`. -/
theorem finiteCapture [Fintype Move]
    [MeasurableSpace History] [MeasurableSpace State] [MeasurableSpace Move]
    [MeasurableSingletonClass Move]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (history : History) (incumbent : State) (trust : ℝ≥0)
    (objective : Objective State) (gain : ℝ) :
    capture (finiteProposal (History := History) (State := State) law)
        history incumbent trust candidate objective gain =
      ENNReal.ofReal
        (ObjectiveSearch.goodMass law objective incumbent candidate gain) := by
  classical
  have goodMeas : MeasurableSet (candidate ⁻¹' betterSet objective incumbent gain) :=
    Set.Finite.measurableSet (Set.toFinite _)
  unfold capture finiteProposal finiteMeasure ObjectiveSearch.goodMass
  rw [Kernel.const_apply, Measure.sum_apply _ goodMeas]
  rw [ENNReal.ofReal_sum_of_nonneg]
  · simp_rw [Measure.smul_apply, smul_eq_mul,
      Measure.dirac_apply' _ goodMeas]
    rw [tsum_fintype, Finset.sum_filter]
    apply Finset.sum_congr rfl
    intro move _
    by_cases good : gain ≤ progress objective incumbent (candidate move)
    · simp [betterSet, good]
    · simp [betterSet, good]
  · intro move member
    exact law.nonnegative move

/-- Enlarging the accepted objective width enlarges the near-optimal set. -/
theorem nearMono (objective : Objective State) (best : State) {small large : ℝ}
    (bound : small ≤ large) :
    nearSet objective best small ⊆ nearSet objective best large := by
  intro state member
  change objective state ≤ objective best + small at member
  change objective state ≤ objective best + large
  linarith

/-- A near-optimal set of larger width cannot require fewer external covering
centers solely by set inclusion. -/
theorem coverMono [PseudoMetricSpace State]
    (objective : Objective State) (best : State) {small large : ℝ} (scale : ℝ≥0)
    (bound : small ≤ large) :
    Metric.externalCoveringNumber scale (nearSet objective best small) ≤
      Metric.externalCoveringNumber scale (nearSet objective best large) :=
  Metric.externalCoveringNumber_mono_set (nearMono objective best bound)

/-- The comparison state belongs to every nonnegative-width near-optimal set. -/
theorem bestMember (objective : Objective State) (best : State) (width : ℝ)
    (nonnegative : 0 ≤ width) :
    best ∈ nearSet objective best width := by
  change objective best ≤ objective best + width
  linarith

/-- Membership in a near-optimal set gives a derivative-free lower bound on
incumbent-to-comparison progress in terms of incumbent simple regret. -/
theorem nearProgress (objective : Objective State) (incumbent best center : State)
    (width : ℝ) (near : center ∈ nearSet objective best width) :
    Regret.simpleRegret objective best incumbent - width ≤
      progress objective incumbent center := by
  change objective center ≤ objective best + width at near
  unfold Regret.simpleRegret progress
  linarith

/-- A surrogate score gap certifies true objective improvement when it clears
the separate incumbent and guide-state error envelopes.  This is the precise
alignment obligation for a history-selected guide; no derivative is used. -/
theorem guideBetter (objective : Objective State) (incumbent center : State)
    (incumbentScore centerScore incumbentError centerError gain : ℝ)
    (incumbentAccurate :
      |incumbentScore - objective incumbent| ≤ incumbentError)
    (centerAccurate : |centerScore - objective center| ≤ centerError)
    (predicted : gain + incumbentError + centerError ≤
      incumbentScore - centerScore) :
    gain ≤ progress objective incumbent center := by
  have incumbentLower := (abs_le.mp incumbentAccurate).2
  have centerUpper := (abs_le.mp centerAccurate).1
  unfold progress
  linarith

/-- A Mathlib metric cover of a region by executable candidate states supplies
an actual candidate within the cover radius of every state in that region. -/
theorem coverHit [Fintype Move] [PseudoMetricSpace State]
    (candidate : Move → State) (region : Set State) (scale : ℝ≥0)
    (covered : Metric.IsCover scale region (Set.range candidate))
    (center : State) (member : center ∈ region) :
    ∃ move, dist (candidate move) center ≤ (scale : ℝ) := by
  obtain ⟨point, ⟨move, rfl⟩, near⟩ := covered member
  refine ⟨move, ?_⟩
  have near' : nndist center (candidate move) ≤ scale :=
    edist_le_coe.mp near
  simpa [dist_comm] using near'

/-- Any particular executable move inside a metric ball contributes at least
its own proposal weight to that ball's finite-law mass. -/
theorem moveMass [Fintype Move] [PseudoMetricSpace State]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (center : State) (radius : ℝ) (move : Move)
    (hit : dist (candidate move) center ≤ radius) :
    law.weight move ≤
      ObjectiveSearch.nearMass law candidate center radius := by
  classical
  unfold ObjectiveSearch.nearMass
  have member : move ∈ Finset.filter
      (fun slot => dist (candidate slot) center ≤ radius) Finset.univ := by
    simp [hit]
  exact Finset.single_le_sum
    (fun slot _ => law.nonnegative slot) member

/-- A nonnegative-radius ball centered at one executable proposal contains that
proposal, so its law weight is an immediate local mass floor. -/
theorem guideMass [Fintype Move] [PseudoMetricSpace State]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (guide : Move) (radius : ℝ) (radiusNonneg : 0 ≤ radius) :
    law.weight guide ≤
      ObjectiveSearch.nearMass law candidate (candidate guide) radius := by
  apply moveMass law candidate (candidate guide) radius guide
  simpa using radiusNonneg

/-- A geometric cover and a uniform lower bound on executable move weights
produce a quantitative proposal-mass floor around every covered state. -/
theorem coverMass [Fintype Move] [PseudoMetricSpace State]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (region : Set State) (scale : ℝ≥0)
    (covered : Metric.IsCover scale region (Set.range candidate))
    (weightFloor : ℝ) (weightBound : ∀ move, weightFloor ≤ law.weight move)
    (center : State) (member : center ∈ region) :
    weightFloor ≤ ObjectiveSearch.nearMass law candidate center (scale : ℝ) := by
  obtain ⟨move, hit⟩ := coverHit candidate region scale covered center member
  exact (weightBound move).trans (moveMass law candidate center scale move hit)

/-- An executable candidate cover cannot use fewer moves than Mathlib's
external covering number at the same resolution. -/
theorem coverCard [Fintype Move] [PseudoMetricSpace State]
    (candidate : Move → State) (region : Set State) (scale : ℝ≥0)
    (covered : Metric.IsCover scale region (Set.range candidate)) :
    Metric.externalCoveringNumber scale region ≤
      (Fintype.card Move : ℕ∞) := by
  calc
    Metric.externalCoveringNumber scale region ≤
        (Set.range candidate).encard :=
      covered.externalCoveringNumber_le_encard
    _ ≤ (Set.univ : Set Move).encard := by
      rw [← Set.image_univ]
      exact Set.encard_image_le candidate Set.univ
    _ = ENat.card Move := Set.encard_univ Move
    _ = (Fintype.card Move : ℕ∞) := ENat.card_eq_coe_fintype_card

/-- If the available candidate count is below the external covering number,
no enumeration by those candidates can cover the declared region. -/
theorem coverObstruct [Fintype Move] [PseudoMetricSpace State]
    (candidate : Move → State) (region : Set State) (scale : ℝ≥0)
    (tooSmall : (Fintype.card Move : ℕ∞) <
      Metric.externalCoveringNumber scale region) :
    ¬ Metric.IsCover scale region (Set.range candidate) := by
  intro covered
  exact (not_le_of_gt tooSmall) (coverCard candidate region scale covered)

/-- Objective regularity turns a ball around a better state into an improvement
set.  This implication contains no derivative or linear-response assumption. -/
theorem ballSubset [PseudoMetricSpace State]
    (objective : Objective State) (modulus : ℝ → ℝ)
    (incumbent center : State) (gain radius : ℝ)
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center) :
    Metric.closedBall center radius ⊆
      betterSet objective incumbent (gain - modulus radius) := by
  intro candidate near
  exact ObjectiveSearch.nearGood objective modulus incumbent center id
    gain radius candidate regular monotone better near

/-- Metric ball mass is a lower bound on improvement capture mass whenever the
ball is centered at a sufficiently better state. -/
theorem massCapture [MeasurableSpace History] [MeasurableSpace State]
    [MeasurableSpace Sample]
    [PseudoMetricSpace State]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent center : State)
    (trust : ℝ≥0) (realize : Sample → State)
    (objective : Objective State) (modulus : ℝ → ℝ)
    (gain radius : ℝ)
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center) :
    ballMass proposal history incumbent trust realize center radius ≤
      capture proposal history incumbent trust realize objective
        (gain - modulus radius) :=
  measure_mono (Set.preimage_mono
    (ballSubset objective modulus incumbent center gain radius
      regular monotone better))

/-- Any proved accessibility lower bound therefore becomes a lower bound on
the probability of proposing a true improvement. -/
theorem accessibleCapture [MeasurableSpace History] [MeasurableSpace State]
    [MeasurableSpace Sample]
    [PseudoMetricSpace State]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent center : State)
    (trust : ℝ≥0) (realize : Sample → State)
    (objective : Objective State) (modulus : ℝ → ℝ)
    (gain radius : ℝ) (mass : ℝ≥0∞)
    (access : Accessible proposal history incumbent trust realize center radius mass)
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center) :
    mass ≤ capture proposal history incumbent trust realize objective
      (gain - modulus radius) :=
  access.trans (massCapture proposal history incumbent center trust realize
    objective modulus gain radius regular monotone better)

/-- Mathlib's local doubling law can be applied directly to one conditional
proposal measure.  It compares scales but does not assert positive ball mass. -/
theorem massScale [MeasurableSpace History] [MeasurableSpace State]
    [PseudoMetricSpace State]
    (proposal : Proposal History State State) [IsMarkovKernel proposal]
    (history : History) (incumbent : State)
    (trust : ℝ≥0) (center : State) (factor radius cap : ℝ)
    [IsUnifLocDoublingMeasure (proposal (history, incumbent, trust))]
    (factorBound : factor ∈ Ioc 0 cap)
    (radiusBound : radius ≤
      IsUnifLocDoublingMeasure.scalingScaleOf
        (proposal (history, incumbent, trust)) cap) :
    ballMass proposal history incumbent trust id center (factor * radius) ≤
      IsUnifLocDoublingMeasure.scalingConstantOf
          (proposal (history, incumbent, trust)) cap *
        ballMass proposal history incumbent trust id center radius :=
  IsUnifLocDoublingMeasure.measure_mul_le_scalingConstantOf_mul
    (proposal (history, incumbent, trust)) factorBound radiusBound

/-- Under a local doubling law, mass known at an outer radius gives an explicit
inner-radius mass floor after division by Mathlib's scaling constant. -/
theorem scaleAccess [MeasurableSpace History] [MeasurableSpace State]
    [PseudoMetricSpace State]
    (proposal : Proposal History State State) [IsMarkovKernel proposal]
    (history : History) (incumbent : State)
    (trust : ℝ≥0) (center : State) (factor radius cap : ℝ)
    [IsUnifLocDoublingMeasure (proposal (history, incumbent, trust))]
    (mass : ℝ≥0∞)
    (outer : mass ≤
      ballMass proposal history incumbent trust id center (factor * radius))
    (factorBound : factor ∈ Ioc 0 cap)
    (radiusBound : radius ≤
      IsUnifLocDoublingMeasure.scalingScaleOf
        (proposal (history, incumbent, trust)) cap) :
    mass / IsUnifLocDoublingMeasure.scalingConstantOf
        (proposal (history, incumbent, trust)) cap ≤
      ballMass proposal history incumbent trust id center radius := by
  exact ENNReal.div_le_of_le_mul' <|
    outer.trans (massScale proposal history incumbent trust center factor radius cap
      factorBound radiusBound)

/-- The same rescaling certificate transfers through objective regularity to
true improvement mass at the inner resolution. -/
theorem scaleCapture [MeasurableSpace History] [MeasurableSpace State]
    [PseudoMetricSpace State]
    (proposal : Proposal History State State) [IsMarkovKernel proposal]
    (history : History) (incumbent center : State)
    (trust : ℝ≥0) (objective : Objective State) (modulus : ℝ → ℝ)
    (factor radius cap gain : ℝ)
    [IsUnifLocDoublingMeasure (proposal (history, incumbent, trust))]
    (mass : ℝ≥0∞)
    (outer : mass ≤
      ballMass proposal history incumbent trust id center (factor * radius))
    (factorBound : factor ∈ Ioc 0 cap)
    (radiusBound : radius ≤
      IsUnifLocDoublingMeasure.scalingScaleOf
        (proposal (history, incumbent, trust)) cap)
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center) :
    mass / IsUnifLocDoublingMeasure.scalingConstantOf
          (proposal (history, incumbent, trust)) cap ≤
      capture proposal history incumbent trust id objective
        (gain - modulus radius) := by
  exact (scaleAccess proposal history incumbent trust center factor radius cap mass
    outer factorBound radiusBound).trans
      (massCapture proposal history incumbent center trust id objective modulus
        gain radius regular monotone better)

/-- A proposal kernel assigns at most unit mass to every improvement set once
it is certified as a Markov kernel. -/
theorem captureLe [MeasurableSpace History] [MeasurableSpace State]
    [MeasurableSpace Sample]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent : State) (trust : ℝ≥0)
    (realize : Sample → State) (objective : Objective State) (gain : ℝ) :
    capture proposal history incumbent trust realize objective gain ≤ 1 := by
  calc
    capture proposal history incumbent trust realize objective gain ≤
        proposal (history, incumbent, trust) Set.univ :=
      measure_mono (Set.subset_univ _)
    _ = 1 := measure_univ

/-- A probability proposal cannot assign a common positive mass floor to
arbitrarily many pairwise-disjoint useful regions.  For `N` regions this is the
exact constraint `N * mass ≤ 1`, written as a finite sum to avoid division at
zero.  Any dimension-free accessibility theorem must therefore restrict the
simultaneously plausible useful regions or use history to align the proposal. -/
theorem disjointMass [Fintype Target]
    [MeasurableSpace History] [MeasurableSpace State] [MeasurableSpace Sample]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent : State) (trust : ℝ≥0)
    (region : Target → Set Sample) (mass : ℝ≥0∞)
    (measurable : ∀ target, MeasurableSet (region target))
    (disjoint : (Set.univ : Set Target).PairwiseDisjoint region)
    (lower : ∀ target,
      mass ≤ proposal (history, incumbent, trust) (region target)) :
    (∑ _ : Target, mass) ≤ 1 := by
  let measure := proposal (history, incumbent, trust)
  have unionMeasure :
      measure (⋃ target ∈ (Finset.univ : Finset Target), region target) =
        ∑ target ∈ (Finset.univ : Finset Target), measure (region target) := by
    apply measure_biUnion_finset
    · simpa using disjoint
    · intro target member
      exact measurable target
  calc
    (∑ _ : Target, mass) ≤ ∑ target : Target, measure (region target) := by
      exact Finset.sum_le_sum fun target _ => lower target
    _ = measure (⋃ target ∈ (Finset.univ : Finset Target), region target) := by
      symm
      simpa using unionMeasure
    _ ≤ measure Set.univ := measure_mono (Set.subset_univ _)
    _ = 1 := measure_univ

/-- An accepted-update map is safe when it never increases the true objective. -/
def Safe (objective : Objective State) (incumbent : State) (commit : Sample → State) : Prop :=
  ∀ sample, 0 ≤ progress objective incumbent (commit sample)

/-- An accepted-update map retains a declared gain on every candidate in the
corresponding true improvement set. -/
def Retains (objective : Objective State) (incumbent : State)
    (realize : Sample → State) (commit : Sample → State) (gain : ℝ) : Prop :=
  ∀ sample, realize sample ∈ betterSet objective incumbent gain →
    gain ≤ progress objective incumbent (commit sample)

/-- The incumbent-relative executable commit is safe whenever its acceptance
margin covers incumbent and candidate score error. -/
theorem commitSafe (objective : Objective State) (incumbent : State)
    (candidate : Move → State) (scores : ObjectiveSearch.Scores Move)
    (baseError candidateError margin : ℝ)
    (accurate : ObjectiveSearch.ScoreError objective incumbent candidate scores
      baseError candidateError)
    (marginSafe : baseError + candidateError ≤ margin) :
    Safe objective incumbent
      (ObjectiveSearch.commit incumbent candidate scores margin) := by
  intro move
  exact ObjectiveSearch.commitNonneg objective incumbent candidate scores
    baseError candidateError margin move accurate marginSafe

/-- The same executable commit retains every captured gain whose separation
exceeds its margin and both score-error terms. -/
theorem commitRetains (objective : Objective State) (incumbent : State)
    (candidate : Move → State) (scores : ObjectiveSearch.Scores Move)
    (baseError candidateError margin gain : ℝ)
    (accurate : ObjectiveSearch.ScoreError objective incumbent candidate scores
      baseError candidateError)
    (separated : baseError + candidateError + margin ≤ gain) :
    Retains objective incumbent candidate
      (ObjectiveSearch.commit incumbent candidate scores margin) gain := by
  intro move good
  exact ObjectiveSearch.goodCommit objective incumbent candidate scores
    baseError candidateError margin gain move accurate good separated

/-- Candidate-specific score error, retaining the local uncertainty envelope
instead of replacing it by a pool-wide maximum. -/
def ScoreEnvelope (objective : Objective State) (incumbent : State)
    (candidate : Move → State) (scores : ObjectiveSearch.Scores Move)
    (baseError : ℝ) (candidateError : Move → ℝ) : Prop :=
  |scores.incumbent - objective incumbent| ≤ baseError ∧
    ∀ move, |scores.candidate move - objective (candidate move)| ≤ candidateError move

/-- Candidate-specific score envelopes prove safety when every local error is
covered by the single executable acceptance margin. -/
theorem pointSafe (objective : Objective State) (incumbent : State)
    (candidate : Move → State) (scores : ObjectiveSearch.Scores Move)
    (baseError margin : ℝ) (candidateError : Move → ℝ)
    (accurate : ScoreEnvelope objective incumbent candidate scores
      baseError candidateError)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin) :
    Safe objective incumbent
      (ObjectiveSearch.commit incumbent candidate scores margin) := by
  intro move
  by_cases accepted : ObjectiveSearch.Accept scores margin move
  · rw [ObjectiveSearch.commit, if_pos accepted]
    have baseUpper := (abs_le.mp accurate.1).2
    have candidateLower := (abs_le.mp (accurate.2 move)).1
    unfold ObjectiveSearch.Accept at accepted
    unfold progress
    linarith [marginSafe move]
  · rw [ObjectiveSearch.commit, if_neg accepted]
    unfold progress
    linarith

/-- Candidate-specific separation is needed only for moves belonging to the
captured gain event; it then proves that the commit retains that gain. -/
theorem pointRetains (objective : Objective State) (incumbent : State)
    (candidate : Move → State) (scores : ObjectiveSearch.Scores Move)
    (baseError margin gain : ℝ) (candidateError : Move → ℝ)
    (accurate : ScoreEnvelope objective incumbent candidate scores
      baseError candidateError)
    (separated : ∀ move,
      gain ≤ progress objective incumbent (candidate move) →
        baseError + candidateError move + margin ≤ gain) :
    Retains objective incumbent candidate
      (ObjectiveSearch.commit incumbent candidate scores margin) gain := by
  intro move good
  change gain ≤ progress objective incumbent (candidate move) at good
  have accepted : ObjectiveSearch.Accept scores margin move := by
    have baseLower := (abs_le.mp accurate.1).1
    have candidateUpper := (abs_le.mp (accurate.2 move)).2
    unfold ObjectiveSearch.Accept
    unfold progress at good
    linarith [separated move good]
  simpa [ObjectiveSearch.commit, accepted] using good

/-- Candidate-specific score accuracy converts captured proposal mass into
expected accepted progress.  Unlike the uniform-error theorem, only captured
moves must satisfy the gain-separation condition. -/
theorem pointMean [Fintype Move]
    (law : ObjectiveSearch.ProposalLaw Move)
    (objective : Objective State) (incumbent : State)
    (candidate : Move → State) (scores : ObjectiveSearch.Scores Move)
    (baseError margin gain : ℝ) (candidateError : Move → ℝ)
    (accurate : ScoreEnvelope objective incumbent candidate scores
      baseError candidateError)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin)
    (separated : ∀ move,
      gain ≤ progress objective incumbent (candidate move) →
        baseError + candidateError move + margin ≤ gain) :
    ObjectiveSearch.goodMass law objective incumbent candidate gain * gain ≤
      ObjectiveSearch.meanProgress law objective incumbent candidate scores margin := by
  classical
  unfold ObjectiveSearch.goodMass ObjectiveSearch.meanProgress
  rw [Finset.sum_filter, Finset.sum_mul]
  apply Finset.sum_le_sum
  intro move _
  by_cases good : gain ≤ progress objective incumbent (candidate move)
  · simp only [good, if_true]
    exact mul_le_mul_of_nonneg_left
      ((pointRetains objective incumbent candidate scores baseError margin gain
        candidateError accurate separated) move (by
          change gain ≤ progress objective incumbent (candidate move)
          exact good))
      (law.nonnegative move)
  · simp only [good, if_false, zero_mul]
    exact mul_nonneg (law.nonnegative move)
      ((pointSafe objective incumbent candidate scores baseError margin
        candidateError accurate marginSafe) move)

/-- For a finite executable law, expected next simple regret plus expected
accepted progress is exactly incumbent simple regret.  This is the real-valued
bridge from a proposal transaction to the finite-oracle trajectory semantics. -/
theorem sumSplit [Fintype Move]
    (law : ObjectiveSearch.ProposalLaw Move)
    (objective : Objective State) (incumbent best : State)
    (candidate : Move → State) (scores : ObjectiveSearch.Scores Move)
    (margin : ℝ) :
    (∑ move, law.weight move * Regret.simpleRegret objective best
      (ObjectiveSearch.commit incumbent candidate scores margin move)) +
        ObjectiveSearch.meanProgress law objective incumbent candidate scores margin =
      Regret.simpleRegret objective best incumbent := by
  classical
  unfold ObjectiveSearch.meanProgress
  rw [← Finset.sum_add_distrib]
  calc
    _ = ∑ move, law.weight move *
        Regret.simpleRegret objective best incumbent := by
      apply Finset.sum_congr rfl
      intro move _
      unfold Regret.simpleRegret progress
      ring
    _ = Regret.simpleRegret objective best incumbent := by
      rw [← Finset.sum_mul, law.total, one_mul]

/-- Scores whose candidate component is the implemented weighted neighbor mean.
The retained incumbent score remains explicit because it may use a different
evaluation path. -/
noncomputable def knnScores {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : Move → Forward.Vector capacity)
    (nearest : Move → List (Fin capacity)) (incumbentScore : ℝ) :
    ObjectiveSearch.Scores Move where
  incumbent := incumbentScore
  candidate := fun move =>
    Neighbor.mean data params scale (distances move) (nearest move)

/-- The implemented weighted KNN mean satisfies the transaction score-error
contract when the retained incumbent score and every pool member satisfy their
respective deterministic error bounds. -/
theorem knnError {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : Move → Forward.Vector capacity)
    (nearest : Move → List (Fin capacity))
    (objective : Objective State) (incumbent : State)
    (candidate : Move → State) (incumbentScore baseError candidateError : ℝ)
    (baseAccurate : |incumbentScore - objective incumbent| ≤ baseError)
    (nonnegative : ∀ move (position : Fin (nearest move).length),
      0 ≤ Neighbor.weight data params scale (distances move)
        (nearest move)[position.val])
    (floorBound : ∀ move,
      (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale
        (distances move) (nearest move))
    (errors : ∀ move (position : Fin (nearest move).length),
      |data.values (nearest move)[position.val] - objective (candidate move)| ≤
        candidateError) :
    ObjectiveSearch.ScoreError objective incumbent candidate
      (knnScores data params scale distances nearest incumbentScore)
      baseError candidateError := by
  refine ⟨baseAccurate, ?_⟩
  exact Accuracy.poolBound data params scale distances nearest
    (fun move => objective (candidate move)) candidateError
    nonnegative floorBound errors

/-- The candidate-specific version retains each weighted KNN neighborhood's
own deterministic error envelope. -/
theorem knnEnvelope {capacity : ℕ}
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : Move → Forward.Vector capacity)
    (nearest : Move → List (Fin capacity))
    (objective : Objective State) (incumbent : State)
    (candidate : Move → State) (incumbentScore baseError : ℝ)
    (candidateError : Move → ℝ)
    (baseAccurate : |incumbentScore - objective incumbent| ≤ baseError)
    (nonnegative : ∀ move (position : Fin (nearest move).length),
      0 ≤ Neighbor.weight data params scale (distances move)
        (nearest move)[position.val])
    (floorBound : ∀ move,
      (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale
        (distances move) (nearest move))
    (errors : ∀ move (position : Fin (nearest move).length),
      |data.values (nearest move)[position.val] - objective (candidate move)| ≤
        candidateError move) :
    ScoreEnvelope objective incumbent candidate
      (knnScores data params scale distances nearest incumbentScore)
      baseError candidateError := by
  refine ⟨baseAccurate, ?_⟩
  exact Accuracy.pointPool data params scale distances nearest
    (fun move => objective (candidate move)) candidateError
    nonnegative floorBound errors

/-- Conditional expected nonnegative improvement after applying an arbitrary
accepted-update map to a proposed complete parameter state. -/
def meanGain [MeasurableSpace History] [MeasurableSpace State] [MeasurableSpace Sample]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent : State) (trust : ℝ≥0)
    (objective : Objective State) (commit : Sample → State) : ℝ≥0∞ :=
  ∫⁻ sample, ENNReal.ofReal (progress objective incumbent (commit sample))
    ∂proposal (history, incumbent, trust)

/-- Conditional expected terminal simple regret after applying the accepted
update map to a proposed complete parameter state. -/
def meanRegret [MeasurableSpace History] [MeasurableSpace State] [MeasurableSpace Sample]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent : State) (trust : ℝ≥0)
    (objective : Objective State) (best : State) (commit : Sample → State) : ℝ≥0∞ :=
  ∫⁻ sample, ENNReal.ofReal
    (Regret.simpleRegret objective best (commit sample))
    ∂proposal (history, incumbent, trust)

/-- For a safe accepted-update map and a global comparator, expected terminal
regret plus expected accepted improvement equals incumbent simple regret. -/
theorem regretSplit [MeasurableSpace History] [MeasurableSpace State]
    [MeasurableSpace Sample]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent best : State) (trust : ℝ≥0)
    (objective : Objective State) (commit : Sample → State)
    (optimal : Regret.Optimal objective best)
    (safe : Safe objective incumbent commit)
    (objectiveMeas : Measurable objective)
    (commitMeas : Measurable commit) :
    meanRegret proposal history incumbent trust objective best commit +
        meanGain proposal history incumbent trust objective commit =
      ENNReal.ofReal (Regret.simpleRegret objective best incumbent) := by
  let measure := proposal (history, incumbent, trust)
  have regretMeas : Measurable (fun candidate => ENNReal.ofReal
      (Regret.simpleRegret objective best (commit candidate))) := by
    exact ENNReal.measurable_ofReal.comp
      ((objectiveMeas.comp commitMeas).sub measurable_const)
  have pointwise (sample : Sample) :
      ENNReal.ofReal (Regret.simpleRegret objective best (commit sample)) +
          ENNReal.ofReal (progress objective incumbent (commit sample)) =
        ENNReal.ofReal (Regret.simpleRegret objective best incumbent) := by
    rw [← ENNReal.ofReal_add
      (Regret.nonnegative objective best (commit sample) optimal)
      (safe sample)]
    congr 1
    unfold Regret.simpleRegret progress
    ring
  calc
    meanRegret proposal history incumbent trust objective best commit +
          meanGain proposal history incumbent trust objective commit =
        ∫⁻ candidate,
          (ENNReal.ofReal (Regret.simpleRegret objective best (commit candidate)) +
            ENNReal.ofReal (progress objective incumbent (commit candidate)))
          ∂measure := by
      rw [lintegral_add_left regretMeas]
      rfl
    _ = ∫⁻ _sample : Sample,
        ENNReal.ofReal (Regret.simpleRegret objective best incumbent) ∂measure := by
      apply lintegral_congr
      exact fun sample => pointwise sample
    _ = ENNReal.ofReal (Regret.simpleRegret objective best incumbent) := by
      simp [measure]

/-- Capture probability times retained gain lower-bounds expected nonnegative
improvement.  The proposal and update remain otherwise unrestricted. -/
theorem captureMean [MeasurableSpace History] [MeasurableSpace State]
    [MeasurableSpace Sample]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent : State) (trust : ℝ≥0)
    (realize : Sample → State) (objective : Objective State)
    (commit : Sample → State) (gain : ℝ)
    (goodMeas : MeasurableSet
      (realize ⁻¹' betterSet objective incumbent gain))
    (retains : Retains objective incumbent realize commit gain) :
    ENNReal.ofReal gain *
        capture proposal history incumbent trust realize objective gain ≤
      meanGain proposal history incumbent trust objective commit := by
  let measure := proposal (history, incumbent, trust)
  let good := realize ⁻¹' betterSet objective incumbent gain
  calc
    ENNReal.ofReal gain *
        capture proposal history incumbent trust realize objective gain =
        ∫⁻ _ in good, ENNReal.ofReal gain ∂measure := by
      rw [setLIntegral_const]
      rfl
    _ = ∫⁻ candidate, good.indicator (fun _ ↦ ENNReal.ofReal gain) candidate
        ∂measure := by
      rw [lintegral_indicator goodMeas]
    _ ≤ ∫⁻ candidate, ENNReal.ofReal
        (progress objective incumbent (commit candidate)) ∂measure := by
      apply lintegral_mono
      intro candidate
      by_cases member : candidate ∈ good
      · simp only [Set.indicator_of_mem member]
        exact ENNReal.ofReal_le_ofReal (retains candidate member)
      · simp [member]
    _ = meanGain proposal history incumbent trust objective commit := rfl

/-- Retained improvement mass is therefore a certified one-round decrement of
expected terminal simple regret. -/
theorem captureRegret [MeasurableSpace History] [MeasurableSpace State]
    [MeasurableSpace Sample]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent best : State) (trust : ℝ≥0)
    (realize : Sample → State) (objective : Objective State)
    (commit : Sample → State) (gain : ℝ)
    (optimal : Regret.Optimal objective best)
    (safe : Safe objective incumbent commit)
    (goodMeas : MeasurableSet
      (realize ⁻¹' betterSet objective incumbent gain))
    (objectiveMeas : Measurable objective)
    (commitMeas : Measurable commit)
    (retains : Retains objective incumbent realize commit gain) :
    meanRegret proposal history incumbent trust objective best commit +
        ENNReal.ofReal gain *
          capture proposal history incumbent trust realize objective gain ≤
      ENNReal.ofReal (Regret.simpleRegret objective best incumbent) := by
  calc
    meanRegret proposal history incumbent trust objective best commit +
          ENNReal.ofReal gain *
            capture proposal history incumbent trust realize objective gain ≤
        meanRegret proposal history incumbent trust objective best commit +
          meanGain proposal history incumbent trust objective commit := by
      gcongr
      exact captureMean proposal history incumbent trust realize objective commit
        gain goodMeas retains
    _ = ENNReal.ofReal (Regret.simpleRegret objective best incumbent) :=
      regretSplit proposal history incumbent best trust objective commit
        optimal safe objectiveMeas commitMeas

/-- The finite executable proposal law, score-error contract, and
incumbent-relative acceptance rule compose directly into a one-round expected
simple-regret certificate. -/
theorem finiteRegret [Fintype Move]
    [MeasurableSpace History] [MeasurableSpace State] [MeasurableSpace Move]
    [MeasurableSingletonClass Move]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (scores : ObjectiveSearch.Scores Move)
    (history : History) (incumbent best : State) (trust : ℝ≥0)
    (objective : Objective State)
    (baseError candidateError margin gain : ℝ)
    (optimal : Regret.Optimal objective best)
    (objectiveMeas : Measurable objective)
    (accurate : ObjectiveSearch.ScoreError objective incumbent candidate scores
      baseError candidateError)
    (marginSafe : baseError + candidateError ≤ margin)
    (separated : baseError + candidateError + margin ≤ gain) :
    meanRegret (finiteProposal (History := History) (State := State) law)
        history incumbent trust objective best
          (ObjectiveSearch.commit incumbent candidate scores margin) +
        ENNReal.ofReal gain * ENNReal.ofReal
          (ObjectiveSearch.goodMass law objective incumbent candidate gain) ≤
      ENNReal.ofReal (Regret.simpleRegret objective best incumbent) := by
  rw [← finiteCapture law candidate history incumbent trust objective gain]
  exact captureRegret
    (finiteProposal (History := History) (State := State) law)
    history incumbent best trust candidate objective
    (ObjectiveSearch.commit incumbent candidate scores margin) gain optimal
    (commitSafe objective incumbent candidate scores baseError candidateError
      margin accurate marginSafe)
    (Set.Finite.measurableSet (Set.toFinite _)) objectiveMeas
    (measurable_of_finite _)
    (commitRetains objective incumbent candidate scores baseError candidateError
      margin gain accurate separated)

/-- The finite-law regret certificate with candidate-specific estimator error.
No worst-case KNN envelope is introduced by this composition. -/
theorem pointRegret [Fintype Move]
    [MeasurableSpace History] [MeasurableSpace State] [MeasurableSpace Move]
    [MeasurableSingletonClass Move]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (scores : ObjectiveSearch.Scores Move)
    (history : History) (incumbent best : State) (trust : ℝ≥0)
    (objective : Objective State) (baseError margin gain : ℝ)
    (candidateError : Move → ℝ)
    (optimal : Regret.Optimal objective best)
    (objectiveMeas : Measurable objective)
    (accurate : ScoreEnvelope objective incumbent candidate scores
      baseError candidateError)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin)
    (separated : ∀ move,
      gain ≤ progress objective incumbent (candidate move) →
        baseError + candidateError move + margin ≤ gain) :
    meanRegret (finiteProposal (History := History) (State := State) law)
        history incumbent trust objective best
          (ObjectiveSearch.commit incumbent candidate scores margin) +
        ENNReal.ofReal gain * ENNReal.ofReal
          (ObjectiveSearch.goodMass law objective incumbent candidate gain) ≤
      ENNReal.ofReal (Regret.simpleRegret objective best incumbent) := by
  rw [← finiteCapture law candidate history incumbent trust objective gain]
  exact captureRegret
    (finiteProposal (History := History) (State := State) law)
    history incumbent best trust candidate objective
    (ObjectiveSearch.commit incumbent candidate scores margin) gain optimal
    (pointSafe objective incumbent candidate scores baseError margin
      candidateError accurate marginSafe)
    (Set.Finite.measurableSet (Set.toFinite _)) objectiveMeas
    (measurable_of_finite _)
    (pointRetains objective incumbent candidate scores baseError margin gain
      candidateError accurate separated)

/-- Metric proposal mass near a better state, candidate-specific estimator
error, and incumbent-relative acceptance compose into expected simple regret. -/
theorem finiteAccess [Fintype Move]
    [MeasurableSpace History] [MeasurableSpace State] [MeasurableSpace Move]
    [MeasurableSingletonClass Move] [PseudoMetricSpace State]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (scores : ObjectiveSearch.Scores Move)
    (history : History) (incumbent center best : State) (trust : ℝ≥0)
    (objective : Objective State) (modulus : ℝ → ℝ)
    (gain radius baseError margin : ℝ) (candidateError : Move → ℝ)
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (optimal : Regret.Optimal objective best)
    (objectiveMeas : Measurable objective)
    (accurate : ScoreEnvelope objective incumbent candidate scores
      baseError candidateError)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin)
    (separated : ∀ move,
      gain - modulus radius ≤ progress objective incumbent (candidate move) →
        baseError + candidateError move + margin ≤ gain - modulus radius) :
    meanRegret (finiteProposal (History := History) (State := State) law)
        history incumbent trust objective best
          (ObjectiveSearch.commit incumbent candidate scores margin) +
        ENNReal.ofReal (gain - modulus radius) * ENNReal.ofReal
          (ObjectiveSearch.nearMass law candidate center radius) ≤
      ENNReal.ofReal (Regret.simpleRegret objective best incumbent) := by
  have massBound := ObjectiveSearch.massBound law objective modulus incumbent center
    candidate gain radius regular monotone better
  calc
    meanRegret (finiteProposal (History := History) (State := State) law)
          history incumbent trust objective best
            (ObjectiveSearch.commit incumbent candidate scores margin) +
          ENNReal.ofReal (gain - modulus radius) * ENNReal.ofReal
            (ObjectiveSearch.nearMass law candidate center radius) ≤
        meanRegret (finiteProposal (History := History) (State := State) law)
          history incumbent trust objective best
            (ObjectiveSearch.commit incumbent candidate scores margin) +
          ENNReal.ofReal (gain - modulus radius) * ENNReal.ofReal
            (ObjectiveSearch.goodMass law objective incumbent candidate
              (gain - modulus radius)) := by
      gcongr
    _ ≤ ENNReal.ofReal (Regret.simpleRegret objective best incumbent) :=
      pointRegret law candidate scores history incumbent best trust objective
        baseError margin (gain - modulus radius) candidateError optimal objectiveMeas
        accurate marginSafe separated

/-- Metric proposal mass, objective regularity, candidate-specific score error,
and acceptance yield the real one-step regret inequality used by finite-horizon
trajectory theorems. -/
theorem accessLocal [Fintype Move] [PseudoMetricSpace State]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (scores : ObjectiveSearch.Scores Move)
    (incumbent center best : State)
    (objective : Objective State) (modulus : ℝ → ℝ)
    (gain radius baseError margin : ℝ) (candidateError : Move → ℝ)
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (useful : modulus radius ≤ gain)
    (accurate : ScoreEnvelope objective incumbent candidate scores
      baseError candidateError)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin)
    (separated : ∀ move,
      gain - modulus radius ≤ progress objective incumbent (candidate move) →
        baseError + candidateError move + margin ≤ gain - modulus radius) :
    (∑ move, law.weight move * Regret.simpleRegret objective best
        (ObjectiveSearch.commit incumbent candidate scores margin move)) +
      ObjectiveSearch.nearMass law candidate center radius *
        (gain - modulus radius) ≤
      Regret.simpleRegret objective best incumbent := by
  have massBound := ObjectiveSearch.massBound law objective modulus incumbent center
    candidate gain radius regular monotone better
  have massMean : ObjectiveSearch.nearMass law candidate center radius *
      (gain - modulus radius) ≤
        ObjectiveSearch.meanProgress law objective incumbent candidate scores margin := by
    calc
      ObjectiveSearch.nearMass law candidate center radius *
            (gain - modulus radius) ≤
          ObjectiveSearch.goodMass law objective incumbent candidate
              (gain - modulus radius) * (gain - modulus radius) :=
        mul_le_mul_of_nonneg_right massBound (sub_nonneg.mpr useful)
      _ ≤ ObjectiveSearch.meanProgress law objective incumbent candidate scores margin :=
        pointMean law objective incumbent candidate scores baseError margin
          (gain - modulus radius) candidateError accurate marginSafe separated
  have split := sumSplit law objective incumbent best candidate scores margin
  linarith

/-- A lower bound on captured local gain discharges the exact local-contraction
shape consumed by `Rate.expected`.  The factor and penalty remain symbolic. -/
theorem rateLocal [Fintype Move] [PseudoMetricSpace State]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (scores : ObjectiveSearch.Scores Move)
    (incumbent center best : State)
    (objective : Objective State) (modulus : ℝ → ℝ)
    (gain radius baseError margin : ℝ) (candidateError : Move → ℝ)
    (factor penalty : ℝ)
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (useful : modulus radius ≤ gain)
    (accurate : ScoreEnvelope objective incumbent candidate scores
      baseError candidateError)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin)
    (separated : ∀ move,
      gain - modulus radius ≤ progress objective incumbent (candidate move) →
        baseError + candidateError move + margin ≤ gain - modulus radius)
    (dominance :
      (1 - factor) * Regret.simpleRegret objective best incumbent - penalty ≤
        ObjectiveSearch.nearMass law candidate center radius *
          (gain - modulus radius)) :
    (∑ move, law.weight move * Regret.simpleRegret objective best
      (ObjectiveSearch.commit incumbent candidate scores margin move)) ≤
        factor * Regret.simpleRegret objective best incumbent + penalty := by
  have bound := accessLocal law candidate scores incumbent center best objective modulus
    gain radius baseError margin candidateError regular monotone better useful accurate
    marginSafe separated
  linarith

/-- A lower bound on proposal mass and a lower bound on retained local gain
derive the contraction-dominance inequality.  Thus the contraction factor
`1 - massFloor * gainShare` is a consequence, not a supplied optimizer rate. -/
theorem productBound (regret nearMass localGain massFloor gainShare loss : ℝ)
    (massNonneg : 0 ≤ massFloor)
    (massBound : massFloor ≤ nearMass)
    (gainNonneg : 0 ≤ localGain)
    (gainBound : gainShare * regret - loss ≤ localGain) :
    (1 - (1 - massFloor * gainShare)) * regret - massFloor * loss ≤
      nearMass * localGain := by
  have first : massFloor * (gainShare * regret - loss) ≤
      massFloor * localGain :=
    mul_le_mul_of_nonneg_left gainBound massNonneg
  have second : massFloor * localGain ≤ nearMass * localGain :=
    mul_le_mul_of_nonneg_right massBound gainNonneg
  calc
    (1 - (1 - massFloor * gainShare)) * regret - massFloor * loss =
        massFloor * (gainShare * regret - loss) := by ring
    _ ≤ massFloor * localGain := first
    _ ≤ nearMass * localGain := second

/-- Explicit accessibility and retained-gain bounds derive a local contraction
without postulating the contraction inequality itself. -/
theorem derivedLocal [Fintype Move] [PseudoMetricSpace State]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (scores : ObjectiveSearch.Scores Move)
    (incumbent center best : State)
    (objective : Objective State) (modulus : ℝ → ℝ)
    (gain radius baseError margin : ℝ) (candidateError : Move → ℝ)
    (massFloor gainShare loss : ℝ)
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (useful : modulus radius ≤ gain)
    (accurate : ScoreEnvelope objective incumbent candidate scores
      baseError candidateError)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin)
    (separated : ∀ move,
      gain - modulus radius ≤ progress objective incumbent (candidate move) →
        baseError + candidateError move + margin ≤ gain - modulus radius)
    (massNonneg : 0 ≤ massFloor)
    (massBound : massFloor ≤
      ObjectiveSearch.nearMass law candidate center radius)
    (gainBound : gainShare * Regret.simpleRegret objective best incumbent - loss ≤
      gain - modulus radius) :
    (∑ move, law.weight move * Regret.simpleRegret objective best
      (ObjectiveSearch.commit incumbent candidate scores margin move)) ≤
        (1 - massFloor * gainShare) *
          Regret.simpleRegret objective best incumbent + massFloor * loss := by
  apply rateLocal law candidate scores incumbent center best objective modulus
    gain radius baseError margin candidateError
    (1 - massFloor * gainShare) (massFloor * loss) regular monotone better useful
    accurate marginSafe separated
  exact productBound
    (Regret.simpleRegret objective best incumbent)
    (ObjectiveSearch.nearMass law candidate center radius)
    (gain - modulus radius) massFloor gainShare loss massNonneg massBound
    (sub_nonneg.mpr useful) gainBound

/-- A history-selected, locally accurate guide proposal supplies its own mass
floor.  Its predicted score gap therefore yields an explicit contraction rate
without any ambient-dimensional coverage assumption.  The remaining premises
are estimator accuracy, acceptance separation, and retained gain relative to
incumbent regret. -/
theorem guidedLocal [Fintype Move] [PseudoMetricSpace State]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (scores : ObjectiveSearch.Scores Move) (guide : Move)
    (incumbent best : State) (objective : Objective State)
    (modulus : ℝ → ℝ) (gain radius baseError margin : ℝ)
    (candidateError : Move → ℝ) (gainShare loss : ℝ)
    (radiusNonneg : 0 ≤ radius)
    (predicted : gain + baseError + candidateError guide ≤
      scores.incumbent - scores.candidate guide)
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (useful : modulus radius ≤ gain)
    (accurate : ScoreEnvelope objective incumbent candidate scores
      baseError candidateError)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin)
    (separated : ∀ move,
      gain - modulus radius ≤ progress objective incumbent (candidate move) →
        baseError + candidateError move + margin ≤ gain - modulus radius)
    (gainBound : gainShare * Regret.simpleRegret objective best incumbent - loss ≤
      gain - modulus radius) :
    (∑ move, law.weight move * Regret.simpleRegret objective best
      (ObjectiveSearch.commit incumbent candidate scores margin move)) ≤
        (1 - law.weight guide * gainShare) *
          Regret.simpleRegret objective best incumbent + law.weight guide * loss := by
  apply derivedLocal law candidate scores incumbent (candidate guide) best
    objective modulus gain radius baseError margin candidateError
    (law.weight guide) gainShare loss regular monotone
  · exact guideBetter objective incumbent (candidate guide)
      scores.incumbent (scores.candidate guide) baseError
      (candidateError guide) gain accurate.1 (accurate.2 guide) predicted
  · exact useful
  · exact accurate
  · exact marginSafe
  · exact separated
  · exact law.nonnegative guide
  · exact guideMass law candidate guide radius radiusNonneg
  · exact gainBound

/-- The guide-state contraction specialized to the implemented weighted-neighbor
score.  The estimator obligation is discharged from the source data, neighbor
lists, weight normalization, and candidate-specific observation envelopes. -/
theorem knnGuided [Fintype Move] [PseudoMetricSpace State] {capacity : ℕ}
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : Move → Forward.Vector capacity)
    (nearest : Move → List (Fin capacity)) (guide : Move)
    (incumbent best : State) (objective : Objective State)
    (modulus : ℝ → ℝ) (incumbentScore gain radius baseError margin : ℝ)
    (candidateError : Move → ℝ) (gainShare loss : ℝ)
    (radiusNonneg : 0 ≤ radius)
    (baseAccurate : |incumbentScore - objective incumbent| ≤ baseError)
    (nonnegative : ∀ move (position : Fin (nearest move).length),
      0 ≤ Neighbor.weight data params scale (distances move)
        (nearest move)[position.val])
    (floorBound : ∀ move,
      (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale
        (distances move) (nearest move))
    (errors : ∀ move (position : Fin (nearest move).length),
      |data.values (nearest move)[position.val] - objective (candidate move)| ≤
        candidateError move)
    (predicted : gain + baseError + candidateError guide ≤
      incumbentScore - Neighbor.mean data params scale
        (distances guide) (nearest guide))
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (useful : modulus radius ≤ gain)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin)
    (separated : ∀ move,
      gain - modulus radius ≤ progress objective incumbent (candidate move) →
        baseError + candidateError move + margin ≤ gain - modulus radius)
    (gainBound : gainShare * Regret.simpleRegret objective best incumbent - loss ≤
      gain - modulus radius) :
    (∑ move, law.weight move * Regret.simpleRegret objective best
      (ObjectiveSearch.commit incumbent candidate
        (knnScores data params scale distances nearest incumbentScore)
        margin move)) ≤
      (1 - law.weight guide * gainShare) *
        Regret.simpleRegret objective best incumbent + law.weight guide * loss := by
  apply guidedLocal law candidate
    (knnScores data params scale distances nearest incumbentScore) guide
    incumbent best objective modulus gain radius baseError margin candidateError
    gainShare loss radiusNonneg predicted regular monotone useful
  · exact knnEnvelope data params scale distances nearest objective incumbent
      candidate incumbentScore baseError candidateError baseAccurate
      nonnegative floorBound errors
  · exact marginSafe
  · exact separated
  · exact gainBound

/-- One safe proposal atom with certified gain is sufficient for an expected
regret contraction.  This is the probability-theoretic core, independent of
how the atom was generated, scored, or accepted. -/
theorem atomRegret [Fintype Move]
    (law : ObjectiveSearch.ProposalLaw Move) (objective : Objective State)
    (incumbent best : State) (commit : Move → State) (guide : Move)
    (gain gainShare loss : ℝ)
    (safe : Safe objective incumbent commit)
    (guideGain : gain ≤ progress objective incumbent (commit guide))
    (gainBound : gainShare * Regret.simpleRegret objective best incumbent - loss ≤ gain) :
    (∑ move, law.weight move * Regret.simpleRegret objective best (commit move)) ≤
        (1 - law.weight guide * gainShare) *
          Regret.simpleRegret objective best incumbent + law.weight guide * loss := by
  classical
  have gainLower : law.weight guide * gain ≤
      ∑ move, law.weight move * progress objective incumbent (commit move) := by
    calc
      law.weight guide * gain ≤ law.weight guide *
          progress objective incumbent (commit guide) :=
        mul_le_mul_of_nonneg_left guideGain (law.nonnegative guide)
      _ ≤ ∑ move, law.weight move * progress objective incumbent (commit move) :=
        Finset.single_le_sum
          (fun move _ => mul_nonneg (law.nonnegative move) (safe move))
          (Finset.mem_univ guide)
  have split :
      (∑ move, law.weight move * Regret.simpleRegret objective best (commit move)) +
          (∑ move, law.weight move * progress objective incumbent (commit move)) =
        Regret.simpleRegret objective best incumbent := by
    rw [← Finset.sum_add_distrib]
    calc
      _ = ∑ move, law.weight move *
          Regret.simpleRegret objective best incumbent := by
        apply Finset.sum_congr rfl
        intro move _
        unfold Regret.simpleRegret progress
        ring
      _ = Regret.simpleRegret objective best incumbent := by
        rw [← Finset.sum_mul, law.total, one_mul]
  calc
    (∑ move, law.weight move * Regret.simpleRegret objective best (commit move)) =
        Regret.simpleRegret objective best incumbent -
          (∑ move, law.weight move * progress objective incumbent (commit move)) := by
      linarith
    _ ≤ Regret.simpleRegret objective best incumbent - law.weight guide * gain :=
      sub_le_sub_left gainLower _
    _ ≤ Regret.simpleRegret objective best incumbent - law.weight guide *
        (gainShare * Regret.simpleRegret objective best incumbent - loss) :=
      sub_le_sub_left
        (mul_le_mul_of_nonneg_left gainBound (law.nonnegative guide)) _
    _ = (1 - law.weight guide * gainShare) *
          Regret.simpleRegret objective best incumbent + law.weight guide * loss := by
      ring

/-- One accepted guide atom is sufficient for contraction.  This sequential
result uses neither a metric cover nor a neighborhood regularity assumption:
all other proposal outcomes need only be safe, while the guide's score gap
certifies its gain and its law weight pays for that gain in expectation. -/
theorem guideRegret [Fintype Move]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (scores : ObjectiveSearch.Scores Move) (guide : Move)
    (incumbent best : State) (objective : Objective State)
    (baseError margin : ℝ) (candidateError : Move → ℝ)
    (gain gainShare loss : ℝ)
    (accurate : ScoreEnvelope objective incumbent candidate scores
      baseError candidateError)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin)
    (accepted : ObjectiveSearch.Accept scores margin guide)
    (predicted : gain + baseError + candidateError guide ≤
      scores.incumbent - scores.candidate guide)
    (gainBound : gainShare * Regret.simpleRegret objective best incumbent - loss ≤ gain) :
    (∑ move, law.weight move * Regret.simpleRegret objective best
      (ObjectiveSearch.commit incumbent candidate scores margin move)) ≤
        (1 - law.weight guide * gainShare) *
          Regret.simpleRegret objective best incumbent + law.weight guide * loss := by
  let commit := ObjectiveSearch.commit incumbent candidate scores margin
  apply atomRegret law objective incumbent best commit guide gain gainShare loss
  · exact pointSafe objective incumbent candidate scores baseError margin
      candidateError accurate marginSafe
  · rw [show commit guide = candidate guide by
      simp only [commit, ObjectiveSearch.commit, accepted, if_true]]
    exact guideBetter objective incumbent (candidate guide)
      scores.incumbent (scores.candidate guide) baseError
      (candidateError guide) gain accurate.1 (accurate.2 guide) predicted
  · exact gainBound

/-- A deterministic one-proposal transaction pays no candidate-pool mass
penalty.  Its certified retained gain gives the full factor
`1 - gainShare`, matching the sequential one-update regime. -/
theorem singleRegret
    (candidate : Fin 1 → State) (scores : ObjectiveSearch.Scores (Fin 1))
    (incumbent best : State) (objective : Objective State)
    (baseError margin candidateError gain gainShare loss : ℝ)
    (accurate : ScoreEnvelope objective incumbent candidate scores
      baseError (fun _ => candidateError))
    (marginSafe : baseError + candidateError ≤ margin)
    (accepted : ObjectiveSearch.Accept scores margin 0)
    (predicted : gain + baseError + candidateError ≤
      scores.incumbent - scores.candidate 0)
    (gainBound : gainShare * Regret.simpleRegret objective best incumbent - loss ≤ gain) :
    Regret.simpleRegret objective best
        (ObjectiveSearch.commit incumbent candidate scores margin 0) ≤
      (1 - gainShare) * Regret.simpleRegret objective best incumbent + loss := by
  have bound := guideRegret (ObjectiveSearch.uniformLaw 1) candidate scores 0
    incumbent best objective baseError margin (fun _ => candidateError)
    gain gainShare loss accurate (fun _ => marginSafe) accepted predicted gainBound
  simpa [ObjectiveSearch.uniformLaw] using bound

/-- The direct sequential contraction for the implemented weighted-neighbor
score.  Its premises expose the exact source-data facts needed to certify one
history-selected proposal without a global candidate-cover assumption. -/
theorem knnGuide [Fintype Move] {capacity : ℕ}
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : Move → Forward.Vector capacity)
    (nearest : Move → List (Fin capacity)) (guide : Move)
    (incumbent best : State) (objective : Objective State)
    (incumbentScore baseError margin : ℝ) (candidateError : Move → ℝ)
    (gain gainShare loss : ℝ)
    (baseAccurate : |incumbentScore - objective incumbent| ≤ baseError)
    (nonnegative : ∀ move (position : Fin (nearest move).length),
      0 ≤ Neighbor.weight data params scale (distances move)
        (nearest move)[position.val])
    (floorBound : ∀ move,
      (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale
        (distances move) (nearest move))
    (errors : ∀ move (position : Fin (nearest move).length),
      |data.values (nearest move)[position.val] - objective (candidate move)| ≤
        candidateError move)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin)
    (accepted : margin ≤ incumbentScore - Neighbor.mean data params scale
      (distances guide) (nearest guide))
    (predicted : gain + baseError + candidateError guide ≤
      incumbentScore - Neighbor.mean data params scale
        (distances guide) (nearest guide))
    (gainBound : gainShare * Regret.simpleRegret objective best incumbent - loss ≤ gain) :
    (∑ move, law.weight move * Regret.simpleRegret objective best
      (ObjectiveSearch.commit incumbent candidate
        (knnScores data params scale distances nearest incumbentScore)
        margin move)) ≤
      (1 - law.weight guide * gainShare) *
        Regret.simpleRegret objective best incumbent + law.weight guide * loss := by
  apply guideRegret law candidate
    (knnScores data params scale distances nearest incumbentScore) guide
    incumbent best objective baseError margin candidateError gain gainShare loss
  · exact knnEnvelope data params scale distances nearest objective incumbent
      candidate incumbentScore baseError candidateError baseAccurate
      nonnegative floorBound errors
  · exact marginSafe
  · change Neighbor.mean data params scale (distances guide) (nearest guide) +
        margin ≤ incumbentScore
    linarith
  · exact predicted
  · exact gainBound

/-- Covering a near-optimal set at scale `scale` yields a completely
derivative-free local contraction.  The additive term is exactly the objective
resolution lost through the modulus at that scale. -/
theorem coverLocal [Fintype Move] [PseudoMetricSpace State]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (scores : ObjectiveSearch.Scores Move)
    (incumbent best : State) (objective : Objective State)
    (modulus : ℝ → ℝ) (width : ℝ) (scale : ℝ≥0)
    (baseError margin massFloor : ℝ) (candidateError : Move → ℝ)
    (widthNonneg : 0 ≤ width)
    (covered : Metric.IsCover scale (nearSet objective best width)
      (Set.range candidate))
    (massNonneg : 0 ≤ massFloor)
    (weightBound : ∀ move, massFloor ≤ law.weight move)
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (useful : modulus scale ≤ Regret.simpleRegret objective best incumbent)
    (accurate : ScoreEnvelope objective incumbent candidate scores
      baseError candidateError)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin)
    (separated : ∀ move,
      Regret.simpleRegret objective best incumbent - modulus scale ≤
          progress objective incumbent (candidate move) →
        baseError + candidateError move + margin ≤
          Regret.simpleRegret objective best incumbent - modulus scale) :
    (∑ move, law.weight move * Regret.simpleRegret objective best
      (ObjectiveSearch.commit incumbent candidate scores margin move)) ≤
        (1 - massFloor) * Regret.simpleRegret objective best incumbent +
          massFloor * modulus scale := by
  have massBound : massFloor ≤
      ObjectiveSearch.nearMass law candidate best scale :=
    coverMass law candidate (nearSet objective best width) scale covered
      massFloor weightBound best (bestMember objective best width widthNonneg)
  have better : Regret.simpleRegret objective best incumbent ≤
      progress objective incumbent best := by
    unfold Regret.simpleRegret progress
    rfl
  have gainBound : 1 * Regret.simpleRegret objective best incumbent - modulus scale ≤
      Regret.simpleRegret objective best incumbent - modulus scale := by
    exact le_of_eq (by ring)
  simpa only [mul_one] using
    derivedLocal law candidate scores incumbent best best objective modulus
      (Regret.simpleRegret objective best incumbent) scale baseError margin
      candidateError massFloor 1 (modulus scale) regular monotone better useful
      accurate marginSafe separated massNonneg massBound gainBound

/-- For a uniform nonempty pool of `count` executable moves, near-optimal
coverage gives mass floor `1 / count` and hence its exact local regret bound. -/
theorem uniformCover (count : ℕ) [NeZero count]
    [PseudoMetricSpace State]
    (candidate : Fin count → State) (scores : ObjectiveSearch.Scores (Fin count))
    (incumbent best : State) (objective : Objective State)
    (modulus : ℝ → ℝ) (width : ℝ) (scale : ℝ≥0)
    (baseError margin : ℝ) (candidateError : Fin count → ℝ)
    (widthNonneg : 0 ≤ width)
    (covered : Metric.IsCover scale (nearSet objective best width)
      (Set.range candidate))
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (useful : modulus scale ≤ Regret.simpleRegret objective best incumbent)
    (accurate : ScoreEnvelope objective incumbent candidate scores
      baseError candidateError)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin)
    (separated : ∀ move,
      Regret.simpleRegret objective best incumbent - modulus scale ≤
          progress objective incumbent (candidate move) →
        baseError + candidateError move + margin ≤
          Regret.simpleRegret objective best incumbent - modulus scale) :
    (∑ move, (ObjectiveSearch.uniformLaw count).weight move *
      Regret.simpleRegret objective best
        (ObjectiveSearch.commit incumbent candidate scores margin move)) ≤
      (1 - (count : ℝ)⁻¹) * Regret.simpleRegret objective best incumbent +
        (count : ℝ)⁻¹ * modulus scale := by
  exact coverLocal (ObjectiveSearch.uniformLaw count) candidate scores incumbent best
    objective modulus width scale baseError margin (count : ℝ)⁻¹
    candidateError widthNonneg covered (inv_nonneg.mpr (Nat.cast_nonneg count))
    (fun move ↦ by simp [ObjectiveSearch.uniformLaw]) regular monotone useful
    accurate marginSafe separated

/-- The conditional weights of the existing finite oracle, at one time,
memory, and query state, form an executable proposal law. -/
def oracleLaw {Feedback Memory Noise : Type*} [Fintype Noise]
    (oracle : Synthesis.Oracle State Feedback Memory Noise)
    (time : ℕ) (memory : Memory) (state : State) :
    ObjectiveSearch.ProposalLaw Noise where
  weight := oracle.weight time memory state
  nonnegative := oracle.nonnegative time memory state
  total := oracle.mass time memory state

/-- All local obligations needed to turn one adaptive optimizer round into a
regret contraction.  Its candidate map denotes proposed parameter states;
`realizes` separately identifies the accepted state retained by the policy. -/
structure RoundCertificate {Feedback Memory Noise : Type*} [Fintype Noise]
    [PseudoMetricSpace State]
    (policy : Synthesis.Policy State Feedback Memory)
    (oracle : Synthesis.Oracle State Feedback Memory Noise)
    (objective : Objective State) (best : State)
    (factor penalty : ℝ) (time : ℕ) (memory : Memory) where
  center : State
  candidate : Noise → State
  scores : ObjectiveSearch.Scores Noise
  modulus : ℝ → ℝ
  gain : ℝ
  radius : ℝ
  baseError : ℝ
  margin : ℝ
  candidateError : Noise → ℝ
  regular : ObjectiveSearch.ValueModulus objective modulus
  monotone : Monotone modulus
  better : gain ≤ progress objective (policy.recommend memory) center
  useful : modulus radius ≤ gain
  accurate : ScoreEnvelope objective (policy.recommend memory) candidate scores
    baseError candidateError
  marginSafe : ∀ noise, baseError + candidateError noise ≤ margin
  separated : ∀ noise,
    gain - modulus radius ≤
        progress objective (policy.recommend memory) (candidate noise) →
      baseError + candidateError noise + margin ≤ gain - modulus radius
  realizes : ∀ noise,
    policy.recommend (Synthesis.advance policy oracle time memory noise) =
      ObjectiveSearch.commit (policy.recommend memory) candidate scores margin noise
  dominance :
    (1 - factor) *
          Regret.simpleRegret objective best (policy.recommend memory) - penalty ≤
      ObjectiveSearch.nearMass
        (oracleLaw oracle time memory (policy.query time memory))
        candidate center radius * (gain - modulus radius)

/-- A round certificate proves the exact local inequality expected by the
existing finite-oracle recurrence. -/
theorem RoundCertificate.local {Feedback Memory Noise : Type*} [Fintype Noise]
    [PseudoMetricSpace State]
    {policy : Synthesis.Policy State Feedback Memory}
    {oracle : Synthesis.Oracle State Feedback Memory Noise}
    {objective : Objective State} {best : State}
    {factor penalty : ℝ} {time : ℕ} {memory : Memory}
    (certificate : RoundCertificate policy oracle objective best
      factor penalty time memory) :
    (∑ noise, oracle.weight time memory (policy.query time memory) noise *
      Regret.simpleRegret objective best
        (policy.recommend (Synthesis.advance policy oracle time memory noise))) ≤
      factor * Regret.simpleRegret objective best (policy.recommend memory) +
        penalty := by
  have bound := rateLocal
    (oracleLaw oracle time memory (policy.query time memory))
    certificate.candidate certificate.scores (policy.recommend memory)
    certificate.center best objective certificate.modulus certificate.gain
    certificate.radius certificate.baseError certificate.margin
    certificate.candidateError factor penalty certificate.regular
    certificate.monotone certificate.better certificate.useful
    certificate.accurate certificate.marginSafe certificate.separated
    certificate.dominance
  calc
    (∑ noise, oracle.weight time memory (policy.query time memory) noise *
        Regret.simpleRegret objective best
          (policy.recommend (Synthesis.advance policy oracle time memory noise))) =
        ∑ noise,
          (oracleLaw oracle time memory (policy.query time memory)).weight noise *
            Regret.simpleRegret objective best
              (ObjectiveSearch.commit (policy.recommend memory)
                certificate.candidate certificate.scores certificate.margin noise) := by
      apply Finset.sum_congr rfl
      intro noise _
      rw [certificate.realizes noise]
      rfl
    _ ≤ factor * Regret.simpleRegret objective best (policy.recommend memory) +
        penalty := bound

/-- Per-round geometric and estimator certificates therefore imply the
finite-horizon regret envelope for the actual adaptive policy and oracle. -/
theorem trajectoryBound {Feedback Memory Noise : Type*} [Fintype Noise]
    [PseudoMetricSpace State]
    (policy : Synthesis.Policy State Feedback Memory)
    (oracle : Synthesis.Oracle State Feedback Memory Noise)
    (objective : Objective State) (best : State)
    (factor penalty : ℝ) (factorNonneg : 0 ≤ factor)
    (certificate : ∀ time memory,
      RoundCertificate policy oracle objective best factor penalty time memory)
    (time steps : ℕ) (memory : Memory) :
    Synthesis.expected policy oracle objective best time steps memory ≤
      factor ^ steps *
          Regret.simpleRegret objective best (policy.recommend memory) +
        penalty * ∑ index ∈ Finset.range steps, factor ^ index := by
  exact Rate.expected policy oracle objective best factor penalty factorNonneg
    (fun round state ↦ (certificate round state).local) time steps memory

/-- End-to-end one-round regret certificate for the implemented weighted KNN
score, finite proposal law, and incumbent-relative acceptance rule. -/
theorem knnRegret [Fintype Move] {capacity : ℕ}
    [MeasurableSpace History] [MeasurableSpace State] [MeasurableSpace Move]
    [MeasurableSingletonClass Move]
    (law : ObjectiveSearch.ProposalLaw Move) (candidate : Move → State)
    (data : Neighbor.Data capacity) (params : Neighbor.Parameters)
    (scale : ℝ) (distances : Move → Forward.Vector capacity)
    (nearest : Move → List (Fin capacity)) (incumbentScore : ℝ)
    (history : History) (incumbent best : State) (trust : ℝ≥0)
    (objective : Objective State) (baseError margin gain : ℝ)
    (candidateError : Move → ℝ)
    (optimal : Regret.Optimal objective best)
    (objectiveMeas : Measurable objective)
    (baseAccurate : |incumbentScore - objective incumbent| ≤ baseError)
    (nonnegative : ∀ move (position : Fin (nearest move).length),
      0 ≤ Neighbor.weight data params scale (distances move)
        (nearest move)[position.val])
    (floorBound : ∀ move,
      (1 / 10^12 : ℝ) ≤ Accuracy.weightTotal data params scale
        (distances move) (nearest move))
    (errors : ∀ move (position : Fin (nearest move).length),
      |data.values (nearest move)[position.val] - objective (candidate move)| ≤
        candidateError move)
    (marginSafe : ∀ move, baseError + candidateError move ≤ margin)
    (separated : ∀ move,
      gain ≤ progress objective incumbent (candidate move) →
        baseError + candidateError move + margin ≤ gain) :
    meanRegret (finiteProposal (History := History) (State := State) law)
        history incumbent trust objective best
          (ObjectiveSearch.commit incumbent candidate
            (knnScores data params scale distances nearest incumbentScore) margin) +
        ENNReal.ofReal gain * ENNReal.ofReal
          (ObjectiveSearch.goodMass law objective incumbent candidate gain) ≤
      ENNReal.ofReal (Regret.simpleRegret objective best incumbent) := by
  exact pointRegret law candidate
    (knnScores data params scale distances nearest incumbentScore)
    history incumbent best trust objective baseError margin gain candidateError
    optimal objectiveMeas
    (knnEnvelope data params scale distances nearest objective incumbent candidate
      incumbentScore baseError candidateError baseAccurate nonnegative floorBound errors)
    marginSafe separated

/-- If retained gain times capture mass covers a declared fraction of incumbent
simple regret, the same fraction is removed from the one-round expected-regret
upper bound.  This is the local contraction obligation for proposal search. -/
theorem contractBound [MeasurableSpace History] [MeasurableSpace State]
    [MeasurableSpace Sample]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent best : State) (trust : ℝ≥0)
    (realize : Sample → State) (objective : Objective State)
    (commit : Sample → State)
    (gain : ℝ) (fraction : ℝ≥0∞)
    (optimal : Regret.Optimal objective best)
    (safe : Safe objective incumbent commit)
    (goodMeas : MeasurableSet
      (realize ⁻¹' betterSet objective incumbent gain))
    (objectiveMeas : Measurable objective)
    (commitMeas : Measurable commit)
    (retains : Retains objective incumbent realize commit gain)
    (coverage : fraction *
        ENNReal.ofReal (Regret.simpleRegret objective best incumbent) ≤
      ENNReal.ofReal gain *
        capture proposal history incumbent trust realize objective gain) :
    meanRegret proposal history incumbent trust objective best commit ≤
      ENNReal.ofReal (Regret.simpleRegret objective best incumbent) -
        fraction * ENNReal.ofReal
          (Regret.simpleRegret objective best incumbent) := by
  have captureFinite :
      capture proposal history incumbent trust realize objective gain ≠ ∞ :=
    ne_top_of_le_ne_top ENNReal.one_ne_top
      (captureLe proposal history incumbent trust realize objective gain)
  have coveredFinite :
      ENNReal.ofReal gain *
          capture proposal history incumbent trust realize objective gain ≠ ∞ :=
    ENNReal.mul_ne_top ENNReal.ofReal_ne_top captureFinite
  have fractionFinite : fraction *
      ENNReal.ofReal (Regret.simpleRegret objective best incumbent) ≠ ∞ :=
    ne_top_of_le_ne_top coveredFinite coverage
  apply ENNReal.le_sub_of_add_le_right fractionFinite
  calc
    meanRegret proposal history incumbent trust objective best commit +
          fraction * ENNReal.ofReal
            (Regret.simpleRegret objective best incumbent) ≤
        meanRegret proposal history incumbent trust objective best commit +
          ENNReal.ofReal gain *
            capture proposal history incumbent trust realize objective gain := by
      gcongr
    _ ≤ ENNReal.ofReal (Regret.simpleRegret objective best incumbent) :=
      captureRegret proposal history incumbent best trust realize objective commit gain
        optimal safe goodMeas objectiveMeas commitMeas retains

/-- Quantitative accessibility, local objective regularity, and retained
acceptance compose into an expected-improvement certificate. -/
theorem accessMean [MeasurableSpace History] [MeasurableSpace State]
    [MeasurableSpace Sample]
    [PseudoMetricSpace State]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent center : State) (trust : ℝ≥0)
    (realize : Sample → State) (objective : Objective State)
    (modulus : ℝ → ℝ) (commit : Sample → State)
    (gain radius : ℝ) (mass : ℝ≥0∞)
    (access : Accessible proposal history incumbent trust realize center radius mass)
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (goodMeas : MeasurableSet
      (realize ⁻¹' betterSet objective incumbent (gain - modulus radius)))
    (retains : Retains objective incumbent realize commit (gain - modulus radius)) :
    ENNReal.ofReal (gain - modulus radius) * mass ≤
      meanGain proposal history incumbent trust objective commit := by
  have massBound := accessibleCapture proposal history incumbent center trust realize
    objective modulus gain radius mass access regular monotone better
  calc
    ENNReal.ofReal (gain - modulus radius) * mass ≤
        ENNReal.ofReal (gain - modulus radius) *
          capture proposal history incumbent trust realize objective
            (gain - modulus radius) :=
      by gcongr
    _ ≤ meanGain proposal history incumbent trust objective commit :=
      captureMean proposal history incumbent trust realize objective commit
        (gain - modulus radius) goodMeas retains

/-- Accessibility and objective regularity consequently certify a one-round
expected simple-regret decrement without choosing a derivative model. -/
theorem accessRegret [MeasurableSpace History] [MeasurableSpace State]
    [MeasurableSpace Sample]
    [PseudoMetricSpace State]
    (proposal : Proposal History State Sample) [IsMarkovKernel proposal]
    (history : History) (incumbent center best : State) (trust : ℝ≥0)
    (realize : Sample → State) (objective : Objective State)
    (modulus : ℝ → ℝ) (commit : Sample → State)
    (gain radius : ℝ) (mass : ℝ≥0∞)
    (access : Accessible proposal history incumbent trust realize center radius mass)
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (optimal : Regret.Optimal objective best)
    (safe : Safe objective incumbent commit)
    (objectiveMeas : Measurable objective)
    (commitMeas : Measurable commit)
    (goodMeas : MeasurableSet
      (realize ⁻¹' betterSet objective incumbent (gain - modulus radius)))
    (retains : Retains objective incumbent realize commit (gain - modulus radius)) :
    meanRegret proposal history incumbent trust objective best commit +
        ENNReal.ofReal (gain - modulus radius) * mass ≤
      ENNReal.ofReal (Regret.simpleRegret objective best incumbent) := by
  calc
    meanRegret proposal history incumbent trust objective best commit +
          ENNReal.ofReal (gain - modulus radius) * mass ≤
        meanRegret proposal history incumbent trust objective best commit +
          meanGain proposal history incumbent trust objective commit := by
      gcongr
      exact accessMean proposal history incumbent center trust realize objective modulus commit
        gain radius mass access regular monotone better goodMeas retains
    _ = ENNReal.ofReal (Regret.simpleRegret objective best incumbent) :=
      regretSplit proposal history incumbent best trust objective commit
        optimal safe objectiveMeas commitMeas

/-- A reference-scale mass certificate and Mathlib's local doubling law yield
an inner-scale one-round regret decrement.  The scaling constant, rather than
ambient parameter count, controls the probability loss across scales. -/
theorem scaleRegret [MeasurableSpace History] [MeasurableSpace State]
    [PseudoMetricSpace State]
    (proposal : Proposal History State State) [IsMarkovKernel proposal]
    (history : History) (incumbent center best : State) (trust : ℝ≥0)
    (objective : Objective State) (modulus : ℝ → ℝ)
    (commit : State → State) (factor radius cap gain : ℝ)
    [IsUnifLocDoublingMeasure (proposal (history, incumbent, trust))]
    (mass : ℝ≥0∞)
    (outer : mass ≤
      ballMass proposal history incumbent trust id center (factor * radius))
    (factorBound : factor ∈ Ioc 0 cap)
    (radiusBound : radius ≤
      IsUnifLocDoublingMeasure.scalingScaleOf
        (proposal (history, incumbent, trust)) cap)
    (regular : ObjectiveSearch.ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (optimal : Regret.Optimal objective best)
    (safe : Safe objective incumbent commit)
    (objectiveMeas : Measurable objective)
    (commitMeas : Measurable commit)
    (retains : Retains objective incumbent id commit
      (gain - modulus radius)) :
    meanRegret proposal history incumbent trust objective best commit +
        ENNReal.ofReal (gain - modulus radius) *
          (mass / IsUnifLocDoublingMeasure.scalingConstantOf
            (proposal (history, incumbent, trust)) cap) ≤
      ENNReal.ofReal (Regret.simpleRegret objective best incumbent) := by
  apply accessRegret proposal history incumbent center best trust id objective
    modulus commit gain radius
    (mass / IsUnifLocDoublingMeasure.scalingConstantOf
      (proposal (history, incumbent, trust)) cap)
  · exact scaleAccess proposal history incumbent trust center factor radius cap
      mass outer factorBound radiusBound
  · exact regular
  · exact monotone
  · exact better
  · exact optimal
  · exact safe
  · exact objectiveMeas
  · exact commitMeas
  · exact (measurableBetter objective incumbent (gain - modulus radius)
      objectiveMeas).preimage measurable_id
  · exact retains

/-- Isometric re-encodings preserve the internal covering complexity of the
same near-optimal state set. -/
theorem coverImage [MetricSpace State] [PseudoMetricSpace Target]
    (objective : Objective State) (best : State) (width : ℝ) (scale : ℝ≥0)
    (encode : State → Target) (isometry : Isometry encode) :
    Metric.coveringNumber scale (encode '' nearSet objective best width) =
      coverProfile objective best width scale := by
  exact isometry.coveringNumber_image

#print axioms coverMass
#print axioms coverObstruct
#print axioms coverLocal
#print axioms uniformCover
#print axioms scaleAccess
#print axioms scaleRegret
#print axioms disjointMass
#print axioms derivedLocal
#print axioms guideBetter
#print axioms guideMass
#print axioms guidedLocal
#print axioms knnGuided
#print axioms atomRegret
#print axioms guideRegret
#print axioms singleRegret
#print axioms knnGuide
#print axioms trajectoryBound

end

end BauxResearch.Optimization.Reachability
