import BauxResearch.Optimization.Step
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Topology.MetricSpace.Basic

/-!
# Objective-query search

This module gives the central optimizer proof path no derivative vocabulary.
It uses objective values, a metric, a finite proposal law, observation error,
and an acceptance margin.

One proposal evaluation compares a candidate score with a retained incumbent
score. A sufficiently good proposal is accepted despite bounded score error;
every accepted proposal has a certified true decrease when the margin covers
that error. The final expectation theorem turns proposal capture mass into a
one-step progress bound.
-/

namespace BauxResearch.Optimization.ObjectiveSearch

open scoped BigOperators
noncomputable section

variable {History State Move Seed Slot : Type*}

/-- Value continuity stated without a derivative or linear approximation. -/
def ValueModulus [PseudoMetricSpace State]
    (objective : Objective State) (modulus : ℝ → ℝ) : Prop :=
  ∀ first second,
    |objective first - objective second| ≤ modulus (dist first second)

/-- A value decrease is reachable inside a metric trust region. -/
def Reachable [PseudoMetricSpace State]
    (objective : Objective State) (incumbent : State)
    (radius gain : ℝ) : Prop :=
  ∃ candidate ∈ Metric.closedBall incumbent radius,
    gain ≤ progress objective incumbent candidate

/-- A finite proposal law used by one optimizer transaction. -/
structure ProposalLaw (Move : Type*) [Fintype Move] where
  weight : Move → ℝ
  nonnegative : ∀ move, 0 ≤ weight move
  total : ∑ move, weight move = 1

/-- A history-conditioned seed law and its deterministic executable pool.
Selection and acceptance remain separate optimizer components. -/
structure ProposalPolicy
    (History State Seed Slot : Type*) [Fintype Seed] where
  law : History → State → ProposalLaw Seed
  generate : History → State → Seed → Slot → State

/-- Uniform choice from a nonempty finite executable pool. -/
noncomputable def uniformLaw (count : ℕ) [NeZero count] :
    ProposalLaw (Fin count) where
  weight := fun _ => (count : ℝ)⁻¹
  nonnegative := fun _ => inv_nonneg.mpr (Nat.cast_nonneg count)
  total := by
    simp [NeZero.ne count]

/-- Probability mass assigned to proposals attaining a declared true gain. -/
noncomputable def goodMass [Fintype Move]
    (law : ProposalLaw Move) (objective : Objective State)
    (incumbent : State) (candidate : Move → State) (gain : ℝ) : ℝ :=
  ∑ move with gain ≤ progress objective incumbent (candidate move), law.weight move

/-- Proposal mass landing in a metric neighborhood of a comparison state. -/
noncomputable def nearMass [Fintype Move] [PseudoMetricSpace State]
    (law : ProposalLaw Move) (candidate : Move → State)
    (center : State) (radius : ℝ) : ℝ :=
  ∑ move with dist (candidate move) center ≤ radius, law.weight move

/-- One captured slot in a uniform finite pool contributes at least one over
the pool cardinality. No independence or continuous density is assumed. -/
theorem hitMass (count : ℕ) [NeZero count] [PseudoMetricSpace State]
    (candidate : Fin count → State) (center : State) (radius : ℝ)
    (move : Fin count) (hit : dist (candidate move) center ≤ radius) :
    (count : ℝ)⁻¹ ≤ nearMass (uniformLaw count) candidate center radius := by
  classical
  unfold nearMass
  have member : move ∈ Finset.filter
      (fun slot => dist (candidate slot) center ≤ radius) Finset.univ := by
    simp [hit]
  have bound : (uniformLaw count).weight move ≤
      ∑ slot ∈ Finset.filter
        (fun slot => dist (candidate slot) center ≤ radius) Finset.univ,
          (uniformLaw count).weight slot :=
    Finset.single_le_sum
      (fun slot _ => (uniformLaw count).nonnegative slot) member
  simpa [uniformLaw] using bound

/-- The source-sized four-slot pool assigns at least one quarter of its
uniform selection mass to any ball containing one executable slot. -/
theorem quarterMass [PseudoMetricSpace State]
    (candidate : Fin 4 → State) (center : State) (radius : ℝ)
    (move : Fin 4) (hit : dist (candidate move) center ≤ radius) :
    (1 : ℝ) / 4 ≤ nearMass (uniformLaw 4) candidate center radius := by
  simpa using hitMass 4 candidate center radius move hit

/-- A scale-local lower bound on proposal mass around a comparison state.
The radius interval is explicit because a finite executable law cannot satisfy
a positive power-law bound at every arbitrarily small radius. -/
def BallMass [Fintype Move] [PseudoMetricSpace State]
    (law : ProposalLaw Move) (candidate : Move → State) (center : State)
    (coefficient : ℝ) (exponent : ℕ) (minRadius maxRadius : ℝ) : Prop :=
  ∀ radius, minRadius ≤ radius → radius ≤ maxRadius →
    coefficient * radius ^ exponent ≤ nearMass law candidate center radius

/-- A proposal law assigning no executable candidate to a ball has zero mass
there, independent of how the candidate scores are modeled. -/
theorem zeroMass [Fintype Move] [PseudoMetricSpace State]
    (law : ProposalLaw Move) (candidate : Move → State)
    (center : State) (radius : ℝ)
    (missed : ∀ move, radius < dist (candidate move) center) :
    nearMass law candidate center radius = 0 := by
  classical
  unfold nearMass
  have empty : Finset.filter
      (fun move => dist (candidate move) center ≤ radius) Finset.univ = ∅ := by
    ext move
    simp [not_le.mpr (missed move)]
  rw [empty]
  simp

/-- Metric proximity to a better state preserves its gain up to the declared
value modulus. This is the derivative-free perturbation lemma. -/
theorem nearGood [PseudoMetricSpace State]
    (objective : Objective State) (modulus : ℝ → ℝ)
    (incumbent center : State) (candidate : Move → State)
    (gain radius : ℝ) (move : Move)
    (continuous : ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (near : dist (candidate move) center ≤ radius) :
    gain - modulus radius ≤ progress objective incumbent (candidate move) := by
  have valueBound : objective (candidate move) - objective center ≤
      modulus (dist (candidate move) center) :=
    (le_abs_self (objective (candidate move) - objective center)).trans
      (continuous (candidate move) center)
  have radiusBound := monotone near
  unfold progress at better ⊢
  linarith

/-- If one executable pool member lands near a better state, simultaneous score
accuracy transfers its local gain to the actually selected member. This is the
source-shaped deterministic selector theorem; it does not choose a slot
uniformly and does not assume independent candidates. -/
theorem selectBound [PseudoMetricSpace State]
    (objective : Objective State) (modulus : ℝ → ℝ)
    (incumbent center : State) (candidate : Move → State)
    (estimate : Move → ℝ) (selected comparison : Move)
    (gain radius epsilon slack : ℝ)
    (continuous : ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (near : dist (candidate comparison) center ≤ radius)
    (accurate : UniformError
      (fun move => objective (candidate move)) estimate epsilon)
    (chosen : ApproxMin estimate selected slack) :
    gain - modulus radius - (2 * epsilon + slack) ≤
      progress objective incumbent (candidate selected) := by
  have nearGain := nearGood objective modulus incumbent center candidate
    gain radius comparison continuous monotone better near
  have selectedLower := Progress.lowerBound objective incumbent candidate estimate
    selected comparison epsilon slack accurate chosen
  unfold opportunity at selectedLower
  unfold progress at nearGain selectedLower ⊢
  linarith

/-- A random seed hits when at least one member of its deterministically
generated pool enters the declared metric neighborhood. Randomness belongs to
the seed law; the selector still examines the complete generated pool. -/
def PoolHit [PseudoMetricSpace State]
    (candidate : Seed → Slot → State) (center : State)
    (radius : ℝ) (seed : Seed) : Prop :=
  ∃ slot, dist (candidate seed slot) center ≤ radius

/-- Probability that a random seed generates a pool containing a nearby
candidate. Dependence among slots is retained inside the existential event. -/
noncomputable def poolMass [Fintype Seed] [PseudoMetricSpace State]
    (law : ProposalLaw Seed) (candidate : Seed → Slot → State)
    (center : State) (radius : ℝ) : ℝ :=
  by
    classical
    exact ∑ seed with PoolHit candidate center radius seed, law.weight seed

/-- Conditional capture probability of a history-conditioned proposal policy. -/
noncomputable def captureMass [Fintype Seed] [PseudoMetricSpace State]
    (policy : ProposalPolicy History State Seed Slot)
    (history : History) (incumbent center : State) (radius : ℝ) : ℝ :=
  poolMass (policy.law history incumbent) (policy.generate history incumbent)
    center radius

/-- Expected progress of the candidate selected from each seeded pool. -/
noncomputable def selectMean [Fintype Seed]
    (law : ProposalLaw Seed) (objective : Objective State)
    (incumbent : State) (candidate : Seed → Slot → State)
    (selected : Seed → Slot) : ℝ :=
  ∑ seed, law.weight seed *
    progress objective incumbent (candidate seed (selected seed))

/-- A seeded pool hit transfers to its selected member under simultaneous
within-pool score accuracy. No independence between slots is used. -/
theorem hitSelect [PseudoMetricSpace State]
    (objective : Objective State) (modulus : ℝ → ℝ)
    (incumbent center : State) (candidate : Seed → Slot → State)
    (estimate : Seed → Slot → ℝ) (selected : Seed → Slot) (seed : Seed)
    (gain radius epsilon slack : ℝ)
    (continuous : ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (hit : PoolHit candidate center radius seed)
    (accurate : UniformError
      (fun slot => objective (candidate seed slot)) (estimate seed) epsilon)
    (chosen : ApproxMin (estimate seed) (selected seed) slack) :
    gain - modulus radius - (2 * epsilon + slack) ≤
      progress objective incumbent (candidate seed (selected seed)) := by
  rcases hit with ⟨comparison, near⟩
  exact selectBound objective modulus incumbent center (candidate seed)
    (estimate seed) (selected seed) comparison gain radius epsilon slack
    continuous monotone better near accurate chosen

/-- Conditional pool-hit probability becomes expected selected progress when
miss rounds are protected from objective increase. This is the probability
interface needed from a history-conditioned proposal generator. -/
theorem seedBound [Fintype Seed] [PseudoMetricSpace State]
    (law : ProposalLaw Seed) (objective : Objective State)
    (modulus : ℝ → ℝ) (incumbent center : State)
    (candidate : Seed → Slot → State) (estimate : Seed → Slot → ℝ)
    (selected : Seed → Slot) (gain radius epsilon slack : ℝ)
    (continuous : ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (accurate : ∀ seed, UniformError
      (fun slot => objective (candidate seed slot)) (estimate seed) epsilon)
    (chosen : ∀ seed, ApproxMin (estimate seed) (selected seed) slack)
    (safe : ∀ seed, 0 ≤ progress objective incumbent
      (candidate seed (selected seed))) :
    poolMass law candidate center radius *
        (gain - modulus radius - (2 * epsilon + slack)) ≤
      selectMean law objective incumbent candidate selected := by
  classical
  unfold poolMass selectMean
  rw [Finset.sum_filter]
  rw [Finset.sum_mul]
  apply Finset.sum_le_sum
  intro seed _
  by_cases hit : PoolHit candidate center radius seed
  · simp only [hit, if_true]
    exact mul_le_mul_of_nonneg_left
      (hitSelect objective modulus incumbent center candidate estimate selected
        seed gain radius epsilon slack continuous monotone better hit
        (accurate seed) (chosen seed))
      (law.nonnegative seed)
  · simp only [hit, if_false, zero_mul]
    exact mul_nonneg (law.nonnegative seed) (safe seed)

/-- Neighborhood capture mass is a lower bound on good-proposal mass. -/
theorem massBound [Fintype Move] [PseudoMetricSpace State]
    (law : ProposalLaw Move) (objective : Objective State)
    (modulus : ℝ → ℝ) (incumbent center : State)
    (candidate : Move → State) (gain radius : ℝ)
    (continuous : ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center) :
    nearMass law candidate center radius ≤
      goodMass law objective incumbent candidate (gain - modulus radius) := by
  classical
  unfold nearMass goodMass
  apply Finset.sum_le_sum_of_subset_of_nonneg
  · intro move
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact nearGood objective modulus incumbent center candidate gain radius move
      continuous monotone better
  · intro move _ _
    exact law.nonnegative move

/-- One retained incumbent score and one candidate score per proposal. -/
structure Scores (Move : Type*) where
  incumbent : ℝ
  candidate : Move → ℝ

/-- Score accuracy is stated directly against objective values. -/
def ScoreError
    (objective : Objective State) (incumbent : State)
    (candidate : Move → State) (scores : Scores Move)
    (baseError candidateError : ℝ) : Prop :=
  |scores.incumbent - objective incumbent| ≤ baseError ∧
    ∀ move, |scores.candidate move - objective (candidate move)| ≤ candidateError

/-- A candidate is accepted only after clearing the incumbent score margin. -/
abbrev Accept (scores : Scores Move) (margin : ℝ) (move : Move) : Prop :=
  scores.candidate move + margin ≤ scores.incumbent

/-- Rejecting a proposal retains the incumbent state. -/
noncomputable def commit
    (incumbent : State) (candidate : Move → State)
    (scores : Scores Move) (margin : ℝ) (move : Move) : State :=
  if Accept scores margin move then candidate move else incumbent

/-- Every accepted proposal has a true objective-progress bound. -/
theorem acceptBound
    (objective : Objective State) (incumbent : State)
    (candidate : Move → State) (scores : Scores Move)
    (baseError candidateError margin : ℝ) (move : Move)
    (accurate : ScoreError objective incumbent candidate scores baseError candidateError)
    (accepted : Accept scores margin move) :
    margin - baseError - candidateError ≤
      progress objective incumbent (candidate move) := by
  have baseUpper := (abs_le.mp accurate.1).2
  have candidateLower := (abs_le.mp (accurate.2 move)).1
  unfold Accept at accepted
  unfold progress
  linarith

/-- A proposal whose true gain clears margin plus observation error is accepted. -/
theorem goodAccepted
    (objective : Objective State) (incumbent : State)
    (candidate : Move → State) (scores : Scores Move)
    (baseError candidateError margin gain : ℝ) (move : Move)
    (accurate : ScoreError objective incumbent candidate scores baseError candidateError)
    (good : gain ≤ progress objective incumbent (candidate move))
    (separated : baseError + candidateError + margin ≤ gain) :
    Accept scores margin move := by
  have baseLower := (abs_le.mp accurate.1).1
  have candidateUpper := (abs_le.mp (accurate.2 move)).2
  unfold Accept
  unfold progress at good
  linarith

/-- If the acceptance margin covers score error, commit never increases the
true objective. Rejection has exactly zero progress. -/
theorem commitNonneg
    (objective : Objective State) (incumbent : State)
    (candidate : Move → State) (scores : Scores Move)
    (baseError candidateError margin : ℝ) (move : Move)
    (accurate : ScoreError objective incumbent candidate scores baseError candidateError)
    (safe : baseError + candidateError ≤ margin) :
    0 ≤ progress objective incumbent
      (commit incumbent candidate scores margin move) := by
  by_cases accepted : Accept scores margin move
  · rw [commit, if_pos accepted]
    have bound := acceptBound objective incumbent candidate scores
      baseError candidateError margin move accurate accepted
    linarith
  · rw [commit, if_neg accepted]
    unfold progress
    linarith

/-- Every sufficiently good proposal is committed and retains its true gain. -/
theorem goodCommit
    (objective : Objective State) (incumbent : State)
    (candidate : Move → State) (scores : Scores Move)
    (baseError candidateError margin gain : ℝ) (move : Move)
    (accurate : ScoreError objective incumbent candidate scores baseError candidateError)
    (good : gain ≤ progress objective incumbent (candidate move))
    (separated : baseError + candidateError + margin ≤ gain) :
    gain ≤ progress objective incumbent
      (commit incumbent candidate scores margin move) := by
  have accepted := goodAccepted objective incumbent candidate scores
    baseError candidateError margin gain move accurate good separated
  simpa [commit, accepted] using good

/-- Expected true progress of the incumbent-relative commit under the proposal law. -/
noncomputable def meanProgress [Fintype Move]
    (law : ProposalLaw Move) (objective : Objective State)
    (incumbent : State) (candidate : Move → State)
    (scores : Scores Move) (margin : ℝ) : ℝ :=
  ∑ move, law.weight move *
    progress objective incumbent (commit incumbent candidate scores margin move)

/-- Proposal capture mass becomes expected progress. The theorem needs no
candidate independence because one transaction evaluates one proposal. -/
theorem meanBound [Fintype Move]
    (law : ProposalLaw Move) (objective : Objective State)
    (incumbent : State) (candidate : Move → State)
    (scores : Scores Move) (baseError candidateError margin gain : ℝ)
    (accurate : ScoreError objective incumbent candidate scores baseError candidateError)
    (safe : baseError + candidateError ≤ margin)
    (separated : baseError + candidateError + margin ≤ gain) :
    goodMass law objective incumbent candidate gain * gain ≤
      meanProgress law objective incumbent candidate scores margin := by
  classical
  unfold goodMass meanProgress
  rw [Finset.sum_filter]
  rw [Finset.sum_mul]
  apply Finset.sum_le_sum
  intro move _
  by_cases good : gain ≤ progress objective incumbent (candidate move)
  · simp only [good, if_true]
    exact mul_le_mul_of_nonneg_left
      (goodCommit objective incumbent candidate scores baseError candidateError
        margin gain move accurate good separated)
      (law.nonnegative move)
  · simp only [good, if_false, zero_mul]
    exact mul_nonneg (law.nonnegative move)
      (commitNonneg objective incumbent candidate scores baseError candidateError
        margin move accurate safe)

/-- A reachable better state, proposal coverage around it, and bounded score
error compose into a one-transaction expected progress certificate. -/
theorem localBound [Fintype Move] [PseudoMetricSpace State]
    (law : ProposalLaw Move) (objective : Objective State)
    (modulus : ℝ → ℝ) (incumbent center : State)
    (candidate : Move → State) (scores : Scores Move)
    (gain radius baseError candidateError margin : ℝ)
    (continuous : ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (useful : modulus radius ≤ gain)
    (accurate : ScoreError objective incumbent candidate scores baseError candidateError)
    (safe : baseError + candidateError ≤ margin)
    (separated : baseError + candidateError + margin ≤ gain - modulus radius) :
    nearMass law candidate center radius * (gain - modulus radius) ≤
      meanProgress law objective incumbent candidate scores margin := by
  calc
    nearMass law candidate center radius * (gain - modulus radius) ≤
        goodMass law objective incumbent candidate (gain - modulus radius) *
          (gain - modulus radius) :=
      mul_le_mul_of_nonneg_right
        (massBound law objective modulus incumbent center candidate gain radius
          continuous monotone better)
        (sub_nonneg.mpr useful)
    _ ≤ meanProgress law objective incumbent candidate scores margin :=
      meanBound law objective incumbent candidate scores baseError candidateError
        margin (gain - modulus radius) accurate safe separated

/-- For the concrete four-slot pool, one captured slot is already an exact
one-quarter mass certificate. This is a finite-pool result, not a density or
independence claim. -/
theorem slotBound [PseudoMetricSpace State]
    (objective : Objective State) (modulus : ℝ → ℝ)
    (incumbent center : State) (candidate : Fin 4 → State)
    (scores : Scores (Fin 4))
    (gain radius baseError candidateError margin : ℝ) (move : Fin 4)
    (hit : dist (candidate move) center ≤ radius)
    (continuous : ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (useful : modulus radius ≤ gain)
    (accurate : ScoreError objective incumbent candidate scores baseError candidateError)
    (safe : baseError + candidateError ≤ margin)
    (separated : baseError + candidateError + margin ≤ gain - modulus radius) :
    ((1 : ℝ) / 4) * (gain - modulus radius) ≤
      meanProgress (uniformLaw 4) objective incumbent candidate scores margin := by
  have mass := quarterMass candidate center radius move hit
  have gap : 0 ≤ gain - modulus radius := sub_nonneg.mpr useful
  exact (mul_le_mul_of_nonneg_right mass gap).trans
    (localBound (uniformLaw 4) objective modulus incumbent center candidate scores
      gain radius baseError candidateError margin continuous monotone better useful
      accurate safe separated)

/-- A power-law ball-mass certificate turns local proposal geometry into an
explicit progress rate over the scale range where that certificate is valid. -/
theorem scaledBound [Fintype Move] [PseudoMetricSpace State]
    (law : ProposalLaw Move) (objective : Objective State)
    (modulus : ℝ → ℝ) (incumbent center : State)
    (candidate : Move → State) (scores : Scores Move)
    (coefficient : ℝ) (exponent : ℕ) (minRadius maxRadius : ℝ)
    (gain radius baseError candidateError margin : ℝ)
    (coverage : BallMass law candidate center coefficient exponent minRadius maxRadius)
    (radiusLower : minRadius ≤ radius) (radiusUpper : radius ≤ maxRadius)
    (continuous : ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (useful : modulus radius ≤ gain)
    (accurate : ScoreError objective incumbent candidate scores baseError candidateError)
    (safe : baseError + candidateError ≤ margin)
    (separated : baseError + candidateError + margin ≤ gain - modulus radius) :
    coefficient * radius ^ exponent * (gain - modulus radius) ≤
      meanProgress law objective incumbent candidate scores margin := by
  have massLower := coverage radius radiusLower radiusUpper
  exact (mul_le_mul_of_nonneg_right massLower (sub_nonneg.mpr useful)).trans
    (localBound law objective modulus incumbent center candidate scores gain radius
      baseError candidateError margin continuous monotone better useful accurate safe separated)

/-- For a linear value modulus, selecting the radius `gain / (2 * L)` retains
half the reachable gain. The exponent is a proposal-mass exponent, not the
ambient parameter count and not an asserted intrinsic dimension. -/
theorem linearBound [Fintype Move] [PseudoMetricSpace State]
    (law : ProposalLaw Move) (objective : Objective State)
    (incumbent center : State) (candidate : Move → State) (scores : Scores Move)
    (coefficient : ℝ) (exponent : ℕ) (minRadius maxRadius : ℝ)
    (L gain baseError candidateError margin : ℝ)
    (coverage : BallMass law candidate center coefficient exponent minRadius maxRadius)
    (radiusLower : minRadius ≤ gain / (2 * L))
    (radiusUpper : gain / (2 * L) ≤ maxRadius)
    (positiveL : 0 < L) (gainNonneg : 0 ≤ gain)
    (continuous : ValueModulus objective (fun radius => L * radius))
    (better : gain ≤ progress objective incumbent center)
    (accurate : ScoreError objective incumbent candidate scores baseError candidateError)
    (safe : baseError + candidateError ≤ margin)
    (separated : baseError + candidateError + margin ≤ gain / 2) :
    coefficient * (gain / (2 * L)) ^ exponent * (gain / 2) ≤
      meanProgress law objective incumbent candidate scores margin := by
  have nonzeroL : L ≠ 0 := ne_of_gt positiveL
  have halfGain : L * (gain / (2 * L)) = gain / 2 := by
    field_simp
  have gapGain : gain - gain / 2 = gain / 2 := by
    ring
  have useful : L * (gain / (2 * L)) ≤ gain := by
    rw [halfGain]
    linarith
  have monotone : Monotone (fun radius : ℝ => L * radius) := by
    intro first second ordered
    exact mul_le_mul_of_nonneg_left ordered (le_of_lt positiveL)
  have bound := scaledBound law objective (fun radius => L * radius)
    incumbent center candidate scores coefficient exponent minRadius maxRadius
    gain (gain / (2 * L)) baseError candidateError margin coverage
    radiusLower radiusUpper continuous monotone better useful accurate safe
    (by simpa [halfGain, gapGain] using separated)
  simpa [halfGain, gapGain] using bound

/-- Reachability inside a trust region exposes a comparison state whose local
proposal coverage yields the same expected progress certificate. -/
theorem reachableBound [Fintype Move] [PseudoMetricSpace State]
    (law : ProposalLaw Move) (objective : Objective State)
    (modulus : ℝ → ℝ) (incumbent : State)
    (candidate : Move → State) (scores : Scores Move)
    (trust gain radius baseError candidateError margin : ℝ)
    (continuous : ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (reachable : Reachable objective incumbent trust gain)
    (useful : modulus radius ≤ gain)
    (accurate : ScoreError objective incumbent candidate scores baseError candidateError)
    (safe : baseError + candidateError ≤ margin)
    (separated : baseError + candidateError + margin ≤ gain - modulus radius) :
    ∃ center ∈ Metric.closedBall incumbent trust,
      nearMass law candidate center radius * (gain - modulus radius) ≤
        meanProgress law objective incumbent candidate scores margin := by
  rcases reachable with ⟨center, within, better⟩
  exact ⟨center, within,
    localBound law objective modulus incumbent center candidate scores gain radius
      baseError candidateError margin continuous monotone better useful accurate safe separated⟩

/-- Two-stage single-evaluation expected progress theorem:
Stage 1 (ENN surrogate selection) chooses `selected seed : Slot` from the procedural
pool `candidate seed : Slot → State` under within-pool surrogate error `surrogateEps`
and selection slack `slack`.
Stage 2 (outer evaluation + commit) evaluates only the single materialized candidate
`candidate seed (selected seed)` with observation errors `baseError`, `candidateError`
and commits when it clears `margin`.
Unlike `seedBound`, no unverified nonnegativity premise on raw un-committed proposals
is required: `baseError + candidateError ≤ margin` protects miss rounds via `commitNonneg`. -/
theorem poolCommit [Fintype Seed] [PseudoMetricSpace State]
    (law : ProposalLaw Seed) (objective : Objective State)
    (modulus : ℝ → ℝ) (incumbent center : State)
    (candidate : Seed → Slot → State)
    (estimate : Seed → Slot → ℝ) (selected : Seed → Slot)
    (scores : Scores Seed)
    (gain radius surrogateEps slack baseError candidateError margin : ℝ)
    (continuous : ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (poolAccurate : ∀ seed, UniformError
      (fun slot => objective (candidate seed slot)) (estimate seed) surrogateEps)
    (chosen : ∀ seed, ApproxMin (estimate seed) (selected seed) slack)
    (evalAccurate : ScoreError objective incumbent
      (fun seed => candidate seed (selected seed)) scores baseError candidateError)
    (safe : baseError + candidateError ≤ margin)
    (separated :
      baseError + candidateError + margin ≤
        gain - modulus radius - (2 * surrogateEps + slack)) :
    poolMass law candidate center radius *
        (gain - modulus radius - (2 * surrogateEps + slack)) ≤
      meanProgress law objective incumbent
        (fun seed => candidate seed (selected seed)) scores margin := by
  classical
  let picked : Seed → State := fun seed => candidate seed (selected seed)
  let netGain : ℝ := gain - modulus radius - (2 * surrogateEps + slack)
  unfold poolMass meanProgress
  rw [Finset.sum_filter, Finset.sum_mul]
  apply Finset.sum_le_sum
  intro seed _
  by_cases hit : PoolHit candidate center radius seed
  · simp only [hit, ite_true]
    have hSelectedGain : netGain ≤ progress objective incumbent (picked seed) :=
      hitSelect objective modulus incumbent center candidate estimate selected
        seed gain radius surrogateEps slack continuous monotone better hit
        (poolAccurate seed) (chosen seed)
    have hCommitted :
        netGain ≤ progress objective incumbent
          (commit incumbent picked scores margin seed) :=
      goodCommit objective incumbent picked scores baseError candidateError
        margin netGain seed evalAccurate hSelectedGain separated
    exact mul_le_mul_of_nonneg_left hCommitted (law.nonnegative seed)
  · simp only [hit, ite_false, zero_mul]
    exact mul_nonneg (law.nonnegative seed)
      (commitNonneg objective incumbent picked scores baseError candidateError
        margin seed evalAccurate safe)

/-- Strictly positive expected committed progress for the two-stage single-evaluation
optimizer transaction whenever `poolMass > 0` and net post-ENN gain strictly exceeds `0`. -/
theorem poolPositive [Fintype Seed] [PseudoMetricSpace State]
    (law : ProposalLaw Seed) (objective : Objective State)
    (modulus : ℝ → ℝ) (incumbent center : State)
    (candidate : Seed → Slot → State)
    (estimate : Seed → Slot → ℝ) (selected : Seed → Slot)
    (scores : Scores Seed)
    (gain radius surrogateEps slack baseError candidateError margin : ℝ)
    (continuous : ValueModulus objective modulus)
    (monotone : Monotone modulus)
    (better : gain ≤ progress objective incumbent center)
    (poolAccurate : ∀ seed, UniformError
      (fun slot => objective (candidate seed slot)) (estimate seed) surrogateEps)
    (chosen : ∀ seed, ApproxMin (estimate seed) (selected seed) slack)
    (evalAccurate : ScoreError objective incumbent
      (fun seed => candidate seed (selected seed)) scores baseError candidateError)
    (safe : baseError + candidateError ≤ margin)
    (separated :
      baseError + candidateError + margin ≤
        gain - modulus radius - (2 * surrogateEps + slack))
    (positiveMass : 0 < poolMass law candidate center radius)
    (positiveNet : 0 < gain - modulus radius - (2 * surrogateEps + slack)) :
    0 < meanProgress law objective incumbent
      (fun seed => candidate seed (selected seed)) scores margin := by
  have hBound := poolCommit law objective modulus incumbent center
    candidate estimate selected scores gain radius surrogateEps slack
    baseError candidateError margin continuous monotone better
    poolAccurate chosen evalAccurate safe separated
  have hProdPos :
      0 < poolMass law candidate center radius *
        (gain - modulus radius - (2 * surrogateEps + slack)) :=
    mul_pos positiveMass positiveNet
  linarith

/-- Shift both incumbent and candidate scores by a common window difficulty
offset `windowShift` (paired evaluation on the same `WorldId`). -/
def shiftScores (scores : Scores Move) (windowShift : ℝ) : Scores Move where
  incumbent := scores.incumbent + windowShift
  candidate := fun move => scores.candidate move + windowShift

/-- Paired evaluation on a shared `WorldId` makes the acceptance predicate
strictly invariant to any additive window difficulty shift `windowShift`. -/
theorem acceptShift (scores : Scores Move) (windowShift margin : ℝ)
    (move : Move) :
    Accept (shiftScores scores windowShift) margin move ↔
      Accept scores margin move := by
  unfold Accept shiftScores
  constructor <;> intro h <;> linarith

/-- Paired evaluation on a shared `WorldId` makes the committed state
strictly invariant to any additive window difficulty shift `windowShift`. -/
theorem commitShift (incumbent : State) (candidate : Move → State)
    (scores : Scores Move) (windowShift margin : ℝ) (move : Move) :
    commit incumbent candidate (shiftScores scores windowShift) margin move =
      commit incumbent candidate scores margin move := by
  unfold commit
  by_cases h : Accept scores margin move
  · have hShift : Accept (shiftScores scores windowShift) margin move :=
      (acceptShift scores windowShift margin move).mpr h
    simp [h, hShift]
  · have hShift : ¬ Accept (shiftScores scores windowShift) margin move :=
      fun hc => h ((acceptShift scores windowShift margin move).mp hc)
    simp [h, hShift]

/-- Expected single-round progress `meanProgress` is strictly invariant to any
common additive window difficulty shift `windowShift` when incumbent and
challenger are paired on the same `WorldId`. -/
theorem progressShift [Fintype Move]
    (law : ProposalLaw Move) (objective : Objective State)
    (incumbent : State) (candidate : Move → State)
    (scores : Scores Move) (windowShift margin : ℝ) :
    meanProgress law objective incumbent candidate
        (shiftScores scores windowShift) margin =
      meanProgress law objective incumbent candidate scores margin := by
  unfold meanProgress
  apply Finset.sum_congr rfl
  intro move _
  rw [commitShift]

#print axioms acceptBound
#print axioms goodAccepted
#print axioms commitNonneg
#print axioms goodCommit
#print axioms meanBound
#print axioms nearGood
#print axioms selectBound
#print axioms hitSelect
#print axioms seedBound
#print axioms massBound
#print axioms localBound
#print axioms hitMass
#print axioms quarterMass
#print axioms slotBound
#print axioms zeroMass
#print axioms scaledBound
#print axioms linearBound
#print axioms reachableBound
#print axioms poolCommit
#print axioms poolPositive
#print axioms acceptShift
#print axioms commitShift
#print axioms progressShift

end
end BauxResearch.Optimization.ObjectiveSearch
